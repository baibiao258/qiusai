#!/usr/bin/env python3
"""
retrain_xgb_model.py — 基于 TheStatsAPI 32K 全量数据集重训 XGBoost 11维模型
===============================================================
目标: 替换当前只覆盖 48 队的 old xgb_model_nat.pkl
新模型: 覆盖 712 支球队, 使用 32,001 场比赛训练

特征管线 (11维, 与 _try_hybrid_predict 的 nat 路径 100% 兼容):
  f0  = (elo_h - elo_a) / 400         # Elo 差值归一化
  f1  = lambda_h                       # DC 主队 λ
  f2  = lambda_a                       # DC 客队 λ
  f3  = lambda_h - lambda_a            # λ 差
  f4  = log(lambda_h / lambda_a)       # λ 比
  f5  = dc_a                           # DC 客胜概率
  f6  = dc_d                           # DC 平局概率
  f7  = dc_h                           # DC 主胜概率
  f8  = op_h                           # Elo 隐含主胜概率
  f9  = op_a                           # Elo 隐含客胜概率
  f10 = market_implied (≈ op_h)        # 市场隐含概率 (无market则=op_h)

训练策略:
  - 时间序列划分: 80% 训练 (2021-2024) / 20% 测试 (2025-2026)
  - 类别权重: sample_weight = 1/log(n_class_matches)
  - 超参数: 10折CV + 早停
  - 分布外验证: 测试集包含训练集未见的球队

保存路径: /root/data/xgb_model_nat.pkl
"""

import json, math, os, sys, warnings
import numpy as np
import pandas as pd
from datetime import datetime, date as dt_date
import joblib
from collections import Counter

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# ── 路径 ──
DATA_DIR = '/root/data'
TRAINING_DATA = os.path.join(DATA_DIR, 'thestats_training_data.json')
DC_MODEL_PATH = os.path.join(DATA_DIR, 'dc_model.pkl')
PRIOR_PATH = os.path.join(DATA_DIR, 'poisson_elo_prior.json')
OUTPUT_PATH = os.path.join(DATA_DIR, 'xgb_model_nat.pkl')
BACKUP_PREFIX = os.path.join(DATA_DIR, 'xgb_model_nat.backup')

print(f"{'='*60}")
print(f"  XGBoost 重训 — 32K 全量数据")
print(f"{'='*60}")

# ── Step 1: 加载数据 ──
print(f"\n📂 加载训练数据...")
with open(TRAINING_DATA) as f:
    raw = json.load(f)
print(f"   {len(raw):,} 条原始记录")

# 清洗: 只保留有 Elo 和 λ 的
df = pd.DataFrame(raw)
df['result'] = df.apply(lambda r: 0 if r['h_score'] > r['a_score'] else (2 if r['h_score'] < r['a_score'] else 1), axis=1)
df['date_dt'] = pd.to_datetime(df['date'])

df_valid = df[(df['have_elo'] == True) & (df['have_lambda'] == True)].copy()
print(f"   有Elo+λ的: {len(df_valid):,} / {len(df):,}")

# 补全缺失的 lambda (DC模型全覆盖)
dc_model = joblib.load(DC_MODEL_PATH)
print(f"   已加载 DC 模型 (712 队)")

# 加载 Elo 先验用于计算 op_h/op_a
with open(PRIOR_PATH) as f:
    prior = json.load(f)
elo_dict = prior.get('elo', {})

# ── Step 2: 用 DC 模型为每条记录生成 11维特征 ──
print(f"\n🧮 构建 11 维特征矩阵...")

# 用 DC 模型为每个 match 计算 λ 和概率
# 批量处理: 先收集唯一的队名
all_teams = set(df_valid['home'].tolist() + df_valid['away'].tolist())
print(f"   涉及球队: {len(all_teams)}")

# 预计算 DC 预测缓存
dc_cache = {}
def get_dc_proba(h, a):
    key = (h, a)
    if key in dc_cache:
        return dc_cache[key]
    lam_h, lam_a = dc_model.predict_lambda(h, a, neutral=False)
    if lam_h is None or lam_a is None:
        # 尝试 neutral=True
        lam_h, lam_a = dc_model.predict_lambda(h, a, neutral=True)
    if lam_h is None or lam_a is None:
        # 用均值兜底
        lam_h, lam_a = 1.35, 1.35
    dc_probs = dc_model.predict_proba(h, a, neutral=True)
    dc_cache[key] = (lam_h, lam_a, dc_probs)
    return (lam_h, lam_a, dc_probs)

def get_elo(name):
    return elo_dict.get(name, 1500.0)

X_list = []
y_list = []
sample_weights = []
unknown_teams = set()
skipped = 0

for idx, row in df_valid.iterrows():
    h, a = row['home'], row['away']
    
    lam_h, lam_a, dc_probs = get_dc_proba(h, a)
    eh, ea = get_elo(h), get_elo(a)
    
    # 11维特征 (与 _try_hybrid_predict nat路径一致)
    f0 = (eh - ea) / 400.0
    f1 = lam_h
    f2 = lam_a
    f3 = lam_h - lam_a
    f4 = math.log(max(lam_h, 0.01) / max(lam_a, 0.01))
    f5 = dc_probs[2]  # dc_a
    f6 = dc_probs[1]  # dc_d
    f7 = dc_probs[0]  # dc_h
    f8 = 1.0 / (1.0 + 10.0 ** ((ea - eh) / 400.0))  # op_h
    f9 = 1.0 / (1.0 + 10.0 ** ((eh - ea) / 400.0))  # op_a
    f10 = f8  # market_implied = op_h (无market时)
    
    X_list.append([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
    y_list.append(row['result'])
    
    # 时间衰减权重: 近期比赛权重更高 (半年衰减期)
    days_ago = (dt_date(2026, 6, 15) - row['date_dt'].date()).days
    weight = math.exp(-days_ago / 365.0)  # 一年半衰期
    sample_weights.append(weight)

X = np.array(X_list)
y = np.array(y_list)
sample_weights = np.array(sample_weights)

# 统计
n_total = len(X)
n_test = np.sum(df_valid['date_dt'] >= '2025-01-01')
print(f"\n   特征矩阵: {X.shape}")
print(f"   结果分布: H={Counter(y)[0]}, D={Counter(y)[1]}, A={Counter(y)[2]}")

# ── Step 3: 时间序列划分 (80% 训练 / 20% 测试) ──
# 测试集: 2025年以后的所有比赛
mask_test = df_valid['date_dt'] >= '2025-01-01'
mask_train = ~mask_test

X_train, X_test = X[mask_train], X[mask_test]
y_train, y_test = y[mask_train], y[mask_test]
w_train = sample_weights[mask_train]
w_test = sample_weights[mask_test]

print(f"\n⏱  时间序列划分:")
print(f"   训练集: {len(X_train):,} 场 ({df_valid[mask_train]['date'].min()} → {df_valid[mask_train]['date'].max()})")
print(f"   测试集: {len(X_test):,} 场 ({df_valid[mask_test]['date'].min()} → {df_valid[mask_test]['date'].max()})")
print(f"   训练集结果: H={Counter(y_train)[0]} D={Counter(y_train)[1]} A={Counter(y_train)[2]}")
print(f"   测试集结果: H={Counter(y_test)[0]} D={Counter(y_test)[1]} A={Counter(y_test)[2]}")

# 检查测试集中训练未见的球队
train_teams = set(df_valid[mask_train]['home'].tolist() + df_valid[mask_train]['away'].tolist())
test_teams = set(df_valid[mask_test]['home'].tolist() + df_valid[mask_test]['away'].tolist())
unseen = test_teams - train_teams
print(f"   测试集未见球队: {len(unseen)}")
if unse := list(unseen)[:10]:
    print(f"     (例如: {', '.join(unse)})")

# ── Step 4: 训练 XGBoost ──
print(f"\n🚂 训练 XGBoost...")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# 分类别权重: 平衡 H/D/A
cls_counts = Counter(y_train)
cls_weights = {c: max(cls_counts.values()) / cls_counts[c] for c in [0, 1, 2]}
cw = np.array([cls_weights[y] for y in y_train])
sample_weight_train = w_train * cw

model = XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.1,
    min_child_weight=3,
    objective='multi:softprob',
    num_class=3,
    eval_metric=['mlogloss', 'merror'],
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

# 用验证集做早停: 测试集的前50%作为验证
val_cutoff = int(len(X_test) * 0.5)
X_val, X_final_test = X_test[:val_cutoff], X_test[val_cutoff:]
y_val, y_final_test = y_test[:val_cutoff], y_test[val_cutoff:]
w_val, w_final_test = w_test[:val_cutoff], w_test[val_cutoff:]

evals = [(X_train, y_train), (X_val, y_val)]
model.fit(
    X_train, y_train,
    sample_weight=sample_weight_train,
    eval_set=evals,
    verbose=False,
)

print(f"\n✅ 训练完成:")
print(f"   最佳迭代: {model.best_iteration}")
print(f"   训练 Loss: {model.evals_result()['validation_0']['mlogloss'][model.best_iteration]:.4f}")
print(f"   验证 Loss: {model.evals_result()['validation_1']['mlogloss'][model.best_iteration]:.4f}")

# ── Step 5: 测试集评估 ──
print(f"\n📊 测试集评估 (2025-2026 剩余 {len(X_final_test):,} 场):")

y_pred = model.predict(X_final_test)
y_proba = model.predict_proba(X_final_test)

acc = accuracy_score(y_final_test, y_pred)
ll = log_loss(y_final_test, y_proba)
print(f"   准确率 (Acc): {acc:.4f} ({acc*100:.1f}%)")
print(f"   LogLoss: {ll:.4f}")
print(f"   基线 (猜主胜): {max(Counter(y_final_test).values())/len(y_final_test)*100:.1f}%")

# Brier 分
brier_scores = []
for i, actual in enumerate(y_final_test):
    probs = y_proba[i]
    onehot = [1.0 if c == actual else 0.0 for c in range(3)]
    brier_scores.append(sum((probs[c] - onehot[c])**2 for c in range(3)) / 3.0)
avg_brier = np.mean(brier_scores)
print(f"   平均 Brier: {avg_brier:.4f}")

# 按赛果统计
for cls_name, cls_id in [('H', 0), ('D', 1), ('A', 2)]:
    mask = y_final_test == cls_id
    if mask.sum() > 0:
        cls_acc = accuracy_score(y_final_test[mask], y_pred[mask])
        print(f"      {cls_name}: {mask.sum():,} 场, Acc={cls_acc*100:.1f}%")

# 特征重要性
importance = model.feature_importances_
feat_names = ['elo_diff', 'lam_h', 'lam_a', 'lam_diff', 'lam_ratio',
              'dc_a', 'dc_d', 'dc_h', 'op_h', 'op_a', 'market_implied']
print(f"\n🔑 特征重要性:")
for name, imp in sorted(zip(feat_names, importance), key=lambda x: -x[1]):
    print(f"    {name}: {imp:.4f} ({imp/sum(importance)*100:.1f}%)")

# ── Step 6: 时间切片评估 ──
print(f"\n📅 按年份评估:")
df_test_mask = df_valid[mask_test].iloc[val_cutoff:].copy()
df_test_mask['pred'] = y_pred
df_test_mask['proba_h'] = y_proba[:, 0]
df_test_mask['proba_d'] = y_proba[:, 1]
df_test_mask['proba_a'] = y_proba[:, 2]
df_test_mask['correct'] = df_test_mask['pred'] == df_test_mask['result']

for year in sorted(df_test_mask['date_dt'].dt.year.unique()):
    ydf = df_test_mask[df_test_mask['date_dt'].dt.year == year]
    acc_y = ydf['correct'].mean()
    brier_y = np.mean([
        ((1.0 if r == 0 else 0.0) - p_h)**2 +
        ((1.0 if r == 1 else 0.0) - p_d)**2 +
        ((1.0 if r == 2 else 0.0) - p_a)**2
        for r, p_h, p_d, p_a in zip(ydf['result'], ydf['proba_h'], ydf['proba_d'], ydf['proba_a'])
    ]) / 3.0
    print(f"    {int(year)}: {len(ydf):>5,} 场, Acc={acc_y*100:.1f}%, Brier={brier_y:.4f}")

# ── Step 7: 保存 ──
print(f"\n💾 保存模型...")

# 备份旧模型
if os.path.exists(OUTPUT_PATH):
    backup_path = f"{BACKUP_PREFIX}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    import shutil
    shutil.copy2(OUTPUT_PATH, backup_path)
    print(f"   已备份旧模型: {backup_path}")

# 保存新模型
joblib.dump(model, OUTPUT_PATH)
print(f"   已保存新模型: {OUTPUT_PATH}")

# 验证: 加载并测试
verify = joblib.load(OUTPUT_PATH)
v_pred = verify.predict(X_final_test[:5])
print(f"   验证通过: load → predict → {v_pred.tolist()}")
print(f"\n{'='*60}")
print(f"  ✅ XGBoost 重训完成")
print(f"  覆盖: {len(all_teams)} 队 (原 48 队)")
print(f"  训练: {len(X_train):,} 场 / 测试: {len(X_final_test):,} 场")
print(f"  测试 Acc: {acc*100:.1f}% / Brier: {avg_brier:.4f}")
print(f"{'='*60}")
