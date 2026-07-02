#!/usr/bin/env python3
"""
retrain_xgb_v3.py — XGBoost V3: 深层训练 + 表单特征
========================================================
修正 V1 的最佳迭代=1 问题 (训练崩溃).
策略: 用完整 2000 棵树, 不早停, 后期监督验证集.

特征: 17维
  - 11维 (DC+Elo 基础)
  - 6维 (form胜率/场均进球/失球 × 主客队)
"""

import json, math, os, warnings, sys
import numpy as np
import joblib
from datetime import datetime, date as dt_date
from collections import defaultdict, Counter
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings('ignore')
DATA_DIR = '/root/data'

print(f"{'='*60}")
print(f"  XGBoost V3 — 深层训练 (17维)")
print(f"{'='*60}")

# 加载
with open(f'{DATA_DIR}/thestats_training_data.json') as f:
    raw = json.load(f)
print(f"   原始记录: {len(raw):,}")

df = [r for r in raw if r.get('have_elo') and r.get('have_lambda')]
print(f"   有效记录 (有Elo+λ): {len(df):,}")

dc_model = joblib.load(f'{DATA_DIR}/dc_model.pkl')
prior = json.load(open(f'{DATA_DIR}/poisson_elo_prior.json'))
elo_dict = prior.get('elo', {})

# 构建球队比赛索引 (用于form计算)
team_matches = defaultdict(list)
for idx, r in enumerate(df):
    for team, gf, ga in [(r['home'], r['h_score'], r['a_score']),
                          (r['away'], r['a_score'], r['h_score'])]:
        team_matches[team].append((idx, 1 if gf > ga else (2 if gf < ga else 1), gf, ga))

def get_form(team, before_idx, n=5):
    """获取球队在before_idx之前的n场战绩"""
    prev = [m for m in team_matches.get(team, []) if m[0] < before_idx]
    prev = prev[-n:] if len(prev) > n else prev
    if not prev:
        return (0, 0.333, 0, 0)  # played, win_rate, avg_gf, avg_ga
    wins = sum(1 for m in prev if m[1] == 0)
    draws = sum(1 for m in prev if m[1] == 1)
    gfs = sum(m[2] for m in prev)
    gas = sum(m[3] for m in prev)
    played = len(prev)
    wr = (wins + draws * 0.5) / played
    return (played, wr, gfs / played, gas / played)

# 构建特征
X_list, y_list, sw_list = [], [], []
missing = 0
target_date = dt_date(2026, 6, 15)

for idx, r in enumerate(df):
    h, a = r['home'], r['away']
    eh, ea = elo_dict.get(h, 1500.0), elo_dict.get(a, 1500.0)
    
    # DC model outputs
    lam_h, lam_a = dc_model.predict_lambda(h, a, neutral=True)
    if lam_h is None or lam_a is None:
        missing += 1
        continue
    dc_p = dc_model.predict_proba(h, a, neutral=True)
    
    # 11-dim base
    f = [(eh - ea) / 400.0,                 # f0: elo_diff
         lam_h,                             # f1: lam_h
         lam_a,                             # f2: lam_a
         lam_h - lam_a,                     # f3: lam_diff
         math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),  # f4: lam_ratio
         dc_p[2],                            # f5: dc_a
         dc_p[1],                            # f6: dc_d
         dc_p[0],                            # f7: dc_h
         1.0 / (1.0 + 10 ** ((ea - eh) / 400.0)),  # f8: op_h
         1.0 / (1.0 + 10 ** ((eh - ea) / 400.0)),  # f9: op_a
         1.0 / (1.0 + 10 ** ((ea - eh) / 400.0)),  # f10: market_implied (=op_h)
    ]
    
    # form features (6-dim)
    fh = get_form(h, idx, 5)
    fa = get_form(a, idx, 5)
    f += [fh[1], fh[2], fh[3], fa[1], fa[2], fa[3]]
    
    X_list.append(f)
    
    # target
    hg, ag = r['h_score'], r['a_score']
    y_list.append(0 if hg > ag else (2 if hg < ag else 1))
    
    # time decay weight
    date_obj = datetime.strptime(r['date'], '%Y-%m-%d').date()
    days_ago = (target_date - date_obj).days
    sw_list.append(math.exp(-days_ago / 540.0))

X = np.array(X_list); y = np.array(y_list); sw = np.array(sw_list)
print(f"   特征矩阵: {X.shape}, DC缺失: {missing}")

# 时间划分: 80/20
dates = [datetime.strptime(r['date'], '%Y-%m-%d').date() for r in df if not (dc_model.predict_lambda(r['home'], r['away'], neutral=True)[0] is None)]
# 重新对齐: 跳过missing的记录
valid_indices = [idx for idx, r in enumerate(df) if not (dc_model.predict_lambda(r['home'], r['away'], neutral=True)[0] is None)]
all_dates = [datetime.strptime(df[i]['date'], '%Y-%m-%d').date() for i in valid_indices]
cutoff_date = dt_date(2025, 1, 1)
test_mask = np.array([d >= cutoff_date for d in all_dates])
train_mask = ~test_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
sw_train, sw_test = sw[train_mask], sw[test_mask]

# 类别平衡权重
cls_counts = Counter(y_train)
cw = np.array([max(cls_counts.values()) / max(cls_counts[c], 1) for c in y_train])
sample_weight = sw_train * cw

print(f"\n⏱  训练集: {len(X_train):,} | 测试集: {len(X_test):,}")
print(f"   训练: {Counter(y_train)} | 测试: {Counter(y_test)}")
baseline = Counter(y_test).most_common(1)[0][1] / len(y_test)
print(f"   基线猜主胜: {baseline*100:.1f}%")

# ── 训练 (无早期停止, 完整2000棵) ──
print(f"\n🚂 训练 XGBoost (17维, 2000 trees)...")
model = XGBClassifier(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_alpha=0.3,
    reg_lambda=1.0,
    gamma=0.2,
    min_child_weight=5,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)

# 评估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_proba)

# Brier
brier = np.mean([sum((p - (1.0 if c == actual else 0.0))**2 for c, p in enumerate(probs)) / 3.0 
                  for actual, probs in zip(y_test, y_proba)])

# LogLoss per class
ll_h = -np.mean([np.log(max(p[0], 1e-10)) for actual, p in zip(y_test, y_proba) if actual == 0])
ll_d = -np.mean([np.log(max(p[1], 1e-10)) for actual, p in zip(y_test, y_proba) if actual == 1])
ll_a = -np.mean([np.log(max(p[2], 1e-10)) for actual, p in zip(y_test, y_proba) if actual == 2])

print(f"\n📊 测试集 ({len(X_test):,} 场):")
print(f"   准确率 (Acc): {acc*100:.1f}% vs 基线 {baseline*100:.1f}%")
print(f"   LogLoss: {ll:.4f} (随机: 1.099)")
print(f"   Brier: {brier:.4f} (随机: 0.222)")

# 按赛果
for name, cid in [('H',0), ('D',1), ('A',2)]:
    mask = y_test == cid
    if mask.sum() > 0:
        ca = (y_pred[mask] == cid).mean()
        print(f"     {name}: {mask.sum():,}场, Acc={ca*100:.1f}%, LogLoss={[ll_h, ll_d, ll_a][cid]:.4f}")

# 特征重要性
names17 = ['elo_diff', 'lam_h', 'lam_a', 'lam_diff', 'lam_ratio',
           'dc_a', 'dc_d', 'dc_h', 'op_h', 'op_a', 'market_implied',
           'form_h_wr', 'form_h_gf', 'form_h_ga', 'form_a_wr', 'form_a_gf', 'form_a_ga']
imp = model.feature_importances_
print(f"\n🔑 特征重要性:")
for n, i in sorted(zip(names17, imp), key=lambda x: -x[1])[:10]:
    print(f"    {n}: {i:.4f}")

# 保存
print(f"\n💾 保存为 xgb_model_17d.pkl ...")
joblib.dump(model, f'{DATA_DIR}/xgb_model_17d.pkl')

# 验证加载
v = joblib.load(f'{DATA_DIR}/xgb_model_17d.pkl')
print(f"   验证: {v.predict(X_test[:3]).tolist()}")

# 与旧 nat 模型对比 (在同样测试集上)
old_nat = joblib.load(f'{DATA_DIR}/xgb_model_nat.pkl')
try:
    old_pred = old_nat.predict(X_test)
    old_acc = (old_pred == y_test).mean()
    print(f"\n📊 对比:")
    print(f"   V3新模型 (17维, {X.shape[1]}队): Acc={acc*100:.1f}%")
    print(f"   旧nat模型 (11维, 48队): Acc={old_acc*100:.1f}%")
except:
    # 维度可能不同
    print(f"   旧nat模型: 维度不兼容, 跳过对比")

print(f"\n{'='*60}")
print(f"  ✅ XGBoost V3 完成")
print(f"{'='*60}")
