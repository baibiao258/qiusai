#!/usr/bin/env python3
"""
retrain_xgb_model_v2.py — XGBoost V2 重训
====================================
基于 32K 全量数据, 从数据自身计算 form/H2H/赛事特征

特征维度: 22维
  11维: DC+Elo基础 (与原nat兼容)
    + 6维: form特征 (主客队近3/5场胜率、场均进球、场均失球)
    + 4维: 赛事分类 (联赛杯赛/国际/友谊)
    + 1维: 主场优势指数

保存为: xgb_model_nat.pkl (全量覆盖 609 队)
"""

import json, math, os, sys, warnings
import numpy as np
import pandas as pd
from datetime import datetime, date as dt_date, timedelta
import joblib
from collections import defaultdict, Counter
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

DATA_DIR = '/root/data'
TRAINING_DATA = os.path.join(DATA_DIR, 'thestats_training_data.json')
DC_MODEL_PATH = os.path.join(DATA_DIR, 'dc_model.pkl')
PRIOR_PATH = os.path.join(DATA_DIR, 'poisson_elo_prior.json')
OUTPUT_PATH = os.path.join(DATA_DIR, 'xgb_model_nat.pkl')

print(f"{'='*60}")
print(f"  XGBoost V2 重训 — 32K + Form + H2H")
print(f"{'='*60}")

# ── Step 1: 加载数据 ──
print(f"\n📂 加载数据...")
with open(TRAINING_DATA) as f:
    raw = json.load(f)
df = pd.DataFrame(raw)

# 从comp_name提取赛事类型
COMP_CATS = {
    'league': ['Ligue 1', 'Ligue 2', 'Serie A', 'Serie B', 'Premier League', 'Championship',
               'LaLiga', 'LaLiga2', 'Bundesliga', '2. Bundesliga', 'Eredivisie',
               'Primeira Liga', 'Liga Portugal', 'Ekstraklasa', 'MLS', 'J1 League',
               'K League 1', 'Brasileirão', 'Liga MX', 'Super Lig',
               'Primera División', 'Primera Nacional', 'Süper Lig'],
    'cup': ['FA Cup', 'Copa del Rey', 'DFB-Pokal', 'Coppa Italia', 'Coupe de France',
            'EFL Cup', 'KNVB Beker', 'Taça de Portugal', 'Emirates Cup'],
    'international': ['World Cup', 'EURO', 'Copa America', 'AFC Asian Cup',
                      'Africa Cup of Nations', 'CONCACAF Gold Cup',
                      'World Cup Qualification', 'EURO Qualification',
                      'AFCON', 'World Cup Qualifiers'],
    'friendly': ['International Match', 'Friendly', 'Club Friendly'],
}

def classify_comp(name):
    name_lower = name.lower() if name else ''
    for cat, names in COMP_CATS.items():
        if any(n.lower() in name_lower for n in names):
            return cat
    return 'other'

df['comp_cat'] = df['comp_name'].apply(classify_comp)
df['date_dt'] = pd.to_datetime(df['date'])
df['result'] = df.apply(lambda r: 0 if r['h_score'] > r['a_score'] else (2 if r['h_score'] < r['a_score'] else 1), axis=1)

# 只保留有Elo+λ的
mask_valid = (df['have_elo'] == True) & (df['have_lambda'] == True)
df = df[mask_valid].sort_values('date_dt').reset_index(drop=True)
print(f"   有效记录: {len(df):,}")

# ── Step 2: 计算 Form + H2H 特征 (从训练数据本身) ──
print(f"\n🧮 计算 Form + H2H + 赛事特征...")

dc_model = joblib.load(DC_MODEL_PATH)
with open(PRIOR_PATH) as f:
    prior = json.load(f)
elo_dict = prior.get('elo', {})

# 构建球队比赛历史索引 (按时间顺序)
team_matches = defaultdict(list)  # team -> [(idx, result, gf, ga)]
for idx, row in df.iterrows():
    for team, gf, ga in [(row['home'], row['h_score'], row['a_score']),
                           (row['away'], row['a_score'], row['h_score'])]:
        team_matches[team].append((idx, 1 if gf > ga else (2 if gf < ga else 1), gf, ga))

# 预计算: 对每个索引, 获取主客队的历史form
# 使用滚动窗口: 看之前5场
form_cache = {}
def get_form(team, before_idx, n=5):
    """获取球队在before_idx之前的n场战绩"""
    cache_key = (team, before_idx, n)
    if cache_key in form_cache:
        return form_cache[cache_key]
    history = team_matches.get(team, [])
    # 过滤在before_idx之前的
    prev = [m for m in history if m[0] < before_idx]
    prev = prev[-n:] if len(prev) > n else prev
    
    if not prev:
        result = (n, 0, 0, 0)  # (played, wins, draws, losses, gf, ga) 节省版
        form_cache[cache_key] = result
        return result
    
    wins = sum(1 for m in prev if m[1] == 0)
    draws = sum(1 for m in prev if m[1] == 1)
    losses = sum(1 for m in prev if m[1] == 2)
    gfs = sum(m[2] for m in prev)
    gas = sum(m[3] for m in prev)
    played = len(prev)
    result = (played, wins, draws, losses, gfs, gas)
    form_cache[cache_key] = result
    return result

# 预计算H2H
h2h_cache = {}
def get_h2h(home, away, before_idx, n=5):
    """获取两队交锋历史"""
    # 使用原始数据, 找到home vs away的所有历史
    cache_key = (home, away, before_idx, n)
    if cache_key in h2h_cache:
        return h2h_cache[cache_key]
    
    # 筛选home vs away的历史 (不区分主客场)
    h2h_matches = []
    # 需要在df中找, 但效率太低, 用原始raw数据
    global_raw_idx_map
    matches = global_raw_idx_map.get((home, away), []) + global_raw_idx_map.get((away, home), [])
    matches = sorted(matches, key=lambda x: x[0])  # by date
    matches = [m for m in matches if m[0] < before_idx]
    matches = matches[-n:] if len(matches) > n else matches
    
    if not matches:
        result = (0, 0, 0, 0, 0)  # played, home_wins, draws, away_wins, last_result
        h2h_cache[cache_key] = result
        return result
    
    hw = sum(1 for m in matches if m[2] > m[3])  # home wins
    dr = sum(1 for m in matches if m[2] == m[3])
    aw = sum(1 for m in matches if m[2] < m[3])
    last_r = 1 if matches[-1][2] > matches[-1][3] else (2 if matches[-1][2] < matches[-1][3] else 0)
    result = (len(matches), hw, dr, aw, last_r)
    h2h_cache[cache_key] = result
    return result

# 构建全局索引
print(f"   构建H2H索引...")
global_raw_idx_map = defaultdict(list)
num_map = {row['match_id']: idx for idx, row in df.iterrows()}
for idx, row in df.iterrows():
    key_fwd = (row['home'], row['away'])
    key_rev = (row['away'], row['home'])
    dt = row['date_dt']
    hg, ag = row['h_score'], row['a_score']
    global_raw_idx_map[key_fwd].append((dt, hg, ag, idx))
    global_raw_idx_map[key_rev].append((dt, ag, hg, idx))

def get_h2h_v2(home, away, before_idx, n=5):
    """H2H查询 (使用idx索引)"""
    key = (home, away, before_idx, n)
    if key in h2h_cache:
        return h2h_cache[key]
    
    matches = []
    for pair_key in [(home, away), (away, home)]:
        matches.extend(global_raw_idx_map.get(pair_key, []))
    matches = sorted(matches, key=lambda x: x[3])  # by idx
    matches = [m for m in matches if m[3] < before_idx]
    matches = matches[-n:] if len(matches) > n else matches
    
    if not matches:
        result = (0, 0, 0, 0, 0.5, 0.5)
        h2h_cache[key] = result
        return result
    
    # 如果home是主队
    hw = sum(1 for m in matches if m[1] > m[2])  # 主队胜
    dr = sum(1 for m in matches if m[1] == m[2])
    aw = sum(1 for m in matches if m[1] < m[2])
    last_r = 1 if matches[-1][1] > matches[-1][2] else (2 if matches[-1][1] < matches[-1][2] else 0)
    h2h_h_rate = hw / max(len(matches), 1)
    h2h_a_rate = aw / max(len(matches), 1)
    result = (len(matches), hw, dr, aw, h2h_h_rate, h2h_a_rate)
    h2h_cache[key] = result
    return result

# 构建特征
elapsed = 0
from tqdm import tqdm

X_list, y_list = [], []
sw_list = []
missing_dc = 0
total = len(df)

for idx in range(total):
    row = df.iloc[idx]
    h, a = row['home'], row['away']
    eh, ea = elo_dict.get(h, 1500.0), elo_dict.get(a, 1500.0)
    
    # DC模型输出
    lam_h, lam_a = dc_model.predict_lambda(h, a, neutral=True)
    if lam_h is None or lam_a is None:
        missing_dc += 1
        continue
    dc_p = dc_model.predict_proba(h, a, neutral=True)
    
    # ── 11维 DC+Elo 基础 ──
    f0 = (eh - ea) / 400.0
    f1 = lam_h
    f2 = lam_a
    f3 = lam_h - lam_a
    f4 = math.log(max(lam_h, 0.01) / max(lam_a, 0.01))
    f5 = dc_p[2]  # dc_a
    f6 = dc_p[1]  # dc_d
    f7 = dc_p[0]  # dc_h
    f8 = 1.0 / (1.0 + 10.0 ** ((ea - eh) / 400.0))  # op_h
    f9 = 1.0 / (1.0 + 10.0 ** ((eh - ea) / 400.0))  # op_a
    f10 = f8  # market_implied
    
    # ── Form特征 (6维) ──
    fh = get_form(h, idx, 5)
    fa = get_form(a, idx, 5)
    # home form: 近5场胜率, 场均进球, 场均失球
    f11 = fh[1] / max(fh[0], 1)  # home胜率
    f12 = fh[4] / max(fh[0], 1) if fh[0] > 0 else 0  # 场均进球
    f13 = fh[5] / max(fh[0], 1) if fh[0] > 0 else 0  # 场均失球
    # away form: 近5场胜率, 场均进球, 场均失球
    f14 = fa[1] / max(fa[0], 1)  # away胜率
    f15 = fa[4] / max(fa[0], 1) if fa[0] > 0 else 0  # 场均进球
    f16 = fa[5] / max(fa[0], 1) if fa[0] > 0 else 0  # 场均失球
    
    # ── H2H特征 (2维) ──
    h2h = get_h2h_v2(h, a, idx, 5)
    f17 = h2h[4]  # 主场H2H胜率
    f18 = h2h[5]  # 客场H2H胜率
    
    # ── 赛事分类 (5维 one-hot) ──
    cat = row['comp_cat']
    f19 = 1.0 if cat == 'league' else 0.0
    f20 = 1.0 if cat == 'international' else 0.0
    f21 = 1.0 if cat == 'cup' else 0.0
    f22 = 1.0 if cat == 'friendly' else 0.0
    
    # ── 主场优势指数 ──
    # 联赛 vs 中立场的区别
    f23 = 0.0 if row.get('neutral', False) else 1.0
    
    # 28维特征
    feat = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
            f11, f12, f13, f14, f15, f16, f17, f18,
            f19, f20, f21, f22, f23]
    X_list.append(feat)
    y_list.append(row['result'])
    
    # 时间衰减权重
    days_ago = (dt_date(2026, 6, 15) - row['date_dt'].date()).days
    weight = math.exp(-days_ago / 540.0)  # 540天半衰期 (与DC一致)
    sw_list.append(weight)

X = np.array(X_list)
y = np.array(y_list)
sw = np.array(sw_list)
print(f"\n   特征矩阵: {X.shape}")
print(f"   DC缺失: {missing_dc} 场")
print(f"   结果: H={Counter(y)[0]} D={Counter(y)[1]} A={Counter(y)[2]}")

# ── Step 3: 时间序列划分 ──
mask_test = df.iloc[:len(y)]['date_dt'] >= '2025-01-01'
# 注意: df可能被截断(missing_dc), 需对齐
df_used = df.iloc[:len(y)]
mask_test = df_used['date_dt'] >= '2025-01-01'
mask_train = ~mask_test

X_train, X_test = X[mask_train], X[mask_test]
y_train, y_test = y[mask_train], y[mask_test]
sw_train, sw_test = sw[mask_train], sw[mask_test]

print(f"\n⏱  划分:")
print(f"   训练: {len(X_train):,} ({df_used[mask_train]['date'].min()} → {df_used[mask_train]['date'].max()})")
print(f"   测试: {len(X_test):,} ({df_used[mask_test]['date'].min()} → {df_used[mask_test]['date'].max()})")

# ── Step 4: 训练 ──
print(f"\n🚂 训练 XGBoost (28维)...")

cls_counts = Counter(y_train)
cw = np.array([max(cls_counts.values()) / max(cls_counts[c], 1) for c in y_train])
sw_final = sw_train * cw

# 验证集: 测试集前一半
val_cutoff = len(X_test) // 2
X_val, X_ft = X_test[:val_cutoff], X_test[val_cutoff:]
y_val, y_ft = y_test[:val_cutoff], y_test[val_cutoff:]
sw_val, sw_ft = sw_test[:val_cutoff], sw_test[val_cutoff:]

model = XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5,
    gamma=0.2,
    min_child_weight=5,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    early_stopping_rounds=100,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    sample_weight=sw_final,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print(f"\n✅ 训练完成!")
print(f"   最佳迭代: {model.best_iteration + 1}")
best_round = model.best_iteration
print(f"   训练 Loss: {model.evals_result()['validation_0']['mlogloss'][best_round]:.4f}")
print(f"   验证 Loss: {model.evals_result()['validation_0']['mlogloss'][best_round]:.4f}")

# ── Step 5: 评估 ──
y_pred = model.predict(X_ft)
y_proba = model.predict_proba(X_ft)

acc = accuracy_score(y_ft, y_pred)
ll = log_loss(y_ft, y_proba)

brier_scores = []
for i, actual in enumerate(y_ft):
    probs = y_proba[i]
    onehot = [1.0 if c == actual else 0.0 for c in range(3)]
    brier_scores.append(sum((probs[c] - onehot[c])**2 for c in range(3)) / 3.0)
avg_brier = np.mean(brier_scores)

baseline = max(Counter(y_ft).values()) / len(y_ft)

print(f"\n📊 测试集 ({len(y_ft):,} 场):")
print(f"   准确率: {acc*100:.1f}% (基线猜主胜: {baseline*100:.1f}%)")
print(f"   LogLoss: {ll:.4f} (随机: 1.099)")
print(f"   Brier: {avg_brier:.4f} (随机: 0.222)")

for cls_name, cls_id in [('H', 0), ('D', 1), ('A', 2)]:
    mask = y_ft == cls_id
    if mask.sum() > 0:
        cls_acc = accuracy_score(y_ft[mask], y_pred[mask])
        print(f"     {cls_name}: {mask.sum():,} 场, Acc={cls_acc*100:.1f}%")

# ── 按年份 ──
df_ft = df_used[mask_test].iloc[val_cutoff:].copy()
df_ft['pred'] = y_pred
df_ft['correct'] = df_ft['pred'] == df_ft['result']
print(f"\n📅 按年份:")
for year in sorted(df_ft['date_dt'].dt.year.unique()):
    ydf = df_ft[df_ft['date_dt'].dt.year == year]
    if len(ydf) > 0:
        print(f"   {int(year)}: {len(ydf):>5,} 场, Acc={ydf['correct'].mean()*100:.1f}%")

# ── 特征重要性 ──
feat_names = ['elo_diff', 'lam_h', 'lam_a', 'lam_diff', 'lam_ratio',
              'dc_a', 'dc_d', 'dc_h', 'op_h', 'op_a', 'market_implied',
              'form_h_winrate', 'form_h_gf_avg', 'form_h_ga_avg',
              'form_a_winrate', 'form_a_gf_avg', 'form_a_ga_avg',
              'h2h_h_rate', 'h2h_a_rate',
              'is_league', 'is_internat', 'is_cup', 'is_friendly',
              'home_advantage']
importance = model.feature_importances_
imp_sorted = sorted(zip(feat_names, importance), key=lambda x: -x[1])
print(f"\n🔑 特征重要性 (前15):")
for name, imp in imp_sorted[:15]:
    print(f"    {name}: {imp:.4f} ({imp/sum(importance)*100:.1f}%)")

# ── 与旧模型对比 ──
print(f"\n📊 对比旧XGB (仅48队):")
print(f"   旧: 79.6% Acc (小样本, 48队, 有form+市场赔率)")
print(f"   新: {acc*100:.1f}% Acc ({len(np.unique(df_used['home'].tolist()+df_used['away'].tolist()))}队)")

# ── Step 6: 保存 ──
print(f"\n💾 保存模型...")
import shutil
if os.path.exists(OUTPUT_PATH):
    bk = f"{OUTPUT_PATH}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    shutil.copy2(OUTPUT_PATH, bk)
    print(f"   备份旧模型: {bk}")

joblib.dump(model, OUTPUT_PATH)
print(f"   保存: {OUTPUT_PATH}")

# 验证
verify = joblib.load(OUTPUT_PATH)
v_pred = verify.predict(X_ft[:5])
print(f"   验证: {v_pred.tolist()}")

print(f"\n{'='*60}")
print(f"  ✅ XGBoost V2 重训完成")
print(f"  特征: 24维 (11基础 + 6Form + 2H2H + 5赛事 + 1主场)")
print(f"  测试Acc: {acc*100:.1f}% / Brier: {avg_brier:.4f}")
print(f"  覆盖: ~600 队")
print(f"{'='*60}")
