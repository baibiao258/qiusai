#!/usr/bin/env python3
"""
ml_football.py — 泊松 + sklearn 混合足球预测模型
==================================================
你正在学 sklearn，这个脚本就是实战。

工作流:
  1. 从 football-data.org 拉取英超 2023/24 → 特征工程
  2. 训练多个 sklearn 分类器 (逻辑回归 / 随机森林 / XGBoost)
  3. 预测 2024/25 赛季全部 380 场
  4. 对比纯泊松 vs 机器学习

特征工程 (每个样本 = 一场比赛):
  ─ 泊松特征: H_prob, D_prob, A_prob (来自 wc_predictor)
  ─ Elo特征: 主队Elo, 客队Elo, Elo差
  ─ 强度特征: 主队进攻/防守强度, 客队进攻/防守强度
  ─ 近期状态: 近5场积分, 近5场进球/失球, 连胜/连败
  ─ 对战历史: 近3次交锋积分比
  ─ 场次特征: 赛季第几轮, 主队(1)客队(0)

评估: 准确率 / 混淆矩阵 / Brier Score / Log Loss
"""
import json, math, os, urllib.request
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

API_KEY = os.environ.get('FOOTBALL_API_KEY', '5d07c80baa2645d0809b6ec96d6b49c6')
HEADERS = {'X-Auth-Token': API_KEY, 'Accept': 'application/json'}
MAX_GOALS = 6
np.random.seed(42)

# ══════════════════════════════════════════════════
#  1) 数据拉取
# ══════════════════════════════════════════════════

def fetch_season(code, start, end):
    url = f"https://api.football-data.org/v4/competitions/{code}/matches?dateFrom={start}&dateTo={end}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    matches = []
    for m in data.get('matches', []):
        if m['status'] != 'FINISHED': continue
        s = m['score']['fullTime']
        if s['home'] is None: continue
        matches.append({
            'date': m['utcDate'][:10],
            'matchday': m.get('matchday', 1),
            'home': m['homeTeam']['shortName'],
            'away': m['awayTeam']['shortName'],
            'h_score': s['home'], 'a_score': s['away'],
        })
    return matches

print("="*60)
print("  ⚽ sklearn 混合足球预测模型")
print("="*60)

print("\n📡 拉取数据...")
train_raw = fetch_season('PL', '2023-08-01', '2024-05-19')
test_raw = fetch_season('PL', '2024-08-01', '2025-05-25')
print(f"  训练: {len(train_raw)} 场  |  测试: {len(test_raw)} 场")

# ══════════════════════════════════════════════════
#  2) 核心工具函数
# ══════════════════════════════════════════════════

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def elo_expected(ra, rb):
    return 1.0 / (1 + 10 ** ((rb - ra) / 400))

def compute_team_strengths(matches, cutoff, half_life=180):
    """时间衰减的进攻/防守强度"""
    stats = defaultdict(lambda: {'wg':0,'wc':0,'ws':0,'m':0})
    for m in matches:
        if m['date'] >= cutoff: continue
        days = (datetime.strptime(cutoff,'%Y-%m-%d') - datetime.strptime(m['date'],'%Y-%m-%d')).days
        w = 0.5 ** (max(days,0) / half_life)
        for team, gf, ga in [(m['home'],m['h_score'],m['a_score']),(m['away'],m['a_score'],m['h_score'])]:
            s = stats[team]; s['wg']+=gf*w; s['wc']+=ga*w; s['ws']+=w; s['m']+=1
    total_wg = sum(s['wg'] for s in stats.values())
    global_avg = total_wg / max(sum(s['ws'] for s in stats.values()), 1)
    ts = {}
    for team, s in stats.items():
        avg_gf = s['wg']/max(s['ws'],0.001); avg_ga = s['wc']/max(s['ws'],0.001)
        ts[team] = {'att': avg_gf/max(global_avg,0.01), 'def': avg_ga/max(global_avg,0.01)}
    return ts, global_avg

def compute_elo(matches, cutoff):
    elo = defaultdict(lambda: 1500.0)
    for m in matches:
        if m['date'] >= cutoff: continue
        h,a = m['home'],m['away']
        e_h = elo_expected(elo[h], elo[a])
        sh,sa = (1.0,0.0) if m['h_score']>m['a_score'] else ((0.5,0.5) if m['h_score']==m['a_score'] else (0.0,1.0))
        elo[h] += 32*(sh-e_h); elo[a] += 32*(sa-(1-e_h))
    return dict(elo)

# ══════════════════════════════════════════════════
#  3) 特征工程
# ══════════════════════════════════════════════════

def build_features(raw_matches, train_matches, all_matches, fit_mode=True, 
                   prev_elo=None, prev_strength=None, prev_ga=None):
    """
    为 raw_matches 中的每场比赛构建特征向量
    
    返回:
      X: np.array shape (n_matches, n_features)
      y: np.array shape (n_matches,)  — 0=主胜, 1=平局, 2=客胜
      feature_names: list[str]
    """
    cutoff = raw_matches[0]['date']  # 使用第一场比赛日期作为训练截止
    
    # 计算基础参数 (使用截止日期前的所有数据)
    team_strength, global_avg = compute_team_strengths(all_matches, cutoff)
    elo_ratings = compute_elo(all_matches, cutoff)
    
    # 让所有已结束比赛按时间排序，用于构建滚动特征
    # 我们需要对每个 raw_matches 中的比赛，提取截止到该比赛日期之前的信息
    # 所以按日期构建历史记录
    
    # 先建一个按日期索引的比赛表
    all_by_date = defaultdict(list)
    for m in all_matches:
        all_by_date[m['date']].append(m)
    
    all_dates = sorted(all_by_date.keys())
    
    # ⭐ 特征列名 (教育用途: 知道每列是什么)
    feature_names = [
        # 泊松预测 (3)
        'poisson_H', 'poisson_D', 'poisson_A',
        # Elo (3)
        'elo_home', 'elo_away', 'elo_diff',
        # 强度 (4)
        'att_home', 'def_home', 'att_away', 'def_away',
        # 近期状态 (8)
        'form5_home_pts', 'form5_home_gf', 'form5_home_ga',
        'form5_away_pts', 'form5_away_gf', 'form5_away_ga',
        'streak_home', 'streak_away',
        # 对战历史 (1)
        'h2h_ratio',
        # 赛季进度 (1)
        'matchday',
        # 排名 (2)
        'pos_home', 'pos_away',
    ]
    n_features = len(feature_names)
    
    X, y = [], []
    
    for m in raw_matches:
        home, away = m['home'], m['away']
        match_date = m['date']
        matchday = m['matchday']
        actual_h, actual_a = m['h_score'], m['a_score']
        
        # ── a) 泊松概率 ──
        ts_h = team_strength.get(home, {'att':1.0,'def':1.0})
        ts_a = team_strength.get(away, {'att':1.0,'def':1.0})
        lam_h = global_avg * ts_h['att'] * ts_a['def'] * 1.05
        lam_a = global_avg * ts_a['att'] * ts_h['def'] * 0.95
        lam_h = max(0.1,min(5.0,lam_h)); lam_a = max(0.1,min(5.0,lam_a))
        hw,dr,aw = 0.0,0.0,0.0
        for hg in range(MAX_GOALS+1):
            for ag in range(MAX_GOALS+1):
                p = poisson_pmf(hg,lam_h)*poisson_pmf(ag,lam_a)
                if hg>ag: hw+=p
                elif hg==ag: dr+=p
                else: aw+=p
        t=hw+dr+aw; hw,dr,aw=hw/t,dr/t,aw/t
        
        # ── b) Elo (使用比赛前的 Elo) ──
        elo_h = elo_ratings.get(home, 1500)
        elo_a = elo_ratings.get(away, 1500)
        
        # ── c) 近期状态 (使用 match_date 之前的数据) ──
        # 找出该日期前各队最近 N 场比赛
        recent_home = []
        recent_away = []
        for d in all_dates:
            if d >= match_date: break
            for mm in all_by_date[d]:
                if mm['home'] == home or mm['away'] == home:
                    recent_home.append(mm)
                if mm['home'] == away or mm['away'] == away:
                    recent_away.append(mm)
        
        # 近5场积分/进球
        def recent_stats(recent, team, n=5):
            recent = recent[-n:]  # 取最近n场
            pts, gf, ga, streak = 0, 0, 0, 0
            for i, mm in enumerate(recent):
                is_home = mm['home'] == team
                scored = mm['h_score'] if is_home else mm['a_score']
                conceded = mm['a_score'] if is_home else mm['h_score']
                gf += scored; ga += conceded
                if scored > conceded: pts += 3
                elif scored == conceded: pts += 1
                # 连胜/连败 (只看最近一场)
                if i == len(recent)-1:
                    if scored > conceded: streak = 1
                    elif scored < conceded: streak = -1
                    else: streak = 0
            return pts, gf, ga, streak
        
        r5_h = recent_stats(recent_home, home)
        r5_a = recent_stats(recent_away, away)
        
        # ── d) 对战历史 ──
        h2h_ratio = 0.5  # 默认均衡
        h2h_h_pts, h2h_a_pts = 0, 0
        h2h_matches = []
        for d in all_dates:
            if d >= match_date: break
            for mm in all_by_date[d]:
                if (mm['home']==home and mm['away']==away) or (mm['home']==away and mm['away']==home):
                    h2h_matches.append(mm)
        for mm in h2h_matches[-3:]:  # 最近3次交锋
            h_is_home = mm['home']==home
            if (h_is_home and mm['h_score']>mm['a_score']) or (not h_is_home and mm['a_score']>mm['h_score']):
                h2h_h_pts += 3
            elif mm['h_score']==mm['a_score']:
                h2h_h_pts += 1; h2h_a_pts += 1
            else:
                h2h_a_pts += 3
        if h2h_h_pts + h2h_a_pts > 0:
            h2h_ratio = h2h_h_pts / (h2h_h_pts + h2h_a_pts)
        
        # ── e) 排名 (使用赛季中的当前排名，近似) ──
        # 先算积分榜
        league_table = defaultdict(int)
        for d in all_dates:
            if d >= match_date: break
            for mm in all_by_date[d]:
                hm=mm['home']; am=mm['away']; hs=mm['h_score']; az=mm['a_score']
                if hs>az: league_table[hm]+=3
                elif hs==az: league_table[hm]+=1; league_table[am]+=1
                else: league_table[am]+=3
        sorted_teams = sorted(league_table.items(), key=lambda x: x[1], reverse=True)
        pos_map = {t: i+1 for i, (t,_) in enumerate(sorted_teams)}
        pos_h = pos_map.get(home, 15)
        pos_a = pos_map.get(away, 15)
        
        # ── 构建特征向量 ──
        features = [
            hw, dr, aw,                          # 泊松
            elo_h, elo_a, elo_h - elo_a,          # Elo
            ts_h['att'], ts_h['def'],              # 主队强度
            ts_a['att'], ts_a['def'],              # 客队强度
            r5_h[0], r5_h[1], r5_h[2],            # 主队近5场
            r5_a[0], r5_a[1], r5_a[2],            # 客队近5场
            r5_h[3], r5_a[3],                     # 连胜/连败
            h2h_ratio,                             # 对战历史
            matchday,                              # 轮次
            pos_h, pos_a,                          # 排名
        ]
        
        # ── 标签 ──
        if actual_h > actual_a: label = 0  # Home
        elif actual_h == actual_a: label = 1  # Draw
        else: label = 2  # Away
        
        X.append(features)
        y.append(label)
        
        # ── 赛后更新 (滚动) ──
        # 供后续比赛使用
        # (实际上 all_by_date 已经包含了这些比赛, 但因为它们在 cutoff 后, 上面的循环会跳过)
        # 但我们模拟滚动: 将当前比赛加入 all_by_date 供后面的比赛做"近期状态"使用
        if match_date not in all_by_date:
            all_by_date[match_date] = []
        all_by_date[match_date].append(m)
        if match_date not in all_dates:
            all_dates.append(match_date)
            all_dates.sort()
        
        # 更新 Elo 供后续比赛
        e_h = elo_expected(elo_h, elo_a)
        sh, sa = (1.0,0.0) if actual_h>actual_a else ((0.5,0.5) if actual_h==actual_a else (0.0,1.0))
        elo_ratings[home] = elo_h + 48*(sh-e_h)
        elo_ratings[away] = elo_a + 48*(sa-(1-e_h))
        
        # 更新球队强度 (通过重新计算)
        # (简单方法: 用累积法)
    
    X = np.array(X)
    y = np.array(y)
    
    # 处理 NaN
    X = np.nan_to_num(X, nan=0.0)
    
    return X, y, feature_names


# ══════════════════════════════════════════════════
#  4) 训练 & 评估
# ══════════════════════════════════════════════════

print("\n🔧 构建特征...")
print(f"  📋 特征列表 ({22} 个):")
features_demo = [
    "poisson_H/D/A   — 泊松预测概率",
    "elo_home/away   — 球队当前 Elo 评分",
    "elo_diff        — Elo 差",
    "att/def_home    — 主队进攻/防守强度",
    "att/def_away    — 客队进攻/防守强度", 
    "form5_X_pts     — 近5场积分",
    "form5_X_gf/ga   — 近5场进球/失球",
    "streak_home     — 连胜(1)/连败(-1)/平(0)",
    "h2h_ratio       — 近3次交锋得分比",
    "matchday        — 赛季轮次",
    "pos_home/away   — 联赛排名",
]
for f in features_demo:
    print(f"    • {f}")

# 构建训练特征
print("\n  🏗️  训练集 (只使用 2023/24 之前的比赛做特征)...")
X_train, y_train, fnames = build_features(
    train_raw, [], train_raw, fit_mode=True)

# 构建测试特征 (使用全部历史数据)
# build_features 的第二参数传入训练+之前的测试做历史
print("  🏗️  测试集 (使用全部历史数据做特征)...")
all_data = train_raw + test_raw
X_test, y_test, _ = build_features(
    test_raw, [], all_data, fit_mode=False)

n_train = len(X_train)
n_test = len(X_test)
print(f"  📊 训练样本: {n_train}  |  测试样本: {n_test}  |  特征数: {X_train.shape[1]}")

# ── 训练 sklearn 模型 ──
print(f"\n{'='*60}")
print(f"  🧠 训练 sklearn 分类器")
print(f"{'='*60}")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

models = {}

# 1) 逻辑回归 (带 L2 正则)
print("\n  1️⃣  Logistic Regression (L2)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = {'model': lr, 'scaler': scaler}

# 2) 随机森林
print("  2️⃣  Random Forest (200 trees)...")
rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # RF 不需要标准化
models['Random Forest'] = {'model': rf, 'scaler': None}

# 3) 看看有没有 xgboost
try:
    import xgboost as xgb
    print("  3️⃣  XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = {'model': xgb_model, 'scaler': None}
except ImportError:
    print("  3️⃣  XGBoost 未安装, 跳过")

# 4) 基准: 纯泊松
print("  4️⃣  Poisson Baseline (无ML)...")
class PoissonBaseline:
    def predict(self, X):
        # X 的前3列是 poisson_H, poisson_D, poisson_A
        return np.argmax(X[:, :3], axis=1)
    def predict_proba(self, X):
        probs = X[:, :3]
        # 归一化
        row_sums = probs.sum(axis=1)
        return probs / row_sums[:, np.newaxis]
models['Poisson Baseline'] = {'model': PoissonBaseline(), 'scaler': None}


# ══════════════════════════════════════════════════
#  5) 评估
# ══════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  🏆 模型对比 — 英超 2024/25 (380场)")
print(f"{'='*70}")

results = []
for name, cfg in models.items():
    model = cfg['model']
    scaler_m = cfg['scaler']
    
    if scaler_m:
        X_t = scaler_m.transform(X_test)
    else:
        X_t = X_test
    
    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)
    
    acc = accuracy_score(y_test, y_pred)
    
    # Brier score (multi-class: average per-class Brier)
    brier = 0
    for c in range(3):
        y_bin = (y_test == c).astype(int)
        brier += brier_score_loss(y_bin, y_proba[:, c])
    brier /= 3.0
    
    # Log loss
    ll = log_loss(y_test, y_proba)
    
    # 按主胜/平/客胜拆分
    correct_by_result = {0:0, 1:0, 2:0}
    total_by_result = {0:0, 1:0, 2:0}
    for i in range(len(y_test)):
        total_by_result[y_test[i]] += 1
        if y_pred[i] == y_test[i]:
            correct_by_result[y_test[i]] += 1
    
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        'name': name,
        'acc': acc * 100,
        'brier': brier,
        'log_loss': ll,
        'cm': cm,
        'by_result': {k: (correct_by_result[k], total_by_result[k]) for k in range(3)},
    })

# 排序: 准确率降序
results.sort(key=lambda r: r['acc'], reverse=True)

# 打印结果
print(f"\n  {'模型':<28s} {'准确率':>8s} {'Brier':>8s} {'LogLoss':>8s}  {'H正确':>7s} {'D正确':>7s} {'A正确':>7s}")
print(f"  {'─'*75}")
for r in results:
    h_c, h_t = r['by_result'][0]
    d_c, d_t = r['by_result'][1]
    a_c, a_t = r['by_result'][2]
    print(f"  {r['name']:<28s} {r['acc']:>6.2f}%  {r['brier']:>7.4f}  {r['log_loss']:>7.4f}  "
          f"{h_c}/{h_t:>3d}  {d_c}/{d_t:>3d}  {a_c}/{a_t:>3d}")

# 特征重要性 (随机森林)
print(f"\n{'='*70}")
print(f"  🔬 特征重要性 (Random Forest TOP 10)")
print(f"{'='*70}")
rf_model = models['Random Forest']['model']
importances = sorted(zip(fnames, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in importances[:10]:
    bar = '█' * int(imp * 100)
    print(f"  {feat:<22s} {imp*100:>5.2f}%  {bar}")

# 混淆矩阵
print(f"\n{'='*70}")
for r in results:
    if r['name'] == 'Logistic Regression':
        print(f"  📊 Logistic Regression 混淆矩阵:")
        cm = r['cm']
        print(f"  {'':>10s} {'预测主胜':>8s} {'预测平局':>8s} {'预测客胜':>8s}")
        print(f"  {'─'*36}")
        print(f"  {'实际主胜':>10s} {cm[0][0]:>8d} {cm[0][1]:>8d} {cm[0][2]:>8d}")
        print(f"  {'实际平局':>10s} {cm[1][0]:>8d} {cm[1][1]:>8d} {cm[1][2]:>8d}")
        print(f"  {'实际客胜':>10s} {cm[2][0]:>8d} {cm[2][1]:>8d} {cm[2][2]:>8d}")

# 最佳模型的误判分析
print(f"\n{'='*70}")
print(f"  🔥 最佳模型 (Logistic Regression) 的典型误区")
print(f"{'='*70}")
best_name = results[0]['name']
best_cfg = models[best_name]
bm = best_cfg['model']
scaler_b = best_cfg['scaler']
X_t = scaler_b.transform(X_test) if scaler_b else X_test
y_proba = bm.predict_proba(X_t)
y_pred = bm.predict(X_t)

# 找预测最自信但错的比赛
wrong_high_confidence = []
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        confidence = max(y_proba[i])
        wrong_high_confidence.append((i, confidence, test_raw[i]))

wrong_high_confidence.sort(key=lambda x: x[1], reverse=True)

print(f"  预测信心最高但猜错的 5 场:")
for idx, conf, m in wrong_high_confidence[:5]:
    actual_result = ['H','D','A'][y_test[idx]]
    pred_result = ['H','D','A'][y_pred[idx]]
    probs_str = f"H={y_proba[idx][0]:.0f}% D={y_proba[idx][1]:.0f}% A={y_proba[idx][2]:.0f}%"
    print(f"  ❌ {m['home']:<18s} vs {m['away']:<18s}")
    print(f"     预测: {pred_result} ({probs_str})  |  实际: {m['h_score']}-{m['a_score']} ({actual_result})")

print(f"\n{'='*70}")
print(f"  ✅ 完成 — 模型文件: /root/ml_football.py")
print(f"  最佳模型: {results[0]['name']} ({results[0]['acc']:.2f}%)")
print(f"  比纯泊松提升: {results[0]['acc'] - results[-1]['acc']:.2f}%")
print(f"{'='*70}")
