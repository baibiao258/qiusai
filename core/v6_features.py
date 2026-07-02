"""
v6: v3 + 三个新特征
- rolling_window: 滚动时间窗 (赛前 N 场)
- phase_factor: 阶段因子 (小组赛 vs 淘汰赛)
- dynamic_elo: 锦标赛内动态 Elo (K 因子随阶段变化)
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/root')
from v3_backtest import build_team_features, load_match_data, run_backtest_v3, baseline_v3_accuracy

# 加载数据
matches, elo_long = load_match_data()
print(f"Loaded {len(matches)} matches, {len(elo_long)} elo records")

# 跑 v3 baseline
v3_acc, v3_preds = baseline_v3_accuracy(matches, elo_long)
print(f"\nv3 baseline accuracy: {v3_acc:.4f}")

# === v6_a: 滚动时间窗 ===
# 用近 N=5 场历史, 替换全场平均
def build_features_v6a(matches_df, elo_long_df, N=5):
    """v3 + 滚动时间窗 (近 N 场)"""
    matches_df = matches_df.copy().sort_values('date').reset_index(drop=True)
    feats = []
    for idx, row in matches_df.iterrows():
        date = row['date']
        home, away = row['home_team'], row['away_team']
        # 历史: 截止 date, 各队近 N 场
        home_hist = elo_long_df[(elo_long_df['team']==home) & (elo_long_df['date']<date)].tail(N)
        away_hist = elo_long_df[(elo_long_df['team']==away) & (elo_long_df['date']<date)].tail(N)
        # 当前 Elo
        home_elo = home_hist['elo'].iloc[-1] if len(home_hist) else 1500
        away_elo = away_hist['elo'].iloc[-1] if len(away_hist) else 1500
        # 近 N 场均
        home_gf = home_hist['gf'].mean() if len(home_hist) else 1.3
        home_ga = home_hist['ga'].mean() if len(home_hist) else 1.3
        away_gf = away_hist['gf'].mean() if len(away_hist) else 1.3
        away_ga = away_hist['ga'].mean() if len(away_hist) else 1.3
        # DC λ
        home_xg = (home_gf + away_ga) / 2
        away_xg = (away_gf + home_ga) / 2
        from scipy.stats import poisson
        pH = 1 - poisson.cdf(0, home_xg) * poisson.cdf(0, away_xg)
        pD = sum(poisson.pmf(i, home_xg) * poisson.pmf(i, away_xg) for i in range(0, 8))
        pA = 1 - pH - pD
        # Elo prob
        elo_diff = home_elo - away_elo + 50
        pEloH = 1 / (1 + 10**(-elo_diff/400))
        feats.append({
            'match_idx': idx, 'home_team': home, 'away_team': away, 'date': date,
            'phase': row.get('phase', 'group'),
            'pH_dc': pH, 'pD_dc': pD, 'pA_dc': pA,
            'pEloH': pEloH,
            'home_elo': home_elo, 'away_elo': away_elo,
        })
    return pd.DataFrame(feats)

# === v6_b: 阶段因子 ===
def build_features_v6b(matches_df, elo_long_df):
    """v3 + 阶段因子 (group/knockout)"""
    base = build_features_v6a(matches_df, elo_long_df)  # 用滚动窗
    # 阶段: group=0, R16=1, QF=2, SF=3, Final=4
    phase_map = {'group': 0, 'R16': 1, 'QF': 2, 'SF': 3, 'Final': 4}
    base['phase_id'] = base['phase'].map(phase_map).fillna(0).astype(int)
    return base

# === v6_c: 锦标赛内动态 Elo K 因子 ===
# 复盘: 小组赛 K=30, 淘汰赛 K=60
# 重写 Elo 训练
def compute_elo_v6c(matches_df, K_group=30, K_ko=60):
    """动态 K 因子"""
    matches_df = matches_df.copy().sort_values('date').reset_index(drop=True)
    elo = {}  # team -> elo
    history = []
    for idx, row in matches_df.iterrows():
        h, a = row['home_team'], row['away_team']
        if h not in elo: elo[h] = 1500
        if a not in elo: elo[a] = 1500
        eh, ea = elo[h], elo[a]
        # 期望
        sh = 1 / (1 + 10**(-(eh-ea+50)/400))
        # 实际 (90min)
        hg, ag = row['home_score'], row['away_score']
        if hg > ag: rh = 1
        elif hg == ag: rh = 0.5
        else: rh = 0
        # K 因子
        K = K_ko if row.get('phase', 'group') != 'group' else K_group
        # 更新
        elo[h] += K * (rh - sh)
        elo[a] += K * ((1-rh) - (1-sh))
        # 记录
        history.append({'date': row['date'], 'team': h, 'elo': eh, 'gf': hg, 'ga': ag})
        history.append({'date': row['date'], 'team': a, 'elo': ea, 'gf': ag, 'ga': hg})
    return pd.DataFrame(history)

# === 跑 10 届回测 ===
def run_v6_backtest(version='a', K_group=30, K_ko=60):
    """跑 v6 a/b/c 任一版本"""
    wc_years = [1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
    all_acc = []
    for year in wc_years:
        train = matches[matches['year'] < year].copy()
        test = matches[matches['year'] == year].copy()
        if len(test) == 0: continue
        # 重算 Elo
        if version == 'c':
            elo_hist = compute_elo_v6c(train, K_group, K_ko)
        else:
            elo_hist = compute_elo_v6c(train, 30, 30)  # 普通 K=30

        if version == 'a':
            feats = build_features_v6a(test, elo_hist, N=5)
        elif version == 'b':
            feats = build_features_v6b(test, elo_hist)
        else:  # c
            feats = build_features_v6a(test, elo_hist, N=5)  # 同样用滚动窗

        # 加 phase_id (用 one-hot)
        if version == 'b':
            for p in ['group', 'R16', 'QF', 'SF', 'Final']:
                col = f'is_{p}'
                feats[col] = (feats['phase'] == p).astype(int)
        elif version == 'c':
            feats['is_ko'] = (feats['phase'] != 'group').astype(int)

        # 合并标签
        test_reset = test.reset_index(drop=True)
        feats['y'] = test_reset['home_score'] - test_reset['away_score']
        feats['y_class'] = feats['y'].apply(lambda x: 0 if x > 0 else (1 if x == 0 else 2))

        # 简化: 用 pH_dc / pD_dc / pA_dc + pEloH + (可选) phase one-hot
        if version == 'b':
            X = feats[['pH_dc', 'pD_dc', 'pA_dc', 'pEloH', 'is_R16', 'is_QF', 'is_SF', 'is_Final']].fillna(0).values
        elif version == 'c':
            X = feats[['pH_dc', 'pD_dc', 'pA_dc', 'pEloH', 'is_ko']].fillna(0).values
        else:
            X = feats[['pH_dc', 'pD_dc', 'pA_dc', 'pEloH']].fillna(0).values
        y = feats['y_class'].values

        # 预测: argmax(pH, pD, pA)
        preds = np.argmax(X[:, :3], axis=1)
        acc = (preds == y).mean()
        all_acc.append(acc)
    return np.mean(all_acc), all_acc


if __name__ == "__main__":
    print("\n=== v6_a (滚动时间窗 N=5) ===")
    acc_a, acc_list_a = run_v6_backtest('a')
    print(f"v6_a acc: {acc_a:.4f}")

    print("\n=== v6_b (阶段因子) ===")
    acc_b, acc_list_b = run_v6_backtest('b')
    print(f"v6_b acc: {acc_b:.4f}")

    print("\n=== v6_c (动态 Elo K) ===")
    # 尝试不同 K 组合
    for kg, kk in [(20, 50), (30, 60), (30, 80), (40, 60)]:
        acc_c, _ = run_v6_backtest('c', K_group=kg, K_ko=kk)
        print(f"  K_group={kg}, K_ko={kk}: acc={acc_c:.4f}")

    print(f"\nv3 baseline: {v3_acc:.4f}")
