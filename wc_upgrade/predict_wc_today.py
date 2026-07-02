#!/usr/bin/env python3
"""predict_wc_today.py — 世界杯首日预测 (v28 + 双DC + OddsAPI)"""
import json, os, sys, math, numpy as np, joblib
sys.path.insert(0, '/root')
DATA_DIR = '/root/data'

# 加载模型
print("📡 加载模型...")
xgb = joblib.load(f'{DATA_DIR}/xgb_model_28.pkl')
dc = joblib.load(f'{DATA_DIR}/dc_model.pkl')
dc_club = joblib.load(f'{DATA_DIR}/dc_club.pkl')
elo = joblib.load(f'{DATA_DIR}/elo_ratings.pkl')
wc_odds = json.load(open(f'{DATA_DIR}/wc_2026_odds_today.json'))

def get_dc_probs(home, away):
    """双DC回退"""
    for model, neutral in [(dc, True), (dc_club, False)]:
        try:
            lam_h, lam_a = model.predict_lambda(home, away, neutral=neutral)
            if lam_h is None: continue
            return model.predict_proba(home, away, neutral=neutral), lam_h, lam_a
        except: continue
    return None, None, None

def predict_hybrid(home, away, market_implied):
    """v28 + DC融合"""
    elo_h = elo.get(home, 1500)
    elo_a = elo.get(away, 1500)
    dc_probs = np.array([1/3, 1/3, 1/3])
    lam_h = lam_a = 1.5
    
    dc_r = get_dc_probs(home, away)
    if dc_r[0] is not None:
        dc_pred, lam_h, lam_a = dc_r
        # dc_pred = [ph, pd, pa] from predict_proba
        # 转成 [A, D, H] 顺序以匹配XGBoost标签
        dc_probs = np.array([dc_pred[2], dc_pred[1], dc_pred[0]])
    
    # Winsorize
    dc_probs = np.clip(dc_probs, 0.01, 0.99)
    lam_h = max(0.1, min(5.0, lam_h))
    lam_a = max(0.1, min(5.0, lam_a))
    
    op_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
    op_a = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))
    
    f28 = np.array([[
        (elo_h - elo_a) / 400, lam_h, lam_a, lam_h - lam_a,
        math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
        dc_probs[0], dc_probs[1], dc_probs[2],  # A,D,H (匹配训练时的顺序)
        op_h, op_a, market_implied
    ]])
    
    xgb_p = xgb.predict_proba(f28)[0]  # [A, D, H]
    
    # 简单融合: 30% DC + 70% XGBoost
    hybrid = 0.3 * dc_probs + 0.7 * xgb_p
    hybrid /= hybrid.sum()
    
    return hybrid  # [H, D, A] = [胜, 平, 负]

# 过滤今天的比赛
today = '2026-06-14'
today_matches = [m for m in wc_odds if m['date'] == today]

print(f"\n{'='*90}")
print(f"  🌍 世界杯 {today} 战预测 (v28+双DC+市场赔率)")
print(f"{'='*90}")

results = []
for m in today_matches:
    home, away = m['home_en'], m['away_en']
    market_prob = m['market_implied_prob']
    
    hybrid = predict_hybrid(home, away, market_prob)
    h_prob, d_prob, a_prob = hybrid
    
    # DC标签
    dc_r = get_dc_probs(home, away)
    dc_label = f'dc={dc_r[0][0]:.0f}/{dc_r[0][1]:.0f}/{dc_r[0][2]:.0f}' if dc_r[0] is not None else 'dc=—'
    
    results.append({
        'home': home, 'away': away,
        'odds': (m['odds_h'], m['odds_d'], m['odds_a']),
        'probs': hybrid,
        'market_h': market_prob,
        'dc_label': dc_label,
    })

# 按主胜概率从高到低排序
results.sort(key=lambda r: -r['probs'][0])

print(f"\n{'主队':>20} {'客队':>20} {'主胜':>7} {'平':>7} {'客胜':>7} "
      f"{'赔率主':>6} {'赔率平':>6} {'赔率客':>6} {'市场H':>6} {'DC':>20}")
print('-' * 110)
for r in results:
    h, d, a = r['probs']
    oh, od, oa = r['odds']
    print(f'{r["home"][:18]:>20} {r["away"][:18]:>20} '
          f'{h*100:>5.1f}% {d*100:>5.1f}% {a*100:>5.1f}% '
          f'{oh:>6.2f} {od:>6.2f} {oa:>6.2f} {r["market_h"]*100:>5.1f}% '
          f'{r["dc_label"]:>20}')

# 价值信号
print(f"\n{'='*90}")
print(f"  价值信号 (模型 vs 市场分歧最大)")
print(f"{'='*90}")
print(f"{'比赛':>42} {'模型H':>7} {'市场H':>7} {'偏差':>7} {'赔率':>6} {'EV':>8}")
print('-' * 80)
for r in sorted(results, key=lambda x: -abs(x['probs'][0] - x['market_h']))[:10]:
    diff = r['probs'][0] - r['market_h']
    ev = r['probs'][0] * r['odds'][0] - 1
    label = f'{r["home"][:18]:>18} vs {r["away"][:18]:<20}'
    print(f'{label} {r["probs"][0]*100:>6.1f}% {r["market_h"]*100:>6.1f}% '
          f'{diff*100:>+6.1f}% {r["odds"][0]:>6.2f} {ev:>+7.3f}')
