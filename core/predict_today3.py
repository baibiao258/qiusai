#!/usr/bin/env python3
"""竞彩今日3场预测 — 胜平负/让球/半全场/比分/总进球 (90分钟口径)"""
import sys, os, json, math
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')

import numpy as np
from scipy.stats import poisson
import joblib
from datetime import datetime

from wc_2026_phase1 import *
from mc_uncertainty_helper import jitter_prob, summarize_probs
from mc_market_weight_helper import market_weight_for_match

DATA_DIR = '/root/data'
MAX_GOALS = 6
DC_WEIGHT = 0.4
XGB_WEIGHT = 0.6
MARKET_WEIGHT = 0.40
MODEL_WEIGHT = 1.0 - MARKET_WEIGHT

OPTUNA_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.03218571685398262,
    'n_estimators': 369,
    'reg_alpha': 3.0540401601028355,
    'reg_lambda': 2.694513099210833,
    'colsample_bytree': 0.4500553009276969,
    'subsample': 0.6426882590232543,
    'min_child_weight': 8.22712251093365,
}

def log(s=""): print(s, flush=True)

xgb_model = (
    joblib.load(os.path.join(DATA_DIR, 'xgb_model_29.pkl'))
    if os.path.exists(os.path.join(DATA_DIR, 'xgb_model_29.pkl'))
    else joblib.load(os.path.join(DATA_DIR, 'xgb_model_20_3.pkl'))
)
dc = joblib.load(os.path.join(DATA_DIR, 'dc_model.pkl'))
elo = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))
if isinstance(elo, dict):
    elo = {k: float(v) for k, v in elo.items()}

market_path = os.path.join(DATA_DIR, 'theodds_api_data.json')
winner_odds = {}
if os.path.exists(market_path):
    with open(market_path) as f:
        md = json.load(f)
    winner_odds = md.get('winner_odds', {})

NAME_MAP = {
    '克罗地亚': 'Croatia', '比利时': 'Belgium',
    '格鲁吉亚': 'Georgia', '罗马尼亚': 'Romania',
    '威尔士': 'Wales', '加纳': 'Ghana',
}

def pred_match(home_en, away_en):
    dc_p = dc.predict_proba(home_en, away_en, neutral=True)
    lam_h, lam_a = dc.predict_lambda(home_en, away_en, neutral=True)
    lam_h = max(0.1, min(5.0, lam_h or 1.0))
    lam_a = max(0.1, min(5.0, lam_a or 1.0))
    eh = elo.get(home_en, 1500); ea = elo.get(away_en, 1500)
    op = make_odds_from_elo(eh, ea)
    feat = np.array([[
        (eh-ea)/400, lam_h, lam_a, lam_h-lam_a,
        math.log(max(lam_h,0.01)/max(lam_a,0.01)),
        dc_p[0], dc_p[1], dc_p[2],
        0.5, 0.5, 0, 0, 0, 0, 1,
        0.0, 1, 0, 0.0, 0.0,
        op[0], op[1], op[2]
    ]])
    xgb_p = xgb_model.predict_proba(feat)[0]
    hybrid = DC_WEIGHT * np.array([dc_p[2], dc_p[1], dc_p[0]]) + XGB_WEIGHT * xgb_p
    return {
        'home_en': home_en, 'away_en': away_en,
        'hyb_away': float(hybrid[0]), 'hyb_draw': float(hybrid[1]), 'hyb_home': float(hybrid[2]),
        'lam_h': lam_h, 'lam_a': lam_a,
    }

# ═══ 批量MC (numpy向量化) ═══
def mc_htft(lam_h, lam_a, n=20000, rho=None):
    """向量化半全场MC"""
    if rho is None:
        rho = float(dc.rho_) if hasattr(dc, 'rho_') else 0.25
    rng = np.random.default_rng(42)
    
    # 半场进球 (截断到MAX_GOALS)
    ht_h = rng.poisson(lam_h * 0.45, n).clip(0, MAX_GOALS)
    ht_a = rng.poisson(lam_a * 0.45, n).clip(0, MAX_GOALS)
    
    # Dixon-Coles低比分修正
    low_mask = ((ht_h == 0) & (ht_a == 0)) | ((ht_h == 1) & (ht_a == 0)) | \
               ((ht_h == 0) & (ht_a == 1)) | ((ht_h == 1) & (ht_a == 1))
    adj = 1.0 - rho * ht_h * ht_a / (MAX_GOALS + 1)**2
    swap = rng.random(n) > adj
    swap_low = swap & low_mask
    ht_h = np.where((swap_low) & (ht_h > 0), ht_h - 1, ht_h)
    ht_a = np.where((swap_low) & (ht_a > 0), ht_a - 1, ht_a)
    
    # 全场进球 (半场起累计)
    ft_h = ht_h + rng.poisson(lam_h * 0.55, n).clip(0, MAX_GOALS)
    ft_a = ht_a + rng.poisson(lam_a * 0.55, n).clip(0, MAX_GOALS)
    
    labels = ['HH','HD','HA','DH','DD','DA','AH','AD','AA']
    counts = {}
    for l in labels:
        counts[l] = 0
    
    for i in range(n):
        ht_l = 'H' if ht_h[i] > ht_a[i] else ('D' if ht_h[i] == ht_a[i] else 'A')
        ft_l = 'H' if ft_h[i] > ft_a[i] else ('D' if ft_h[i] == ft_a[i] else 'A')
        counts[ht_l + ft_l] += 1
    
    probs = {l: counts[l] / n for l in labels}
    top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
    return probs, top3

def mc_scoreline(lam_h, lam_a, n=50000):
    """向量化比分MC"""
    rng = np.random.default_rng(42)
    hg = rng.poisson(lam_h, n).clip(0, MAX_GOALS)
    ag = rng.poisson(lam_a, n).clip(0, MAX_GOALS)
    unique, counts = np.unique(list(zip(hg, ag)), axis=0, return_counts=True)
    pairs = [(f"{u[0]}-{u[1]}", int(c)) for u, c in zip(unique, counts)]
    pairs.sort(key=lambda x: -x[1])
    return pairs[:5]

def mc_totalgoals(lam_h, lam_a, n=50000):
    """总进球数MC"""
    rng = np.random.default_rng(42)
    hg = rng.poisson(lam_h, n).clip(0, MAX_GOALS)
    ag = rng.poisson(lam_a, n).clip(0, MAX_GOALS)
    tg = hg + ag
    tg = np.clip(tg, 0, 7)
    unique, counts = np.unique(tg, return_counts=True)
    probs = {int(k): int(v)/n for k, v in zip(unique, counts)}
    sorted_tg = sorted(probs.items(), key=lambda x: -x[1])
    
    under2 = sum(probs.get(i, 0) for i in range(3))
    mid = sum(probs.get(i, 0) for i in range(3, 5))
    over4 = sum(probs.get(i, 0) for i in range(5, 8))
    
    return {
        'probs': probs, 'sorted': sorted_tg,
        'under2': under2, 'mid': mid, 'over4': over4,
        'lambda_total': lam_h + lam_a,
    }

# ═══ Run ═══
matches_cn = [
    ('克罗地亚', '比利时'),
    ('格鲁吉亚', '罗马尼亚'),
    ('威尔士', '加纳'),
]

results = []
for home_cn, away_cn in matches_cn:
    home = NAME_MAP.get(home_cn, home_cn)
    away = NAME_MAP.get(away_cn, away_cn)
    pred = pred_match(home, away)
    
    hp, dp, ap = pred['hyb_home'], pred['hyb_draw'], pred['hyb_away']
    pick_name = max([('主胜', hp), ('平局', dp), ('客胜', ap)], key=lambda x: x[1])[0]
    
    # 让球MC
    rng = np.random.default_rng(99)
    hg = rng.poisson(pred['lam_h'], 10000).clip(0, MAX_GOALS)
    ag = rng.poisson(pred['lam_a'], 10000).clip(0, MAX_GOALS)
    diff = hg - ag
    rq_home = np.sum(diff > 1) / 10000 + np.sum(diff == 1) / 20000
    rq_away = 1 - rq_home
    rq_pick = '主让胜' if rq_home > 0.5 else '客让胜'
    
    # 半全场
    htft_probs, htft_top3 = mc_htft(pred['lam_h'], pred['lam_a'])
    
    # 比分
    score_top3 = mc_scoreline(pred['lam_h'], pred['lam_a'])
    
    # 总进球
    tg = mc_totalgoals(pred['lam_h'], pred['lam_a'])
    
    results.append((home_cn, away_cn, pick_name, hp, dp, ap, rq_home, rq_away, rq_pick, htft_top3, score_top3, tg, pred))

# ═══ Output ═══
print(f"\n{'='*75}")
print(f" 🏟  2026-06-02 竞彩足球预测报告 (90分钟口径)")
print(f"{'='*75}")
print(f" 模型: DC×{DC_WEIGHT} + XGB×{XGB_WEIGHT} | 20+3黄金 | MC 50K")
print(f"    克罗地亚#{13} vs 比利时#{8}")
print(f"    格鲁吉亚#{68} vs 罗马尼亚#{38}")
print(f"    威尔士#{29} vs 加纳#{77}")
print(f"{'='*75}")

CN_1X2 = {'主胜': '胜', '平局': '平', '客胜': '负'}
CN_HTFT = {'HH':'胜胜','HD':'胜平','HA':'胜负','DH':'平胜','DD':'平平','DA':'平负','AH':'负胜','AD':'负平','AA':'负负'}

for home_cn, away_cn, pick_name, hp, dp, ap, rq_home, rq_away, rq_pick, htft_top3, score_top3, tg, pred in results:
    print(f"\n{'─'*75}")
    print(f" 📋 周二 {home_cn} VS {away_cn}")
    print(f" Elo: {pred['home_en']}={elo.get(pred['home_en'],1500):.0f} vs {pred['away_en']}={elo.get(pred['away_en'],1500):.0f} | 让球: -1")
    print(f"{'─'*75}")
    
    # 1. 胜平负
    print(f"\n  【1】胜平负")
    print(f"  主胜: {hp*100:>5.1f}%  |  平局: {dp*100:>5.1f}%  |  客胜: {ap*100:>5.1f}%")
    pick_cn = CN_1X2[pick_name]
    print(f"  ✅ 预测: {pick_cn} ({pick_name} {max(hp,dp,ap)*100:.1f}%)")
    
    # 2. 让球
    print(f"\n  【2】让球 -1")
    print(f"  主让胜: {rq_home*100:>5.1f}%  |  客让胜: {rq_away*100:>5.1f}%")
    print(f"  ✅ 预测: {rq_pick}")
    
    # 3. 半全场
    print(f"\n  【3】半全场")
    for label, prob in htft_top3:
        cn = CN_HTFT[label]
        print(f"  {cn}({label}): {prob*100:>5.1f}%")
    
    # 4. 比分
    print(f"\n  【4】比分 Top3")
    for score, cnt in score_top3[:3]:
        print(f"  {score}: {cnt/50000*100:.1f}%")
    
    # 5. 总进球
    print(f"\n  【5】总进球 (λ={tg['lambda_total']:.2f})")
    print(f"  小球(0-2): {tg['under2']*100:>5.1f}%  |  中球(3-4): {tg['mid']*100:>5.1f}%  |  大球(5+): {tg['over4']*100:>5.1f}%")
    print(f"  Top: ", end='')
    for n, p in tg['sorted'][:4]:
        print(f"{n}球{p*100:.0f}% ", end='')
    print()

print(f"\n{'='*75}")
print(f" ⚠️ 仅代表模型预测，不构成投注建议。")
print(f"    结算口径: 90分钟常规时间(含伤停补时)，不含加时/点球。")
print(f"{'='*75}")
