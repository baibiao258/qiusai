#!/usr/bin/env python3
"""预测今日3场国际赛：胜平负/让球/半全场/比分/总进球"""
import sys, os, json, math, random
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')

import numpy as np
from scipy.stats import poisson
import joblib
from datetime import datetime

from wc_2026_phase1 import *
from mc_uncertainty_helper import jitter_prob, summarize_probs
from mc_market_weight_helper import market_weight_for_match

# 与 wc_2026_final.py 同步的东道主加成
from wc_2026_final import HOST_TEAMS, HOST_BONUS_BY_TEAM

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

# Load models
log(" 📦 加载模型...")
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

def make_cdf(lam):
    cdf = []; s = 0
    for k in range(MAX_GOALS + 1):
        s += poisson.pmf(k, lam)
        cdf.append(s)
    return cdf

def predict_match(home, away, neutral=True, host_bonus=0.0):
    """单场预测
    host_bonus>0 时: neutral=False, DC+XGB 都用提升后的 λ
    """
    is_host = host_bonus > 0 and home in HOST_TEAMS and not neutral
    dc_p = dc.predict_proba(home, away, neutral=neutral, host_bonus=host_bonus if is_host else 0.0)
    lam_h, lam_a = dc.predict_lambda(home, away, neutral=neutral, host_bonus=host_bonus if is_host else 0.0)
    lam_h = max(0.1, min(5.0, lam_h))
    lam_a = max(0.1, min(5.0, lam_a))
    
    eh_elo = elo.get(home, 1500); ea_elo = elo.get(away, 1500)
    op = make_odds_from_elo(eh_elo, ea_elo)
    fh5 = [0.5, 0.0, 0.0, 0.0]; fa5 = [0.5, 0.0, 0.0, 0.0]
    b15 = [
        (eh_elo - ea_elo) / 400, lam_h, lam_a, lam_h - lam_a,
        math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
        dc_p[0], dc_p[1], dc_p[2],
        fh5[0], fa5[0],
        fh5[1] - fa5[2], fa5[1] - fh5[2],
        fh5[1] - fa5[1], fh5[0] - fa5[0],
        0 if is_host else 1,  # neutral flag
    ]
    gold = [0.0, 1, 0, 0.0, 0.0]
    odds_feat = [op[0], op[1], op[2]]
    feat = np.array([b15 + gold + odds_feat])
    xgb_p = xgb_model.predict_proba(feat)[0]
    
    dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])
    hybrid = DC_WEIGHT * dc_ado + XGB_WEIGHT * xgb_p
    
    mh = winner_odds.get(home, 0)
    ma = winner_odds.get(away, 0)
    if mh > 0 and ma > 0:
        mw = MARKET_WEIGHT
        blended_h = hybrid[2] * MODEL_WEIGHT + (1/mh) / (1/mh + 1/ma + 0.01) * mw
        blended_a = hybrid[0] * MODEL_WEIGHT + (1/ma) / (1/mh + 1/ma + 0.01) * mw
        blended_d = max(0, 1 - blended_h - blended_a)
        hybrid = np.array([blended_a, blended_d, blended_h])
    
    return {
        'home': home, 'away': away,
        'dc_away': float(dc_p[0]), 'dc_draw': float(dc_p[1]), 'dc_home': float(dc_p[2]),
        'xgb_away': float(xgb_p[0]), 'xgb_draw': float(xgb_p[1]), 'xgb_home': float(xgb_p[2]),
        'hyb_away': float(hybrid[0]), 'hyb_draw': float(hybrid[1]), 'hyb_home': float(hybrid[2]),
        'lam_h': lam_h, 'lam_a': lam_a,
        'elo_h': eh_elo, 'elo_a': ea_elo,
        'cdf_h': make_cdf(lam_h), 'cdf_a': make_cdf(lam_a),
    }

def sim_htft(pred, n_sims=20000):
    """模拟半全场9宫格"""
    h, a = pred['home'], pred['away']
    cdf_h, cdf_a = pred['cdf_h'], pred['cdf_a']
    r_ht = 0.45  # 默认半场比例
    
    labels = ['HH','HD','HA','DH','DD','DA','AH','AD','AA']
    counts = {l: 0 for l in labels}
    
    rng = random.Random(42)
    
    # DC rho 低比分修正
    rho = float(dc.rho_) if hasattr(dc, 'rho_') else 0.25
    
    for _ in range(n_sims):
        # 半场
        hg_ht = 0; ag_ht = 0
        for k in range(MAX_GOALS + 1):
            if rng.random() <= cdf_h[k]:
                hg_ht = k; break
        for k in range(MAX_GOALS + 1):
            if rng.random() <= cdf_a[k]:
                ag_ht = k; break
        
        # Dixon-Coles低比分修正
        if (hg_ht == 0 and ag_ht == 0) or (hg_ht == 1 and ag_ht == 0) or \
           (hg_ht == 0 and ag_ht == 1) or (hg_ht == 1 and ag_ht == 1):
            adj = (1 - rho * hg_ht * ag_ht / (MAX_GOALS + 1)**2)
            if rng.random() > adj:
                # swap goal to lower bin
                if hg_ht > 0 and ag_ht > 0:
                    hg_ht -= 1; ag_ht -= 1
                elif hg_ht > 0:
                    hg_ht -= 1
                elif ag_ht > 0:
                    ag_ht -= 1
        
        # 全场
        hg_ft = hg_ht; ag_ft = ag_ht
        for k in range(hg_ht, MAX_GOALS + 1):
            prob_ft_h = poisson.pmf(k, pred['lam_h']) / sum(poisson.pmf(j, pred['lam_h']) for j in range(hg_ht, MAX_GOALS+1))
            if rng.random() <= prob_ft_h:
                hg_ft = k; break
        for k in range(ag_ht, MAX_GOALS + 1):
            prob_ft_a = poisson.pmf(k, pred['lam_a']) / sum(poisson.pmf(j, pred['lam_a']) for j in range(ag_ht, MAX_GOALS+1))
            if rng.random() <= prob_ft_a:
                ag_ft = k; break
        
        # Label
        ht = 'H' if hg_ht > ag_ht else ('D' if hg_ht == ag_ht else 'A')
        ft = 'H' if hg_ft > ag_ft else ('D' if hg_ft == ag_ft else 'A')
        counts[ht + ft] += 1
    
    total = sum(counts.values())
    probs = {l: counts[l] / total for l in labels}
    top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
    return probs, top3

def best_scoreline(pred, n_sims=50000):
    """MC模拟找最佳比分"""
    lam_h, lam_a = pred['lam_h'], pred['lam_a']
    counts = {}
    rng = random.Random(42)
    
    for _ in range(n_sims):
        hg = 0; cum = 0
        for k in range(MAX_GOALS + 1):
            cum += poisson.pmf(k, lam_h)
            if rng.random() <= cum:
                hg = k; break
        ag = 0; cum = 0
        for k in range(MAX_GOALS + 1):
            cum += poisson.pmf(k, lam_a)
            if rng.random() <= cum:
                ag = k; break
        key = f"{hg}-{ag}"
        counts[key] = counts.get(key, 0) + 1
    
    sorted_scores = sorted(counts.items(), key=lambda x: -x[1])
    return sorted_scores[:5]

def total_goals_analysis(pred, n_sims=50000):
    """总进球数分析"""
    lam_h, lam_a = pred['lam_h'], pred['lam_a']
    rng = random.Random(42)
    tg_counts = {i: 0 for i in range(8)}
    
    for _ in range(n_sims):
        hg = 0; cum = 0
        for k in range(MAX_GOALS + 1):
            cum += poisson.pmf(k, lam_h)
            if rng.random() <= cum:
                hg = k; break
        ag = 0; cum = 0
        for k in range(MAX_GOALS + 1):
            cum += poisson.pmf(k, lam_a)
            if rng.random() <= cum:
                ag = k; break
        tg = hg + ag
        if tg > 7: tg = 7
        tg_counts[tg] += 1
    
    total = sum(tg_counts.values())
    probs = {k: v/total for k, v in tg_counts.items()}
    
    # 分类
    under_prob = sum(probs.get(i, 0) for i in range(3))  # 0,1,2球
    mid_prob = sum(probs.get(i, 0) for i in range(3, 5))  # 3,4球
    over_prob = sum(probs.get(i, 0) for i in range(5, 8))  # 5,6,7+球
    
    # Best ranges
    sorted_tg = sorted(probs.items(), key=lambda x: -x[1])
    
    return {
        'probs': probs,
        'sorted': sorted_tg,
        'under2': under_prob,
        'mid': mid_prob,
        'over4': over_prob,
        'lambda_total': lam_h + lam_a,
    }

# ═══ Predict 3 matches ═══
# Name mapping: 500.com Chinese → DC model English
NAME_MAP = {
    '克罗地亚': 'Croatia',
    '比利时': 'Belgium',
    '格鲁吉亚': 'Georgia',
    '罗马尼亚': 'Romania',
    '威尔士': 'Wales',
    '加纳': 'Ghana',
}

def norm_name(name):
    return NAME_MAP.get(name, name)

matches = [
    ('克罗地亚', '比利时'),
    ('格鲁吉亚', '罗马尼亚'),
    ('威尔士', '加纳'),
]

results = []
for home_cn, away_cn in matches:
    home = norm_name(home_cn)
    away = norm_name(away_cn)
    pred = predict_match(home, away)
    pred['home_cn'] = home_cn
    pred['away_cn'] = away_cn
    
    # 1. 胜平负 pick
    pick_1x2 = max([('主胜', pred['hyb_home']), ('平局', pred['hyb_draw']), ('客胜', pred['hyb_away'])],
                   key=lambda x: x[1])
    
    # 2. 让球 (当前让-1球)
    # 让-1: 主队需赢2球+ → 等价于调整后的主胜概率
    rq = -1
    # 让球后概率估算: 主让1球 = 主队必须净胜2球
    # 简化: 用MC模拟
    rng = random.Random(99)
    rq_home_win = 0
    rq_sims = 10000
    for _ in range(rq_sims):
        hg = 0; cum = 0
        for k in range(MAX_GOALS + 1):
            cum += poisson.pmf(k, pred['lam_h'])
            if rng.random() <= cum: hg = k; break
        ag = 0; cum = 0
        for k in range(MAX_GOALS + 1):
            cum += poisson.pmf(k, pred['lam_a'])
            if rng.random() <= cum: ag = k; break
        if hg - ag > abs(rq):
            rq_home_win += 1
        elif hg - ag == abs(rq):
            # 走盘处理 → 概率平均
            rq_home_win += 0.5
    
    rq_home_p = rq_home_win / rq_sims
    rq_away_p = 1 - rq_home_p
    
    rq_pick = '主让胜' if rq_home_p > 0.5 else '客让胜'
    
    # 3. 半全场
    htft_probs, htft_top3 = sim_htft(pred)
    
    # 4. 比分Top3
    score_top3 = best_scoreline(pred)
    
    # 5. 总进球数
    tg_analysis = total_goals_analysis(pred)
    
    results.append({
        'home': home, 'away': away,
        'pick_1x2': pick_1x2,
        'hyb': (pred['hyb_home'], pred['hyb_draw'], pred['hyb_away']),
        'rangqiu_nspf': (pred['hyb_home'] - 0.15, pred['hyb_draw'], pred['hyb_away'] + 0.15),  # 让球调整
        'rq_home_p': rq_home_p,
        'rq_away_p': rq_away_p,
        'rq_pick': rq_pick,
        'htft': htft_top3,
        'score_top3': score_top3,
        'total_goals': tg_analysis,
    })

# ═══ Output Report ═══
print(f"\n{'='*75}")
print(f" 🏟  2026-06-02 竞彩足球预测报告 (90分钟口径)")
print(f"{'='*75}")
print(f" 模型: DC×{DC_WEIGHT} + XGB×{XGB_WEIGHT} | 20+3黄金 | MC 50K")
print(f"{'='*75}")

for r in results:
    h, a = r['home'], r['away']
    hp, dp, ap = r['hyb']
    h_cn, a_cn = r.get('home_cn', h), r.get('away_cn', a)
    
    print(f"\n{'─'*75}")
    print(f" 📋 {h_cn} VS {a_cn}")
    print(f"{'─'*75}")
    
    # 1. 胜平负
    pick_name, pick_p = r['pick_1x2']
    print(f"\n  【1】胜平负")
    print(f"  {'主胜':<6s}: {hp*100:>5.1f}%")
    print(f"  {'平局':<6s}: {dp*100:>5.1f}%")
    print(f"  {'客胜':<6s}: {ap*100:>5.1f}%")
    print(f"  ✅ 预测: {pick_name} ({pick_p*100:.1f}%)")
    
    # 2. 让球
    print(f"\n  【2】让球胜平负 (-1)")
    print(f"  {'主让-1胜':<8s}: {r['rq_home_p']*100:>5.1f}%")
    print(f"  {'客让+1胜':<8s}: {r['rq_away_p']*100:>5.1f}%")
    print(f"  ✅ 预测: {r['rq_pick']}")
    
    # 3. 半全场
    print(f"\n  【3】半全场 (MC 20K)")
    for label, prob in r['htft']:
        cn = {'HH':'胜胜','HD':'胜平','HA':'胜负','DH':'平胜','DD':'平平','DA':'平负','AH':'负胜','AD':'负平','AA':'负负'}[label]
        print(f"  {cn:<4s} ({label}): {prob*100:>5.1f}%")
    
    # 4. 比分
    print(f"\n  【4】比分 Top3 (MC 50K)")
    for score, cnt in r['score_top3']:
        print(f"  {score:<6s}: {cnt/50000*100:.1f}%")
    
    # 5. 总进球数
    tg = r['total_goals']
    print(f"\n  【5】总进球数 (λ_total={tg['lambda_total']:.2f})")
    print(f"  {'小球(0-2)':<10s}: {tg['under2']*100:>5.1f}%")
    print(f"  {'中球(3-4)':<10s}: {tg['mid']*100:>5.1f}%")
    print(f"  {'大球(5+)':<10s}: {tg['over4']*100:>5.1f}%")
    print(f"  Top分布: ", end='')
    for n, p in tg['sorted'][:4]:
        print(f"{n}球{p*100:.0f}% ", end='')
    print()

print(f"\n{'='*75}")
print(f" ⚠️ 仅代表模型预测，不构成投注建议。理性购彩。")
print(f"{'='*75}")
