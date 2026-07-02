#!/usr/bin/env python3
"""
wc_detail_pred.py — 世界杯详细5玩法预测
=========================================
输出: 胜平负 / 让球 / 半全场 / 比分 / 总进球
"""
import json, os, sys, math, numpy as np
from scipy.stats import poisson
sys.path.insert(0, '/root')
DATA_DIR = '/root/data'

# 加载模型
from calibrated_predictor import predict as spf_predict, _load
_load()

# 加载 DC 和 Elo
import joblib
dc = joblib.load(f'{DATA_DIR}/dc_model.pkl')
dc_club = joblib.load(f'{DATA_DIR}/dc_club.pkl')
elo = joblib.load(f'{DATA_DIR}/elo_ratings.pkl')

def _norm(name):
    m = {'USA':'United States','Türkiye':'Turkey','Bosnia & Herzegovina':'Bosnia and Herzegovina',"Côte d'Ivoire":'Ivory Coast'}
    return m.get(name, name)

def get_lam(home, away):
    """获取两队 λ (DC→club→默认)"""
    for model, neut in [(dc, True), (dc_club, False)]:
        try:
            lh, la = model.predict_lambda(home, away, neutral=neut)
            if lh is not None: return lh, la
        except: continue
    return 1.2, 1.2

def poisson_probs(lam_h, lam_a, max_g=8):
    """泊松比分概率网格"""
    ph = [poisson.pmf(i, lam_h) for i in range(max_g)]
    pa = [poisson.pmf(i, lam_a) for i in range(max_g)]
    probs = {}
    for i in range(max_g):
        for j in range(max_g):
            p = ph[i] * pa[j]
            if p > 1e-6:
                probs[(i, j)] = p
    total = sum(probs.values())
    for k in probs: probs[k] /= total
    return probs

def predict_detail(home, away, market_h=0.0, odds_h=1.0, odds_d=1.0, odds_a=1.0):
    home_n, away_n = _norm(home), _norm(away)
    
    # SPF
    spf, _ = spf_predict(home_n, away_n, market_h)
    
    # DC λ
    lam_h, lam_a = get_lam(home_n, away_n)
    lam_h = max(0.1, min(5.0, lam_h))
    lam_a = max(0.1, min(5.0, lam_a))
    
    # 比分概率
    score_probs = poisson_probs(lam_h, lam_a, max_g=8)
    sorted_scores = sorted(score_probs.items(), key=lambda x: -x[1])
    
    # 总进球概率 (0-12)
    goals_probs = {}
    for g in range(13):
        p = sum(v for (i,j),v in score_probs.items() if i+j == g)
        goals_probs[g] = p
    
    # 让球 SPF (默认让球值基于 Elo 差)
    eh = elo.get(home_n, 1500)
    ea = elo.get(away_n, 1500)
    elo_diff = eh - ea
    
    # 估算让球: Elo差每150分约1球
    hcap = round(elo_diff / 150)
    hcap = max(-3, min(3, hcap))  # -3 ~ +3
    
    rq_probs = {}
    for i in range(8):
        for j in range(8):
            p = score_probs.get((i, j), 0)
            adj_h = i + hcap
            if adj_h > j:
                key = 'rq_win'
            elif adj_h == j:
                key = 'rq_draw'
            else:
                key = 'rq_loss'
            rq_probs[key] = rq_probs.get(key, 0) + p
    rq_total = sum(rq_probs.values())
    for k in rq_probs:
        rq_probs[k] /= rq_total
    
    # 半全场概率 (HT λ ≈ 0.45 * FT λ)
    ht_lam_h, ht_lam_a = lam_h * 0.45, lam_a * 0.45
    ht_probs = poisson_probs(ht_lam_h, ht_lam_a, max_g=6)
    ft_probs = poisson_probs(lam_h, lam_a, max_g=6)
    
    htft_probs = {}
    for (ht_h, ht_a), p_ht in ht_probs.items():
        for (ft_h, ft_a), p_ft in ft_probs.items():
            if p_ht * p_ft < 1e-8: continue
            ht_label = 'H' if ht_h > ht_a else ('D' if ht_h == ht_a else 'A')
            ft_label = 'H' if ft_h > ft_a else ('D' if ft_h == ft_a else 'A')
            label = ht_label + ft_label
            htft_probs[label] = htft_probs.get(label, 0) + p_ht * p_ft
    
    htft_total = sum(htft_probs.values())
    for k in htft_probs: htft_probs[k] /= htft_total
    sorted_htft = sorted(htft_probs.items(), key=lambda x: -x[1])
    
    return {
        'spf': spf,
        'lam_h': lam_h, 'lam_a': lam_a,
        'rq_probs': rq_probs,
        'hcap': hcap,
        'scores': sorted_scores[:15],
        'goals': goals_probs,
        'htft': sorted_htft,
    }

# ── 今晚3场 ──
matches = [
    ('Germany', 'Curaçao', 0.928, 1.03, 21.00, 36.00),
    ('Netherlands', 'Japan', 0.466, 2.00, 3.40, 3.60),
    ('Ivory Coast', 'Ecuador', 0.289, 3.25, 2.80, 2.50),
]

for home, away, mh, oh, od, oa in matches:
    d = predict_detail(home, away, mh, oh, od, oa)
    
    print(f'\n{"="*90}')
    print(f'  🌍 {home} vs {away}     | 市场: {oh:.2f} / {od:.2f} / {oa:.2f}')
    print(f'  {"="*90}')
    print(f'  90分钟常规时间（含伤停补时），不含加时赛和点球大战')
    print()
    
    # 1. 胜平负
    spf = d['spf']
    sorted_spf = sorted([('主胜', spf[2]), ('平局', spf[1]), ('客胜', spf[0])], key=lambda x: -x[1])
    print(f'  📊 胜平负 (SPF)')
    for i, (label, prob) in enumerate(sorted_spf, 1):
        medal = '🏆' if i == 1 else ('  ' if i > 3 else f'  {i}.')
        print(f'    {medal} {label}: {prob:.1f}%')
    print(f'  → 推荐: {sorted_spf[0][0]} ({sorted_spf[0][1]:.1f}%)')
    print()
    
    # 2. 让球
    hcap = d['hcap']
    rq = d['rq_probs']
    hcap_str = f'{"+" if hcap > 0 else ""}{hcap}'
    rq_label = {-1:'让平', 0:'不让'}
    # 正确显示
    rq_items = sorted([('让胜', rq.get('rq_win',0)*100), ('让平', rq.get('rq_draw',0)*100), ('让负', rq.get('rq_loss',0)*100)], key=lambda x: -x[1])
    rq_sign = f'主让{hcap}' if hcap > 0 else (f'客让{abs(hcap)}' if hcap < 0 else '平手')
    print(f'  📊 让球 ({rq_sign})')
    for i, (label, prob) in enumerate(rq_items, 1):
        medal = '🏆' if i == 1 else ('  ' if i > 3 else f'  {i}.')
        print(f'    {medal} {label}: {prob:.1f}%')
    print(f'  → 推荐: {rq_items[0][0]} ({rq_items[0][1]:.1f}%)')
    print()
    
    # 3. 半全场
    htft = d['htft']
    print(f'  📊 半全场 (HT/FT)')
    for i, (label, prob) in enumerate(htft[:9], 1):
        medal = '🏆' if i == 1 else f'  {i}.'
        label_cn = {'HH':'胜-胜','HD':'胜-平','HA':'胜-负',
                     'DH':'平-胜','DD':'平-平','DA':'平-负',
                     'AH':'负-胜','AD':'负-平','AA':'负-负'}.get(label, label)
        print(f'    {medal} {label_cn}: {prob*100:.1f}%')
    print(f'  → 推荐: {htft[0][0]} ({htft[0][1]*100:.1f}%)')
    print()
    
    # 4. 比分
    scores = d['scores']
    print(f'  📊 比分 Top 15')
    for i, ((sh, sa), prob) in enumerate(scores[:15], 1):
        medal = '🏆' if i == 1 else f'  {i:>2}.'
        print(f'    {medal} {sh}-{sa}: {prob*100:.1f}%')
    print(f'  → 推荐: {scores[0][0][0]}-{scores[0][0][1]} ({scores[0][1]*100:.1f}%)')
    print()
    
    # 5. 总进球
    goals = d['goals']
    print(f'  📊 总进球数 (全部13档)')
    for g in range(13):
        medal = '🏆' if g == max(goals, key=goals.get) else ''
        prob = goals.get(g, 0) * 100
        bar = '█' * int(prob / 3) if prob > 0 else ''
        print(f'    {medal} {g}球: {prob:.1f}% {bar}')
    print(f'  → 推荐: {max(goals, key=goals.get)}球 ({goals[max(goals, key=goals.get)]*100:.1f}%)')
