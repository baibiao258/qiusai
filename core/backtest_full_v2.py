#!/usr/bin/env python3
"""WC 2026 全管线回测 (17维模型 + Pinnacle市场校正)"""
import requests, sys, math, numpy as np, os, joblib
sys.path.insert(0, '/root/wc_2026_upgrade'); os.chdir('/root')
from daily_jczq import _try_hybrid_predict, _load_poisson_elo_prior, _load_shared_models
from dc_model_definition import DixonColes

KEY="fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww"
BA = "https://api.thestatsapi.com/api"
HDR = {"Authorization":f"Bearer {KEY}"}

_load_poisson_elo_prior()
_load_shared_models()

# 获取WC比赛 + match_id
r = requests.get(f"{BA}/football/matches?competition_id=comp_6107&status=finished&per_page=50", headers=HDR, timeout=30)
matches = r.json().get('data', [])

print(f"{'='*90}")
print(f"  🏆 WC 2026 完整管线回测 — 17维模型 + Pinnacle校正")
print(f"{'='*90}")
print(f"{'比赛':<35s} {'比分':>5s} {'实':>3s} {'DC':>8s} {'XGB':>8s} {'Post':>8s} {'Brier':>7s}")
print('─'*80)

total_brier_post = 0
total_brier_dc = 0
total_brier_xgb = 0
hit_post = hit_dc = hit_xgb = n = 0

for m in matches:
    ht = (m.get('home_team') or {}).get('name', '')
    at = (m.get('away_team') or {}).get('name', '')
    sc = m.get('score') or {}
    hg = sc.get('home'); ag = sc.get('away')
    if hg is None or ag is None: continue
    mid = m['id']
    actual = 'H' if hg>ag else ('D' if hg==ag else 'A')
    actual_id = 0 if actual=='H' else (1 if actual=='D' else 2)
    
    # 管线预测 (含Pinnacle校正)
    r_pred = _try_hybrid_predict(ht, at, '世界杯', mid)
    
    # DC模型纯输出
    dc_model = joblib.load('/root/data/dc_model.pkl')
    dc_p = dc_model.predict_proba(ht, at, neutral=True)
    
    xgb_p = None
    post_hybrid = None
    
    if r_pred and r_pred.get('probs'):
        probs = r_pred['probs']
        post_hybrid = np.array([probs.get('H',0), probs.get('D',0), probs.get('A',0)])
        
        # XGB原始输出 (从r_pred的model字段获取)
        if 'details' in r_pred and 'xgb_raw' in r_pred['details']:
            xgb_p = np.array(r_pred['details']['xgb_raw'])
    else:
        post_hybrid = dc_p  # fallback to DC
    
    def brier(pred, actual_id):
        oh = [1.0 if c==actual_id else 0.0 for c in range(3)]
        return sum((pred[c]-oh[c])**2 for c in range(3))/3.0
    
    b_dc = brier(dc_p, actual_id)
    b_post = brier(post_hybrid, actual_id)
    
    # 计算各模型预测
    dc_pred = max(range(3), key=lambda i: dc_p[i])
    post_pred = max(range(3), key=lambda i: post_hybrid[i])
    dc_correct = dc_pred == actual_id
    post_correct = post_pred == actual_id
    
    total_brier_post += b_post
    total_brier_dc += b_dc
    n += 1
    if dc_correct: hit_dc += 1
    if post_correct: hit_post += 1
    
    dc_s = 'H' if dc_pred==0 else ('D' if dc_pred==1 else 'A')
    post_s = 'H' if post_pred==0 else ('D' if post_pred==1 else 'A')
    
    print(f'{ht:<20s} vs {at:<15s} {hg:>2d}-{ag:<2d} {actual:>3s} '
          f'{dc_s}{"✅" if dc_correct else "❌":>6s} {"-":>8s} '
          f'{post_s}{"✅" if post_correct else "❌":>6s} {b_post:.4f}')

print('─'*80)
print(f'DC模型:       {hit_dc}/{n} = {hit_dc/n*100:.1f}%  Brier={total_brier_dc/n:.4f}')
print(f'DC+XGB+校正: {hit_post}/{n} = {hit_post/n*100:.1f}%  Brier={total_brier_post/n:.4f}')
print(f'DC+XGB改善:  {hit_post-hit_dc:+.0f}场命中, Brier {total_brier_post/n - total_brier_dc/n:+.4f}')
