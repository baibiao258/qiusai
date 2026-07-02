#!/usr/bin/env python3
"""
backtest_comparison.py — DC+XGB vs 纯泊松+Elo 对比回测
"""
import requests, json, sys, os
sys.path.insert(0, '/root/wc_2026_upgrade')
os.chdir('/root')
from daily_jczq import _try_hybrid_predict, predict_match_legacy, _load_poisson_elo_prior
import numpy as np

KEY = "fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww"
HDR = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api"

_load_poisson_elo_prior()

url = f"{BASE}/football/matches?competition_id=comp_6107&status=finished&date_from=2026-05-01&date_to=2026-06-15&per_page=100"
r = requests.get(url, headers=HDR, timeout=30)
matches = r.json().get("data", [])

results = []
for m in matches:
    sc = m.get('score', {})
    ht = m.get('home_team', {}) or {}
    at = m.get('away_team', {}) or {}
    home = ht.get('name', '')
    away = at.get('name', '')
    hg = sc.get('home')
    ag = sc.get('away')
    if hg is None or ag is None: continue
    
    if hg > ag: actual = 'H'
    elif hg == ag: actual = 'D'
    else: actual = 'A'

    # 模型1: DC+XGB hybrid
    r1 = _try_hybrid_predict(home, away, '世界杯', '')
    
    # 模型2: 纯泊松+Elo (用先验)
    r2 = predict_match_legacy(home, away, {}, 2.5, {})
    
    def calc(r_pred, name):
        if not r_pred or not r_pred.get('probs'): return None
        p = r_pred['probs']
        prd = max(p, key=p.get)
        iH=1.0 if actual=='H' else 0.0; iD=1.0 if actual=='D' else 0.0; iA=1.0 if actual=='A' else 0.0
        b = ((iH-p['H'])**2+(iD-p['D'])**2+(iA-p['A'])**2)/3.0
        return {'pred': prd, 'correct': prd==actual, 'brier': round(b,4), 'probs': p, 'model': name}
    
    res1 = calc(r1, 'DC+XGB')
    res2 = calc(r2, '纯泊松+Elo')
    
    results.append({
        'match': f"{home} vs {away}", 'score': f"{hg}-{ag}", 'actual': actual,
        'dcxgb': res1, 'poisson': res2,
    })

print(f"\n{'='*100}")
print(f"  🏆 DC+XGB vs 纯泊松+Elo 对比回测")
print(f"{'='*100}")
print(f"{'比赛':<30s} {'比分':>5s} {'实':>3s} {'DC-XGB':>8s} {'Brier':>7s} {'泊松Elo':>8s} {'Brier':>7s}")
print(f"{'─'*75}")
for r in results:
    r1, r2 = r['dcxgb'], r['poisson']
    if r1 and r2:
        c1 = '✅' if r1['correct'] else '❌'
        c2 = '✅' if r2['correct'] else '❌'
        print(f"{r['match']:<30s} {r['score']:>5s} {r['actual']:>3s} {r1['pred']+c1:>8s} {r1['brier']:>7.4f} {r2['pred']+c2:>8s} {r2['brier']:>7.4f}")

# 汇总
n=len(results)
hit1=sum(1 for r in results if r['dcxgb'] and r['dcxgb']['correct'])
hit2=sum(1 for r in results if r['poisson'] and r['poisson']['correct'])
b1=sum(r['dcxgb']['brier'] for r in results if r['dcxgb'])/n
b2=sum(r['poisson']['brier'] for r in results if r['poisson'])/n
print(f"{'─'*75}")
print(f"DC+XGB:    命中 {hit1}/{n}={hit1/n*100:.1f}%  平均Brier={b1:.4f}")
print(f"纯泊松+Elo: 命中 {hit2}/{n}={hit2/n*100:.1f}%  平均Brier={b2:.4f}")

# 输在哪儿
print(f"\n  🔍 DC+XGB 失败场次:")
for r in results:
    if r['dcxgb'] and not r['dcxgb']['correct']:
        p = r['dcxgb']['probs']
        print(f"    {r['match']:<30s} 实{r['actual']} 预{r['dcxgb']['pred']}  "
              f"H={p['H']:.0%} D={p['D']:.0%} A={p['A']:.0%}  "
              f"泊松确实{'✅' if r['poisson'] and r['poisson']['correct'] else '❌'}")
