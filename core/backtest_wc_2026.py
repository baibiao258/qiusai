#!/usr/bin/env python3
"""
backtest_wc_2026.py — 世界杯 2026 已完赛场次回测
"""
import requests, json, sys, os
sys.path.insert(0, '/root/wc_2026_upgrade')
os.chdir('/root')
from daily_jczq import _try_hybrid_predict, _load_poisson_elo_prior

KEY = "fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww"
HDR = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api"

# 预加载先验 (让日志输出优雅)
_load_poisson_elo_prior()

# 拉取已完赛的世界杯 2026 比赛
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
    if hg is None or ag is None:
        continue
    
    if hg > ag: actual_hda = 'H'
    elif hg == ag: actual_hda = 'D'
    else: actual_hda = 'A'
    
    # 使用 DC+XGB 预测 (空match_id跳过高阶特征)
    r_pred = _try_hybrid_predict(home, away, '世界杯', '')
    
    if r_pred and r_pred.get('probs'):
        probs = r_pred['probs']
        pred_hda = max(probs, key=probs.get)
        pred_h = probs.get('H', 0)
        pred_d = probs.get('D', 0)
        pred_a = probs.get('A', 0)
        
        iH = 1.0 if actual_hda == 'H' else 0.0
        iD = 1.0 if actual_hda == 'D' else 0.0
        iA = 1.0 if actual_hda == 'A' else 0.0
        brier = ((iH - pred_h)**2 + (iD - pred_d)**2 + (iA - pred_a)**2) / 3.0
        
        correct = pred_hda == actual_hda
        results.append({
            'match': f"{home} vs {away}",
            'date': m['utc_date'][:10],
            'score': f"{hg}-{ag}",
            'actual': actual_hda,
            'pred': pred_hda,
            'probs': {'H': round(pred_h,3), 'D': round(pred_d,3), 'A': round(pred_a,3)},
            'correct': correct,
            'brier': round(brier, 4),
            'model': r_pred.get('model', '?'),
        })

# 输出
print(f"\n{'='*90}")
print(f"  🏆 世界杯 2026 已完赛回测 ({len(results)} 场)")
print(f"{'='*90}")
print(f"{'比赛':<35s} {'比分':>7s} {'实':>3s} {'预':>3s} {'结':>4s} {'Brier':>7s} {'H':>6s} {'D':>6s} {'A':>6s} {'模型':>12s}")
print(f"{'─'*90}")
hit = 0; total_brier = 0.0
for r in results:
    ok = '✅' if r['correct'] else '❌'
    probs = r['probs']
    print(f"{r['match']:<35s} {r['score']:>7s} {r['actual']:>3s} {r['pred']:>3s} {ok:>4s} {r['brier']:>7.4f} {probs['H']:>5.1%} {probs['D']:>5.1%} {probs['A']:>5.1%} {r['model']:>12s}")
    if r['correct']: hit += 1
    total_brier += r['brier']

n = len(results)
avg_brier = total_brier/n if n > 0 else 0

# 分级统计
hda_brier = {'H': [], 'D': [], 'A': []}
for r in results:
    hda_brier[r['actual']].append(r['brier'])

print(f"{'─'*90}")
print(f"  📊 汇总")
print(f"    总场次: {n}")
print(f"    命中: {hit}/{n} = {hit/n*100:.1f}%")
print(f"    平均 Brier: {avg_brier:.4f}  (随机=0.222, 完美=0)")
print(f"    Brier 按赛果: ", end="")
for hda in ['H', 'D', 'A']:
    bvs = hda_brier[hda]
    if bvs:
        print(f"{hda}={sum(bvs)/len(bvs):.4f}({len(bvs)}场)  ", end="")
print()

# 输掉的场次分析
print(f"\n  ❌ 预测错误场次:")
for r in results:
    if not r['correct']:
        probs = r['probs']
        print(f"    {r['match']:<30s} 实{r['actual']} 预{r['pred']}  "
              f"probs=H{probs['H']:.0%}/D{probs['D']:.0%}/A{probs['A']:.0%}  "
              f"Brier={r['brier']:.4f}")
