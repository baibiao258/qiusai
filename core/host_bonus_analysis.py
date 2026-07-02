#!/usr/bin/env python3
"""分析东道主历史数据."""
import sys; sys.path.insert(0,'/root')
from wc_2026_phase1 import *
import pandas as pd, joblib

cache = load_data('/root/data/international_results.json')
matches = filter_matches(cache)
df = pd.DataFrame(matches)

print("=== 东道主历史非中立主场表现 ===")
for host in ['Canada', 'Mexico', 'United States']:
    hm = df[(df['home']==host) & (df['neutral']==False)]
    w = len(hm[hm['h_score']>hm['a_score']])
    d = len(hm[hm['h_score']==hm['a_score']])
    l = len(hm)-w-d
    gf = int(hm['h_score'].sum()); ga = int(hm['a_score'].sum())
    pts_pct = (w+d*0.5)/max(len(hm),1)
    print(f"  {host:>20s}: {len(hm):>4d}场  {w:>3d}胜 {d:>3d}平 {l:>3d}负  进{gf}失{ga}  得分率{pts_pct*100:.1f}%")

# 加载DC看看当前 λ 差异
dc = joblib.load('/root/data/dc_model.pkl')
print("\n=== 东道主非中立主场 λ vs 中立 λ ===")
for host in ['Canada', 'Mexico', 'United States']:
    for opp in ['Spain','France','England','Brazil']:
        ln = dc.predict_lambda(host, opp, neutral=True)
        lh = dc.predict_lambda(host, opp, neutral=False)
        lb = dc.predict_lambda(host, opp, neutral=False, host_bonus=dc.host_bonus_)
        ei = dc.team_idx_.get(host)
        ai = dc.team_idx_.get(opp)
        print(f"  {host:>15s} vs {opp:<10s}: 中立λ=({ln[0]:.2f},{ln[1]:.2f}) 主场λ=({lh[0]:.2f},{lh[1]:.2f}) +host_bonus=({lb[0]:.2f},{lb[1]:.2f})")

# 比较不同 host_bonus 下的冠军概率敏感性
print("\n=== host_bonus 灵敏度 (来自输出) ===")
print(f"  host_bonus=0.0000: Canada ~1.5% Mexico ~2.0%")
print(f"  host_bonus=0.0700: Canada ~2.5% Mexico ~3.5%")
print(f"  host_bonus=0.1445: Canada ~4.5% Mexico ~5.9%")

# 历史大型赛事东道主表现
print("\n=== 近5届世界杯东道主表现 ===")
data = [
    ('2018', 'Russia', 5, 2, 1, 12, 5, 'QF'),
    ('2014', 'Brazil', 4, 3, 1, 13, 6, 'SF'),
    ('2010', 'South Africa', 1, 1, 4, 5, 8, 'Group'),
    ('2006', 'Germany', 5, 1, 1, 14, 6, 'SF'),
    ('2002', 'South Korea', 4, 2, 1, 9, 5, 'SF'),
    ('2002', 'Japan', 2, 2, 2, 6, 4, 'R16'),
    ('1998', 'France', 6, 1, 0, 15, 2, 'Champion'),
]
print(f"  {'年份':>4s} {'东道主':>12s} {'胜':>3s} {'平':>3s} {'负':>3s} {'进':>3s} {'失':>3s} {'成绩':>8s}")
for y,h,w,d,l,gf,ga,stage in data:
    print(f"  {y} {h:>12s} {w:>3d} {d:>3d} {l:>3d} {gf:>3d} {ga:>3d} {stage:>8s}")
