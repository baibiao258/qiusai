#!/usr/bin/env python3
"""测试OOV模糊匹配 + 平局膨胀"""
import sys, os
sys.path.insert(0, '/root/wc_2026_upgrade')
os.chdir('/root')
sys.path.insert(0, '/root')
from daily_jczq import _try_hybrid_predict, _load_poisson_elo_prior, _load_shared_models

_load_poisson_elo_prior()
_load_shared_models()

tests = [
    ("Côte d'Ivoire", "Ecuador", "世界杯"),
    ("Australia", "Türkiye", "世界杯"),
    ("USA", "Paraguay", "世界杯"),
    ("Canada", "Bosnia & Herzegovina", "世界杯"),
    ("South Korea", "Czechia", "世界杯"),
]

print(f"{'比赛':<40s} {'H':>6s} {'D':>6s} {'A':>6s} {'模型':>14s} {'intl':>6s}")
print("─"*75)
for h, a, league in tests:
    r = _try_hybrid_predict(h, a, league, None)
    if r and r.get('probs'):
        p=r['probs']
        route=r.get('routing',{})
        is_i=route.get('is_intl','?')
        m=r.get('model','?')
        print(f"{h:<18s} vs {a:<18s} {p['H']:>5.0%} {p['D']:>5.0%} {p['A']:>5.0%} {m:>14s} {str(is_i):>6s}")
    else:
        print(f"{h:<18s} vs {a:<18s} {'N/A':>18s}")

# 平局膨胀
tight = [("Netherlands","Japan"), ("Brazil","Morocco"), ("Qatar","Switzerland"), ("Canada","Bosnia & Herzegovina")]
print(f"\n{'='*60}")
print(f"平局膨胀测试 (Elo差 < 100)")
print(f"{'比赛':<40s} {'H':>6s} {'D':>6s} {'A':>6s} {'模型'}")
print("─"*60)
for h, a in tight:
    r = _try_hybrid_predict(h, a, '世界杯', None)
    if r:
        p=r['probs']
        print(f"{h:<18s} vs {a:<18s} {p['H']:>5.0%} {p['D']:>5.0%} {p['A']:>5.0%} {r.get('model','?')}")
