#!/usr/bin/env python3
"""轻量干跑 - 只测路由分离, 跳过高阶特征 API"""
import sys, os
sys.path.insert(0, '/root/wc_2026_upgrade'); os.chdir('/root')
from daily_jczq import _try_hybrid_predict, _load_poisson_elo_prior, _load_shared_models

_load_poisson_elo_prior()
_load_shared_models()

# 不传 match_id 跳过 API 调用
tests = [
    # (home, away, league, expected_route, label)
    ("Sweden", "Tunisia", "世界杯", "A:DC+Pinnacle", "🌍 世界杯"),
    ("Netherlands", "Japan", "世界杯", "A:DC+Pinnacle", "🌍 世界杯"),
    ("Brazil", "Morocco", "International Friendly", "A:DC+Pinnacle", "🌍 友谊赛"),
    ("Nice", "Saint-Étienne", "法甲", "B:DC+XGB", "🏟 法甲"),
    ("LA Galaxy", "LAFC", "美职", "B:DC+XGB", "🏟 MLS"),
    ("Arsenal", "Chelsea", "英超", "B:DC+XGB", "🏟 英超"),
    ("Barcelona", "Real Madrid", "西甲", "B:DC+XGB", "🏟 西甲"),
    ("Toyota", "Verdy", "日职", "B:DC+XGB", "🏟 日职"),
]

for home, away, league, expected, label in tests:
    r = _try_hybrid_predict(home, away, league, None)
    if r:
        p = r['probs']
        model = r.get('model', '?')
        route = r.get('routing', {})
        is_intl = route.get('is_intl', '?')
        ok = '✅' if (expected.startswith('A') and is_intl) or (expected.startswith('B') and not is_intl) else '❌'
        print(f"{ok} {label:<12s} | {home:<18s} vs {away:<18s} | "
              f"model={model:<12s} | H={p['H']:.0%} D={p['D']:.0%} A={p['A']:.0%} | "
              f"intl={is_intl} (期望{expected})")
    else:
        print(f"❌ {label:<12s} | {home:<18s} vs {away:<18s} | None")
