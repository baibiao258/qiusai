#!/usr/bin/env python3
"""打印官方分组版冠军概率."""
import json

d = json.load(open('/root/data/final_results.json'))

print("=" * 70)
print(f"  🏆 WC 2026 冠军概率 | 官方分组 | 修复版签表")
print(f"  🕐 {d['ts'][:19]} | 模拟: {d['sims']:,} 次")
print(f"  淘汰赛: {d['summary']['bracket_info']}")
print(f"  验证 acc={d['validation']['acc']:.1%} | 2022回测 acc={d['backtest_wc2022']['hybrid_acc']:.1%}")
print("=" * 70)

champs = d['champs'][:25]
best = champs[0][2]
odds_map = d['winner_odds']

# Build tier1 lookup
tier1_teams = {t for t,_,_,_,_ in d.get('tier1', [])}

for i, (t, c, p) in enumerate(champs, 1):
    rp = d['runner_prob'].get(t, 0) * 100
    cp = p  # already percent
    odds = odds_map.get(t, 0)
    ev = cp/100 * odds - 1 if odds > 0 else 0

    bar_len = int(cp / best * 30) if best > 0 else 0
    bar = '█' * bar_len

    if ev > 0:
        pm = '🟢'
        tier_tag = ' ✓'
    elif ev > -0.20:
        pm = '🟡'
        tier_tag = ' △'
    else:
        pm = '🔴'
        tier_tag = ''

    print(f"  {i:>2d} {pm}{tier_tag} {t:<22s} {cp:>6.2f}% 亚{rp:>5.2f}% 决{(cp+rp):>5.2f}% 赔{odds:>5.1f} EV{ev*100:+6.1f}% |{bar}")

print("\n" + "=" * 70)
print("  🥇 正EV (市场低估)")
print("=" * 70)
for t, p, odds, ev, kelly in d.get('tier1', []):
    rp = d['runner_prob'].get(t, 0) * 100
    cp = p * 100
    print(f"    ✓ {t:<22s} {cp:>5.2f}% vs 市场{1/odds*100:>5.1f}% 赔{odds:>5.1f} EV{ev*100:+6.1f}%")
