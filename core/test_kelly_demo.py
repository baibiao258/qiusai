#!/usr/bin/env python3
"""Kelly 仓位空跑展示"""
import sys, os
sys.path.insert(0, '/root')
os.chdir('/root')
import daily_jczq
from daily_jczq import _try_hybrid_predict, _load_poisson_elo_prior, _load_shared_models

_load_poisson_elo_prior()
_load_shared_models()

# 模拟一场世界杯 RECOMMEND (有赔率)
# 构造 bundle 级别的 Kelly展示
print("=" * 70)
print("  Kelly 仓位计算展示")
print("=" * 70)

# 拿一场世界杯预测
home, away, league = "Sweden", "Tunisia", "世界杯"
r = _try_hybrid_predict(home, away, league, None)

if r and r.get('probs'):
    p = r['probs']
    print(f"\n  {home} vs {away}  [{league}]")
    print(f"  H={p['H']:.1%}  D={p['D']:.1%}  A={p['A']:.1%}")
    print(f"  模型: {r.get('model')}")
    
    # 模拟竞彩赔率 (使用 DC 模型的公平赔率反推)
    prob_h, prob_d, prob_a = p['H'], p['D'], p['A']
    
    # 假设竞彩赔率 (取自 500.com 的正常范围)
    odds_h, odds_d, odds_a = 2.10, 3.20, 3.80
    
    # Kelly 计算
    def _kelly(prob_pct, odds):
        p = prob_pct / 100.0 if prob_pct > 1 else prob_pct
        if odds <= 1: return 0.0
        ev = p * (odds - 1) - (1 - p)
        if ev <= 0: return 0.0
        kelly_f = ev / (odds - 1)
        return kelly_f / 4.0  # Quarter-Kelly
    
    print(f"\n  📊 Quarter-Kelly 仓位演示:")
    print(f"{'方向':>6s} {'模型概率':>10s} {'竞彩赔率':>10s} {'EV':>10s} {'Kelly':>10s} {'仓位%':>8s}")
    print("  " + "-" * 55)
    
    for label, prob, odd in [('主胜', prob_h, odds_h), ('平', prob_d, odds_d), ('客胜', prob_a, odds_a)]:
        p_val = prob / 100.0 if prob > 1 else prob
        if odd > 1:
            ev = p_val * (odd - 1) - (1 - p_val)
            k = _kelly(prob, odd)
            print(f"  {label:>4s}  {prob*100 if prob<=1 else prob:>7.1f}%  {odd:>8.2f}  "
                  f"{ev:>+.4f}  {k:>8.4f}  {k*100:>6.1f}%")
    
    # SPF推荐方向的Kelly
    spf_pick = '主胜' if prob_h > prob_d and prob_h > prob_a else ('平' if prob_d > prob_a else '客胜')
    pick_odds = {'主胜': odds_h, '平': odds_d, '客胜': odds_a}[spf_pick]
    pick_kelly = _kelly({'主胜': prob_h, '平': prob_d, '客胜': prob_a}[spf_pick], pick_odds)
    
    print(f"\n  → 推荐: {spf_pick} (建议仓位: {pick_kelly*100:.1f}% 总资金)")
    
    # 模拟多条匹RECOMMEND
    print(f"\n  📋 模拟当日多场 RECOMMEND:")
    # 假设有3场 RECOMMEND 各自的 Kelly
    kellys = [0.035, 0.042, 0.028, 0.050]  # 3.5% + 4.2% + 2.8% + 5.0% = 15.5%
    total = sum(kellys)
    for i, k in enumerate(kellys):
        print(f"    场次{i+1}: {k*100:.1f}%")
    print(f"    合计: {total*100:.1f}%")
    if total * 100 > 15:
        print(f"    ⚠️ 当日并发总仓位超过 15% 上限, 建议按比例缩减")
    print()

print("=" * 70)
print("  ✅ Kelly 仓位计算已集成到 daily_jczq.py")
print("=" * 70)
