#!/usr/bin/env python3
"""分析东道主冠军概率"""
import json

d = json.load(open('/root/data/final_results.json'))
champs = {t: c for t, c, _ in d.get('champs', [])}
runner = d.get('runner_prob', {})
winner_odds = d.get('winner_odds', {})

print("=== 当前 host_bonus=0.1445 下的东道主 ===")
for host in ['Mexico', 'Canada', 'United States']:
    cp = champs.get(host, 0) * 100
    rp = runner.get(host, 0) * 100
    odds = winner_odds.get(host, 0)
    ev = cp/100 * odds - 1 if odds > 0 else 0
    print(f"  {host:>15s}: 冠军 {cp:.2f}% 亚军 {rp:.2f}% 赔率 {odds:.0f} EV {ev*100:+.1f}%")

print("\n=== 灵敏度参考 (MC 50K) ===")
print("  host_bonus=0.0000 → Canada ~1.5%  Mexico ~2.0%  US ~1.8%")
print("  host_bonus=0.0700 → Canada ~2.5%  Mexico ~3.5%  US ~2.5%")
print("  host_bonus=0.1000 → Canada ~3.0%  Mexico ~4.5%  US ~3.0% (插值)")
print("  host_bonus=0.1445 → Canada ~4.5%  Mexico ~5.9%  US ~3.4% (当前)")

print("\n=== 历史东道主非中立主场数据 (训练集) ===")
print("  Canada:  18场  11胜5平2负  得分率75%  (小样本)")
print("  Mexico:  17场  10胜7平0负  得分率79%  (小样本, 0负)")
print("  United States: 68场  42胜11平15负  得分率70%  (大样本)")

print("\n=== 建议 ===")
print("  1. 加拿大样本仅18场 → host_bonus不可靠, 建议下调至 0.07-0.10")
print("  2. 墨西哥主场0负, 但仅17场 → 有数据偏差, 建议 0.10")
print("  3. 美国数据集最完整(68场) → 建议维持 0.14")
print("  4. 淘汰赛阶段东道主优势递减 → 可考虑阶段衰减")
