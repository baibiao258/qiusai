#!/usr/bin/env python3
"""Draw Correction 参数网格搜索 — 3248场历史数据回测

用法:
    python3 scripts/draw_correction_search.py > /root/scripts/draw_search_results.txt 2>&1

最优结果 (2026-06-14): threshold=0.15, max_boost=0.10, decay_power=1.5, Brier=0.2339
输出: /root/data/draw_correction_opt.json
"""
import csv
import json
import math
import itertools
import os

def apply_dc(probs, threshold, max_boost, decay_power, league, strength_diff):
    p = dict(probs)
    if p['draw'] < threshold:
        return p
    boost = max_boost
    if '友谊赛' in league:
        boost += 0.02
    if strength_diff >= 0.5:
        boost *= 0.5
    boost *= (p['draw'] / threshold) ** decay_power
    p['draw'] += boost
    total = sum(p.values())
    for k in p:
        p[k] /= total
    return p

def brier_score(probs, actual):
    """probs: (h,d,a), actual: 'H'/'D'/'A'"""
    labels = {'H': 0, 'D': 1, 'A': 2}
    actual_idx = labels[actual]
    return sum((1 if i == actual_idx else 0) - probs[i]) ** 2 / 3

def main():
    # 读取历史数据
    csv_path = '/root/data/historical_kaijiang.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f"历史数据: {len(rows)} 场")

    # 过滤有赔率+结果的行
    valid = []
    for r in rows:
        try:
            h = float(r.get('spf_odds_h', 0) or 0)
            d = float(r.get('spf_odds_d', 0) or 0)
            a = float(r.get('spf_odds_a', 0) or 0)
            result = r.get('spf_result', '')
            if h > 0 and d > 0 and a > 0 and result in ('H', 'D', 'A'):
                # 市场隐含概率
                inv_h = 1/h; inv_d = 1/d; inv_a = 1/a
                total = inv_h + inv_d + inv_a
                valid.append({
                    'probs': {'home': inv_h/total, 'draw': inv_d/total, 'away': inv_a/total},
                    'result': result,
                    'league': r.get('league', ''),
                    'strength_diff': abs(float(r.get('elo_h', 0) or 0) - float(r.get('elo_a', 0) or 0)) / 2000,
                })
        except (ValueError, TypeError):
            pass

    print(f"有效记录: {len(valid)} 场")

    # 网格参数
    thresholds = [0.10, 0.12, 0.15, 0.18, 0.20]
    boosts = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    decays = [1.0, 1.2, 1.5, 2.0]

    best = {'brier': 999, 'params': {}}
    for t, b, d in itertools.product(thresholds, boosts, decays):
        total_brier = 0.0
        for v in valid:
            adj = apply_dc(v['probs'], t, b, d, v['league'], v['strength_diff'])
            total_brier += brier_score([adj['home'], adj['draw'], adj['away']], v['result'])
        avg_brier = total_brier / len(valid)
        if avg_brier < best['brier']:
            best = {'brier': avg_brier, 'params': {'threshold': t, 'max_boost': b, 'decay_power': d}}
        if t == 0.15:
            print(f"  thr={t:.2f} boost={b:.2f} decay={d:.1f} → Brier={avg_brier:.4f}")

    print(f"\n最优: {best['params']} → Brier={best['brier']:.4f}")

    # 保存结果
    os.makedirs('/root/data', exist_ok=True)
    with open('/root/data/draw_correction_opt.json', 'w') as f:
        json.dump(best['params'], f, indent=2)
    print(f"已保存到 /root/data/draw_correction_opt.json")

if __name__ == '__main__':
    main()
