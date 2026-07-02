#!/usr/bin/env python3
"""
friendly_calibration.py — 友谊赛自适应折扣参数计算
==================================================
基于历史友谊赛回测，计算最优折扣因子和门控阈值。

输出: /root/data/friendly_calib.json
  - discount_map: {strength_bin: optimal_discount}
  - best_margin_threshold: BET/SKIP 分割阈值
  - default_discount: 全局默认折扣
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

DATA_DIR = '/root/data'
KAJ_PATH = os.path.join(DATA_DIR, 'historical_kaijiang.csv')


def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def compute_team_stats(rows):
    """计算简化的球队强度统计."""
    from collections import defaultdict
    stats = defaultdict(lambda: {'gf': 0, 'ga': 0, 'n': 0})
    for r in rows:
        try:
            ft_h = int(r['ft_h'])
            ft_a = int(r['ft_a'])
        except (ValueError, TypeError):
            continue
        h, a = r.get('home', ''), r.get('away', '')
        stats[h]['gf'] += ft_h
        stats[h]['ga'] += ft_a
        stats[h]['n'] += 1
        stats[a]['gf'] += ft_a
        stats[a]['ga'] += ft_h
        stats[a]['n'] += 1
    return stats


def elo_expected(ra, rb):
    return 1.0 / (1 + 10 ** ((rb - ra) / 400))


def compute_elo_fast(rows):
    """快速 Elo 计算."""
    from collections import defaultdict
    elo = defaultdict(lambda: 1500.0)
    for r in rows:
        try:
            ft_h = int(r['ft_h'])
            ft_a = int(r['ft_a'])
        except (ValueError, TypeError):
            continue
        h, a = r.get('home', ''), r.get('away', '')
        e_h = elo_expected(elo[h], elo[a])
        sh, sa = (1.0, 0.0) if ft_h > ft_a else ((0.5, 0.5) if ft_h == ft_a else (0.0, 1.0))
        elo[h] += 32 * (sh - e_h)
        elo[a] += 32 * (sa - (1 - e_h))
    return dict(elo)


def main():
    print("=" * 60)
    print("  📊 友谊赛自适应折扣参数计算")
    print("=" * 60)

    # ── 加载数据 ──
    with open(KAJ_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    friendly = [r for r in all_rows if '友谊' in r.get('league', '') or 'Friendly' in r.get('league', '')]
    non_friendly = [r for r in all_rows if '友谊' not in r.get('league', '') and 'Friendly' not in r.get('league', '')]

    print(f"\n   全部: {len(all_rows)} 场")
    print(f"   友谊赛: {len(friendly)} 场")
    print(f"   非友谊赛: {len(non_friendly)} 场")

    # ── 计算全局统计 (用于泊松基线) ──
    stats = compute_team_stats(all_rows)
    elo = compute_elo_fast(all_rows)
    ga = sum(s['gf'] for s in stats.values()) / max(sum(s['n'] for s in stats.values()), 1)

    # ── 为每场友谊赛生成基线预测 ──
    friendly_data = []
    for r in friendly:
        try:
            ft_h = int(r['ft_h'])
            ft_a = int(r['ft_a'])
        except (ValueError, TypeError):
            continue
        home, away = r.get('home', ''), r.get('away', '')
        hs = stats.get(home, {'gf': 0, 'ga': 0, 'n': 5})
        as_ = stats.get(away, {'gf': 0, 'ga': 0, 'n': 5})
        eh = elo.get(home, 1500)
        ea = elo.get(away, 1500)

        # 泊松 λ
        lam_h = ga * (hs['gf'] / max(hs['n'], 1) / ga) * (as_['ga'] / max(as_['n'], 1) / ga)
        lam_a = ga * (as_['gf'] / max(as_['n'], 1) / ga) * (hs['ga'] / max(hs['n'], 1) / ga)
        lam_h = max(0.2, min(5.0, lam_h))
        lam_a = max(0.2, min(5.0, lam_a))

        hw, dr, aw = 0.0, 0.0, 0.0
        for hg in range(8):
            for ag in range(8):
                p = poisson_pmf(hg, lam_h) * poisson_pmf(ag, lam_a)
                if hg > ag: hw += p
                elif hg == ag: dr += p
                else: aw += p
        t = hw + dr + aw
        if t > 0: hw, dr, aw = hw / t, dr / t, aw / t

        strength_diff = (hs['gf'] / max(hs['n'], 1) - as_['gf'] / max(as_['n'], 1)) / ga

        result = 'H' if ft_h > ft_a else ('D' if ft_h == ft_a else 'A')
        pred = 'H' if max(hw, dr, aw) == hw else ('D' if max(hw, dr, aw) == dr else 'A')

        # Brier components
        actual = [1.0 if result == 'A' else 0, 1.0 if result == 'D' else 0, 1.0 if result == 'H' else 0]
        pred_p = [aw, dr, hw]
        brier_raw = sum((a - p) ** 2 for a, p in zip(actual, pred_p))

        # Apply discounts 0-50% and compute Brier for each
        discount_effects = {}
        for discount in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            adj = (1 - discount) * sum(p * e for p, e in zip(pred_p, range(3))) + discount * (1/3)
            # renormalize
            adj_hw = (1 - discount) * hw + discount / 3
            adj_dr = (1 - discount) * dr + discount / 3
            adj_aw = (1 - discount) * aw + discount / 3
            st = adj_hw + adj_dr + adj_aw
            adj_hw, adj_dr, adj_aw = adj_hw / st, adj_dr / st, adj_aw / st
            brier_adj = sum((a - p) ** 2 for a, p in zip(actual, [adj_aw, adj_dr, adj_hw]))
            discount_effects[discount] = brier_adj - brier_raw  # negative = improvement

        friendly_data.append({
            'home': home, 'away': away,
            'strength_diff': strength_diff,
            'abs_strength_diff': abs(strength_diff),
            'brier_raw': brier_raw,
            'discount_effects': discount_effects,
            'result': result, 'pred': pred,
            'date': r.get('date', ''),
        })

    # ── 按强度差异分组分析 ──
    print(f"\n按强度差异分组分析 ({len(friendly_data)} 场友谊赛):")
    bins = [('|Δ|<0.5 (接近)', -0.5, 0.5), ('|Δ|≥0.5 (有差距)', 0.5, 999),
            ('Δ≥0.5 (主队强)', 0.5, 999), ('Δ≤-0.5 (客队强)', -999, -0.5)]

    results_by_bin = {}
    for label, lo, hi in bins:
        matches = [f for f in friendly_data if lo <= f['strength_diff'] < hi]
        if not matches:
            continue
        avg_brier = sum(f['brier_raw'] for f in matches) / len(matches)
        # Find best discount for this bin
        best_disc = 0.0
        best_brier = avg_brier
        for disc in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            brier_at_disc = sum(
                f['brier_raw'] + f['discount_effects'].get(disc, 0)
                for f in matches
            ) / len(matches)
            if brier_at_disc < best_brier:
                best_brier = brier_at_disc
                best_disc = disc

        results_by_bin[label] = {
            'n': len(matches),
            'avg_brier_raw': round(avg_brier, 4),
            'best_discount': best_disc,
            'brier_at_best': round(best_brier, 4),
            'brier_improvement': round(avg_brier - best_brier, 4),
        }
        print(f"  {label}: n={len(matches):>3d} | raw_brier={avg_brier:.4f} "
              f"|最佳折扣={best_disc:.0%} → brier={best_brier:.4f} (改善={avg_brier-best_brier:+.4f})",
              end='')
        # Also show 30% discount result
        brier_30 = sum(
            f['brier_raw'] + f['discount_effects'].get(0.30, 0)
            for f in matches
        ) / len(matches)
        print(f" | 30%折扣→brier={brier_30:.4f}")

    # ── 计算门控阈值分析 ──
    print(f"\n📈 门控阈值分析 (margin_pp):")
    for margin_cut in [5, 10, 15, 20, 25]:
        below = [f for f in friendly_data if f['abs_strength_diff'] * 100 < margin_cut]
        above = [f for f in friendly_data if f['abs_strength_diff'] * 100 >= margin_cut]
        if below:
            brier_below = sum(f['brier_raw'] for f in below) / len(below)
            acc_below = sum(1 for f in below if f['pred'] == f['result']) / len(below)
            print(f"  |Δ|×100 < {margin_cut:>2d}pp: n={len(below):>3d} acc={acc_below:.1%} brier={brier_below:.4f}")
        if above:
            brier_above = sum(f['brier_raw'] for f in above) / len(above)
            acc_above = sum(1 for f in above if f['pred'] == f['result']) / len(above)
            print(f"  |Δ|×100 ≥ {margin_cut:>2d}pp: n={len(above):>3d} acc={acc_above:.1%} brier={brier_above:.4f}")

    # ── 输出最优参数 ──
    # 构建 strength_diff → discount 映射
    discount_map = {}
    for label, data in results_by_bin.items():
        if '|Δ|<0.5' in label:
            discount_map['low_diff'] = data['best_discount']
        elif '有差距' in label:
            discount_map['high_diff'] = data['best_discount']
        if '主队强' in label:
            discount_map['home_favored'] = data['best_discount']
        elif '客队强' in label:
            discount_map['away_favored'] = data['best_discount']

    # 全局最优
    global_best = 0.0
    global_best_brier = sum(f['brier_raw'] for f in friendly_data) / len(friendly_data)
    for disc in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        brier_at_disc = sum(
            f['brier_raw'] + f['discount_effects'].get(disc, 0)
            for f in friendly_data
        ) / len(friendly_data)
        if brier_at_disc < global_best_brier:
            global_best = disc
            global_best_brier = brier_at_disc

    calib_output = {
        'default_discount': round(global_best, 2),
        'discount_by_strength': {k: round(v, 2) for k, v in discount_map.items()},
        'global_brier_raw': round(sum(f['brier_raw'] for f in friendly_data) / len(friendly_data), 4),
        'global_brier_optimal': round(global_best_brier, 4),
        'n_friendly': len(friendly_data),
        'generated': __import__('datetime').date.today().isoformat(),
    }
    print(f"\n🏆 最优友谊赛参数:")
    print(f"   default_discount = {global_best:.0%}")
    for k, v in discount_map.items():
        print(f"   {k} = {v:.0%}")
    print(f"   global_brier_raw = {global_best_brier:.4f}")

    out_path = os.path.join(DATA_DIR, 'friendly_calib.json')
    with open(out_path, 'w') as f:
        json.dump(calib_output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存: {out_path}")


if __name__ == '__main__':
    main()
