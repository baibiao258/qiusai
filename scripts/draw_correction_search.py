#!/usr/bin/env python3
"""
Draw Correction 网格搜索 + 参数化验证
======================================
搜索最优的 draw_correction 参数（threshold, max_boost, confidence_decay），
用 historical_kaijiang 实际开奖结果评估 Brier Score。

用法:
  python3 scripts/draw_correction_search.py

输出:
  - 最优参数写入 /root/data/draw_correction_opt.json
  - 终端打印参数对比表
"""
import csv
import json
import math
import os
import sys
from itertools import product

import numpy as np

DATA_DIR = '/root/data'
KAJ_PATH = os.path.join(DATA_DIR, 'historical_kaijiang.csv')

# ── 搜索空间 ──
PARAM_GRID = {
    'threshold':   [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],   # 触发阈值
    'max_boost':   [0.03, 0.05, 0.08, 0.10, 0.12],         # 最大调整量
    'decay_power': [0.5, 0.7, 1.0, 1.3, 1.5],              # confidence 衰减指数
}

# 不同比赛类型的可选参数 (用于条件化)
LEAGUE_CONDITIONS = {
    '世界杯': {},   # 后续覆盖
    '友谊赛': {},
    '联赛': {},
}


def spf_result(ft_h, ft_a):
    """实际结果标签: H/D/A"""
    if ft_h > ft_a:
        return 'H'
    if ft_h == ft_a:
        return 'D'
    return 'A'


def load_matches():
    """从 historical_kaijiang 加载比赛."""
    matches = []
    with open(KAJ_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ft_h = int(row.get('ft_h', -1))
                ft_a = int(row.get('ft_a', -1))
                if ft_h < 0 or ft_a < 0:
                    continue
                matches.append({
                    'date': row.get('date', ''),
                    'code': row.get('code', ''),
                    'league': row.get('league', ''),
                    'home': row.get('home', ''),
                    'away': row.get('away', ''),
                    'ft_h': ft_h,
                    'ft_a': ft_a,
                    'spf_result': row.get('spf_result', spf_result(ft_h, ft_a)),
                })
            except (ValueError, TypeError):
                continue
    return matches


def apply_draw_correction(hybrid, threshold=0.15, max_boost=0.05, decay_power=1.0):
    """参数化 Draw Correction.

    Args:
        hybrid: np.array [p_away, p_draw, p_home]
        threshold: 平局触发阈值
        max_boost: 最大播种量
        decay_power: confidence 衰减指数 (>1 = 更保守, <1 = 更激进)

    Returns: np.array [p_away, p_draw, p_home]
    """
    h = hybrid.copy()
    if h[1] >= threshold:
        return h

    # 置信度 = max(home, away) 的概率
    # 置信度越高 (=强队越确定) → 需更大的boost才能纠正
    confidence = max(h[2], h[0])
    # decay_power: 控制 confidence 的影响曲线
    adjusted_conf = confidence ** decay_power
    draw_boost = max_boost * (1.0 - adjusted_conf)

    h[1] += draw_boost
    denom = h[2] + h[0] + 1e-10
    h[2] -= draw_boost * (h[2] / denom)
    h[0] -= draw_boost * (h[0] / denom)
    s = h.sum()
    if s > 0:
        h /= s
    return h


def simulate_predictions():
    """生成所有历史比赛的基准预测概率。

    使用泊松+Elo（简化版）生成模拟预测，避免逐个调用完整模型。
    这用于评估 Draw Correction 参数的相对效果。
    """
    matches = load_matches()
    print(f"📊 加载 {len(matches)} 场历史比赛")

    # 计算全局平均进球
    from collections import defaultdict
    team_stats = defaultdict(lambda: {'gf': 0, 'ga': 0, 'n': 0})
    for m in matches:
        h, a = m['home'], m['away']
        team_stats[h]['gf'] += m['ft_h']
        team_stats[h]['ga'] += m['ft_a']
        team_stats[h]['n'] += 1
        team_stats[a]['gf'] += m['ft_a']
        team_stats[a]['ga'] += m['ft_h']
        team_stats[a]['n'] += 1

    total_gf = sum(s['gf'] for s in team_stats.values())
    total_n = sum(s['n'] for s in team_stats.values())
    ga = total_gf / max(total_n, 1)

    def poisson_pmf(k, lam):
        return (lam ** k) * math.exp(-lam) / math.factorial(k)

    results = []
    for m in matches:
        home, away = m['home'], m['away']
        hs = team_stats.get(home, {'gf': 0, 'ga': 0, 'n': 5})
        as_ = team_stats.get(away, {'gf': 0, 'ga': 0, 'n': 5})

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
        if t > 0:
            hw, dr, aw = hw / t, dr / t, aw / t

        # Simulate some distribution of high/low confidence
        strength_diff = (hs['gf']/max(hs['n'],1) - as_['gf']/max(as_['n'],1)) / ga

        results.append({
            'home': home, 'away': away,
            'predicted': np.array([aw, dr, hw]),  # [away, draw, home]
            'actual': m['spf_result'],
            'league': m.get('league', ''),
            'strength_diff': strength_diff,
            'draw_prob': dr,
        })

    return results


def brier_score(y_true_onehot, y_pred):
    return float(np.mean((y_true_onehot - y_pred) ** 2))


def one_hot(label):
    if label == 'H':
        return np.array([0, 0, 1])
    elif label == 'D':
        return np.array([0, 1, 0])
    return np.array([1, 0, 0])


def run_grid():
    predictions = simulate_predictions()
    n = len(predictions)
    print(f"  产生 {n} 个模拟预测\n")

    results = []
    total_configs = len(PARAM_GRID['threshold']) * len(PARAM_GRID['max_boost']) * len(PARAM_GRID['decay_power'])
    idx = 0

    for threshold, max_boost, decay_power in product(
        PARAM_GRID['threshold'], PARAM_GRID['max_boost'], PARAM_GRID['decay_power']
    ):
        idx += 1
        corrected_probs = []
        for p in predictions:
            cp = apply_draw_correction(p['predicted'], threshold, max_boost, decay_power)
            corrected_probs.append(cp)

        y_true = np.array([one_hot(p['actual']) for p in predictions])
        y_pred = np.array(corrected_probs)

        brier = brier_score(y_true, y_pred)

        # 按置信度分层评估
        confident_idx = [i for i, p in enumerate(predictions) if p['draw_prob'] < 0.15]
        uncert_idx = [i for i, p in enumerate(predictions) if p['draw_prob'] >= 0.15]

        brier_low = brier_score(y_true[confident_idx], y_pred[confident_idx]) if confident_idx else 0
        brier_high = brier_score(y_true[uncert_idx], y_pred[uncert_idx]) if uncert_idx else 0

        # 按联赛分层
        friendly_idx = [i for i, p in enumerate(predictions) if '友谊赛' in p.get('league', '') or 'Friendly' in p.get('league', '')]

        brier_friendly = brier_score(y_true[friendly_idx], y_pred[friendly_idx]) if len(friendly_idx) >= 10 else 0

        results.append({
            'threshold': threshold,
            'max_boost': max_boost,
            'decay_power': decay_power,
            'brier': round(brier, 4),
            'brier_lowdraw': round(brier_low, 4),
            'brier_normdraw': round(brier_high, 4),
            'brier_friendly': round(brier_friendly, 4),
            'n_lowdraw': len(confident_idx),
        })

        if idx % 20 == 0:
            print(f"  [{idx}/{total_configs}] thresh={threshold} boost={max_boost} decay={decay_power} → brier={brier:.4f}")

    # 排序: 总 Brier 升序
    results.sort(key=lambda r: r['brier'])

    print(f"\n{'='*90}")
    print(f"  Draw Correction 网格搜索结果 (共 {total_configs} 组)")
    print(f"{'='*90}")
    print(f"  {'#':>3} {'thresh':>7} {'boost':>6} {'decay':>6} {'Brier':>7} {'B_lowD':>7} {'B_normD':>7} {'B_fri':>7} {'n_low':>5}")
    print(f"  {'─'*60}")

    baseline = results[0]  # baseline uses very conservative params
    for i, r in enumerate(results[:15]):
        prefix = ' 🏆' if i == 0 else ''
        print(f"  {prefix}{i+1:>2} {r['threshold']:>7.2f} {r['max_boost']:>6.2f} {r['decay_power']:>6.1f} "
              f"{r['brier']:>7.4f} {r['brier_lowdraw']:>7.4f} {r['brier_normdraw']:>7.4f} "
              f"{r['brier_friendly']:>7.4f} {r['n_lowdraw']:>5d}")

    best = results[0]
    print(f"\n🏆 最优参数:")
    print(f"    threshold   = {best['threshold']:.2f}")
    print(f"    max_boost   = {best['max_boost']:.2f}")
    print(f"    decay_power = {best['decay_power']:.1f}")
    print(f"    Brier Score = {best['brier']:.4f} (低平局区: {best['brier_lowdraw']:.4f})")

    # 保存最优参数
    opt = {
        'threshold': best['threshold'],
        'max_boost': best['max_boost'],
        'decay_power': best['decay_power'],
        'brier': best['brier'],
        'brier_lowdraw': best['brier_lowdraw'],
        'source': 'draw_correction_grid_search',
        'date': __import__('datetime').date.today().isoformat(),
        'n_matches': n,
    }
    opt_path = os.path.join(DATA_DIR, 'draw_correction_opt.json')
    with open(opt_path, 'w') as f:
        json.dump(opt, f, indent=2)
    print(f"\n💾 最优参数已保存到 {opt_path}")

    return best


if __name__ == '__main__':
    run_grid()
