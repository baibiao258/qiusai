#!/usr/bin/env python3
"""
dc_real_odds_test.py — DC-Only vs 真实市场赔率验证
==================================================
核心目的: 诊断 XGB 是否带来 Alpha，还是仅放大噪音
方法: 用 DC 模型概率 vs 500.com 真实赔率，计算 EV 和 Kelly
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import joblib

sys.path.insert(0, '/root')
DATA_DIR = '/root/data'


def load_dc():
    return joblib.load(os.path.join(DATA_DIR, 'dc_model_club.pkl'))


def implied_prob(odds):
    """赔率转隐含概率."""
    return 1.0 / odds if odds > 0 else 0


def ev(prob, odds):
    """期望值: P*(O-1) - (1-P)."""
    return prob * (odds - 1) - (1 - prob)


def kelly(prob, odds, frac=0.25):
    """Quarter-Kelly."""
    b = odds - 1
    if b <= 0: return 0
    f = (prob * b - (1 - prob)) / b
    return max(0, f * frac)


def strip_rank(name):
    """去掉 [7] 这样的排名前缀."""
    import re
    return re.sub(r'\[\d+\]\s*', '', name).strip()


def main():
    print("═" * 60)
    print("  🔬 DC-Only vs 真实市场赔率验证")
    print("═" * 60)

    dc = load_dc()
    print(f"  DC: ρ={dc.rho_:.4f} γ={dc.gamma_:.4f}")

    # 加载今日真实赔率
    with open(os.path.join(DATA_DIR, 'today_500_odds.json')) as f:
        odds_data = json.load(f)

    spf_matches = odds_data.get('SPF', [])
    print(f"  今日竞彩: {len(spf_matches)} 场")

    results = []

    for m in spf_matches:
        home_raw = m.get('home', '')
        away_raw = m.get('away', '')
        home = strip_rank(home_raw)
        away = strip_rank(away_raw)

        odds_raw = m.get('odds', {}).get('spf', {})
        if not odds_raw:
            continue

        # 500.com 赔率: '0'=客胜, '1'=平, '3'=主胜
        try:
            o_away = float(odds_raw.get('0', 0))
            o_draw = float(odds_raw.get('1', 0))
            o_home = float(odds_raw.get('3', 0))
        except:
            continue

        if o_away <= 0 or o_draw <= 0 or o_home <= 0:
            continue

        # 市场隐含概率 (含 overround)
        imp_h = implied_prob(o_home)
        imp_d = implied_prob(o_draw)
        imp_a = implied_prob(o_away)
        overround = imp_h + imp_d + imp_a

        # 归一化市场概率
        mkt_h = imp_h / overround
        mkt_d = imp_d / overround
        mkt_a = imp_a / overround

        # DC 模型预测
        try:
            dc_p = dc.predict_proba(home, away, neutral=True)
            dc_h, dc_d, dc_a = dc_p
            lam_h, lam_a = dc.predict_lambda(home, away, neutral=True)
            if lam_h is None: lam_h, lam_a = 1.0, 1.0
        except Exception as e:
            continue

        # 标记 DC 是否在均匀分布模式
        dc_uniform = (abs(dc_h - 1/3) < 0.02 and abs(dc_d - 1/3) < 0.02 and abs(dc_a - 1/3) < 0.02)

        # EV 计算 (DC 概率 vs 真实赔率)
        ev_h = ev(dc_h, o_home)
        ev_d = ev(dc_d, o_draw)
        ev_a = ev(dc_a, o_away)

        # Kelly 仓位
        k_h = kelly(dc_h, o_home)
        k_d = kelly(dc_d, o_draw)
        k_a = kelly(dc_a, o_away)

        # DC vs 市场分歧
        diff_h = dc_h - mkt_h
        diff_d = dc_d - mkt_d
        diff_a = dc_a - mkt_a

        results.append({
            'home': home_raw,
            'away': away_raw,
            'o_home': o_home, 'o_draw': o_draw, 'o_away': o_away,
            'dc_h': dc_h, 'dc_d': dc_d, 'dc_a': dc_a,
            'mkt_h': mkt_h, 'mkt_d': mkt_d, 'mkt_a': mkt_a,
            'ev_h': ev_h, 'ev_d': ev_d, 'ev_a': ev_a,
            'k_h': k_h, 'k_d': k_d, 'k_a': k_a,
            'diff_h': diff_h, 'diff_d': diff_d, 'diff_a': diff_a,
            'overround': overround,
            'lam_h': lam_h, 'lam_a': lam_a,
            'dc_uniform': dc_uniform,
        })

    # ── 输出 ──
    print(f"\n{'═' * 60}")
    print(f"  📊 DC vs 市场 对比 ({len(results)} 场)")
    print(f"{'═' * 60}")

    # 价值投注扫描 (跳过 DC 均匀分布的场次)
    value_bets = []
    uniform_count = sum(1 for r in results if r['dc_uniform'])
    for r in results:
        if r['dc_uniform']:
            continue  # DC 模型退化为均匀分布, 无参考价值
        for pick, ev_val, kelly_val, odds, dc_p, mkt_p, diff in [
            ('主胜', r['ev_h'], r['k_h'], r['o_home'], r['dc_h'], r['mkt_h'], r['diff_h']),
            ('平局', r['ev_d'], r['k_d'], r['o_draw'], r['dc_d'], r['mkt_d'], r['diff_d']),
            ('客胜', r['ev_a'], r['k_a'], r['o_away'], r['dc_a'], r['mkt_a'], r['diff_a']),
        ]:
            if ev_val > 0.03:  # EV > 3% 才展示
                value_bets.append({
                    'match': f"{r['home']} vs {r['away']}",
                    'pick': pick,
                    'odds': odds,
                    'dc_prob': dc_p,
                    'mkt_prob': mkt_p,
                    'ev': ev_val,
                    'kelly': kelly_val,
                    'diff': diff,
                })

    value_bets.sort(key=lambda x: -x['ev'])

    if value_bets:
        print(f"\n  💰 真实 EV > 3% 的价值投注 ({len(value_bets)} 个, 排除 {uniform_count} 场 DC 均匀分布)")
        print(f"  {'─' * 55}")
        print(f"  {'比赛':<28} {'玩法':<5} {'赔率':>5} {'DC概率':>7} {'市场概率':>8} {'EV':>7} {'Kelly':>6}")
        print(f"  {'─' * 55}")
        for b in value_bets:
            match_str = b['match'][:26]
            print(f"  {match_str:<28} {b['pick']:<5} {b['odds']:>5.2f} "
                  f"{b['dc_prob']*100:>6.1f}% {b['mkt_prob']*100:>7.1f}% "
                  f"{b['ev']*100:>+6.1f}% {b['kelly']*100:>5.1f}%")
    else:
        print(f"\n  ⚠️ 今日无 EV > 3% 的价值投注 (排除 {uniform_count} 场 DC 均匀分布)")

    # ── 详细逐场分析 ──
    print(f"\n{'═' * 60}")
    print(f"  📋 逐场详细分析")
    print(f"{'═' * 60}")

    for r in results:
        print(f"\n  {r['home']} vs {r['away']}")
        print(f"  Overround: {r['overround']:.3f} | λ: {r['lam_h']:.2f} - {r['lam_a']:.2f}")
        print(f"  {'':>4} {'':>5} {'赔率':>5} {'DC':>7} {'市场':>7} {'分歧':>7} {'EV':>7} {'Kelly':>6}")
        for pick, odds, dc_p, mkt_p, diff, ev_val, kelly_val in [
            ('主胜', r['o_home'], r['dc_h'], r['mkt_h'], r['diff_h'], r['ev_h'], r['k_h']),
            ('平局', r['o_draw'], r['dc_d'], r['mkt_d'], r['diff_d'], r['ev_d'], r['k_d']),
            ('客胜', r['o_away'], r['dc_a'], r['mkt_a'], r['diff_a'], r['ev_a'], r['k_a']),
        ]:
            marker = "🔥" if ev_val > 0.05 else ("✅" if ev_val > 0 else "  ")
            print(f"  {marker} {pick:>4} {odds:>5.2f} {dc_p*100:>6.1f}% {mkt_p*100:>6.1f}% "
                  f"{diff*100:>+6.1f}% {ev_val*100:>+6.1f}% {kelly_val*100:>5.1f}%")

    # ── 汇总诊断 ──
    print(f"\n{'═' * 60}")
    print(f"  🔍 诊断汇总")
    print(f"{'═' * 60}")

    all_ev_h = [r['ev_h'] for r in results]
    all_ev_d = [r['ev_d'] for r in results]
    all_ev_a = [r['ev_a'] for r in results]
    all_overround = [r['overround'] for r in results]

    print(f"  平均 Overround: {np.mean(all_overround):.3f} ({(np.mean(all_overround)-1)*100:.1f}%)")
    print(f"  DC 平均 EV:")
    print(f"    主胜: {np.mean(all_ev_h)*100:>+.2f}%")
    print(f"    平局: {np.mean(all_ev_d)*100:>+.2f}%")
    print(f"    客胜: {np.mean(all_ev_a)*100:>+.2f}%")

    pos_ev = sum(1 for r in results if max(r['ev_h'], r['ev_d'], r['ev_a']) > 0)
    print(f"  有正 EV 选项的场次: {pos_ev}/{len(results)} ({pos_ev/len(results)*100:.0f}%)")

    # 主胜偏差分析
    home_bias = [r['diff_h'] for r in results]
    print(f"  DC 主胜偏差: {np.mean(home_bias)*100:>+.2f}% (正=DC比市场更看好主胜)")

    print(f"\n{'═' * 60}")


if __name__ == '__main__':
    main()
