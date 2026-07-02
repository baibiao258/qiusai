#!/usr/bin/env python3
"""
backtest_kelly.py — Kelly 策略回测管线
======================================
核心逻辑:
  1. 用训练好的 XGB+DC+Isotonic 校准模型逐场预测概率
  2. 生成合成市场赔率 (DC概率 + overround)
  3. 计算 EV, 用 Kelly Criterion 决定下注比例
  4. 模拟逐场下注, 追踪累计 P&L 和 ROI 曲线
  5. 输出完整回测报告

风控参数:
  - Quarter-Kelly (f* = 0.25)
  - 单场上限: 总资金的 5%
  - 单日上限: 总资金的 15%
  - 最小 EV 阈值: 5% (过滤噪音)
  - Bankroll: 10000 起始
"""
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import joblib

sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')

DATA_DIR = '/root/data'


# ═══════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════

def load_all():
    with open(os.path.join(DATA_DIR, 'club_matches.json')) as f:
        matches = json.load(f)
    elo = joblib.load(os.path.join(DATA_DIR, 'elo_club.pkl'))
    dc = joblib.load(os.path.join(DATA_DIR, 'dc_model_club.pkl'))
    xgb = joblib.load(os.path.join(DATA_DIR, 'xgb_model_club.pkl'))
    calibrators = joblib.load(os.path.join(DATA_DIR, 'calibrators_club.pkl'))
    with open(os.path.join(DATA_DIR, 'form_club.json')) as f:
        form_state = json.load(f)
    xg_path = os.path.join(DATA_DIR, 'xg_proxy_club.json')
    xg_state = {}
    if os.path.exists(xg_path):
        with open(xg_path) as f:
            xg_state = json.load(f)
    return matches, elo, dc, xgb, calibrators, form_state, xg_state


# ═══════════════════════════════════════════════
# 2. 特征构建 (复用 train_xgb_club.py 的逻辑)
# ═══════════════════════════════════════════════

class ClubFeatureBuffer:
    def __init__(self, elo, form_state):
        self.elo = elo
        self.form_state = form_state
        self.team_games = defaultdict(list)
        self.h2h_cache = defaultdict(lambda: defaultdict(list))

    def add_match(self, m):
        h, a = m['home'], m['away']
        self.team_games[h].append(m)
        self.team_games[a].append(m)
        key = (min(h, a), max(h, a))
        self.h2h_cache[key[0]][key[1]].append(m)

    def recent_form(self, team, n=5):
        games = self.form_state.get(team, [])
        recent = games[-n:] if len(games) >= n else games
        if not recent:
            return [0.5, 0.0, 0.0, 0.0]
        wins = sum(1 for g in recent if g[0] > g[1]) + \
               sum(0.5 for g in recent if g[0] == g[1])
        gf = sum(g[0] for g in recent) / len(recent)
        ga = sum(g[1] for g in recent) / len(recent)
        return [wins / len(recent), gf, ga, gf - ga]

    def h2h(self, home, away, n=3):
        key = (min(home, away), max(home, away))
        raw = self.h2h_cache[key[0]][key[1]][-n:]
        if not raw:
            return [0.5, 0.0, 0.0]
        wins = 0; gf = 0; ga = 0
        for m in raw:
            if m['home'] == home:
                gf += m['h_score']; ga += m['a_score']
                wins += 1 if m['h_score'] > m['a_score'] else (0.5 if m['h_score'] == m['a_score'] else 0)
            else:
                gf += m['a_score']; ga += m['h_score']
                wins += 1 if m['a_score'] > m['h_score'] else (0.5 if m['a_score'] == m['h_score'] else 0)
        return [wins / len(raw), gf / len(raw), ga / len(raw)]


def build_feat(fb, dc, h, a, xg_state=None):
    eh = fb.elo.get(h, 1400)
    ea = fb.elo.get(a, 1400)
    try:
        dc_p = dc.predict_proba(h, a, neutral=True)
        lam_h, lam_a = dc.predict_lambda(h, a, neutral=True)
        if lam_h is None: raise ValueError
    except:
        dc_p = [1/3, 1/3, 1/3]
        lam_h, lam_a = 1.0, 1.0

    fh5 = fb.recent_form(h, 5)
    fa5 = fb.recent_form(a, 5)
    fh12 = fb.recent_form(h, 12)
    fa12 = fb.recent_form(a, 12)
    h2h = fb.h2h(h, a, 3)

    op_h = 1 / (1 + 10 ** ((ea - eh) / 400))
    op_a = 1 / (1 + 10 ** ((eh - ea) / 400))

    b15 = [
        (eh - ea) / 400,
        lam_h, lam_a, lam_h - lam_a,
        math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
        dc_p[0], dc_p[1], dc_p[2],
        fh5[0], fa5[0],
        fh5[1] - fa5[2], fa5[1] - fh5[2],
        fh5[1] - fa5[1], fh5[0] - fa5[0],
        1,
    ]
    gold = [
        h2h[0] - h2h[2],
        0, 0,
        fh12[1] - fa12[2],
        fa12[1] - fh12[0],
    ]
    odds_feat = [op_h, op_a, 0.0]
    form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]

    if xg_state is not None:
        xg_proxy_feat = []
        for team in [h, a]:
            s = xg_state.get(team, {})
            xg_proxy_feat.extend([
                s.get('xg_proxy_5', 0.0),
                s.get('xg_proxy_12', 0.0),
                s.get('xg_streak', 0) / 10.0,
                s.get('xg_volatility', 0.0),
            ])
    else:
        xg_proxy_feat = [0.0] * 8

    return b15 + gold + odds_feat + form_feat + xg_proxy_feat


# ═══════════════════════════════════════════════
# 3. 校准概率 (Hybrid DC + XGB + Isotonic)
# ═══════════════════════════════════════════════

def get_calibrated_probs(xgb, calibrators, dc, feat, home, away):
    """输出校准后的 [away, draw, home] 三路概率."""
    raw = xgb.predict_proba(np.array([feat]))[0]
    p = np.clip(raw, 1e-10, 1.0)
    p = p / p.sum()

    # 动态权重
    e = -np.sum(p * np.log2(p))
    conf = 1.0 - e / math.log2(3)
    xgb_w = max(0.10, min(0.90, 0.30 + 0.50 * conf))
    dc_w = 1.0 - xgb_w

    try:
        dp = dc.predict_proba(home, away, neutral=True)
        dc_ado = [dp[2], dp[1], dp[0]]
    except:
        dc_ado = [1/3, 1/3, 1/3]

    h = dc_w * np.array(dc_ado) + xgb_w * p
    s = h.sum()
    if s > 0: h = h / s

    # Isotonic 校准
    calibrated = np.zeros(3)
    for j, key in enumerate(['away', 'draw', 'home']):
        if key in calibrators:
            calibrated[j] = calibrators[key].predict([h[j]])[0]
        else:
            calibrated[j] = h[j]
    s = calibrated.sum()
    if s > 0: calibrated = calibrated / s

    return calibrated  # [P_away, P_draw, P_home]


# ═══════════════════════════════════════════════
# 4. 合成市场赔率 (DC概率 + overround)
# ═══════════════════════════════════════════════

def generate_market_odds(dc_probs, overround=1.10):
    """
    生成合成市场赔率.
    dc_probs: [P_home, P_draw, P_away] (DC 原始概率)
    overround: 总赔率倒数和 (1.05-1.15 常见)
    """
    fair_odds = [1.0 / max(p, 0.01) for p in dc_probs]
    # 加入 overround: 按比例缩减
    market_odds = [o / overround for o in fair_odds]
    return market_odds  # [odds_home, odds_draw, odds_away]


# ═══════════════════════════════════════════════
# 5. Kelly Criterion 计算
# ═══════════════════════════════════════════════

def kelly_fraction(prob, odds, fraction=0.25):
    """
    计算 Kelly 仓位比例.
    f* = (p * b - q) / b, b = odds - 1, q = 1 - p
    fraction: 0.25 = Quarter-Kelly
    """
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    f_star = (prob * b - q) / b
    if f_star <= 0:
        return 0.0
    return f_star * fraction


# ═══════════════════════════════════════════════
# 6. 回测引擎
# ═══════════════════════════════════════════════

class BacktestEngine:
    def __init__(self, bankroll=10000.0, kelly_frac=0.25,
                 min_ev=0.05, max_per_match=0.05, max_daily=0.15,
                 overround=1.10):
        self.bankroll_init = bankroll
        self.bankroll = bankroll
        self.kelly_frac = kelly_frac
        self.min_ev = min_ev
        self.max_per_match = max_per_match
        self.max_daily = max_daily
        self.overround = overround

        # 记录
        self.bets = []           # 所有下注记录
        self.daily_pnl = {}      # {date: pnl}
        self.bankroll_curve = [] # [(date, bankroll)]
        self.daily_bets_count = defaultdict(int)

    def run(self, matches, fb, dc, xgb, calibrators, xg_state):
        """主回测循环."""
        # 只回测验证集 (后 20%)
        n = len(matches)
        start_idx = int(n * 0.8)  # 与训练切分一致

        # 按日期分组
        date_matches = defaultdict(list)
        for i in range(start_idx, n):
            m = matches[i]
            date = m['date']
            date_matches[date].append((i, m))

        # 按日期排序
        sorted_dates = sorted(date_matches.keys())

        print(f"\n🔄 回测开始: {len(sorted_dates)} 天, {sum(len(v) for v in date_matches.values())} 场")
        print(f"   起始资金: ${self.bankroll_init:,.0f}")
        print(f"   Kelly: {self.kelly_frac*100:.0f}% | EV阈值: {self.min_ev*100:.0f}% | "
              f"单场上限: {self.max_per_match*100:.0f}% | 日上限: {self.max_daily*100:.0f}%")

        for date in sorted_dates:
            day_bets = []
            day_stake = 0.0

            for idx, m in date_matches[date]:
                # 特征
                feat = build_feat(fb, dc, m['home'], m['away'], xg_state)

                # 校准概率
                probs = get_calibrated_probs(xgb, calibrators, dc, feat, m['home'], m['away'])
                # probs: [P_away, P_draw, P_home]

                # DC 概率 (用于生成市场赔率)
                try:
                    dc_p = dc.predict_proba(m['home'], m['away'], neutral=True)
                except:
                    dc_p = [1/3, 1/3, 1/3]

                # 合成市场赔率
                market_odds = generate_market_odds(dc_p, self.overround)

                # 实际结果
                if m['h_score'] > m['a_score']:
                    result = 2  # home win
                elif m['h_score'] == m['a_score']:
                    result = 1  # draw
                else:
                    result = 0  # away win

                # 扫描每个玩法选项
                options = [
                    ('home', 2, probs[2], market_odds[0]),   # 主胜
                    ('draw', 1, probs[1], market_odds[1]),   # 平局
                    ('away', 0, probs[0], market_odds[2]),   # 客胜
                ]

                for pick_name, result_idx, prob, odds in options:
                    ev = prob * (odds - 1.0) - (1.0 - prob)
                    if ev < self.min_ev:
                        continue

                    kelly_f = kelly_fraction(prob, odds, self.kelly_frac)
                    stake_pct = min(kelly_f, self.max_per_match)
                    daily_remaining = self.max_daily - day_stake / self.bankroll if self.bankroll > 0 else 0
                    stake_pct = min(stake_pct, max(daily_remaining, 0))

                    if stake_pct <= 0 or self.bankroll <= 0:
                        continue

                    stake = self.bankroll * stake_pct
                    won = (result == result_idx)
                    pnl = stake * (odds - 1.0) if won else -stake

                    self.bankroll += pnl
                    day_stake += stake
                    day_bets.append({
                        'date': date,
                        'match': f"{m['home']} vs {m['away']}",
                        'pick': pick_name,
                        'odds': odds,
                        'prob': prob,
                        'ev': ev,
                        'stake_pct': stake_pct,
                        'stake': stake,
                        'won': won,
                        'pnl': pnl,
                        'bankroll': self.bankroll,
                    })

            # 记录日终状态
            self.bets.extend(day_bets)
            day_pnl = sum(b['pnl'] for b in day_bets)
            self.daily_pnl[date] = day_pnl
            self.bankroll_curve.append((date, self.bankroll))

            if self.bankroll <= 0:
                print(f"\n💀 破产! 日期: {date}")
                break

        return self.bets

    def summary(self):
        """输出回测摘要."""
        if not self.bets:
            print("\n⚠️ 无下注记录 (所有场次 EV 不达标)")
            return

        total_bets = len(self.bets)
        wins = sum(1 for b in self.bets if b['won'])
        total_stake = sum(b['stake'] for b in self.bets)
        total_pnl = sum(b['pnl'] for b in self.bets)
        roi = total_pnl / self.bankroll_init * 100

        # 最大回撤
        peak = self.bankroll_init
        max_dd = 0
        for date, br in self.bankroll_curve:
            peak = max(peak, br)
            dd = (peak - br) / peak
            max_dd = max(max_dd, dd)

        # 月度收益
        monthly = defaultdict(float)
        monthly_stake = defaultdict(float)
        for b in self.bets:
            month = b['date'][:7]
            monthly[month] += b['pnl']
            monthly_stake[month] += b['stake']

        # 按玩法统计
        play_stats = defaultdict(lambda: {'bets': 0, 'wins': 0, 'pnl': 0, 'stake': 0})
        for b in self.bets:
            ps = play_stats[b['pick']]
            ps['bets'] += 1
            ps['wins'] += 1 if b['won'] else 0
            ps['pnl'] += b['pnl']
            ps['stake'] += b['stake']

        # 按 EV 区间统计
        ev_bins = defaultdict(lambda: {'bets': 0, 'wins': 0})
        for b in self.bets:
            if b['ev'] >= 0.20:
                bucket = 'EV>=20%'
            elif b['ev'] >= 0.10:
                bucket = 'EV 10-20%'
            else:
                bucket = 'EV 5-10%'
            ev_bins[bucket]['bets'] += 1
            ev_bins[bucket]['wins'] += 1 if b['won'] else 0

        # 输出
        print("\n" + "═" * 60)
        print("  📊 KELLY 策略回测报告")
        print("═" * 60)

        print(f"\n  📈 总体表现")
        print(f"  {'─' * 40}")
        print(f"  起始资金:     ${self.bankroll_init:>10,.0f}")
        print(f"  最终资金:     ${self.bankroll:>10,.0f}")
        print(f"  总盈亏:       ${total_pnl:>+10,.0f}")
        print(f"  ROI:          {roi:>+9.1f}%")
        print(f"  下注次数:     {total_bets:>10}")
        print(f"  命中率:       {wins/total_bets*100:>9.1f}%  ({wins}/{total_bets})")
        print(f"  总投入:       ${total_stake:>10,.0f}")
        print(f"  最大回撤:     {max_dd*100:>9.1f}%")

        print(f"\n  🎯 按玩法统计")
        print(f"  {'─' * 40}")
        print(f"  {'玩法':<8} {'下注':>6} {'命中':>6} {'胜率':>7} {'盈亏':>10} {'ROI':>8}")
        for play, ps in sorted(play_stats.items(), key=lambda x: -x[1]['pnl']):
            wr = ps['wins'] / ps['bets'] * 100 if ps['bets'] > 0 else 0
            pr = ps['pnl'] / ps['stake'] * 100 if ps['stake'] > 0 else 0
            print(f"  {play:<8} {ps['bets']:>6} {ps['wins']:>6} {wr:>6.1f}% ${ps['pnl']:>+9,.0f} {pr:>+7.1f}%")

        print(f"\n  🎯 按 EV 区间统计")
        print(f"  {'─' * 40}")
        print(f"  {'区间':<12} {'下注':>6} {'命中':>6} {'胜率':>7}")
        for bucket, bs in sorted(ev_bins.items()):
            wr = bs['wins'] / bs['bets'] * 100 if bs['bets'] > 0 else 0
            print(f"  {bucket:<12} {bs['bets']:>6} {bs['wins']:>6} {wr:>6.1f}%")

        print(f"\n  📅 月度收益")
        print(f"  {'─' * 40}")
        for month in sorted(monthly.keys()):
            pnl = monthly[month]
            stake = monthly_stake[month]
            ret = pnl / stake * 100 if stake > 0 else 0
            bar_len = int(abs(pnl) / 50)
            bar = "█" * min(bar_len, 20)
            sign = "🟢" if pnl >= 0 else "🔴"
            print(f"  {month}  {sign} ${pnl:>+8,.0f}  ROI {ret:>+6.1f}%  {bar}")

        # Top 10 最佳/最差下注
        print(f"\n  🏆 Top 10 盈利下注")
        print(f"  {'─' * 50}")
        sorted_bets = sorted(self.bets, key=lambda b: -b['pnl'])
        for b in sorted_bets[:10]:
            print(f"  {b['date']} {b['match'][:30]:<30} {b['pick']:<5} "
                  f"O{b['odds']:.2f} P{b['prob']*100:.1f}% EV{b['ev']*100:+.1f}% "
                  f"${b['pnl']:>+8,.0f}")

        print(f"\n  💀 Top 10 亏损下注")
        print(f"  {'─' * 50}")
        for b in sorted_bets[-10:]:
            print(f"  {b['date']} {b['match'][:30]:<30} {b['pick']:<5} "
                  f"O{b['odds']:.2f} P{b['prob']*100:.1f}% EV{b['ev']*100:+.1f}% "
                  f"${b['pnl']:>+8,.0f}")

        # ROI 曲线 (ASCII)
        print(f"\n  📈 累计 ROI 曲线")
        print(f"  {'─' * 50}")
        if self.bankroll_curve:
            n_points = min(60, len(self.bankroll_curve))
            step = max(1, len(self.bankroll_curve) // n_points)
            sample = self.bankroll_curve[::step]

            min_br = min(br for _, br in sample)
            max_br = max(br for _, br in sample)
            rng = max_br - min_br if max_br > min_br else 1

            height = 15
            for row in range(height, -1, -1):
                threshold = min_br + rng * row / height
                line = ""
                for _, br in sample:
                    if br >= threshold:
                        line += "█"
                    else:
                        line += " "
                val = min_br + rng * row / height
                roi_val = (val / self.bankroll_init - 1) * 100
                print(f"  {roi_val:>+7.1f}% │{line}")

            print(f"  {'':>8} └{'─' * len(sample)}")
            first_date = self.bankroll_curve[0][0]
            last_date = self.bankroll_curve[-1][0]
            print(f"  {'':>9}{first_date}  →  {last_date}")

        print(f"\n  ⚙️  参数")
        print(f"  {'─' * 40}")
        print(f"  Kelly 折扣:    {self.kelly_frac*100:.0f}%")
        print(f"  EV 阈值:       {self.min_ev*100:.0f}%")
        print(f"  单场上限:      {self.max_per_match*100:.0f}%")
        print(f"  日上限:        {self.max_daily*100:.0f}%")
        print(f"  Overround:     {self.overround:.2f}")

        print(f"\n{'═' * 60}")

        return {
            'bankroll_init': self.bankroll_init,
            'bankroll_final': self.bankroll,
            'total_pnl': total_pnl,
            'roi': roi,
            'total_bets': total_bets,
            'win_rate': wins / total_bets,
            'max_drawdown': max_dd,
        }


# ═══════════════════════════════════════════════
# 7. 主程序
# ═══════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  🎰 Kelly 策略回测 (俱乐部模型 + xG-proxy)")
    print("═" * 60)

    # 加载
    print("\n📦 加载数据...")
    matches, elo, dc, xgb, calibrators, form_state, xg_state = load_all()
    matches.sort(key=lambda m: m['date'])

    # 去重
    seen = set()
    unique = []
    for m in matches:
        key = (m['date'], m['home'], m['away'])
        if key not in seen:
            seen.add(key)
            unique.append(m)
    matches = unique

    print(f"  比赛: {len(matches)} 场")
    print(f"  XGB: {xgb.n_features_in_} 维, {xgb.n_estimators} 棵")
    print(f"  DC: ρ={dc.rho_:.4f} γ={dc.gamma_:.4f}")

    # 特征 buffer (只用前 80% 构建)
    fb = ClubFeatureBuffer(elo, form_state)
    train_end = int(len(matches) * 0.8)
    for m in matches[:train_end]:
        fb.add_match(m)

    # ── 回测 1: 标准参数 ──
    print("\n" + "─" * 60)
    print("  📋 方案 A: Quarter-Kelly, EV≥5%")
    print("─" * 60)
    engine1 = BacktestEngine(
        bankroll=10000, kelly_frac=0.25,
        min_ev=0.05, max_per_match=0.05, max_daily=0.15,
        overround=1.10
    )
    engine1.run(matches, fb, dc, xgb, calibrators, xg_state)
    stats1 = engine1.summary()

    # ── 回测 2: 激进参数 ──
    print("\n" + "─" * 60)
    print("  📋 方案 B: Half-Kelly, EV≥3%")
    print("─" * 60)
    engine2 = BacktestEngine(
        bankroll=10000, kelly_frac=0.50,
        min_ev=0.03, max_per_match=0.08, max_daily=0.20,
        overround=1.08
    )
    engine2.run(matches, fb, dc, xgb, calibrators, xg_state)
    stats2 = engine2.summary()

    # ── 回测 3: 保守参数 ──
    print("\n" + "─" * 60)
    print("  📋 方案 C: Quarter-Kelly, EV≥8%")
    print("─" * 60)
    engine3 = BacktestEngine(
        bankroll=10000, kelly_frac=0.25,
        min_ev=0.08, max_per_match=0.03, max_daily=0.10,
        overround=1.10
    )
    engine3.run(matches, fb, dc, xgb, calibrators, xg_state)
    stats3 = engine3.summary()

    # ── 对比 ──
    if stats1 and stats2 and stats3:
        print("\n" + "═" * 60)
        print("  📊 三方案对比")
        print("═" * 60)
        print(f"  {'方案':<12} {'ROI':>8} {'命中率':>8} {'回撤':>8} {'下注数':>8} {'最终资金':>10}")
        print(f"  {'─' * 55}")
        for label, s in [('Q-Kelly 5%', stats1), ('H-Kelly 3%', stats2), ('Q-Kelly 8%', stats3)]:
            if s:
                print(f"  {label:<12} {s['roi']:>+7.1f}% {s['win_rate']*100:>7.1f}% "
                      f"{s['max_drawdown']*100:>7.1f}% {s['total_bets']:>8} ${s['bankroll_final']:>9,.0f}")
        print(f"  {'═' * 55}")

    print("\n✅ 回测完成!")


if __name__ == '__main__':
    main()
