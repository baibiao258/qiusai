#!/usr/bin/env python3
"""
asian_handicap_scanner.py — 每日亚盘价值扫描器
==============================================

独立只读脚本，不修改任何文件，不干扰主流程。

功能:
  1. 读取 predictions_log.csv 中的模型让球概率 (pred_rq_win/draw/loss)
  2. 加载 500.com 实时让球赔率 (rq_h/d/a)
  3. 计算 EV = prob × (odds - 1) - (1 - prob)
  4. 找出 EV > 5% 的赛事
  5. 输出排版精美的《每日亚盘高价值扫描报告》

数据源:
  - predictions_log.csv (模型预测概率)
  - 500.com 实时赔率 (scrape_500_odds_today)

用法:
  python3 scripts/asian_handicap_scanner.py              # 默认扫描今日
  python3 scripts/asian_handicap_scanner.py --min-ev 0.03  # 调整 EV 阈值
  python3 scripts/asian_handicap_scanner.py --all          # 扫描全部(含已结束)
"""

import csv
import math
import os
import sys
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, '/root')

# ─── 配置 ───
PREDICTIONS_LOG = '/root/data/predictions_log.csv'
MIN_EV = 0.05  # 默认 EV 阈值 5%


def load_predictions():
    """只读加载 predictions_log.csv"""
    if not os.path.exists(PREDICTIONS_LOG):
        print(f"❌ 找不到 {PREDICTIONS_LOG}")
        return []

    rows = []
    with open(PREDICTIONS_LOG, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_500_rq_odds():
    """加载 500.com 实时让球赔率"""
    try:
        from daily_jczq import scrape_500_odds_today
        odds_list = scrape_500_odds_today()
        result = {}
        for m in (odds_list or []):
            key = (m.get('home_cn', ''), m.get('away_cn', ''))
            result[key] = {
                'code': m.get('code', ''),
                'rq_h': m.get('rq_h', 0) or 0,
                'rq_d': m.get('rq_d', 0) or 0,
                'rq_a': m.get('rq_a', 0) or 0,
                'handicap': m.get('handicap', 0) or 0,
            }
        return result
    except Exception as e:
        print(f"⚠️ 500.com 赔率加载失败: {e}")
        return {}


def scan_matches(rows, odds_map, min_ev=MIN_EV, show_all=False):
    """扫描所有比赛，找出亚盘价值"""
    today = date.today().isoformat()
    results = []

    for row in rows:
        # 过滤: 只看今日或未结算的比赛
        match_date = row.get('date', '')
        result_status = row.get('result_status', 'missing')

        if not show_all:
            if match_date != today and result_status != 'missing':
                continue

        # 提取关键字段
        home = row.get('home_cn', '')
        away = row.get('away_cn', '')
        code = row.get('code', '')
        league = row.get('league', '')

        try:
            handicap = int(row.get('rq', 0) or 0)
            pred_rq_win = float(row.get('pred_rq_win', 0)) / 100
            pred_rq_draw = float(row.get('pred_rq_draw', 0)) / 100
            pred_rq_loss = float(row.get('pred_rq_loss', 0)) / 100
        except (ValueError, TypeError):
            continue

        if pred_rq_win <= 0 and pred_rq_loss <= 0:
            continue

        # 获取市场赔率
        key = (home, away)
        m5 = odds_map.get(key, {})
        rq_h_odds = m5.get('rq_h', 0)
        rq_d_odds = m5.get('rq_d', 0)
        rq_l_odds = m5.get('rq_a', 0)

        # 计算 EV
        # EV = prob × (odds - 1) - (1 - prob)
        ev_win = 0
        ev_draw = 0
        ev_lose = 0

        if rq_h_odds > 1 and pred_rq_win > 0:
            ev_win = pred_rq_win * (rq_h_odds - 1) - (1 - pred_rq_win)
        if rq_d_odds > 1 and pred_rq_draw > 0:
            ev_draw = pred_rq_draw * (rq_d_odds - 1) - (1 - pred_rq_draw)
        if rq_l_odds > 1 and pred_rq_loss > 0:
            ev_lose = pred_rq_loss * (rq_l_odds - 1) - (1 - pred_rq_loss)

        # 公平赔率
        fair_win = 1.0 / pred_rq_win if pred_rq_win > 0.001 else 999
        fair_draw = 1.0 / pred_rq_draw if pred_rq_draw > 0.001 else 999
        fair_lose = 1.0 / pred_rq_loss if pred_rq_loss > 0.001 else 999

        # 筛选有价值的比赛
        has_value = abs(ev_win) >= min_ev or abs(ev_draw) >= min_ev or abs(ev_lose) >= min_ev

        result = {
            'code': code,
            'home': home,
            'away': away,
            'league': league,
            'handicap': handicap,
            'pred_rq_win': pred_rq_win,
            'pred_rq_draw': pred_rq_draw,
            'pred_rq_loss': pred_rq_loss,
            'fair_win': fair_win,
            'fair_draw': fair_draw,
            'fair_lose': fair_lose,
            'rq_h_odds': rq_h_odds,
            'rq_d_odds': rq_d_odds,
            'rq_l_odds': rq_l_odds,
            'ev_win': ev_win,
            'ev_draw': ev_draw,
            'ev_lose': ev_lose,
            'has_value': has_value,
            'result_status': result_status,
            'match_date': match_date,
        }
        results.append(result)

    return results


def print_report(results, min_ev=MIN_EV):
    """输出精美报告"""
    # 筛选有价值的比赛
    value_bets = []
    for r in results:
        if r['ev_win'] >= min_ev:
            value_bets.append({**r, 'direction': '让胜', 'ev': r['ev_win'],
                             'market_odds': r['rq_h_odds'], 'fair': r['fair_win'],
                             'prob': r['pred_rq_win']})
        if r['ev_draw'] >= min_ev:
            value_bets.append({**r, 'direction': '让平', 'ev': r['ev_draw'],
                             'market_odds': r['rq_d_odds'], 'fair': r['fair_draw'],
                             'prob': r['pred_rq_draw']})
        if r['ev_lose'] >= min_ev:
            value_bets.append({**r, 'direction': '让负', 'ev': r['ev_lose'],
                             'market_odds': r['rq_l_odds'], 'fair': r['fair_lose'],
                             'prob': r['pred_rq_loss']})

    # 按 EV 排序
    value_bets.sort(key=lambda x: x['ev'], reverse=True)

    print()
    print("=" * 72)
    print("  📊 每日亚盘（AH）高价值扫描报告")
    print(f"  扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  扫描场次: {len(results)} | 有价值: {len(value_bets)} | EV阈值: {min_ev:.0%}")
    print("=" * 72)

    if not value_bets:
        print()
        print("  ℹ️  今日暂无 EV ≥ {:.0%} 的亚盘价值投注".format(min_ev))
        print()
        print("=" * 72)
        return

    print()
    print("  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │  代码   │  对阵                    │ 盘口 │ 方向 │  EV   │ 公平赔率 │")
    print("  ├─────────────────────────────────────────────────────────────────────┤")

    for r in value_bets[:15]:
        code = r['code'][:7]
        matchup = f"{r['home']} vs {r['away']}"
        if len(matchup) > 22:
            matchup = matchup[:22]
        handicap = r['handicap']
        direction = r['direction']
        ev_str = f"{r['ev']:+.1%}"
        fair_str = f"{r['fair']:.2f}"

        print(f"  │ {code:<7} │ {matchup:<22} │ {handicap:>+3}  │ {direction} │ {ev_str:>5} │ {fair_str:>6}  │")

    print("  └─────────────────────────────────────────────────────────────────────┘")

    # 详细分析
    print()
    print("  ─── 详细分析 (Top 5) ───")
    print()

    for i, r in enumerate(value_bets[:5], 1):
        print(f"  #{i} {r['code']} {r['home']} vs {r['away']}")
        print(f"     联赛: {r['league']} | 日期: {r['match_date']}")
        print(f"     盘口: {r['handicap']:+d}")
        print(f"     推荐: {r['direction']}")
        print(f"     模型概率: {r['prob']:.1%}")
        print(f"     公平赔率: {r['fair']:.2f}")
        print(f"     市场赔率: {r['market_odds']:.2f}")
        print(f"     期望值(EV): {r['ev']:+.1%}")
        print(f"     让球概率: 胜{r['pred_rq_win']:.1%} 平{r['pred_rq_draw']:.1%} 负{r['pred_rq_loss']:.1%}")
        print()

    # 风险提示
    print("  ─── 风险提示 ───")
    print("  • 亚盘价值基于模型让球概率 vs 500.com 实时赔率计算")
    print("  • EV > 5% 为高价值投注，但仍需结合临场信息判断")
    print("  • 友谊赛 bet_action=WATCH_FRIENDLY，仅供观察")
    print("  • 市场赔率来自 500.com，实际下注前请二次确认")
    print()
    print("=" * 72)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='亚盘价值扫描器')
    parser.add_argument('--min-ev', type=float, default=MIN_EV, help='最小 EV 阈值 (默认 0.05)')
    parser.add_argument('--all', action='store_true', help='扫描全部比赛 (含已结束)')
    parser.add_argument('--no-odds', action='store_true', help='跳过 500.com 赔率加载')
    args = parser.parse_args()

    # 加载数据
    rows = load_predictions()
    if not rows:
        return

    print(f"📡 加载 {len(rows)} 条预测记录...")

    odds_map = {}
    if not args.no_odds:
        print("📡 加载 500.com 实时让球赔率...")
        odds_map = load_500_rq_odds()
        print(f"  ✅ {len(odds_map)} 场有让球赔率")

    # 扫描
    print("🧠 运行亚盘价值扫描...")
    results = scan_matches(rows, odds_map, min_ev=args.min_ev, show_all=args.all)

    # 输出报告
    print_report(results, min_ev=args.min_ev)


if __name__ == '__main__':
    main()
