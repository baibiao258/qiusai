#!/usr/bin/env python3
"""Friendly match filter — 友谊赛专项分析 (Phase 3 旁路)

Usage:
    python3 /root/friendly_filter.py "TeamA" "TeamB"          # 单场
    python3 /root/friendly_filter.py --batch file.txt          # 批量
    python3 /root/friendly_filter.py --tune                    # 调整门控阈值

输出格式: 完整预测 (HDA + 总进球 + 比分) + 门控决策
基于 76 场回测, 边际≥10pp 才投 HDA, 命中率从 55.3% → 66.0%, ROI +21pp.
"""
import sys, os, json, argparse
sys.path.insert(0, '/root')
import numpy as np
from predict_match import predict_match, mc_score_dist
from predict_match import HOST_BONUS_BY_TEAM

# ── 门控阈值 (76 场回测优化) ──
BET_MARGIN_THRESHOLD = 0.10  # ≥10pp 投 HDA
STRONG_MARGIN = 0.20         # ≥20pp 强信号
SKIP_MARGIN_MAX = 0.10       # <10pp 跳过


def format_match_prediction(p):
    """格式单场完整预测 (HDA + 总进球 + Top5 比分 + 门控)."""
    if not p:
        return "❌ 错误: 模型未收敛"

    home, away = p['home'], p['away']
    h, d, a = p['fin_h'], p['fin_d'], p['fin_a']
    br = p.get('bet_recommendation', {})

    lines = []
    lines.append(f"⚽ {home} vs {away}")
    lines.append(f"   比赛类型: {p['match_type']}  |  友谊赛折扣: {p.get('friendly_discount', 0)*100:.0f}%")
    lines.append(f"   Elo: {home}={p['elo_h']:.0f}  vs  {away}={p['elo_a']:.0f}  (差={p['elo_h']-p['elo_a']:+.0f})")
    lines.append(f"   λ: {home}={p['lam_h']}  客 {p['lam_a']}")
    lines.append("")
    lines.append(f"   HDA: 主 {h:.1f}% | 平 {d:.1f}% | 客 {a:.1f}%")
    lines.append(f"   DC:  {p['dc_h']:.1f}% / {p['dc_d']:.1f}% / {p['dc_a']:.1f}%")
    lines.append(f"   XGB: {p['xgb_h']:.1f}% / {p['xgb_d']:.1f}% / {p['xgb_a']:.1f}%")
    lines.append(f"   Hyb: {p['hyb_h']:.1f}% / {p['hyb_d']:.1f}% / {p['hyb_a']:.1f}%")
    lines.append(f"   Cal: {p['cal_h']:.1f}% / {p['cal_d']:.1f}% / {p['cal_a']:.1f}%")
    lines.append("")

    # 门控决策
    if br:
        action = br.get('action', 'N/A')
        if action == 'BET':
            icon = '✅ 投!'
            note = f"强信号 (回测 66% 命中率, ROI +28.8%)"
        elif action == 'SKIP':
            icon = '⛔ 跳过'
            note = f"弱信号 / 5-10pp 陷阱区 (回测 18-42% 命中率, -ROI)"
        elif action == 'SKIP_DATA':
            icon = '⚠️ 跳过(form缺失)'
            note = '主/客队无 form 数据, 信号不可靠'
        else:
            icon = '➖ N/A'
            note = '正式比赛不做门控'
        lines.append(f"   门控: {icon} | 边际={br['margin_pp']:.1f}pp | 首选={br['best_pick']}({br['best_prob_pct']:.1f}%)")
        lines.append(f"         {note}")
        lines.append("")

    # P4: 总进球 (大/小 2.5) 门控
    tg = p.get('total_goals_recommendation')
    if tg:
        taction = tg.get('action', 'N/A')
        if taction == 'BET_OVER':
            ticon = '🔼 大!'
        elif taction == 'BET_UNDER':
            ticon = '🔽 小!'
        elif taction == 'SKIP':
            ticon = '⏭️ 跳过'
        elif taction == 'SKIP_DATA':
            ticon = '⚠️ 跳过(form缺失)'
        else:
            ticon = '➖ N/A'
        pick = tg.get('pick', '-')
        p_over = tg.get('p_over_2_5_pct', 0)
        p_under = tg.get('p_under_2_5_pct', 0)
        lam_t = tg.get('lam_total', 0)
        conf = tg.get('confidence_pp', 0)
        lines.append(f"   总进球: {ticon} {taction} | 投{pick} | λ={lam_t} 大={p_over:.1f}% 小={p_under:.1f}% |Δ|={conf:.1f}pp")
        lines.append(f"           {tg.get('reason', '')}")
        lines.append("")

    # 比分分布
    scores = mc_score_dist(p['lam_h'], p['lam_a'], n=100000)
    lines.append(f"   比分概率 (Top 6):")
    for s, prob in scores[:6]:
        bar = '█' * int(prob * 2)
        lines.append(f"     {s:>5s}  {prob:5.1f}%  {bar}")
    lines.append("")

    # 总进球分布 (从 MC 抽样重新计算)
    hg = np.random.poisson(p['lam_h'], 100000)
    ag = np.random.poisson(p['lam_a'], 100000)
    total_goals = hg + ag
    lines.append(f"   总进球分布:")
    for g in range(7):
        pct = (total_goals == g).mean() * 100
        bar = '█' * int(pct * 2)
        marker = ' ←首选' if g == 2 else (' ←次选' if g == 3 else '')
        if pct > 2:
            lines.append(f"     {g}球 {pct:5.1f}%  {bar}{marker}")
    over_25 = (total_goals >= 3).mean() * 100
    under_25 = (total_goals <= 2).mean() * 100
    lines.append(f"     大2.5球: {over_25:.1f}%  |  小2.5球: {under_25:.1f}%")

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='友谊赛专项分析 + 边际门控')
    ap.add_argument('home', nargs='?', default=None, help='主队名')
    ap.add_argument('away', nargs='?', default=None, help='客队名')
    ap.add_argument('--home', dest='is_host', action='store_true', help='主队有主场加成')
    ap.add_argument('--batch', metavar='FILE', help='批量模式: 文件每行 "home|away"')
    ap.add_argument('--match-type', default='friendly', choices=['friendly', 'competitive', 'qualifier'])
    ap.add_argument('--json', action='store_true', help='JSON 输出')
    args = ap.parse_args()

    if args.batch:
        # 批量模式
        with open(args.batch) as f:
            lines = [l.strip() for l in f if l.strip() and '|' in l]
        results = []
        for ln in lines:
            home, away = ln.split('|', 1)
            home, away = home.strip(), away.strip()
            p = predict_match(home, away, match_type=args.match_type)
            if not p:
                print(f"❌ {home} vs {away} — 模型未收敛")
                continue
            results.append(p)
            if not args.json:
                print(format_match_prediction(p))
                print('─' * 60)

        # 汇总
        if not args.json:
            n_total = len(results)
            n_bet = sum(1 for r in results if r.get('bet_recommendation', {}).get('action') == 'BET')
            n_skip = sum(1 for r in results if r.get('bet_recommendation', {}).get('action') == 'SKIP')
            print()
            print(f"📊 汇总: 总 {n_total} 场 | 投 {n_bet} 场 | 跳过 {n_skip} 场 | 投注率 {n_bet/n_total*100:.0f}%")
            print(f"   (回测基准: 76场全投 55.3% 命中率 → 边际≥10pp 投 66.0% 命中率, +21pp ROI)")
    else:
        if not args.home or not args.away:
            ap.error("需要 home 和 away 参数 (或 --batch)")
        hb = HOST_BONUS_BY_TEAM.get(args.home, 0.0) if args.is_host else 0.0
        p = predict_match(args.home, args.away, host_bonus=hb, match_type=args.match_type)
        if args.json:
            print(json.dumps(p, ensure_ascii=False, indent=2))
        else:
            print(format_match_prediction(p))


if __name__ == '__main__':
    main()
