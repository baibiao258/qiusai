#!/usr/bin/env python3
"""
bet_math.py — 单场赔率转换 + EV 计算 + Kelly Criterion
======================================================

核心公式:
  EV = P_win × (Odds - 1) - (1 - P_win)
  Kelly f* = (P_win × b - q) / b    其中 b = Odds - 1, q = 1 - P_win

输出:
  analyze_match() → 逐玩法的 EV + Kelly 推荐
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional


# ── 数据结构 ──

@dataclass
class BetScenario:
    """单个投注场景"""
    play: str            # 玩法: SPF / 让球 / 比分 / 总进球 / 半全场
    pick: str            # 推荐选项
    odds: float          # 赔率 (Decimal)
    prob: float          # 模型概率
    ev: float = 0.0      # 期望值
    kelly_full: float = 0.0   # Full Kelly 仓位
    kelly_half: float = 0.0   # Half Kelly
    kelly_quarter: float = 0.0 # Quarter Kelly
    is_value: bool = False     # EV > 0 即为有价值
    edge: float = 0.0    # 概率优势 (prob - 1/odds)
    model_type: str = ''  # hybrid / market_fallback / legacy_poisson


@dataclass
class MatchAnalysis:
    """单场比赛完整分析"""
    home: str
    away: str
    scenarios: List[BetScenario] = field(default_factory=list)
    best_ev: Optional[BetScenario] = None
    bankroll_note: str = ""


# ── 核心计算 ──

def compute_ev(prob: float, odds: float) -> float:
    """
    期望值 (Expected Value)
    
    EV = P_win × (Odds - 1) - (1 - P_win)
    
    Args:
        prob: 模型预测的胜率 (0~1)
        odds: 十进制赔率 (如 2.05)
    Returns:
        EV 值 (正数 = 有价值)
    """
    if odds <= 1.0 or prob <= 0 or prob >= 1:
        return 0.0
    b = odds - 1.0  # 纯利润
    return prob * b - (1.0 - prob)


# 单注最大仓位 (Quarter Kelly 层)
MAX_SINGLE_BET = 0.05  # 单注不超过总资金 5%

def compute_kelly(prob: float, odds: float) -> float:
    """
    Kelly Criterion: 最优仓位比例
    
    f* = (p × b - q) / b
    其中 b = odds - 1, q = 1 - p
    
    Args:
        prob: 模型胜率 (0~1)
        odds: 十进制赔率
    Returns:
        最优仓位比例 (0~1), 负数表示不应下注
    """
    if odds <= 1.0 or prob <= 0 or prob >= 1:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    f_star = (prob * b - q) / b
    # 钳位: 负值→0, 上限→MAX_SINGLE_BET (防极端 edge 导致仓位过大)
    return max(0.0, min(f_star, MAX_SINGLE_BET))


def compute_edge(prob: float, odds: float) -> float:
    """
    概率优势 = 模型概率 - 隐含概率
    
    隐含概率 = 1/odds (博彩公司的"观点")
    """
    if odds <= 1.0:
        return 0.0
    implied = 1.0 / odds
    return prob - implied


# ── 批量分析 ──

def analyze_scenario(play: str, pick: str, odds: float, prob: float, model_type: str = '') -> BetScenario:
    """
    分析单个投注场景, 计算 EV/Kelly/Edge
    """
    if odds <= 1.0 or prob <= 0 or prob >= 1:
        return BetScenario(
            play=play, pick=pick, odds=odds, prob=prob,
            ev=0.0, kelly_full=0.0, kelly_half=0.0,
            kelly_quarter=0.0, is_value=False, edge=0.0,
            model_type=model_type,
        )
    
    ev = compute_ev(prob, odds)
    kelly_f = compute_kelly(prob, odds)
    edge = compute_edge(prob, odds)
    
    return BetScenario(
        play=play,
        pick=pick,
        odds=odds,
        prob=prob,
        ev=ev,
        kelly_full=kelly_f,
        kelly_half=kelly_f / 2.0,
        kelly_quarter=kelly_f / 4.0,
        is_value=(ev > 0),
        edge=edge,
        model_type=model_type,
    )


def analyze_match(home: str, away: str, predictions: dict, odds: dict, model_type: str = '') -> MatchAnalysis:
    """
    单场比赛完整分析
    
    Args:
        home, away: 球队名
        predictions: {
            'spf': {'h': prob, 'd': prob, 'a': prob},
            'rq': {'rq_win': prob, 'rq_draw': prob, 'rq_lose': prob},
            'score': [{'score': '1:0', 'prob': 0.15}, ...],
            'total_goals': [{'goals': 2, 'prob': 0.25}, ...],
            'half_full': [{'hf': '胜-胜', 'prob': 0.12}, ...],
        }
        odds: {
            'spf': {'h': 2.05, 'd': 3.20, 'a': 3.80},
            'rq': {'rq_win': 1.80, 'rq_draw': 3.40, 'rq_lose': 2.10},
            'score': {'1:0': 5.50, '0:0': 8.00, ...},
            'total_goals': {'0': 11.00, '1': 4.40, '2': 3.25, ...},
            'half_full': {'胜-胜': 2.80, ...},
        }
        model_type: hybrid / market_fallback / legacy_poisson
    """
    analysis = MatchAnalysis(home=home, away=away)
    
    # SPF (胜平负)
    spf_pred = predictions.get('spf', {})
    spf_odds = odds.get('spf', {})
    for key, label in [('h', '主胜'), ('d', '平局'), ('a', '客胜')]:
        if key in spf_pred and key in spf_odds:
            s = analyze_scenario('胜平负', label, spf_odds[key], spf_pred[key], model_type)
            analysis.scenarios.append(s)
    
    # 让球
    rq_pred = predictions.get('rq', {})
    rq_odds = odds.get('rq', {})
    for key, label in [('rq_win', '让胜'), ('rq_draw', '让平'), ('rq_lose', '让负')]:
        if key in rq_pred and key in rq_odds:
            s = analyze_scenario('让球', label, rq_odds[key], rq_pred[key], model_type)
            analysis.scenarios.append(s)
    
    # 比分 (取概率最高的 5 个)
    score_pred = predictions.get('score', [])
    score_odds = odds.get('score', {})
    for item in score_pred[:5]:
        score = item['score']
        prob = item['prob']
        if score in score_odds:
            s = analyze_scenario('比分', score, score_odds[score], prob, model_type)
            analysis.scenarios.append(s)
    
    # 总进球
    tg_pred = predictions.get('total_goals', [])
    tg_odds = odds.get('total_goals', {})
    for item in tg_pred[:5]:
        goals = str(item['goals'])
        prob = item['prob']
        if goals in tg_odds:
            s = analyze_scenario('总进球', f'{goals}球', tg_odds[goals], prob, model_type)
            analysis.scenarios.append(s)
    
    # 半全场
    hf_pred = predictions.get('half_full', [])
    hf_odds = odds.get('half_full', {})
    for item in hf_pred[:4]:
        hf = item['hf']
        prob = item['prob']
        if hf in hf_odds:
            s = analyze_scenario('半全场', hf, hf_odds[hf], prob, model_type)
            analysis.scenarios.append(s)
    
    # 找出最佳 EV 场景
    value_bets = [s for s in analysis.scenarios if s.is_value]
    if value_bets:
        analysis.best_ev = max(value_bets, key=lambda s: s.ev)
    
    return analysis


# ── 格式化输出 ──

def format_ev_table(analysis: MatchAnalysis, min_ev: float = 0.0) -> str:
    """
    格式化输出单场比赛的 EV 分析表
    
    Args:
        analysis: MatchAnalysis 对象
        min_ev: 最低 EV 阈值, 低于此值不显示
    """
    lines = []
    lines.append(f"  {'─' * 60}")
    lines.append(f"  💰 赔率分析: {analysis.home} vs {analysis.away}")
    lines.append(f"  {'─' * 60}")
    
    # 过滤有价值的结果
    filtered = [s for s in analysis.scenarios if s.ev >= min_ev]
    
    if not filtered:
        lines.append(f"  ⚠️  无 EV > {min_ev:.0%} 的价值投注")
        return '\n'.join(lines)
    
    # 按 EV 降序排列
    filtered.sort(key=lambda s: -s.ev)
    
    lines.append(f"  {'玩法':<6} {'推荐':<8} {'赔率':>5} {'概率':>6} {'隐含':>6} {'Edge':>6} {'EV':>7} {'Kelly':>7} {'1/4Kelly':>8}")
    lines.append(f"  {'─' * 60}")
    
    for s in filtered:
        implied = 1.0 / s.odds if s.odds > 1.0 else 0.0
        emoji = '🔥' if s.ev > 0.10 else '✅' if s.ev > 0.05 else '📌'
        lines.append(
            f"  {emoji}{s.play:<5} {s.pick:<8} {s.odds:>5.2f} "
            f"{s.prob:>5.1%} {implied:>5.1%} {s.edge:>+5.1%} "
            f"{s.ev:>+6.1%} {s.kelly_half:>6.1%} {s.kelly_quarter:>7.1%}"
        )
    
    # 最佳推荐
    best = analysis.best_ev
    if best and best.ev >= min_ev:
        lines.append(f"  {'─' * 60}")
        lines.append(f"  🎯 最佳: {best.play} {best.pick} (EV={best.ev:+.1%}, "
                     f"Half-Kelly={best.kelly_half:.1%}, "
                     f"建议仓位={best.kelly_quarter:.1%})")
    
    # 风控提示: 同场多注相关性
    value_count = len([s for s in analysis.scenarios if s.is_value and s.ev >= min_ev])
    if value_count > 1:
        lines.append(f"  ⚠️  同场 {value_count} 注高度正相关, 实际风险≈单注最大仓位")
    
    return '\n'.join(lines)


def is_sane_bet(s: 'BetScenario') -> bool:
    """五道保险: 过滤长尾偏差、fallback 幻觉、高赔低概率"""
    # 保险1: 赔率 > 30 倍一律不碰 (数字海市蜃楼)
    if s.odds > 30.0:
        return False
    # 保险2: 概率 < 15% 一律不碰 (低信心不上榜)
    if s.prob < 0.15:
        return False
    # 保险3: market_fallback 场次禁推比分/半全场 (泊松外推不可信)
    if s.model_type == 'market_fallback' and s.play in ('比分', '半全场'):
        return False
    # 保险4: 赔率>5 且概率<25% 的高风险组合 (世界杯爆冷高发区)
    if s.odds > 5.0 and s.prob < 0.25:
        return False
    # 保险5: market_fallback 场次的胜平负/让球也降级 (EV循环论证)
    if s.model_type == 'market_fallback' and s.play in ('胜平负', '让球'):
        return False
    return True


def format_value_summary(all_analyses: list, min_ev: float = 0.05) -> str:
    """
    格式化输出所有有价值投注的汇总 (含风控过滤)
    
    Args:
        all_analyses: MatchAnalysis 列表
        min_ev: 最低 EV 阈值
    """
    value_bets = []
    filtered_out = 0
    for a in all_analyses:
        for s in a.scenarios:
            if s.ev >= min_ev:
                if is_sane_bet(s):
                    value_bets.append((a.home, a.away, s))
                else:
                    filtered_out += 1
    
    if not value_bets:
        return f"  📭 无价值投注 (EV > {min_ev:.0%})" + (f" (已过滤 {filtered_out} 个长尾/低概率选项)" if filtered_out else "")
    
    # 按 EV 降序
    value_bets.sort(key=lambda x: -x[2].ev)
    
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"  💎 价值投注汇总 (EV > {min_ev:.0%}, 已过滤赔率>30/概率<15%/fallback幻觉)")
    lines.append(f"{'=' * 60}")
    lines.append(f"  {'比赛':<20} {'玩法':<6} {'推荐':<8} {'赔率':>5} {'概率':>6} {'EV':>7} {'Kelly':>7}")
    lines.append(f"  {'─' * 55}")
    
    for home, away, s in value_bets[:15]:  # 最多显示 15 个
        match_str = f"{home[:8]} vs {away[:8]}"
        emoji = '🔥' if s.ev > 0.10 else '✅'
        lines.append(
            f"  {emoji}{match_str:<19} {s.play:<6} {s.pick:<8} "
            f"{s.odds:>5.2f} {s.prob:>5.1%} {s.ev:>+6.1%} {s.kelly_half:>6.1%}"
        )
    
    lines.append(f"  {'─' * 55}")
    lines.append(f"  📊 价值投注: {len(value_bets)} 个 (过滤{filtered_out}个) | "
                 f"最高 EV: {value_bets[0][2].ev:+.1%}")
    
    # 建议总仓位 (含相关性折扣)
    # 同一场比赛的多注视为 1 个独立风险单元
    match_groups = {}
    for home, away, s in value_bets:
        key = f"{home}_{away}"
        if key not in match_groups:
            match_groups[key] = []
        match_groups[key].append(s)
    
    # 每组取 max(kelly_quarter)，而非 sum (同场多注高度正相关)
    independent_total = 0.0
    for key, group in match_groups.items():
        max_kelly = max(s.kelly_quarter for s in group)
        independent_total += max_kelly
    
    lines.append(f"  💼 Quarter-Kelly 建议总仓位: {min(independent_total, 0.15):.1%} "
                 f"(上限 15% 单日, 已折扣同场相关性)")
    
    return '\n'.join(lines)


# ── 测试 ──

if __name__ == '__main__':
    print("=" * 50)
    print("bet_math.py 单元测试")
    print("=" * 50)
    
    # 测试 1: EV 计算
    # 模型说主胜 55%, 赔率 2.10 → EV = 0.55*(2.10-1) - 0.45 = 0.55*1.1 - 0.45 = 0.155
    ev1 = compute_ev(0.55, 2.10)
    print(f"\n测试1: prob=55%, odds=2.10 → EV={ev1:+.3f} (期望 +0.155)")
    assert abs(ev1 - 0.155) < 0.001
    
    # 测试 2: Kelly (含钳位: MAX_SINGLE_BET=0.05)
    # 原始 f* = 0.141, 钳位后 = 0.050
    k1 = compute_kelly(0.55, 2.10)
    print(f"测试2: Kelly f* = {k1:.3f} (钳位后期望 0.050)")
    assert abs(k1 - 0.050) < 0.005
    
    # 测试 3: 无价值 (模型概率 < 隐含概率)
    ev2 = compute_ev(0.30, 2.50)  # 隐含 40%, 模型 30%
    print(f"测试3: prob=30%, odds=2.50 → EV={ev2:+.3f} (期望 负值)")
    assert ev2 < 0
    
    # 测试 4: 边界条件
    assert compute_ev(0.0, 2.0) == 0.0
    assert compute_ev(1.0, 2.0) == 0.0
    assert compute_ev(0.5, 1.0) == 0.0
    assert compute_kelly(0.0, 2.0) == 0.0
    print("测试4: 边界条件 OK")
    
    # 测试 5: 完整分析
    analysis = analyze_match(
        home="Arsenal", away="Chelsea",
        predictions={
            'spf': {'h': 0.45, 'd': 0.25, 'a': 0.30},
            'rq': {'rq_win': 0.55, 'rq_draw': 0.20, 'rq_lose': 0.25},
            'score': [
                {'score': '1:0', 'prob': 0.15},
                {'score': '2:1', 'prob': 0.12},
                {'score': '1:1', 'prob': 0.11},
            ],
            'total_goals': [
                {'goals': 2, 'prob': 0.28},
                {'goals': 3, 'prob': 0.22},
                {'goals': 1, 'prob': 0.20},
            ],
        },
        odds={
            'spf': {'h': 2.05, 'd': 3.40, 'a': 3.60},
            'rq': {'rq_win': 1.85, 'rq_draw': 3.50, 'rq_lose': 2.00},
            'score': {'1:0': 5.50, '2:1': 7.00, '1:1': 6.50},
            'total_goals': {'2': 3.30, '3': 3.60, '1': 4.50},
        },
    )
    
    print("\n测试5: 完整分析")
    print(format_ev_table(analysis, min_ev=0.0))
    
    # 输出 Kelly 详情
    for s in analysis.scenarios:
        if s.is_value:
            print(f"  Kelly详解: {s.play} {s.pick} → "
                  f"Full={s.kelly_full:.1%} Half={s.kelly_half:.1%} "
                  f"Quarter={s.kelly_quarter:.1%}")
    
    print("\n✅ 全部测试通过")
