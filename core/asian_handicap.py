"""
asian_handicap.py — 亚盘赢盘概率计算器
========================================
用 Skellam (双泊松差分布) 计算任意亚盘盘口的赢盘概率。

核心原理:
  P(主队净胜 = k) = e^{-(λ₁+λ₂)} * (λ₁/λ₂)^{k/2} * I_{|k|}(2√(λ₁λ₂))
  其中 I_{|k|} 是修正贝塞尔函数。

对于亚盘主队让 h 球 (如 h=0.75, h=1.25):
  - 赢盘: P(主队净胜 > h)
  - 半赢: P(主队净胜 = h) × 0.5  (部分赢)
  - 走水: P(主队净胜 = h) × 0.5  (退款)
  - 半输: P(主队净胜 = h - 0.25) 等复杂情况
  - 输盘: P(主队净胜 < h)

标准处理:
  h 为 quarter (0.25 倍数):
    h % 0.5 == 0   → 整数盘 (无走水)
    h % 0.5 == 0.25 → 四分之一盘 (部分赢/输)

注: 亚盘常见的 h 有:
  - 平手 0     = 主队不输即赢盘
  - 平半 0.25  = 主队赢→全赢, 平→半输
  - 半球 0.5   = 主队赢→全赢, 平/输→全输 (无走水)
  - 半一 0.75  = 主队赢1→半赢, 赢2+→全赢
  - 一球 1.0   = 主队赢1→走水, 赢2+→全赢
  - 一球/球半 1.25 = 主队赢1→半输, 赢2+→全赢

用法:
  from asian_handicap import ah_probs, find_ah_ev

  # 1. 给定 λ_home=1.8, λ_away=0.9, 盘口 h=1.0
  p = ah_probs(1.8, 0.9, 1.0)
  # → {'win': 0.45, 'push': 0.15, 'lose': 0.40}
  
  # 2. EV 对撞
  odds = 1.95  # 亚盘赔率
  ev = find_ah_ev(1.8, 0.9, 1.0, odds)
  # → {'ev': 0.08, 'fair_odds': 2.22, 'edge': 'positive' if ev>0 else 'negative'}
"""

import math
from scipy.special import iv as bessel_i  # 修正贝塞尔函数


# ──────────────────────────────────────────────
# 核心: Skellam 分布 PMF
# ──────────────────────────────────────────────

def skellam_pmf(k, lam1, lam2):
    """
    Skellam 分布概率质量函数: P(净胜球 = k)
    其中 k = 主队进球 - 客队进球
    
    公式: P(K=k) = e^{-(λ₁+λ₂)} (λ₁/λ₂)^{k/2} I_{|k|}(2√(λ₁λ₂))
    
    Args:
        k: 净胜球 (整数, 可为负)
        lam1: 主队预期进球 (λ_home)
        lam2: 客队预期进球 (λ_away)
    
    Returns:
        float: 概率
    """
    log_product = math.exp(-(lam1 + lam2))
    if k == 0:
        return log_product * bessel_i(0, 2 * math.sqrt(lam1 * lam2))
    ratio = (lam1 / lam2) ** (k / 2.0) if lam2 > 0 else 0.0
    return log_product * ratio * bessel_i(abs(k), 2 * math.sqrt(lam1 * lam2))


def skellam_cdf(k, lam1, lam2):
    """
    Skellam 累积分布: P(净胜球 ≤ k)
    
    Args:
        k: 阈值 (整数)
        lam1: 主队λ
        lam2: 客队λ
    
    Returns:
        float: 累积概率
    """
    # 对 k=±30 截断 (概率已可忽略)
    k_max = max(abs(k) + 20, 30)
    total = 0.0
    for i in range(-k_max, k + 1):
        total += skellam_pmf(i, lam1, lam2)
    return total


def skellam_sf(k, lam1, lam2):
    """
    Skellam 生存函数: P(净胜球 > k)
    """
    return 1.0 - skellam_cdf(k, lam1, lam2)


# ──────────────────────────────────────────────
# 亚盘概率计算
# ──────────────────────────────────────────────

def _round_to_quarter(h):
    """四舍五入到最接近的 0.25 倍数"""
    return round(h * 4) / 4.0


def ah_probs(lam_home, lam_away, handicap, max_goals=15):
    """
    计算给定 λ 和亚盘盘口的分布。

    Args:
        lam_home: 主队预期进球
        lam_away: 客队预期进球
        handicap: 亚盘盘口 (主队让球数, 负值=主队受让)
                  如: 1.0 (一球), 0.75 (半一), 0.25 (平半),
                      0 (平手), -0.5 (客让半球)
        max_goals: 最大净胜球计算范围
    
    Returns:
        dict: {
            'handicap': float,  # 标准化后的盘口
            'prob_win': float,  # 赢盘概率 (含半赢折算)
            'prob_push': float, # 走水概率 (退款)
            'prob_lose': float, # 输盘概率 (含半输折算)
            'effective_prob': float,  # 有效赢盘概率 (赢盘+0.5*半赢)
            'fair_odds': float,  # 公平赔率 (无抽水)
            'overround': float,  # 有效抽水率
        }
    
    Example:
        >>> ah_probs(1.8, 0.9, 1.0, max_goals=10)
        {'handicap': 1.0, 'prob_win': 0.45, 'prob_push': 0.15,
         'prob_lose': 0.40, 'effective_prob': 0.525, 'fair_odds': 1.90}
    """
    h = _round_to_quarter(handicap)
    
    # 预计算净胜球分布
    goal_diffs = {}
    for k in range(-max_goals, max_goals + 1):
        goal_diffs[k] = skellam_pmf(k, lam_home, lam_away)
    
    # 用 abs(h) 判断: 负盘口 = 我们从客队角度算
    is_away = h < 0
    abs_h = abs(h)
    
    # 计算赢盘/走水/输盘
    # 亚盘规则:
    # 整数盘 h: P(净胜 > h) = 赢, P(净胜 = h) = 走水, P(净胜 < h) = 输
    # 半盘 h=0.5: P(净胜 > 0) = 赢, P(净胜 ≤ 0) = 输 (无走水)
    # 四分之一盘 h=0.75: P(净胜 > 1) = 赢, P(净胜 = 1)=半赢, P(净胜 ≤ 0)=输
    # 四分之一盘 h=0.25: P(净胜 > 0) = 赢, P(净胜 = 0)=半输, P(净胜 < 0)=输
    
    # 对于客队角度 (h<0), 反转净胜球符号
    if is_away:
        # AH 负数: 从客队看, 净胜 = 客队进球 - 主队进球 = -(主队净胜)
        # 客队赢盘 ⇔ 主队净胜 < abs_h
        # 规则与主队对称:
        a_h = abs_h
        a_int = int(math.floor(a_h))
        a_frac = a_h - a_int
        
        if a_frac == 0:
            # 整数盘: 客赢=主净胜<h, 走水=主净胜=h, 客输=主净胜>h
            win_p = sum(goal_diffs.get(k, 0) for k in range(-max_goals, a_int))
            push_p = goal_diffs.get(a_int, 0)
            lose_p = sum(goal_diffs.get(k, 0) for k in range(a_int + 1, max_goals + 1))
        elif a_frac == 0.5:
            # 半盘: 无走水
            win_p = sum(goal_diffs.get(k, 0) for k in range(-max_goals, a_int + 1))
            push_p = 0.0
            lose_p = sum(goal_diffs.get(k, 0) for k in range(a_int + 1, max_goals + 1))
        elif a_frac == 0.25:
            # 客让0.25: 客赢=主净胜≤0, 半输=主净胜=1, 输=主净胜≥2
            win_p = sum(goal_diffs.get(k, 0) for k in range(-max_goals, a_int + 1))
            half_loss_p = goal_diffs.get(a_int + 1, 0) * 0.5
            push_p = half_loss_p
            lose_p = goal_diffs.get(a_int + 1, 0) * 0.5 + sum(
                goal_diffs.get(k, 0) for k in range(a_int + 2, max_goals + 1))
        elif a_frac == 0.75:
            # 客让半一: 客赢=主净胜≤a_int+0.5×主净胜=a_int+1, 半赢=主净胜=a_int+1, 输=主净胜≥a_int+2
            base_win = sum(goal_diffs.get(k, 0) for k in range(-max_goals, a_int + 1))
            half_win_p = goal_diffs.get(a_int + 1, 0) * 0.5
            win_p = base_win + half_win_p
            push_p = goal_diffs.get(a_int + 1, 0) * 0.5
            lose_p = sum(goal_diffs.get(k, 0) for k in range(a_int + 2, max_goals + 1))
        else:
            win_p = sum(goal_diffs.get(k, 0) for k in range(-max_goals, a_int))
            push_p = goal_diffs.get(a_int, 0)
            lose_p = sum(goal_diffs.get(k, 0) for k in range(a_int + 1, max_goals + 1))
    else:
        # 主队角度
        h_int = int(math.floor(h))
        frac = h - h_int
        
        if frac == 0:
            # 整数盘: 赢=P(净胜>h), 走水=P(净胜=h), 输=P(净胜<h)
            win_p = sum(goal_diffs.get(k, 0) for k in range(h_int + 1, max_goals + 1))
            push_p = goal_diffs.get(h_int, 0)
            lose_p = sum(goal_diffs.get(k, 0) for k in range(-max_goals, h_int))
        elif frac == 0.5:
            # 半盘 (无走水)
            win_p = sum(goal_diffs.get(k, 0) for k in range(h_int + 1, max_goals + 1))
            push_p = 0.0
            lose_p = sum(goal_diffs.get(k, 0) for k in range(-max_goals, h_int + 1))
        elif frac == 0.25:
            # 四分之一: h=0.25 → 赢=净胜≥1, 半输=净胜=0, 输=净胜≤-1
            #   h=1.25 → 赢=净胜≥2, 半输=净胜=1, 输=净胜≤0
            #   核心: 赢=净胜≥h_int+1, 半输=净胜=h_int, 输=净胜≤h_int-1
            win_p = sum(goal_diffs.get(k, 0) for k in range(h_int + 1, max_goals + 1))
            half_loss_p = goal_diffs.get(h_int, 0) * 0.5  # 净胜=h_int时半输
            push_p = half_loss_p
            lose_p = goal_diffs.get(h_int, 0) * 0.5 + sum(
                goal_diffs.get(k, 0) for k in range(-max_goals, h_int))
        elif frac == 0.75:
            # 四分之三: h=0.75 → 赢=净胜≥2+0.5×净胜=1, 半赢=净胜=1, 输=净胜≤0
            #   h=1.75 → 赢=净胜≥3+0.5×净胜=2, 半赢=净胜=2, 输=净胜≤1
            base_win = sum(goal_diffs.get(k, 0) for k in range(h_int + 2, max_goals + 1))
            half_win_p = goal_diffs.get(h_int + 1, 0) * 0.5
            win_p = base_win + half_win_p
            push_p = goal_diffs.get(h_int + 1, 0) * 0.5
            lose_p = sum(goal_diffs.get(k, 0) for k in range(-max_goals, h_int + 1))
        else:
            win_p = sum(goal_diffs.get(k, 0) for k in range(h_int + 1, max_goals + 1))
            push_p = goal_diffs.get(h_int, 0)
            lose_p = sum(goal_diffs.get(k, 0) for k in range(-max_goals, h_int))
    
    # 有效概率 (用于公平赔率计算)
    effective_prob = win_p + push_p * 0.5
    if effective_prob > 0:
        fair_odds = 1.0 / effective_prob
    else:
        fair_odds = 0.0
    
    return {
        'handicap': h,
        'prob_win': round(win_p, 4),
        'prob_push': round(push_p, 4),
        'prob_lose': round(lose_p, 4),
        'effective_prob': round(effective_prob, 4),
        'fair_odds': round(fair_odds, 4),
    }


def find_ah_odds(lam_home, lam_away, handicap, market_odds, overround=0.03):
    """
    计算亚盘价值投注。

    Args:
        lam_home: 主队λ
        lam_away: 客队λ
        handicap: 亚盘盘口
        market_odds: 市场赔率 (主队让球方赔率)
        overround: 市场抽水率 (默认3% 亚盘标准)
    
    Returns:
        dict: {
            'handicap': float,
            'fair_odds': float,  # 公平赔率
            'market_odds': float,  # 市场赔率
            'ev': float,  # 预期价值
            'kelly_pct': float,  # Kelly建议比例
            'edge': str,  # 'positive'/'negative'/'neutral'
            'prob_win_eff': float,  # 模型有效赢盘概率
        }
    """
    probs = ah_probs(lam_home, lam_away, handicap)
    p_eff = probs['effective_prob']
    
    if p_eff <= 0 or market_odds <= 1:
        return None
    
    # 市场真实概率 (去抽水)
    market_prob = 1.0 / market_odds  # 包含抽水
    fair_prob = p_eff
    
    # EV
    ev = p_eff * (market_odds - 1) - (1 - p_eff)
    
    # Kelly
    b = market_odds - 1
    q = 1 - p_eff
    if b > 0:
        kelly = (p_eff * b - q) / b
    else:
        kelly = 0.0
    
    return {
        'handicap': probs['handicap'],
        'fair_odds': probs['fair_odds'],
        'market_odds': market_odds,
        'ev': round(ev, 4),
        'kelly_pct': round(max(kelly * 0.25, 0), 4),  # 1/4 Kelly
        'edge': 'positive' if ev > 0 else ('negative' if ev < 0 else 'neutral'),
        'prob_win_eff': round(p_eff, 4),
    }


def scan_ah_value(lam_home, lam_away, asian_lines):
    """
    批量扫描多个亚盘盘口, 找出价值投注。

    Args:
        lam_home: 主队λ
        lam_away: 客队λ
        asian_lines: list of (handicap, odds)
                     [(0.5, 1.85), (0.75, 2.05), (1.0, 2.20), ...]
    
    Returns:
        list[dict] 按EV排序, 仅>0
    """
    results = []
    for h, odds in asian_lines:
        r = find_ah_odds(lam_home, lam_away, h, odds)
        if r and r['ev'] > 0:
            results.append(r)
    return sorted(results, key=lambda x: x['ev'], reverse=True)


# ──────────────────────────────────────────────
# 测试
# ──────────────────────────────────────────────

if __name__ == '__main__':
    # 示例: 荷兰 vs 乌兹别克 (λ=2.0, 0.8)
    print("=== 荷兰 vs 乌兹别克 (λ_h=2.0, λ_a=0.8) ===")
    for h in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        p = ah_probs(2.0, 0.8, h)
        ev = find_ah_odds(2.0, 0.8, h, 1.0/p['effective_prob'] * 1.05)
        print(f"  AH {h:>5.2f}: win={p['prob_win']:.1%} push={p['prob_push']:.1%} "
              f"lose={p['prob_lose']:.1%} eff={p['effective_prob']:.1%} "
              f"fair={p['fair_odds']:.2f} "
              f"EV={ev['ev']*100 if ev else 0:.1f}%")
    
    print()
    print("=== 法国 vs 北爱尔兰 (λ_h=2.5, λ_a=0.5) ===")
    for h in [0, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5]:
        p = ah_probs(2.5, 0.5, h)
        print(f"  AH {h:>5.2f}: eff={p['effective_prob']:.1%} fair={p['fair_odds']:.2f}")
