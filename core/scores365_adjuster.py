#!/usr/bin/env python3
"""
scores365_adjuster.py — 365scores 后验概率调整器
================================================
将 365scores 的投票/趋势/人气数据作为后验调整因子，
叠加到 DC+XGB 混合模型的预测概率上。

设计原则:
  1. 不改变模型主预测, 只做微调 (±5pp 以内)
  2. 投票数据有信息量时才调整, 否则跳过
  3. 严格校准: 调整方向和幅度基于回测验证

用法:
  from scores365_adjuster import adjust_with_365scores
  adjusted_probs = adjust_with_365scores(home, away, model_probs, match_date)
"""
import csv
import os
from typing import Dict, Optional, Tuple

DATA_DIR = '/root/data'
SCORES365_DIR = os.path.join(DATA_DIR, '365scores')


def load_365scores_for_date(date_str: str) -> Dict[str, dict]:
    """加载指定日期的 365scores 数据, 返回 {(home, away): row}."""
    path = os.path.join(SCORES365_DIR, f'{date_str}.csv')
    if not os.path.exists(path):
        return {}

    result = {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                home = row.get('home', '').strip()
                away = row.get('away', '').strip()
                if home and away:
                    result[(home, away)] = row
    except Exception:
        pass

    return result


def find_match_365scores(home: str, away: str, date_str: str,
                         fuzzy: bool = True) -> Optional[dict]:
    """在 365scores 数据中查找匹配的比赛."""
    data = load_365scores_for_date(date_str)
    if not data:
        return None

    # 精确匹配
    key = (home, away)
    if key in data:
        return data[key]

    # 反向匹配 (主客互换)
    key_rev = (away, home)
    if key_rev in data:
        return data[key_rev]

    if not fuzzy:
        return None

    # 模糊匹配 (子串)
    for (h, a), row in data.items():
        if (home.lower() in h.lower() or h.lower() in home.lower()) and \
           (away.lower() in a.lower() or a.lower() in away.lower()):
            return row

    return None


def compute_vote_signal(row: dict) -> Tuple[float, int]:
    """
    从投票数据计算信号强度和方向.
    
    返回: (adjustment_direction, sample_size)
      adjustment_direction: -1.0 ~ +1.0 (负=倾向客队, 正=倾向主队)
      sample_size: 投票数量 (越大越可靠)
    """
    try:
        vote_h = float(row.get('vote_home', 0))
        vote_d = float(row.get('vote_draw', 0))
        vote_a = float(row.get('vote_away', 0))
        vote_count = int(float(row.get('vote_count', 0)))
    except (ValueError, TypeError):
        return 0.0, 0

    if vote_count < 50:
        return 0.0, vote_count

    # 公众倾向: 主队支持率 - 客队支持率 (归一化到 -1 ~ +1)
    total = vote_h + vote_d + vote_a
    if total <= 0:
        return 0.0, vote_count

    direction = (vote_h - vote_a) / total  # -1 ~ +1
    return direction, vote_count


def compute_trend_signal(row: dict) -> float:
    """
    从趋势数据计算状态信号.
    
    返回: -1.0 ~ +1.0 (负=客队状态更好, 正=主队状态更好)
    """
    try:
        wr_home = float(row.get('trend_win_rate_home', 0.5))
        wr_away = float(row.get('trend_win_rate_away', 0.5))
    except (ValueError, TypeError):
        return 0.0

    # 胜率差异, 归一化到 -1 ~ +1
    diff = wr_home - wr_away
    return max(-1.0, min(1.0, diff))


def compute_popularity_signal(row: dict) -> float:
    """
    从人气排名计算信号.
    
    返回: -1.0 ~ +1.0 (负=客队更受欢迎, 正=主队更受欢迎)
    """
    try:
        rank_home = int(row.get('pop_rank_home', 0))
        rank_away = int(row.get('pop_rank_away', 0))
    except (ValueError, TypeError):
        return 0.0

    if rank_home <= 0 or rank_away <= 0:
        return 0.0

    # 排名差异 (低排名=更受欢迎)
    diff = rank_away - rank_home  # 正=主队更受欢迎
    # 归一化: 差异10位以上才有显著信号
    return max(-1.0, min(1.0, diff / 10.0))


def compute_fifa_rank_signal(row: dict) -> float:
    """
    从 FIFA 排名计算实力差距信号.
    
    返回: -1.0 ~ +1.0 (负=客队更强, 正=主队更强)
    """
    try:
        rank_home = int(row.get('fifa_rank_home', 0))
        rank_away = int(row.get('fifa_rank_away', 0))
    except (ValueError, TypeError):
        return 0.0

    if rank_home <= 0 or rank_away <= 0:
        return 0.0

    # FIFA 排名: 数字越小越强
    # diff = away_rank - home_rank: 正=主队排名更高(更强)
    diff = rank_away - rank_home
    # 归一化: 排名差20位以上才有显著信号
    # 例: 巴西(1) vs 尼日利亚(26) → diff=25 → signal=+1.0
    return max(-1.0, min(1.0, diff / 20.0))


def adjust_with_365scores(home: str, away: str,
                          model_probs: Dict[str, float],
                          date_str: str,
                          max_adjustment: float = 0.05) -> Dict[str, float]:
    """
    用 365scores 数据微调模型预测概率.
    
    Args:
        home: 主队名
        away: 客队名
        model_probs: {'H': p_h, 'D': p_d, 'A': p_a} 模型原始预测
        date_str: 比赛日期 (YYYY-MM-DD)
        max_adjustment: 最大调整幅度 (默认 5pp)
    
    Returns:
        调整后的概率 dict
    """
    row = find_match_365scores(home, away, date_str)
    if row is None:
        return model_probs

    p_h = model_probs.get('H', 1/3)
    p_d = model_probs.get('D', 1/3)
    p_a = model_probs.get('A', 1/3)

    # 计算各信号
    vote_dir, vote_n = compute_vote_signal(row)
    trend_dir = compute_trend_signal(row)
    pop_dir = compute_popularity_signal(row)
    fifa_dir = compute_fifa_rank_signal(row)

    # 信号加权融合
    # FIFA 排名权重最高 (客观实力), 投票次之 (市场情绪), 趋势再次, 人气最弱
    if vote_n >= 200:
        # 大样本投票: FIFA + 投票为主
        combined = 0.35 * fifa_dir + 0.35 * vote_dir + 0.2 * trend_dir + 0.1 * pop_dir
        confidence = min(1.0, vote_n / 500)
    elif vote_n >= 50:
        # 中等样本: FIFA 权重更高
        combined = 0.4 * fifa_dir + 0.25 * vote_dir + 0.25 * trend_dir + 0.1 * pop_dir
        confidence = min(0.7, vote_n / 300)
    else:
        # 小样本或无投票: FIFA + 趋势为主
        combined = 0.5 * fifa_dir + 0.0 * vote_dir + 0.35 * trend_dir + 0.15 * pop_dir
        confidence = 0.4

    # 将 combined 方向转换为概率调整
    # combined > 0 → 主队概率增加, 客队概率减少
    # combined < 0 → 反之
    adj = combined * max_adjustment * confidence

    new_h = p_h + adj
    new_a = p_a - adj
    new_d = p_d  # 平局不变 (投票数据对平局的预测力很弱)

    # 归一化
    total = new_h + new_d + new_a
    if total > 0:
        new_h /= total
        new_d /= total
        new_a /= total

    return {'H': new_h, 'D': new_d, 'A': new_a}


def get_365scores_features(home: str, away: str, date_str: str) -> Optional[Dict[str, float]]:
    """
    提取 365scores 特征 (供未来 XGB 重训练使用).
    
    Returns: dict of feature_name -> value, or None if no data
    """
    row = find_match_365scores(home, away, date_str)
    if row is None:
        return None

    features = {}
    try:
        # 投票特征
        features['vote_home'] = float(row.get('vote_home', 0)) / 100.0
        features['vote_draw'] = float(row.get('vote_draw', 0)) / 100.0
        features['vote_away'] = float(row.get('vote_away', 0)) / 100.0
        features['vote_count'] = float(row.get('vote_count', 0))
        features['vote_count_log'] = __import__('math').log(float(row.get('vote_count', 0)) + 1)

        # 投票 vs 模型差异 (需要模型概率作为输入)
        # 这个在外部计算

        # 人气排名特征
        rh = int(row.get('pop_rank_home', 0))
        ra = int(row.get('pop_rank_away', 0))
        features['pop_rank_home'] = rh
        features['pop_rank_away'] = ra
        features['pop_rank_diff'] = ra - rh  # 正=主队更受欢迎
        features['pop_rank_log_diff'] = __import__('math').log(ra + 1) - __import__('math').log(rh + 1)

        # 趋势特征
        features['trend_win_rate_home'] = float(row.get('trend_win_rate_home', 0.5))
        features['trend_win_rate_away'] = float(row.get('trend_win_rate_away', 0.5))
        features['trend_win_rate_diff'] = float(row.get('trend_win_rate_diff', 0))

        # 趋势积分
        tw_h = int(float(row.get('trend_home_w', 0)))
        td_h = int(float(row.get('trend_home_d', 0)))
        tl_h = int(float(row.get('trend_home_l', 0)))
        tw_a = int(float(row.get('trend_away_w', 0)))
        td_a = int(float(row.get('trend_away_d', 0)))
        tl_a = int(float(row.get('trend_away_l', 0)))
        total_h = tw_h + td_h + tl_h
        total_a = tw_a + td_a + tl_a
        if total_h > 0:
            features['trend_points_home'] = (tw_h * 3 + td_h) / total_h / 3.0
        else:
            features['trend_points_home'] = 1.0
        if total_a > 0:
            features['trend_points_away'] = (tw_a * 3 + td_a) / total_a / 3.0
        else:
            features['trend_points_away'] = 1.0
        features['trend_points_diff'] = features['trend_points_home'] - features['trend_points_away']

        # FIFA 排名特征
        fifa_h = int(row.get('fifa_rank_home', 0))
        fifa_a = int(row.get('fifa_rank_away', 0))
        features['fifa_rank_home'] = fifa_h
        features['fifa_rank_away'] = fifa_a
        features['fifa_rank_diff'] = fifa_a - fifa_h  # 正=主队排名更高(更强)
        if fifa_h > 0 and fifa_a > 0:
            features['fifa_rank_log_diff'] = __import__('math').log(fifa_a) - __import__('math').log(fifa_h)
        else:
            features['fifa_rank_log_diff'] = 0.0

    except (ValueError, TypeError, KeyError):
        return None

    return features


if __name__ == '__main__':
    # 测试
    test_probs = {'H': 0.45, 'D': 0.25, 'A': 0.30}
    adjusted = adjust_with_365scores('France', 'England', test_probs, '2026-06-08')
    print(f"原始: {test_probs}")
    print(f"调整: {adjusted}")

    features = get_365scores_features('France', 'England', '2026-06-08')
    if features:
        print(f"特征: {features}")
    else:
        print("无 365scores 数据")
