"""
fatigue_features.py — 疲劳度特征计算器
========================================
从 500.com 未来赛事数据计算球队疲劳度/轮换风险特征。

核心逻辑:
  - 世界杯前N天的友谊赛 → 轮换概率极高 (coach保护主力)
  - 连续客场远征 → 体能消耗大
  - 3天内有比赛 → 疲劳累积
  - 友谊赛 vs 正赛 → 动机差异

输出特征 (可接入 XGB 模型):
  - days_to_next_match: 距下一场天数
  - next_match_importance: 下一场重要度 (友谊赛=1, 预选赛=2, 世界杯=3)
  - rotation_risk: 轮换风险 (0-1)
  - fatigue_score: 疲劳度分数 (0-1)
  - match_importance: 本场重要度 (友谊赛=1, 预选赛=2, 世界杯=3)

供 daily_jczq.py 使用:
  from fatigue_features import compute_fatigue_features
"""

from datetime import datetime, date, timedelta


# 比赛重要度映射
IMPORTANCE_MAP = {
    '友谊赛': 1,
    '国际友谊': 1,
    '热身赛': 1,
    '球会友谊': 1,
    '中亚杯': 2,
    '麒麟杯': 2,
    '世外欧洲': 2,
    '世外亚洲': 2,
    '世外南美': 2,
    '世外非洲': 2,
    '世外北美': 2,
    '世外附': 2,
    '世界杯预赛': 2,
    '欧国联': 2,
    '亚洲杯': 2,
    '美洲杯': 2,
    '欧洲杯': 2,
    '世界杯': 3,
    '世界杯决赛周': 3,
}


def _parse_date(date_str):
    """解析日期字符串, 支持多种格式"""
    date_str = date_str.strip()
    # MM-DD HH:MM 格式 (500.com常用) → 补年份
    if len(date_str) == 11 and date_str[2] == '-' and date_str[5] == ' ':
        try:
            return datetime.strptime(f"2026-{date_str[:5]}", '%Y-%m-%d').date()
        except ValueError:
            pass
    for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M', '%y-%m-%d', '%y-%m-%d %H:%M',
                '%m-%d %H:%M', '%m-%d']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def _get_importance(competition):
    """获取比赛重要度 (1-3)"""
    for key, val in IMPORTANCE_MAP.items():
        if key in competition:
            return val
    return 1  # 默认友谊赛级别


def compute_fatigue_features(home, away, match_date, match_competition, future_fixtures):
    """
    计算疲劳度特征。

    Args:
        home: 主队名
        away: 客队名
        match_date: 比赛日期 (str: '2026-06-09' 或 date对象)
        match_competition: 本场赛事类型 (str)
        future_fixtures: 未来赛事列表 (从500.com爬取)
            [{'competition': '世界杯', 'date': '2026-06-15', 'home': '荷兰', 'away': '日本'}, ...]

    Returns:
        dict of fatigue features
    """
    if isinstance(match_date, str):
        match_date = _parse_date(match_date)
    if not match_date:
        return {}

    features = {}

    # 本场重要度
    features['match_importance'] = _get_importance(match_competition)

    # 分别计算主客队的疲劳度
    for team_label, team_name in [('home', home), ('away', away)]:
        # 找该队的未来赛事
        team_fixtures = [
            f for f in future_fixtures
            if team_name in f.get('home', '') or team_name in f.get('away', '')
        ]

        if not team_fixtures:
            features[f'{team_label}_days_to_next'] = 99
            features[f'{team_label}_next_importance'] = 0
            features[f'{team_label}_rotation_risk'] = 0.0
            features[f'{team_label}_fatigue'] = 0.0
            continue

        # 找最近的下一场
        next_matches = []
        for f in team_fixtures:
            fdate = _parse_date(f.get('date', ''))
            if fdate and fdate > match_date:
                next_matches.append((fdate, f))
        next_matches.sort(key=lambda x: x[0])

        if next_matches:
            next_date, next_match = next_matches[0]
            days_to_next = (next_date - match_date).days
            next_importance = _get_importance(next_match.get('competition', ''))
        else:
            days_to_next = 99
            next_importance = 0

        features[f'{team_label}_days_to_next'] = days_to_next
        features[f'{team_label}_next_importance'] = next_importance

        # 轮换风险计算
        # 核心逻辑: 下一场越重要 + 距离越近 → 本场轮换概率越高
        rotation_risk = 0.0
        if next_importance >= 3:  # 下一场是世界杯
            if days_to_next <= 3:
                rotation_risk = 0.95  # 几乎确定轮换
            elif days_to_next <= 7:
                rotation_risk = 0.70  # 高概率轮换
            elif days_to_next <= 14:
                rotation_risk = 0.40  # 中等轮换
            else:
                rotation_risk = 0.10  # 低轮换
        elif next_importance >= 2:  # 下一场是预选赛/杯赛
            if days_to_next <= 3:
                rotation_risk = 0.60
            elif days_to_next <= 7:
                rotation_risk = 0.30
            else:
                rotation_risk = 0.05
        else:
            rotation_risk = 0.0  # 下一场也是友谊赛, 无需保护

        # 如果本场就是世界杯, 不存在轮换
        if features['match_importance'] >= 3:
            rotation_risk = 0.0

        features[f'{team_label}_rotation_risk'] = round(rotation_risk, 2)

        # 疲劳度 (综合轮换风险 + 比赛密度)
        fatigue = rotation_risk * 0.7  # 轮换是主要因素
        if days_to_next <= 3:
            fatigue += 0.2  # 密集赛程加成
        features[f'{team_label}_fatigue'] = round(min(fatigue, 1.0), 2)

    # 双方轮换差异 (用于调整预测)
    h_rot = features.get('home_rotation_risk', 0)
    a_rot = features.get('away_rotation_risk', 0)
    features['rotation_diff'] = round(h_rot - a_rot, 2)  # 正=主队更可能轮换

    return features


def fatigue_adjustment(features, base_probs):
    """
    基于疲劳度调整预测概率。

    Args:
        features: compute_fatigue_features 的输出
        base_probs: {'H': 0.5, 'D': 0.25, 'A': 0.25} 原始预测

    Returns:
        dict: 调整后的概率
    """
    rot_diff = features.get('rotation_diff', 0)

    if abs(rot_diff) < 0.1:
        return base_probs  # 无显著差异, 不调整

    # 主队轮换风险更高 → 降低主胜概率
    # 客队轮换风险更高 → 降低客胜概率
    adjustment = rot_diff * 0.05  # 最大调整5pp

    h = base_probs['H'] - adjustment
    a = base_probs['A'] + adjustment
    d = base_probs['D']

    # 归一化
    total = h + d + a
    if total > 0:
        h /= total
        d /= total
        a /= total

    return {'H': max(h, 0.01), 'D': max(d, 0.01), 'A': max(a, 0.01)}


def format_fatigue_lines(features):
    """格式化疲劳度特征为终端展示行"""
    lines = []

    for team_label, emoji in [('home', '🏠'), ('away', '✈️')]:
        days = features.get(f'{team_label}_days_to_next', 99)
        importance = features.get(f'{team_label}_next_importance', 0)
        rot = features.get(f'{team_label}_rotation_risk', 0)
        fatigue = features.get(f'{team_label}_fatigue', 0)

        if days < 99:
            imp_label = ['', '友谊赛', '预选赛/杯赛', '世界杯'][importance] if importance else '-'
            risk_emoji = '🟢' if rot < 0.2 else ('🟡' if rot < 0.5 else '🔴')
            lines.append(
                f"     {emoji} {team_label}: 下一场{days}天后({imp_label}) "
                f"轮换风险{rot:.0%} {risk_emoji} 疲劳{fatigue:.0%}"
            )

    rot_diff = features.get('rotation_diff', 0)
    if abs(rot_diff) >= 0.1:
        if rot_diff > 0:
            lines.append(f"     ⚠️ 主队轮换风险更高 ({rot_diff:+.0%}), 可能降低主胜概率")
        else:
            lines.append(f"     ⚠️ 客队轮换风险更高 ({rot_diff:+.0%}), 可能降低客胜概率")

    return lines


# 测试
if __name__ == '__main__':
    # 荷兰 vs 乌兹别克斯坦 (世界杯前6天友谊赛)
    fixtures = [
        {'competition': '世界杯', 'date': '2026-06-15', 'home': '荷兰', 'away': '日本'},
        {'competition': '世界杯', 'date': '2026-06-18', 'home': '乌兹别克斯坦', 'away': '哥伦比亚'},
    ]
    feats = compute_fatigue_features('荷兰', '乌兹别克斯坦', '2026-06-09', '友谊赛', fixtures)
    print("疲劳度特征:")
    for k, v in feats.items():
        print(f"  {k}: {v}")
    print("\n展示:")
    for line in format_fatigue_lines(feats):
        print(line)
