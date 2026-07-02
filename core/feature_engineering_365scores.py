#!/usr/bin/env python3
"""
365scores 特征工程
=================
将 365scores 数据转换为 XGBoost 特征

用法:
  python3 feature_engineering_365scores.py
"""

import numpy as np

def extract_features(scores365_match):
    """从 365scores 数据提取特征"""
    if not scores365_match:
        return None
    
    features = {}
    
    # 1. 人气排名特征
    pop_rank_home = scores365_match.get('pop_rank_home')
    pop_rank_away = scores365_match.get('pop_rank_away')
    
    if pop_rank_home and pop_rank_away:
        features['pop_rank_home'] = pop_rank_home
        features['pop_rank_away'] = pop_rank_away
        features['pop_rank_diff'] = pop_rank_away - pop_rank_home
        
        # 人气排名对数差异 (归一化)
        features['pop_rank_log_diff'] = np.log(pop_rank_away + 1) - np.log(pop_rank_home + 1)
    
    # 2. 趋势数据特征
    trend_home = scores365_match.get('trend_home', [])
    trend_away = scores365_match.get('trend_away', [])
    
    if trend_home and trend_away and len(trend_home) >= 3 and len(trend_away) >= 3:
        wins_home = trend_home[0]
        draws_home = trend_home[1]
        losses_home = trend_home[2]
        total_home = wins_home + draws_home + losses_home
        
        wins_away = trend_away[0]
        draws_away = trend_away[1]
        losses_away = trend_away[2]
        total_away = wins_away + draws_away + losses_away
        
        if total_home > 0 and total_away > 0:
            win_rate_home = wins_home / total_home
            win_rate_away = wins_away / total_away
            
            features['trend_win_rate_home'] = win_rate_home
            features['trend_win_rate_away'] = win_rate_away
            features['trend_win_rate_diff'] = win_rate_home - win_rate_away
            
            # 近期状态积分 (胜=3, 平=1, 负=0)
            points_home = wins_home * 3 + draws_home * 1
            points_away = wins_away * 3 + draws_away * 1
            features['trend_points_home'] = points_home / total_home
            features['trend_points_away'] = points_away / total_away
            features['trend_points_diff'] = (points_home / total_home) - (points_away / total_away)
    
    # 3. 投票数据特征
    votes = scores365_match.get('votes')
    if votes and votes.get('total', 0) > 0:
        features['vote_home'] = votes['home'] / 100
        features['vote_draw'] = votes['draw'] / 100
        features['vote_away'] = votes['away'] / 100
        features['vote_count'] = votes['total']
        features['vote_count_log'] = np.log(votes['total'] + 1)
        
        # 投票与模型差异 (需要模型概率)
        # 这里暂时不计算，需要在预测时添加
    
    return features

def main():
    """主函数"""
    print("=== 365scores 特征工程 ===")
    print()
    
    # 测试数据
    test_match = {
        'home': 'Spain',
        'away': 'Iraq',
        'votes': {'home': 39.1, 'draw': 10.3, 'away': 50.6, 'total': 222839},
        'trend_home': [2, 1, 2, 0, 0],
        'trend_away': [1, 1, 0, 0, 0],
        'pop_rank_home': 25377,
        'pop_rank_away': 11211,
    }
    
    features = extract_features(test_match)
    
    print("提取的特征:")
    for k, v in features.items():
        print(f"  {k}: {v}")
    
    print()
    print("=== 特征列表 ===")
    print()
    print("1. 人气排名特征 (3个):")
    print("   - pop_rank_home: 主队人气排名")
    print("   - pop_rank_away: 客队人气排名")
    print("   - pop_rank_diff: 人气排名差异")
    print("   - pop_rank_log_diff: 人气排名对数差异")
    print()
    print("2. 趋势数据特征 (6个):")
    print("   - trend_win_rate_home: 主队近期胜率")
    print("   - trend_win_rate_away: 客队近期胜率")
    print("   - trend_win_rate_diff: 胜率差异")
    print("   - trend_points_home: 主队近期积分")
    print("   - trend_points_away: 客队近期积分")
    print("   - trend_points_diff: 积分差异")
    print()
    print("3. 投票数据特征 (4个):")
    print("   - vote_home: 主队投票比例")
    print("   - vote_draw: 平局投票比例")
    print("   - vote_away: 客队投票比例")
    print("   - vote_count: 投票人数")
    print("   - vote_count_log: 投票人数对数")
    print()
    print("=== 下一步 ===")
    print()
    print("1. 将这些特征添加到 XGBoost 模型")
    print("2. 重新训练模型 (需要大量历史数据)")
    print("3. 验证特征重要性")
    print("4. 优化特征组合")

if __name__ == "__main__":
    main()
