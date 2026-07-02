#!/usr/bin/env python3
"""
365scores A/B 测试框架
=====================
对比不同权重配置的预测效果

用法:
  python3 ab_test_365scores.py
"""

import json
import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, '/root')

from predict_match import predict_match
from fetch_365scores import fetch_365scores_data, extract_games

def predict_with_config(home_en, away_en, vote_data, config):
    """使用指定配置进行预测"""
    p = predict_match(home_en, away_en, match_type='friendly')
    if not isinstance(p, dict):
        return None
    
    if vote_data and vote_data.get('total', 0) > 0:
        # 将模型概率转换为小数形式
        model_probs_decimal = np.array([p['fin_h'], p['fin_d'], p['fin_a']]) / 100
        
        # 提取投票概率
        vote_probs = np.array([
            vote_data['home'] / 100,
            vote_data['draw'] / 100,
            vote_data['away'] / 100
        ])
        
        # 计算融合权重
        vote_count = vote_data['total']
        alpha = min(config['max_alpha'], vote_count / config['scale'])
        
        # 融合
        fused_probs_decimal = (1 - alpha) * model_probs_decimal + alpha * vote_probs
        
        # 归一化
        fused_probs_decimal = fused_probs_decimal / fused_probs_decimal.sum()
        
        # 转换回百分比形式
        fused_probs = fused_probs_decimal * 100
        
        return {
            'fin_h': fused_probs[0],
            'fin_d': fused_probs[1],
            'fin_a': fused_probs[2],
            'alpha': alpha,
        }
    else:
        return {
            'fin_h': p['fin_h'],
            'fin_d': p['fin_d'],
            'fin_a': p['fin_a'],
            'alpha': 0.0,
        }

def main():
    """主函数"""
    print("=== 365scores A/B 测试框架 ===")
    print()
    
    # 权重配置
    configs = [
        {'name': '保守', 'max_alpha': 0.2, 'scale': 50000},
        {'name': '当前', 'max_alpha': 0.3, 'scale': 100000},
        {'name': '激进', 'max_alpha': 0.4, 'scale': 150000},
        {'name': '极度激进', 'max_alpha': 0.5, 'scale': 200000},
    ]
    
    # 获取 365scores 数据
    print("📡 获取 365scores 数据...")
    scores365_raw = fetch_365scores_data()
    scores365_games = extract_games(scores365_raw) if scores365_raw else []
    print(f"✓ 获取到 {len(scores365_games)} 场比赛")
    print()
    
    # 过滤友谊赛
    friendly = [g for g in scores365_games if 'friendly' in g['competition'].lower()]
    print(f"友谊赛: {len(friendly)} 场")
    print()
    
    # 测试比赛
    test_matches = [
        ('Spain', 'Iraq', '西班牙 vs 伊拉克'),
        ('France', 'Ivory Coast', '法国 vs 科特迪瓦'),
    ]
    
    print("=== A/B 测试结果 ===")
    print()
    
    for home_en, away_en, cn_name in test_matches:
        print(f"--- {cn_name} ---")
        
        # 查找对应的 365scores 数据
        vote_data = None
        for g in friendly:
            if g['home'].lower() == home_en.lower() and g['away'].lower() == away_en.lower():
                vote_data = g.get('votes')
                break
        
        if not vote_data:
            print(f"  无投票数据，跳过")
            print()
            continue
        
        print(f"  投票数据: 主{vote_data['home']}% / 平{vote_data['draw']}% / 客{vote_data['away']}% ({vote_data['total']}人)")
        print()
        
        # 测试不同配置
        for config in configs:
            result = predict_with_config(home_en, away_en, vote_data, config)
            if result:
                print(f"  {config['name']} (α={config['max_alpha']}, scale={config['scale']}):")
                print(f"    融合: 主{result['fin_h']:.1f}% / 平{result['fin_d']:.1f}% / 客{result['fin_a']:.1f}%")
                print(f"    实际权重: {result['alpha']:.1%}")
            else:
                print(f"  {config['name']}: 预测失败")
        
        print()
    
    print("=== A/B 测试总结 ===")
    print()
    print("1. 测试目标: 对比不同权重配置的预测效果")
    print("2. 测试方法: 使用相同比赛数据，不同权重配置")
    print("3. 评估指标: 预测概率分布、融合权重")
    print()
    print("下一步:")
    print("  1. 积累足够历史数据")
    print("  2. 对比不同配置的实际准确率")
    print("  3. 选择最优权重配置")

if __name__ == "__main__":
    main()
