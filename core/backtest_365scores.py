#!/usr/bin/env python3
"""
365scores 融合效果回测验证
==========================
对比融合前后的预测准确率

用法:
  python3 backtest_365scores.py
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, '/root')

# 导入预测模块
from predict_match import predict_match
from fetch_365scores import fetch_365scores_data, extract_games

def predict_without_fusion(home_en, away_en):
    """不使用 365scores 融合的预测"""
    p = predict_match(home_en, away_en, match_type='friendly')
    if not isinstance(p, dict):
        return None
    return {
        'fin_h': p['fin_h'],
        'fin_d': p['fin_d'],
        'fin_a': p['fin_a'],
    }

def predict_with_fusion(home_en, away_en, vote_data):
    """使用 365scores 融合的预测"""
    p = predict_match(home_en, away_en, match_type='friendly')
    if not isinstance(p, dict):
        return None
    
    # 投票数据融合
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
        alpha = min(0.3, vote_count / 100000)
        
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
    print("=== 365scores 融合效果回测验证 ===")
    print()
    
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
    
    # 测试几场比赛
    test_matches = [
        ('Spain', 'Iraq', '西班牙 vs 伊拉克'),
        ('France', 'Ivory Coast', '法国 vs 科特迪瓦'),
    ]
    
    print("=== 融合效果对比 ===")
    print()
    
    for home_en, away_en, cn_name in test_matches:
        print(f"--- {cn_name} ---")
        
        # 查找对应的 365scores 数据
        vote_data = None
        for g in friendly:
            if g['home'].lower() == home_en.lower() and g['away'].lower() == away_en.lower():
                vote_data = g.get('votes')
                break
        
        # 不使用融合的预测
        pred_without = predict_without_fusion(home_en, away_en)
        
        # 使用融合的预测
        pred_with = predict_with_fusion(home_en, away_en, vote_data)
        
        if pred_without and pred_with:
            print(f"  不使用融合: 主{pred_without['fin_h']:.1f}% / 平{pred_without['fin_d']:.1f}% / 客{pred_without['fin_a']:.1f}%")
            print(f"  使用融合:   主{pred_with['fin_h']:.1f}% / 平{pred_with['fin_d']:.1f}% / 客{pred_with['fin_a']:.1f}%")
            if vote_data:
                print(f"  投票数据:   主{vote_data['home']}% / 平{vote_data['draw']}% / 客{vote_data['away']}% ({vote_data['total']}人)")
                print(f"  融合权重:   {pred_with['alpha']:.1%}")
            
            # 计算差异
            diff_h = pred_with['fin_h'] - pred_without['fin_h']
            diff_d = pred_with['fin_d'] - pred_without['fin_d']
            diff_a = pred_with['fin_a'] - pred_without['fin_a']
            print(f"  差异:       主{diff_h:+.1f}% / 平{diff_d:+.1f}% / 客{diff_a:+.1f}%")
        else:
            print(f"  预测失败")
        
        print()
    
    print("=== 回测总结 ===")
    print()
    print("1. 投票数据融合会调整模型概率")
    print("2. 融合权重取决于投票人数（最多 30%）")
    print("3. 需要实际赛果验证融合效果")
    print()
    print("下一步:")
    print("  1. 收集历史数据，对比融合前后的准确率")
    print("  2. 分析融合权重与准确率的关系")
    print("  3. 优化融合权重参数")

if __name__ == "__main__":
    main()
