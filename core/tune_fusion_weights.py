#!/usr/bin/env python3
"""
365scores 融合权重调优
=====================
提供多种权重配置供选择

用法:
  python3 tune_fusion_weights.py
"""

import numpy as np

def calculate_alpha(vote_count, max_alpha=0.3, scale=100000):
    """计算融合权重"""
    return min(max_alpha, vote_count / scale)

def simulate_fusion(model_probs, vote_probs, alpha):
    """模拟融合过程"""
    model_decimal = np.array(model_probs) / 100
    vote_decimal = np.array(vote_probs) / 100
    
    fused = (1 - alpha) * model_decimal + alpha * vote_decimal
    fused = fused / fused.sum()
    
    return fused * 100

def main():
    """主函数"""
    print("=== 365scores 融合权重调优 ===")
    print()
    
    # 测试场景
    test_cases = [
        {
            'name': '西班牙 vs 伊拉克',
            'model': [67.3, 17.9, 14.8],
            'vote': [39.1, 10.3, 50.6],
            'vote_count': 222839,
        },
        {
            'name': '法国 vs 科特迪瓦',
            'model': [53.2, 27.7, 19.2],
            'vote': [81.9, 9.1, 9.0],
            'vote_count': 40380,
        },
    ]
    
    # 权重配置
    weight_configs = [
        {'max_alpha': 0.2, 'scale': 50000, 'name': '保守 (20%上限, 5万缩放)'},
        {'max_alpha': 0.3, 'scale': 100000, 'name': '当前 (30%上限, 10万缩放)'},
        {'max_alpha': 0.4, 'scale': 150000, 'name': '激进 (40%上限, 15万缩放)'},
        {'max_alpha': 0.5, 'scale': 200000, 'name': '极度激进 (50%上限, 20万缩放)'},
    ]
    
    print("=== 不同权重配置对比 ===")
    print()
    
    for case in test_cases:
        print(f"--- {case['name']} ---")
        print(f"  模型: 主{case['model'][0]}% / 平{case['model'][1]}% / 客{case['model'][2]}%")
        print(f"  投票: 主{case['vote'][0]}% / 平{case['vote'][1]}% / 客{case['vote'][2]}% ({case['vote_count']}人)")
        print()
        
        for config in weight_configs:
            alpha = calculate_alpha(case['vote_count'], config['max_alpha'], config['scale'])
            fused = simulate_fusion(case['model'], case['vote'], alpha)
            
            print(f"  {config['name']}:")
            print(f"    α = {alpha:.1%}")
            print(f"    融合: 主{fused[0]:.1f}% / 平{fused[1]:.1f}% / 客{fused[2]:.1f}%")
            
            # 计算变化
            diff_h = fused[0] - case['model'][0]
            diff_d = fused[1] - case['model'][1]
            diff_a = fused[2] - case['model'][2]
            print(f"    变化: 主{diff_h:+.1f}% / 平{diff_d:+.1f}% / 客{diff_a:+.1f}%")
            print()
    
    print("=== 权重配置建议 ===")
    print()
    print("1. 保守配置 (20%上限, 5万缩放)")
    print("   - 适合: 投票数据质量不确定")
    print("   - 优点: 降低投票数据的影响")
    print("   - 缺点: 可能错过有价值的投票信息")
    print()
    print("2. 当前配置 (30%上限, 10万缩放)")
    print("   - 适合: 一般情况")
    print("   - 优点: 平衡模型和投票数据")
    print("   - 缺点: 可能需要更多数据验证")
    print()
    print("3. 激进配置 (40%上限, 15万缩放)")
    print("   - 适合: 投票数据质量高")
    print("   - 优点: 充分利用投票信息")
    print("   - 缺点: 可能过度依赖投票数据")
    print()
    print("4. 极度激进配置 (50%上限, 20万缩放)")
    print("   - 适合: 投票数据非常可靠")
    print("   - 优点: 最大化投票数据价值")
    print("   - 缺点: 风险较高")
    print()
    print("=== 下一步 ===")
    print()
    print("1. 积累足够数据后，进行 A/B 测试")
    print("2. 对比不同权重配置的准确率")
    print("3. 选择最优权重配置")

if __name__ == "__main__":
    main()
