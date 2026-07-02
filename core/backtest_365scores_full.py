#!/usr/bin/env python3
"""
365scores 融合效果完整回测
==========================
使用历史数据对比融合前后的预测准确率

用法:
  python3 backtest_365scores_full.py
"""

import csv
import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, '/root')

# 导入预测模块
from predict_match import predict_match

def predict_match_probs(home_cn, away_cn):
    """预测比赛概率"""
    # 这里简化处理，直接使用 predict_match
    # 实际应该使用完整的 predict_today.py 流程
    p = predict_match(home_cn, away_cn, match_type='friendly')
    if not isinstance(p, dict):
        return None
    return {
        'fin_h': p['fin_h'],
        'fin_d': p['fin_d'],
        'fin_a': p['fin_a'],
    }

def get_actual_result(actual_score):
    """从实际比分获取胜平负结果"""
    if not actual_score or ':' not in actual_score:
        return None
    try:
        home, away = map(int, actual_score.split(':'))
        if home > away:
            return 'H'
        elif home == away:
            return 'D'
        else:
            return 'A'
    except:
        return None

def main():
    """主函数"""
    print("=== 365scores 融合效果完整回测 ===")
    print()
    
    # 读取历史数据
    log_file = '/root/data/predictions_log.csv'
    if not os.path.exists(log_file):
        print("❌ 未找到 predictions_log.csv")
        return
    
    with open(log_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"历史记录: {len(rows)} 条")
    
    # 过滤有实际赛果的记录
    checked_rows = [r for r in rows if r.get('checked') == '1' and r.get('actual_score')]
    print(f"有实际赛果: {len(checked_rows)} 条")
    print()
    
    # 分析预测准确率
    print("=== 预测准确率分析 ===")
    print()
    
    correct_without = 0
    correct_with = 0
    total = 0
    
    for r in checked_rows:
        home_cn = r['home_cn']
        away_cn = r['away_cn']
        actual_score = r['actual_score']
        
        # 获取实际结果
        actual_result = get_actual_result(actual_score)
        if not actual_result:
            continue
        
        # 获取模型预测
        pred_h = float(r['pred_h']) if r['pred_h'] else 0
        pred_d = float(r['pred_d']) if r['pred_d'] else 0
        pred_a = float(r['pred_a']) if r['pred_a'] else 0
        
        # 判断模型预测结果
        if pred_h >= pred_d and pred_h >= pred_a:
            pred_result_without = 'H'
        elif pred_d >= pred_h and pred_d >= pred_a:
            pred_result_without = 'D'
        else:
            pred_result_without = 'A'
        
        # 这里简化处理，假设融合后的预测与原始预测相同
        # 实际应该使用 365scores 数据进行融合
        pred_result_with = pred_result_without
        
        # 统计准确率
        total += 1
        if pred_result_without == actual_result:
            correct_without += 1
        if pred_result_with == actual_result:
            correct_with += 1
        
        # 显示详细信息
        print(f"{r['code']} {home_cn} vs {away_cn}:")
        print(f"  预测: 主{pred_h}% 平{pred_d}% 客{pred_a}%")
        print(f"  实际: {actual_score} ({actual_result})")
        print(f"  模型预测: {pred_result_without} | {'✓' if pred_result_without == actual_result else '✗'}")
        print()
    
    # 输出统计结果
    print("=== 统计结果 ===")
    print()
    print(f"总场次: {total}")
    print(f"不使用融合 - 正确: {correct_without}, 准确率: {correct_without/total*100:.1f}%")
    print(f"使用融合   - 正确: {correct_with}, 准确率: {correct_with/total*100:.1f}%")
    print()
    
    print("=== 结论 ===")
    print()
    print("1. 由于缺少历史 365scores 数据，无法进行真实融合对比")
    print("2. 当前回测仅验证模型本身的准确率")
    print("3. 要验证融合效果，需要:")
    print("   - 收集历史 365scores 数据")
    print("   - 或使用当前数据进行 A/B 测试")
    print()
    print("下一步建议:")
    print("  1. 建立 365scores 数据收集机制，每日保存")
    print("  2. 积累足够数据后，进行真实融合回测")
    print("  3. 优化融合权重参数")

if __name__ == "__main__":
    main()
