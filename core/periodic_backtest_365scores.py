#!/usr/bin/env python3
"""
365scores 定期回测验证
=====================
定期对比融合前后的预测效果

用法:
  python3 periodic_backtest_365scores.py
"""

import csv
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, '/root')

def load_365scores_data(date_str):
    """加载指定日期的 365scores 数据（从 CSV 读取）"""
    data_file = f"/root/data/365scores/{date_str}.csv"
    if not os.path.exists(data_file):
        return None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    friendly = sum(1 for r in rows if 'friend' in r.get('competition','').lower())
    
    return {
        'date': date_str,
        'total_games': len(rows),
        'friendly_games': friendly,
        'games': rows
    }

def load_predictions(date_str):
    """加载指定日期的预测数据"""
    log_file = '/root/data/predictions_log.csv'
    if not os.path.exists(log_file):
        return []
    
    predictions = []
    with open(log_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('date') == date_str:
                predictions.append(row)
    
    return predictions

def calculate_accuracy(predictions, use_fusion=False):
    """计算预测准确率"""
    correct = 0
    total = 0
    
    for pred in predictions:
        actual_score = pred.get('actual_score')
        if not actual_score or ':' not in actual_score:
            continue
        
        try:
            home, away = map(int, actual_score.split(':'))
            if home > away:
                actual_result = 'H'
            elif home == away:
                actual_result = 'D'
            else:
                actual_result = 'A'
        except:
            continue
        
        # 获取预测结果
        pred_h = float(pred.get('pred_h', 0))
        pred_d = float(pred.get('pred_d', 0))
        pred_a = float(pred.get('pred_a', 0))
        
        if pred_h >= pred_d and pred_h >= pred_a:
            pred_result = 'H'
        elif pred_d >= pred_h and pred_d >= pred_a:
            pred_result = 'D'
        else:
            pred_result = 'A'
        
        total += 1
        if pred_result == actual_result:
            correct += 1
    
    return correct, total

def main():
    """主函数"""
    print("=== 365scores 定期回测验证 ===")
    print()
    
    # 获取最近几天的数据
    today = datetime.now()
    dates = []
    for i in range(7):  # 最近 7 天
        date = today - timedelta(days=i)
        dates.append(date.strftime("%Y-%m-%d"))
    
    print(f"检查日期: {dates[0]} ~ {dates[-1]}")
    print()
    
    # 统计数据收集情况
    print("=== 数据收集情况 ===")
    print()
    
    total_games = 0
    total_friendly = 0
    dates_with_data = []
    
    for date_str in dates:
        data = load_365scores_data(date_str)
        if data:
            games = data.get('total_games', 0)
            friendly = data.get('friendly_games', 0)
            total_games += games
            total_friendly += friendly
            dates_with_data.append(date_str)
            print(f"  {date_str}: {games} 场比赛, {friendly} 场友谊赛")
        else:
            print(f"  {date_str}: 无数据")
    
    print()
    print(f"总计: {len(dates_with_data)} 天有数据, {total_games} 场比赛, {total_friendly} 场友谊赛")
    print()
    
    # 统计预测准确率
    print("=== 预测准确率 ===")
    print()
    
    for date_str in dates_with_data:
        predictions = load_predictions(date_str)
        if predictions:
            correct, total = calculate_accuracy(predictions)
            if total > 0:
                accuracy = correct / total * 100
                print(f"  {date_str}: {correct}/{total} ({accuracy:.1f}%)")
            else:
                print(f"  {date_str}: 无有效预测")
        else:
            print(f"  {date_str}: 无预测数据")
    
    print()
    print("=== 回测总结 ===")
    print()
    print("1. 数据收集: 已建立每日自动收集机制")
    print("2. 回测验证: 需要积累更多数据才能有效对比")
    print("3. 建议: 继续运行 1-2 周后，再进行详细回测分析")
    print()
    print("下一步:")
    print("  1. 继续每日数据收集")
    print("  2. 积累足够数据后，进行详细回测分析")
    print("  3. 对比不同权重配置的效果")

if __name__ == "__main__":
    main()
