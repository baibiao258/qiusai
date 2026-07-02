#!/usr/bin/env python3
"""
365scores 数据积累监控
=====================
监控数据积累进度，判断何时可以进行下一步

用法:
  python3 monitor_365scores_data.py
"""

import csv
import os
from collections import Counter
from datetime import date, datetime

DATA_DIR = '/root/data/365scores'
MASTER_CSV = f"{DATA_DIR}/football_games.csv"

def main():
    """主函数"""
    print("=== 365scores 数据积累监控 ===")
    print()
    
    if not os.path.exists(DATA_DIR):
        print("❌ 数据目录不存在")
        return
    
    # 统计CSV数据文件
    daily_files = sorted([f for f in os.listdir(DATA_DIR) 
                         if f.endswith('.csv') and f[0].isdigit() and len(f) == 15])
    
    # 读取主CSV
    if not os.path.exists(MASTER_CSV):
        print("❌ 主CSV文件不存在")
        return
    
    with open(MASTER_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total_rows = len(rows)
    
    # 日期统计
    dates = Counter(r.get('date', '') for r in rows if r.get('date', ''))
    unique_dates = len(dates)
    date_range = f"{min(dates.keys())} ~ {max(dates.keys())}" if dates else "N/A"
    
    # 赛事类型统计
    competitions = Counter(r.get('competition', '') for r in rows)
    friendly_count = sum(1 for r in rows if 'friendly' in r.get('competition', '').lower())
    
    # 状态统计
    statuses = Counter(r.get('status', '') for r in rows)
    completed = sum(1 for r in rows if r.get('status') in ('finished', 'completed') or 
                   (r.get('score', '').strip() and r.get('score') not in ('', '-')))
    
    print(f"每日CSV文件数: {len(daily_files)}")
    print(f"主CSV行数: {total_rows}")
    print(f"覆盖天数: {unique_dates}")
    print(f"日期范围: {date_range}")
    print()
    print(f"总比赛数: {total_rows}")
    print(f"总友谊赛数: {friendly_count}")
    print()
    
    # 按日期分布
    print("=== 按日期分布 ===")
    for d in sorted(dates):
        print(f"  {d}: {dates[d]} 行")
    print()
    
    # 进度评估
    print("=== 进度评估 ===")
    print()
    
    if unique_dates >= 14:
        print("✅ 数据充足 (≥14 天)")
        print("   可以进行 A/B 测试和模型重训练")
        print("   建议运行: python3 /root/ab_test_365scores.py")
    elif unique_dates >= 7:
        print("⚠️ 数据中等 (7-13 天)")
        print("   可以进行初步回测分析")
        print("   建议运行: python3 /root/backtest_365scores.py")
        print(f"   还需 {14 - unique_dates} 天达到充足")
    else:
        print("❌ 数据不足 (<7 天)")
        print("   需要继续积累数据")
        print(f"   还需 {7 - unique_dates} 天达到初步可用")
    
    print()
    print("=== 建议 ===")
    print()
    
    day_gap = (date.today() - datetime.strptime(min(dates.keys()), '%Y-%m-%d').date()).days if dates else 0
    
    if unique_dates < 7:
        days_needed = 7 - unique_dates
        print(f"继续运行数据收集 {days_needed} 天，然后进行初步回测")
    elif unique_dates < 14:
        print(f"数据已可初步分析，建议继续收集至满 14 天")
        print(f"  当前: {unique_dates}/14 天")
        print(f"  可运行: python3 /root/backtest_365scores.py")
    else:
        print("✅ 数据充足，可以:")
        print("  1. python3 /root/ab_test_365scores.py — A/B 测试")
        print("  2. python3 /root/backtest_365scores.py — 详细回测")
        print("  3. python3 /root/periodic_backtest_365scores.py — 周期性回测")
    
    # 数据质量检查
    print()
    print("=== 数据质量 ===")
    missing_date = sum(1 for r in rows if not r.get('date', ''))
    if missing_date:
        print(f"  ⚠️ {missing_date} 行缺少日期字段")
    
    # 检查最近3天是否有数据
    today_str = date.today().isoformat()
    recent_dates = [d for d in dates if d >= (date.today().isoformat())]
    if recent_dates:
        print(f"  ✓ 今日({today_str})有数据: {dates.get(today_str, 0)} 行")
    else:
        print(f"  ⚠️ 今日({today_str})暂无数据")

if __name__ == "__main__":
    main()
