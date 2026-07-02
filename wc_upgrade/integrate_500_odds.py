#!/usr/bin/env python3
"""
integrate_500_odds.py — 将500.com赔率集成到预测系统
====================================================
从500.com抓取完整赔率，与历史kaijiang数据JOIN，扩充训练数据

用法:
  python3 integrate_500_odds.py                    # 抓取今天并分析
  python3 integrate_500_odds.py --date 2026-06-15  # 指定日期
  python3 integrate_500_odds.py --history 7        # 回捞过去7天
  python3 integrate_500_odds.py --join             # JOIN到训练数据
"""

import json
import os
import sys
import pandas as pd
from datetime import date, timedelta
import argparse

# 添加路径
sys.path.insert(0, '/root/wc_2026_upgrade')

from fetch_500_complete import fetch_500_odds, fetch_historical_odds, save_odds

DATA_DIR = '/root/data'


def load_kaijiang_data():
    """加载历史kaijiang数据"""
    kaijiang_file = os.path.join(DATA_DIR, 'historical_kaijiang.csv')
    if os.path.exists(kaijiang_file):
        df = pd.read_csv(kaijiang_file)
        return df
    return None


def match_odds_to_kaijiang(odds_data, kaijiang_df):
    """
    将500.com赔率与kaijiang数据JOIN
    
    Args:
        odds_data: 500.com赔率数据列表
        kaijiang_df: 历史kaijiang DataFrame
    
    Returns:
        list[dict]: JOIN后的数据
    """
    matched = []
    
    for match in odds_data:
        fixture_id = match.get('fixture_id')
        home = match.get('home', '')
        away = match.get('away', '')
        match_date = match.get('date', '')
        
        # 尝试通过fixture_id匹配
        kaijiang_row = None
        if fixture_id and kaijiang_df is not None:
            # 转换fixture_id类型
            try:
                fixture_id_int = int(fixture_id)
                kaijiang_row = kaijiang_df[kaijiang_df['fixture_id'] == fixture_id_int]
                if len(kaijiang_row) > 0:
                    kaijiang_row = kaijiang_row.iloc[0]
                else:
                    kaijiang_row = None
            except:
                pass
        
        # 尝试通过队名+日期匹配
        if kaijiang_row is None and kaijiang_df is not None:
            # 标准化队名
            home_normalized = home.replace(' ', '').lower()
            away_normalized = away.replace(' ', '').lower()
            
            for idx, row in kaijiang_df.iterrows():
                k_home = str(row.get('home', '')).replace(' ', '').lower()
                k_away = str(row.get('away', '')).replace(' ', '').lower()
                k_date = str(row.get('date', ''))
                
                if (home_normalized in k_home and away_normalized in k_away and
                    match_date in k_date):
                    kaijiang_row = row
                    break
        
        # 整合数据
        integrated = {
            'fixture_id': fixture_id,
            'date': match_date,
            'match_time': match.get('match_time', ''),
            'home': home,
            'away': away,
            'rang': match.get('rang', 0),
            # 500.com赔率
            'spf_500': match.get('odds', {}).get('spf', {}),
            'nspf_500': match.get('odds', {}).get('nspf', {}),
            'jqs_500': match.get('odds', {}).get('jqs', {}),
            'bf_500': match.get('odds', {}).get('bf', {}),
            'bqc_500': match.get('odds', {}).get('bqc', {}),
            # kaijiang结果（如果匹配到）
            'spf_result': kaijiang_row.get('spf_result') if kaijiang_row is not None else None,
            'nspf_result': kaijiang_row.get('nspf_result') if kaijiang_row is not None else None,
            'jqs_result': kaijiang_row.get('jqs_result') if kaijiang_row is not None else None,
            'bf_result': kaijiang_row.get('bf_result') if kaijiang_row is not None else None,
            'bqc_result': kaijiang_row.get('bqc_result') if kaijiang_row is not None else None,
            'score': kaijiang_row.get('score') if kaijiang_row is not None else None,
            'is_matched': kaijiang_row is not None,
        }
        
        matched.append(integrated)
    
    return matched


def analyze_odds_quality(matched_data):
    """分析赔率数据质量"""
    print('=' * 80)
    print('500.com 赔率数据质量分析')
    print('=' * 80)
    
    total = len(matched_data)
    matched = sum(1 for m in matched_data if m['is_matched'])
    
    print(f'总场次: {total}')
    print(f'匹配到kaijiang: {matched} ({matched/total*100:.1f}%)')
    
    # 分析各玩法覆盖率
    play_types = ['spf_500', 'nspf_500', 'jqs_500', 'bf_500', 'bqc_500']
    for pt in play_types:
        count = sum(1 for m in matched_data if m[pt])
        print(f'{pt}: {count}/{total} ({count/total*100:.1f}%)')
    
    # 示例：显示一场完整数据
    print('\n示例数据:')
    for m in matched_data[:3]:
        print(f'\n{m["home"]} vs {m["away"]}')
        if m['spf_500']:
            print(f'  SPF 500.com: {m["spf_500"]}')
        if m['nspf_500']:
            print(f'  NSPF 500.com: {m["nspf_500"]}')
        if m['spf_result']:
            print(f'  SPF 结果: {m["spf_result"]}')
        if m['score']:
            print(f'  比分: {m["score"]}')


def save_integrated_data(matched_data, filename=None):
    """保存整合后的数据"""
    if filename is None:
        filename = f'500_odds_integrated_{date.today().strftime("%Y%m%d")}.json'
    
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(matched_data, f, ensure_ascii=False, indent=2)
    
    print(f'✅ 已保存到: {filepath}')
    return filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='500.com 赔率集成到预测系统')
    parser.add_argument('--date', type=str, help='目标日期 YYYY-MM-DD')
    parser.add_argument('--history', type=int, help='回捞过去N天历史')
    parser.add_argument('--join', action='store_true', help='JOIN到kaijiang数据')
    parser.add_argument('--output', type=str, help='输出文件名')
    
    args = parser.parse_args()
    
    # 抓取赔率数据
    if args.history:
        print(f'📅 回捞过去 {args.history} 天历史赔率...')
        odds_data = fetch_historical_odds(args.history)
    else:
        print(f'📅 抓取 {args.date or "今天"} 的竞彩赔率...')
        odds_data = fetch_500_odds(target_date=args.date)
    
    print(f'✅ 抓取完成: {len(odds_data)} 场')
    
    # 保存原始赔率
    save_odds(odds_data)
    
    # JOIN到kaijiang数据
    if args.join:
        print('\n🔗 JOIN到kaijiang数据...')
        kaijiang_df = load_kaijiang_data()
        
        if kaijiang_df is not None:
            matched_data = match_odds_to_kaijiang(odds_data, kaijiang_df)
            analyze_odds_quality(matched_data)
            save_integrated_data(matched_data, args.output)
        else:
            print('⚠️ 未找到kaijiang数据，跳过JOIN')
    else:
        # 只显示分析
        analyze_odds_quality(odds_data)
