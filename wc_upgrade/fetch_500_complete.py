#!/usr/bin/env python3
"""
fetch_500_complete.py — 500.com 竞彩赔率完整抓取器
==================================================
基于HTML逆向分析，抓取所有玩法赔率（胜平负/让球/半全场/比分/总进球）

用法:
  python3 fetch_500_complete.py                    # 抓取今天
  python3 fetch_500_complete.py --date 2026-06-15  # 抓取指定日期
  python3 fetch_500_complete.py --history 30       # 回捞过去30天
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
import os
from datetime import date, timedelta
import argparse

DATA_DIR = '/root/data'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://trade.500.com/',
    'Accept-Charset': 'gb2312,utf-8',
}


def parse_subactive(subactive_str):
    """
    解析 data-subactive 字段，判断哪些玩法有效
    
    格式: "nspfdg1,nspfgg1,spfdg0,spfgg1,hcdg1,hcgg1,..."
    返回: {'nspf_dg': True, 'nspf_gg': True, 'spf_dg': False, 'spf_gg': True, ...}
    """
    flags = {}
    if not subactive_str:
        return flags
    
    for item in subactive_str.split(','):
        m = re.match(r'([a-z]+)(dg|gg)(\d)', item)
        if m:
            key = f'{m.group(1)}_{m.group(2)}'
            flags[key] = m.group(3) == '1'
    
    return flags


def parse_odds_from_row(row, row_type='main'):
    """
    从 <tr> 行中提取赔率
    
    Args:
        row: BeautifulSoup Tag 对象
        row_type: 'main' (主行) 或 'more' (展开行)
    
    Returns:
        dict: {data_type: {data_value: odds}}
    """
    odds = {}
    
    # 选择正确的选择器
    if row_type == 'main':
        # 主行: p.betbtn (让球/标准胜平负)
        selector = 'p.betbtn[data-type]'
    else:
        # 展开行: p.sbetbtn (半全场/比分/总进球)
        selector = 'p.sbetbtn[data-type]'
    
    for p in row.select(selector):
        dtype = p.get('data-type')
        dvalue = p.get('data-value', '')
        dsp = p.get('data-sp')
        
        if not dtype or not dsp:
            continue
        
        # 跳过空值（如半全场的"其他"）
        if not dvalue:
            continue
        
        # 跳过异常赔率（1000.00 = 停售）
        try:
            sp = float(dsp)
            if sp >= 1000:
                continue
        except ValueError:
            continue
        
        if dtype not in odds:
            odds[dtype] = {}
        
        odds[dtype][dvalue] = sp
    
    return odds


def fetch_500_odds(target_date=None, playid=None):
    """
    抓取500.com竞彩赔率（完整版，含所有玩法）
    
    Args:
        target_date: 日期格式 YYYY-MM-DD，默认今天
        playid: 指定玩法ID，None则抓取所有玩法
                269=让球胜平负+胜平负, 270=总进球, 271=半全场, 272=比分
    
    Returns:
        list[dict]: 比赛赔率列表
    """
    if target_date is None:
        target_date = date.today().strftime('%Y-%m-%d')
    
    # 定义所有玩法
    PLAYID_MAP = {
        269: ['spf', 'nspf'],      # 让球胜平负+胜平负
        270: ['jqs'],              # 总进球
        271: ['bf'],               # 半全场
        272: ['bqc'],              # 比分
    }
    
    # 确定要抓取的 playid
    if playid:
        playids_to_fetch = [playid]
    else:
        playids_to_fetch = list(PLAYID_MAP.keys())
    
    # 按 fixture_id 收集所有赔率
    odds_by_fixture = {}
    
    for pid in playids_to_fetch:
        url = f'https://trade.500.com/jczq/?playid={pid}g2&date={target_date}'
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = 'gb2312'
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            main_rows = soup.select('tr.bet-tb-tr')
            
            for tr in main_rows:
                d = tr.attrs
                fixture_id = d.get('data-fixtureid')
                
                if not fixture_id:
                    continue
                
                # 初始化 fixture 数据
                if fixture_id not in odds_by_fixture:
                    odds_by_fixture[fixture_id] = {
                        'fixture_id': fixture_id,
                        'match_num': d.get('data-matchnum'),
                        'date': d.get('data-matchdate'),
                        'match_time': d.get('data-matchtime'),
                        'home_id': d.get('data-homeid'),
                        'away_id': d.get('data-awayid'),
                        'home': d.get('data-homesxname', ''),
                        'away': d.get('data-awaysxname', ''),
                        'rang': 0.0,
                        'odds': {},
                    }
                    
                    # 队名兜底
                    if not odds_by_fixture[fixture_id]['home']:
                        home_a = tr.select_one('.team-l')
                        odds_by_fixture[fixture_id]['home'] = home_a.get_text(strip=True) if home_a else ''
                    if not odds_by_fixture[fixture_id]['away']:
                        away_a = tr.select_one('.team-r')
                        odds_by_fixture[fixture_id]['away'] = away_a.get_text(strip=True) if away_a else ''
                
                # 让球值
                rang_el = tr.select_one('.itm-rangA2')
                if rang_el:
                    rang_text = rang_el.get_text(strip=True)
                    rang_val = rang_text.replace('+', '')
                    odds_by_fixture[fixture_id]['rang'] = float(rang_val) if rang_val else 0.0
                
                # 提取赔率
                for p in tr.find_all('p', attrs={'data-type': True}):
                    dtype = p.get('data-type')
                    dvalue = p.get('data-value', '')
                    dsp = p.get('data-sp')
                    
                    if not dtype or not dvalue or not dsp:
                        continue
                    
                    try:
                        sp = float(dsp)
                        if sp >= 1000:  # 停售
                            continue
                    except ValueError:
                        continue
                    
                    if dtype not in odds_by_fixture[fixture_id]['odds']:
                        odds_by_fixture[fixture_id]['odds'][dtype] = {}
                    
                    odds_by_fixture[fixture_id]['odds'][dtype][dvalue] = sp
            
            time.sleep(0.5)  # 礼貌性延迟
            
        except Exception as e:
            print(f'❌ playid={pid} 抓取失败: {e}')
    
    # 转换为列表
    results = list(odds_by_fixture.values())
    return results


def fetch_historical_odds(days_back=30):
    """回捞过去N天的历史赔率"""
    all_data = []
    today = date.today()
    
    for i in range(days_back):
        target = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        try:
            rows = fetch_500_odds(target_date=target)
            # 只保留已结束的比赛
            ended = [r for r in rows if r['is_end']]
            all_data.extend(ended)
            print(f'{target}: {len(ended)} 场已结束')
            time.sleep(1.5)  # 礼貌性延迟
        except Exception as e:
            print(f'{target}: 抓取失败 {e}')
    
    return all_data


def save_odds(data, filename=None):
    """保存赔率数据到JSON"""
    if filename is None:
        filename = f'500_odds_complete_{date.today().strftime("%Y%m%d")}.json'
    
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f'✅ 已保存到: {filepath}')
    return filepath


def print_odds_summary(data):
    """打印赔率摘要"""
    print('=' * 80)
    print('500.com 竞彩赔率抓取结果')
    print('=' * 80)
    print(f'总场次: {len(data)}')
    
    for i, match in enumerate(data[:5], 1):
        print(f'\n{i}. {match["home"]} vs {match["away"]}')
        print(f'   fixture_id: {match["fixture_id"]}')
        print(f'   比赛时间: {match["date"]} {match["match_time"]}')
        print(f'   让球: {match["rang"]}')
        
        odds = match['odds']
        
        # 标准胜平负 (spf)
        if 'spf' in odds:
            print(f'   标准赔率: {odds["spf"]}')
        
        # 让球胜平负 (nspf)
        if 'nspf' in odds:
            print(f'   让球赔率: {odds["nspf"]}')
        
        # 总进球 (jqs)
        if 'jqs' in odds:
            print(f'   总进球: {len(odds["jqs"])} 种结果')
            if len(odds['jqs']) <= 10:
                print(f'     {odds["jqs"]}')
        
        # 半全场 (bf)
        if 'bf' in odds:
            print(f'   半全场: {len(odds["bf"])} 种结果')
        
        # 比分 (bqc)
        if 'bqc' in odds:
            print(f'   比分: {len(odds["bqc"])} 种结果')
            if len(odds['bqc']) <= 10:
                print(f'     {odds["bqc"]}')
        
        # 有效玩法
        active_plays = list(odds.keys())
        print(f'   有效玩法: {", ".join(active_plays)}')
    
    if len(data) > 5:
        print(f'\n... 还有 {len(data)-5} 场')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='500.com 竞彩赔率完整抓取')
    parser.add_argument('--date', type=str, help='目标日期 YYYY-MM-DD')
    parser.add_argument('--history', type=int, help='回捞过去N天历史')
    parser.add_argument('--output', type=str, help='输出文件名')
    
    args = parser.parse_args()
    
    if args.history:
        print(f'📅 回捞过去 {args.history} 天历史赔率...')
        data = fetch_historical_odds(args.history)
        save_odds(data, args.output or '500_odds_history.json')
    else:
        print(f'📅 抓取 {args.date or "今天"} 的竞彩赔率...')
        data = fetch_500_odds(target_date=args.date)
        
        print_odds_summary(data)
        save_odds(data, args.output)
