#!/usr/bin/env python3
"""
fetch_500_odds.py — 500.com 竞彩赔率抓取器
============================================
基于HTML逆向分析，直接从 trade.500.com/jczq 抓取赔率数据

用法:
  python3 fetch_500_odds.py                    # 抓取今天
  python3 fetch_500_odds.py --date 2026-06-15  # 抓取指定日期
  python3 fetch_500_odds.py --history 30       # 回捞过去30天
"""

import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import time
import json
import argparse
import os

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://trade.500.com/',
    'Accept-Charset': 'gb2312,utf-8',
}

DATA_DIR = '/root/data'


def fetch_500_odds(target_date: str = None, playid: int = 269):
    """
    抓取500.com竞彩赔率
    
    Args:
        target_date: 日期格式 YYYY-MM-DD，默认今天
        playid: 269=让球胜平负+胜平负, 312=胜平负单关
    
    Returns:
        list[dict]: 比赛赔率列表
    """
    if target_date is None:
        target_date = date.today().strftime('%Y-%m-%d')
    
    url = f'https://trade.500.com/jczq/?playid={playid}g2&date={target_date}'
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = 'gb2312'
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        rows = soup.select('tr.bet-tb-tr')
        
        results = []
        for row in rows:
            d = row.attrs
            
            # 让球值（从文本内容提取，不是title）
            rang_el = row.select_one('.itm-rangA2')
            if rang_el:
                rang_text = rang_el.get_text(strip=True)
                # 处理 "+1", "-1", "0" 等格式
                rang_val = rang_text.replace('+', '')
            else:
                rang_val = '0'
            
            # 赔率提取
            odds = {'nspf': {}, 'spf': {}}
            for p in row.select('p[data-type]'):
                t = p.get('data-type')
                v = p.get('data-value')
                sp = p.get('data-sp')
                if t in odds and v and sp:
                    odds[t][v] = float(sp)
            
            # 队名（优先从data属性取，否则从a标签）
            home = d.get('data-homesxname', '')
            away = d.get('data-awaysxname', '')
            if not home:
                home_a = row.select_one('.team-l')
                home = home_a.get_text(strip=True) if home_a else ''
            if not away:
                away_a = row.select_one('.team-r')
                away = away_a.get_text(strip=True) if away_a else ''
            
            results.append({
                'fixture_id': d.get('data-fixtureid'),
                'process_id': d.get('data-processid'),
                'match_num': d.get('data-matchnum'),
                'date': d.get('data-matchdate'),
                'match_time': d.get('data-matchtime'),
                'process_date': d.get('data-processdate'),
                'home_id': d.get('data-homeid'),
                'away_id': d.get('data-awayid'),
                'home': home,
                'away': away,
                'rang': float(rang_val) if rang_val else 0.0,
                # 标准胜平负 (500.com 的 spf 字段)
                # 注意: 当让球≠0时，spf可能是让球后的赔率！
                'spf_home': odds['spf'].get('3'),
                'spf_draw': odds['spf'].get('1'),
                'spf_away': odds['spf'].get('0'),
                # 让球胜平负 (仅当让球≠0时有)
                'nspf_home': odds['nspf'].get('3'),
                'nspf_draw': odds['nspf'].get('1'),
                'nspf_away': odds['nspf'].get('0'),
                # 赔率类型说明
                'has_nspf': bool(odds['nspf']),  # 是否有让球赔率
                'is_end': d.get('data-isend') == '1',
            })
        
        return results
        
    except Exception as e:
        print(f'❌ 抓取失败: {e}')
        return []


def fetch_historical_odds(days_back: int = 30):
    """回捞过去N天的历史赔率"""
    all_data = []
    today = date.today()
    
    for i in range(days_back):
        target = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        try:
            rows = fetch_500_odds(target_date=target)
            # 只保留已结束的比赛（有结果）
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
        filename = f'500_odds_{date.today().strftime("%Y%m%d")}.json'
    
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f'✅ 已保存到: {filepath}')
    return filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='500.com 竞彩赔率抓取')
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
        print(f'✅ 共 {len(data)} 场比赛')
        
        # 显示摘要
        for i, m in enumerate(data[:5], 1):
            print(f'{i}. {m["home"]} vs {m["away"]}')
            print(f'   让球: {m["rang"]}')
            print(f'   SPF: H={m["spf_home"]} D={m["spf_draw"]} A={m["spf_away"]}')
            print(f'   让球: H={m["nspf_home"]} D={m["nspf_draw"]} A={m["nspf_away"]}')
        
        if len(data) > 5:
            print(f'... 还有 {len(data)-5} 场')
        
        save_odds(data, args.output)
