#!/usr/bin/env python3
"""
analyze_500_odds.py — 500.com 赔率分析器
=========================================
计算隐含概率、EV、Kelly值

用法:
  python3 analyze_500_odds.py                    # 分析今天的赔率
  python3 analyze_500_odds.py --file odds.json   # 分析指定文件
"""

import json
import os
import argparse

DATA_DIR = '/root/data'


def calculate_implied_prob(h, d, a):
    """计算归一化后的隐含概率"""
    total = 1/h + 1/d + 1/a
    return {
        'home': (1/h) / total,
        'draw': (1/d) / total,
        'away': (1/a) / total,
    }


def calculate_ev(prob, odds):
    """计算期望值 (EV)"""
    return prob * odds - 1


def calculate_kelly(prob, odds, fraction=0.25):
    """计算Kelly值"""
    if odds <= 1:
        return 0
    kelly = (prob * odds - 1) / (odds - 1)
    return max(0, kelly * fraction)


def analyze_odds(data):
    """分析赔率数据"""
    results = []
    
    for match in data:
        home = match['home']
        away = match['away']
        rang = match['rang']
        
        # 获取赔率
        spf_h = match.get('spf_home')
        spf_d = match.get('spf_draw')
        spf_a = match.get('spf_away')
        
        nspf_h = match.get('nspf_home')
        nspf_d = match.get('nspf_draw')
        nspf_a = match.get('nspf_away')
        
        # 分析标准赔率
        spf_analysis = None
        if spf_h and spf_d and spf_a:
            probs = calculate_implied_prob(spf_h, spf_d, spf_a)
            spf_analysis = {
                'odds': {'home': spf_h, 'draw': spf_d, 'away': spf_a},
                'probs': probs,
                'overround': (1/spf_h + 1/spf_d + 1/spf_a - 1) * 100,
            }
        
        # 分析让球赔率
        nspf_analysis = None
        if nspf_h and nspf_d and nspf_a:
            probs = calculate_implied_prob(nspf_h, nspf_d, nspf_a)
            nspf_analysis = {
                'odds': {'home': nspf_h, 'draw': nspf_d, 'away': nspf_a},
                'probs': probs,
                'overround': (1/nspf_h + 1/nspf_d + 1/nspf_a - 1) * 100,
            }
        
        results.append({
            'home': home,
            'away': away,
            'rang': rang,
            'spf': spf_analysis,
            'nspf': nspf_analysis,
            'has_nspf': match.get('has_nspf', False),
        })
    
    return results


def print_analysis(results):
    """打印分析结果"""
    print('=' * 80)
    print('500.com 竞彩赔率分析')
    print('=' * 80)
    
    for i, r in enumerate(results, 1):
        print(f'\n{i}. {r["home"]} vs {r["away"]}')
        print(f'   让球: {r["rang"]}')
        
        if r['spf']:
            spf = r['spf']
            print(f'   标准赔率: H={spf["odds"]["home"]:.2f} D={spf["odds"]["draw"]:.2f} A={spf["odds"]["away"]:.2f}')
            print(f'   隐含概率: H={spf["probs"]["home"]*100:.1f}% D={spf["probs"]["draw"]*100:.1f}% A={spf["probs"]["away"]*100:.1f}%')
            print(f'   Overround: {spf["overround"]:.1f}%')
        
        if r['nspf']:
            nspf = r['nspf']
            print(f'   让球赔率: H={nspf["odds"]["home"]:.2f} D={nspf["odds"]["draw"]:.2f} A={nspf["odds"]["away"]:.2f}')
            print(f'   隐含概率: H={nspf["probs"]["home"]*100:.1f}% D={nspf["probs"]["draw"]*100:.1f}% A={nspf["probs"]["away"]*100:.1f}%')
        
        if not r['has_nspf']:
            print(f'   ⚠️ 无让球盘（仅标准1X2）')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='500.com 赔率分析')
    parser.add_argument('--file', type=str, help='输入文件')
    
    args = parser.parse_args()
    
    # 加载数据
    if args.file:
        filepath = args.file
    else:
        filepath = os.path.join(DATA_DIR, f'500_odds_{date.today().strftime("%Y%m%d")}.json')
    
    if not os.path.exists(filepath):
        print(f'❌ 文件不存在: {filepath}')
        print('请先运行 fetch_500_odds.py 抓取赔率')
        exit(1)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 分析并打印
    results = analyze_odds(data)
    print_analysis(results)
