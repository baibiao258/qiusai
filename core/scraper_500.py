#!/usr/bin/env python3
"""
500.com 竞彩赛事数据批量爬虫
============================
从 500.com 指数中心抓取所有竞彩比赛的:
  - FIFA排名
  - 近期战绩 (10场)
  - 交战历史
  - 预计阵容
  - 欧赔/亚盘
  - 澳门心水推荐

用法:
  python3 scraper_500.py                    # 爬取今天所有竞彩比赛
  python3 scraper_500.py --url URL          # 爬取指定比赛页
  python3 scraper_500.py --ids 1411007,1410357  # 按ID爬取
"""

import re
import json
import time
import sys
import os
import argparse
from datetime import datetime
from html.parser import HTMLParser
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://odds.500.com/',
}

BASE_URL = 'https://odds.500.com'


def fetch(url, timeout=15):
    """抓取网页，返回解码后的文本 (gb2312/gbk)"""
    req = Request(url, headers=HEADERS)
    try:
        resp = urlopen(req, timeout=timeout)
        raw = resp.read()
        # 500.com 用 gb2312 编码
        for enc in ['gb2312', 'gbk', 'gb18030', 'utf-8']:
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return raw.decode('utf-8', errors='replace')
    except (URLError, HTTPError) as e:
        print(f"  ❌ 抓取失败: {url} -> {e}")
        return None


def fetch_match_list():
    """
    从 500.com 获取当前竞彩期次的所有比赛ID和对阵。
    先抓任意一个比赛页, 从中提取同期所有比赛链接。
    """
    # 方法1: 从竞彩主入口获取
    # 方法2: 从已知比赛页提取同期列表
    # 这里用方法2: 先抓第一个已知页面
    print("📡 获取竞彩赛事列表...")
    
    # 从竞彩指数中心主页获取
    html = fetch(f'{BASE_URL}/fenxi/shuju-1411007.shtml')
    if not html:
        return []
    
    matches = []
    seen = set()
    
    # 提取所有 shuju-{id}.shtml 链接
    pattern = r'/fenxi/shuju-(\d+)\.shtml[^"]*"[^>]*>.*?<em class="l">(.*?)</em>.*?<em class="r">(.*?)</em>'
    for m in re.finditer(pattern, html, re.DOTALL):
        mid = m.group(1)
        home = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        away = re.sub(r'<[^>]+>', '', m.group(3)).strip()
        if mid not in seen and home and away:
            seen.add(mid)
            matches.append({'id': mid, 'home': home, 'away': away})
    
    # 也提取带 code 的竞彩编号
    code_pattern = r'<span class="gray">(周[一二三四五六日]\d+)</span>'
    codes = re.findall(code_pattern, html)
    
    return matches


def parse_recent_form(html, team_side):
    """
    解析近期战绩。
    team_side: 'team_a'(左侧/主队) 或 'team_b'(右侧/客队)
    """
    results = []
    
    # 从图表分析区域提取
    # 找到包含 bmatch 的表格行
    if team_side == 'team_a':
        section = re.search(r'id="team_zhanji_1"(.*?)(?:id="team_zhanji_0"|$)', html, re.DOTALL)
    else:
        section = re.search(r'id="team_zhanji_0"(.*?)(?:id="team_zhanji2|$)', html, re.DOTALL)
    
    if not section:
        return results
    
    text = section.group(1)
    
    # 提取每行比赛数据
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', text, re.DOTALL)
    for row in rows:
        # 提取赛事类型
        comp_match = re.search(r'title="([^"]*)"', row)
        comp = comp_match.group(1) if comp_match else ''
        
        # 提取日期
        date_match = re.search(r'>(\d{2}-\d{2}-\d{2})<', row)
        date = date_match.group(1) if date_match else ''
        
        # 提取比分
        score_match = re.search(r'<em>(.*?)</em>', row)
        if score_match:
            score_raw = re.sub(r'<[^>]+>', '', score_match.group(1)).strip()
        else:
            score_raw = ''
        
        # 提取赛果
        result_match = re.search(r'<td[^>]*><span class="(ying|ping|shu)">([胜负平])</span>', row)
        result = result_match.group(2) if result_match else ''
        
        if date and score_raw and 'VS' not in score_raw:
            results.append({
                'date': date,
                'competition': comp,
                'score': score_raw,
                'result': result,
            })
    
    return results[:10]  # 最多10场


def parse_match_page(html, match_id):
    """解析单场比赛的数据分析页"""
    data = {'id': match_id}
    
    # === 比赛基本信息 ===
    title_match = re.search(r'<title>(.*?)</title>', html)
    if title_match:
        data['title'] = title_match.group(1)
    
    # 从页面头部提取队名 (当从列表获取不到时)
    team_pattern = r'<a class="hd_name" href="[^"]*">(.*?)</a>'
    team_names = re.findall(team_pattern, html)
    if len(team_names) >= 2:
        data.setdefault('home', team_names[0])
        data.setdefault('away', team_names[1])
    
    # 比赛时间
    time_match = re.search(r'比赛时间([\d-]+ [\d:]+)', html)
    if time_match:
        data['match_time'] = time_match.group(1)
    
    # === FIFA排名 ===
    fifa = {}
    for side, name in [('荷兰|主队', 'home'), ('客队', 'away')]:
        # 从排名表提取
        rank_pattern = r'<h3 class="lslayout1_stit">(.*?)[\[【](\d+)[\]】]</h3>'
        for m in re.finditer(rank_pattern, html):
            team = re.sub(r'<[^>]+>', '', m.group(1)).strip()
            rank = int(m.group(2))
            fifa[team] = rank
    
    # 也用更通用的方式
    rank_pattern2 = r'lslayout1_stit">(.*?)</h3>'
    for m in re.finditer(rank_pattern2, html):
        text = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        rm = re.search(r'(.*?)\[世(\d+)\]', text)
        if rm:
            fifa[rm.group(1).strip()] = int(rm.group(2))
    
    data['fifa_rankings'] = fifa
    
    # === 盘口/赔率 ===
    # 亚盘 (从数据表格提取)
    handicap_match = re.search(r'title="([^"]*)">\s*<span class="">\s*([\d.]+)\s*</span>\s*<span class="table_pl_center">\s*([^<]+)\s*</span>\s*<span[^>]*>\s*([\d.]+)', html)
    if handicap_match:
        data['asian_handicap'] = {
            'description': handicap_match.group(3).strip(),
            'home_water': float(handicap_match.group(2)),
            'away_water': float(handicap_match.group(4)),
        }
    
    # 欧赔 (从第一个比赛行提取)
    # 格式: <span>1.22</span><span>6.06</span><span>11.44</span>
    odds_pattern = r'<p class="pub_table_pl"><span>([\d.]+)</span><span>([\d.]+)</span><span>([\d.]+)</span></p>'
    odds_matches = re.findall(odds_pattern, html)
    if odds_matches:
        # 第一个通常是当前比赛的赔率
        data['euro_odds'] = {
            'home': float(odds_matches[0][0]),
            'draw': float(odds_matches[0][1]),
            'away': float(odds_matches[0][2]),
        }
    
    # === 近期战绩 ===
    # 主队战绩
    home_form = []
    # 从 bottom_info 提取汇总
    summary_pattern = r'<p><strong>(.*?)</strong>近(\d+)场战绩<span class="mar_left20"><span class="ying">(\d+)胜</span><span class="ping">(\d+)平</span><span class="shu">(\d+)负</span></span><span class="mar_left20">进<span class="ying">(\d+)球</span>失<span class="shu">(\d+)球</span></span></p>'
    summaries = re.findall(summary_pattern, html, re.DOTALL)
    
    if len(summaries) >= 1:
        data['home_summary'] = {
            'team': summaries[0][0],
            'matches': int(summaries[0][1]),
            'wins': int(summaries[0][2]),
            'draws': int(summaries[0][3]),
            'losses': int(summaries[0][4]),
            'goals_for': int(summaries[0][5]),
            'goals_against': int(summaries[0][6]),
        }
    if len(summaries) >= 2:
        data['away_summary'] = {
            'team': summaries[1][0],
            'matches': int(summaries[1][1]),
            'wins': int(summaries[1][2]),
            'draws': int(summaries[1][3]),
            'losses': int(summaries[1][4]),
            'goals_for': int(summaries[1][5]),
            'goals_against': int(summaries[1][6]),
        }
    
    # === 交战历史 ===
    h2h_section = re.search(r'交战历史(.*?)(?:<div class="M_box|近期战绩)', html, re.DOTALL)
    if h2h_section:
        h2h_text = h2h_section.group(1)
        if '暂无交战历史' in h2h_text:
            data['h2h'] = []
        else:
            data['h2h'] = 'has_data'  # 有数据时进一步解析
    
    # === 预计阵容 ===
    lineup = {'home': [], 'away': []}
    lineup_section = re.search(r'预计阵容(.*?)澳门心水', html, re.DOTALL)
    if lineup_section:
        lineup_text = lineup_section.group(1)
        # 提取首发球员
        player_pattern = r'<td class="td_one"><span class="td_sp3">(\d+)</span>(.*?)\((.*?)\)</td>'
        players = re.findall(player_pattern, lineup_text)
        # 前11个是主队首发
        for i, (num, name, pos) in enumerate(players):
            player = {'number': int(num), 'name': name.strip(), 'position': pos.strip()}
            if i < 11:
                lineup['home'].append(player)
    
    # 客队阵容 (在 team_b 区域)
    away_lineup_section = re.search(r'<div class="team_b">(.*?)<div class="clearb">', lineup_section.group(1) if lineup_section else '', re.DOTALL)
    
    # === 赢盘率/大球率 (从数据表格的 record_msg 提取) ===
    record_msgs = re.findall(r'record_msg">(.*?)</p>', html, re.DOTALL)
    for i, msg in enumerate(record_msgs):
        clean = re.sub(r'<[^>]+>', ' ', msg).strip()
        if i == 0:
            data['home_record_msg'] = clean
        elif i == 1:
            data['away_record_msg'] = clean
    
    data['lineup'] = lineup
    
    # === 澳门心水 ===
    recommend = re.search(r'推介\s*-\s*<font[^>]*>(.*?)</font>', html)
    if recommend:
        data['macau_tip'] = recommend.group(1).strip()
    
    recommend_reason = re.search(r'td_no4">\s*(.*?)\s*</td>', html, re.DOTALL)
    if recommend_reason:
        data['macau_reason'] = re.sub(r'<[^>]+>', '', recommend_reason.group(1)).strip()
    
    # === 未来赛事 ===
    future = {'home': [], 'away': []}
    future_section = re.search(r'未来赛事(.*?)平均数据|未来赛事(.*?)预计阵容', html, re.DOTALL)
    if future_section:
        ft = future_section.group(1) or future_section.group(2) or ''
        future_rows = re.findall(r'<td class="td_one matchname"[^>]*>.*?>(.*?)</a>.*?>([\d-]+)</td>.*?class="dz-l"[^>]*>(.*?)</a>.*?class="dz-r"[^>]*>(.*?)</a>', ft, re.DOTALL)
        for comp, date, team1, team2 in future_rows:
            entry = {'competition': comp.strip(), 'date': date.strip(),
                     'home': re.sub(r'<[^>]+>', '', team1).strip(),
                     'away': re.sub(r'<[^>]+>', '', team2).strip()}
            future['home'].append(entry)
    
    data['future_matches'] = future
    
    return data


def scrape_match(match_id):
    """爬取单场比赛的完整数据"""
    url = f'{BASE_URL}/fenxi/shuju-{match_id}.shtml'
    html = fetch(url)
    if not html:
        return None
    return parse_match_page(html, match_id)


def scrape_all_from_page(url):
    """从一个比赛页面提取同期所有比赛ID，然后逐个爬取"""
    html = fetch(url)
    if not html:
        return []
    
    # 提取所有比赛链接
    matches = []
    seen = set()
    
    # 从竞彩列表提取
    pattern = r'href="/fenxi/shuju-(\d+)\.shtml"[^>]*>.*?<em class="l">(.*?)</em>.*?<em class="r">(.*?)</em>'
    for m in re.finditer(pattern, html, re.DOTALL):
        mid = m.group(1)
        home = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        away = re.sub(r'<[^>]+>', '', m.group(3)).strip()
        if mid not in seen and home and away and 'VS' not in home:
            seen.add(mid)
            matches.append({'id': mid, 'home': home, 'away': away})
    
    print(f"  找到 {len(matches)} 场比赛")
    return matches


def main():
    parser = argparse.ArgumentParser(description='500.com 竞彩数据批量爬虫')
    parser.add_argument('--url', help='比赛页面URL (提取同期所有比赛)')
    parser.add_argument('--ids', help='逗号分隔的比赛ID列表')
    parser.add_argument('--output', default='/root/data/500_scraped.json', help='输出文件路径')
    parser.add_argument('--delay', type=float, default=1.5, help='请求间隔秒数')
    args = parser.parse_args()
    
    matches = []
    
    if args.ids:
        for mid in args.ids.split(','):
            matches.append({'id': mid.strip(), 'home': '?', 'away': '?'})
    elif args.url:
        matches = scrape_all_from_page(args.url)
    else:
        # 默认: 从已知页面提取
        matches = scrape_all_from_page(f'{BASE_URL}/fenxi/shuju-1411007.shtml')
    
    if not matches:
        print("❌ 没有找到比赛")
        return
    
    print(f"\n🔄 开始爬取 {len(matches)} 场比赛数据...\n")
    
    results = []
    for i, match in enumerate(matches):
        mid = match['id']
        print(f"  [{i+1}/{len(matches)}] {match['home']} vs {match['away']} (ID:{mid})")
        
        data = scrape_match(mid)
        if data:
            data['home'] = match['home']
            data['away'] = match['away']
            results.append(data)
            
            # 打印摘要
            hs = data.get('home_summary', {})
            aws = data.get('away_summary', {})
            euro = data.get('euro_odds', {})
            tip = data.get('macau_tip', '')
            
            if hs:
                print(f"         {hs.get('team','?')}: {hs.get('wins',0)}胜{hs.get('draws',0)}平{hs.get('losses',0)}负")
            if aws:
                print(f"         {aws.get('team','?')}: {aws.get('wins',0)}胜{aws.get('draws',0)}平{aws.get('losses',0)}负")
            if euro:
                print(f"         欧赔: {euro.get('home','-')} / {euro.get('draw','-')} / {euro.get('away','-')}")
            if tip:
                print(f"         澳门推介: {tip}")
        
        if i < len(matches) - 1:
            time.sleep(args.delay)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 完成! {len(results)}/{len(matches)} 场成功")
    print(f"📁 输出: {args.output}")
    
    # 输出汇总表
    print(f"\n{'='*70}")
    print(f"  {'比赛':<30} {'欧赔H':>6} {'欧赔D':>6} {'欧赔A':>6} {'推介'}")
    print(f"{'='*70}")
    for r in results:
        name = f"{r.get('home','?')} vs {r.get('away','?')}"
        euro = r.get('euro_odds', {})
        tip = r.get('macau_tip', '-')
        print(f"  {name:<30} {euro.get('home','-'):>6} {euro.get('draw','-'):>6} {euro.get('away','-'):>6} {tip}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
