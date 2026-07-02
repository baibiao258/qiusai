#!/usr/bin/env python3
"""
500.com 竞彩历史开奖数据抓取器

目标URL: https://zx.500.com/jczq/kaijiang.php?playid=0&d=YYYY-MM-DD

这个页面是离线回测的"黄金接口"：
  - 每场比赛的最终赛果（胜平负/让球/总进球/半全场）
  - 对应的收盘SP赔率（即当日最后能买到的价格）
  - 全场比分 + 半场比分 + 让球数
  - 所有数据来自官方开奖，无需自行判断赛果

用法:
  python3 historical_kaijiang.py [--start 2024-01-01] [--end 2026-06-09] [--delay 1.0]
  python3 historical_kaijiang.py --single 2026-06-08   # 单日测试
"""

import argparse
import csv
import json
import os
import sys
import time
import re
from datetime import date, datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from collections import OrderedDict

# ── 配置 ──
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://zx.500.com/jczq/',
}
BASE_URL = 'https://zx.500.com/jczq/kaijiang.php?playid=0&d={date}'
OUTPUT_CSV = '/root/data/historical_kaijiang.csv'
OUTPUT_JSON = '/root/data/historical_kaijiang.json'
PROGRESS_FILE = '/root/data/historical_kaijiang_progress.json'

# 结果映射 中文 → 内部编码
RESULT_MAP_SPF = {'胜': '3', '平': '1', '负': '0'}

# 比分正则: (HT:HT) FT:FT
SCORE_RE = re.compile(r'\((\d+):(\d+)\)\s*(\d+):(\d+)')

# 玩法中文 → key
PLAY_LABELS = {
    '让球胜平负': 'rq_spf',
    '胜平负': 'spf',
    '总进球数': 'jqs',
    '半全场': 'bqc',
}


def _fetch(url, timeout=10):
    """抓取单页HTML, GBK解码"""
    req = Request(url, headers=HEADERS)
    try:
        resp = urlopen(req, timeout=timeout)
        raw = resp.read()
        for enc in ['gbk', 'gb2312', 'gb18030', 'utf-8']:
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return raw.decode('utf-8', errors='replace')
    except Exception as e:
        return None


def parse_kaijiang_page(html: str, date_str: str) -> list[dict]:
    """
    解析开奖页面, 返回该日的比赛结果列表。
    
    HTML结构 (ld_table):
      每行19个td:
        [0] 赛事编号  [1] 赛事类型  [2] 比赛时间
        [3] 主队      [4] 让球数    [5] 客队
        [6] 比分 (半场)全场  [7] 分隔
        [8] 让球胜平负彩果   [9] 让球胜平负奖金(span.red)
        [10] 分隔              [11] 胜平负彩果
        [12] 胜平负奖金(span.red)  [13] 分隔
        [14] 总进球彩果      [15] 总进球奖金(span.red)
        [16] 分隔              [17] 半全场彩果
        [18] 半全场奖金(span.red)
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table', class_='ld_table')
    if not table:
        return []

    rows = table.find_all('tr')
    matches = []
    for row in rows:
        tds = row.find_all('td')
        if len(tds) != 19:
            continue

        code = tds[0].text.strip()
        if not code:
            continue

        match = {
            'date': date_str,
            'code': code,
            'league': tds[1].text.strip(),
            'time': tds[2].text.strip(),
            'home': tds[3].text.strip(),
            'away': tds[5].text.strip(),
            'handicap_str': tds[4].text.strip(),
        }

        # 比分
        score_raw = tds[6].text.strip()
        sm = SCORE_RE.match(score_raw)
        if sm:
            match['ht_h'], match['ht_a'] = int(sm.group(1)), int(sm.group(2))
            match['ft_h'], match['ft_a'] = int(sm.group(3)), int(sm.group(4))
            # 胜平负结果
            if match['ft_h'] > match['ft_a']:
                match['spf_result'] = '3'
            elif match['ft_h'] == match['ft_a']:
                match['spf_result'] = '1'
            else:
                match['spf_result'] = '0'
            # 总进球
            match['total_goals'] = match['ft_h'] + match['ft_a']
            # 半全场 -> raw编码: 3-3, 3-1, ...
            if match['ht_h'] > match['ht_a']:
                ht_code = '3'
            elif match['ht_h'] == match['ht_a']:
                ht_code = '1'
            else:
                ht_code = '0'
            if match['ft_h'] > match['ft_a']:
                ft_code = '3'
            elif match['ft_h'] == match['ft_a']:
                ft_code = '1'
            else:
                ft_code = '0'
            match['bqc_result'] = f'{ht_code}-{ft_code}'

        # 让球数 (解析数字, +2 → 2, -2 → -2)
        hcap = match['handicap_str']
        if hcap.startswith('+'):
            match['handicap'] = int(hcap[1:])
        elif hcap.startswith('-'):
            match['handicap'] = -int(hcap[1:])
        else:
            match['handicap'] = 0

        # ── 从 span.red 提取奖金 ──
        # 让球胜平负 [8]=彩果 [9]=奖金
        rq_result = tds[8].text.strip()
        rq_sp = tds[9].find('span', class_='red')
        match['rqspf_result'] = RESULT_MAP_SPF.get(rq_result, rq_result)
        match['rqspf_sp'] = float(rq_sp.text.strip()) if rq_sp and rq_sp.text.strip() not in ('--', '-', '') else 0.0

        # 胜平负 [11]=彩果 [12]=奖金
        spf_result = tds[11].text.strip()
        spf_sp = tds[12].find('span', class_='red')
        match['spf_sp'] = float(spf_sp.text.strip()) if spf_sp and spf_sp.text.strip() not in ('--', '-', '') else 0.0

        # 总进球 [14]=彩果 [15]=奖金
        jqs_result = tds[14].text.strip()
        jqs_sp = tds[15].find('span', class_='red')
        match['jqs_result'] = jqs_result
        match['jqs_sp'] = float(jqs_sp.text.strip()) if jqs_sp and jqs_sp.text.strip() not in ('--', '-', '') else 0.0

        # 半全场 [17]=彩果 [18]=奖金
        bqc_result = tds[17].text.strip()
        bqc_sp = tds[18].find('span', class_='red')
        # bqc_result 是中文: 胜胜, 平平, 负负等
        match['bqc_result_cn'] = bqc_result
        match['bqc_sp'] = float(bqc_sp.text.strip()) if bqc_sp and bqc_sp.text.strip() not in ('--', '-', '') else 0.0

        matches.append(match)

    return matches


def load_progress():
    """加载进度, 返回已处理的日期集合"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                return set(json.load(f).get('done_dates', []))
        except Exception:
            return set()
    return set()


def save_progress(done_dates: set):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'done_dates': sorted(done_dates), 'updated': datetime.now().isoformat()}, f)


def save_results_csv(all_matches: list[dict]):
    """保存为CSV (追加模式)"""
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'date', 'code', 'league', 'time',
            'home', 'away', 'handicap', 'handicap_str',
            'ht_h', 'ht_a', 'ft_h', 'ft_a',
            'spf_result', 'spf_sp',
            'rqspf_result', 'rqspf_sp',
            'total_goals', 'jqs_result', 'jqs_sp',
            'bqc_result', 'bqc_result_cn', 'bqc_sp',
        ])
        if not file_exists:
            writer.writeheader()
        for m in all_matches:
            writer.writerow(m)


def save_results_json(all_matches: list[dict]):
    """保存为JSON (覆盖写入, 按日期分组)"""
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    # 按日期分组
    by_date = OrderedDict()
    for m in all_matches:
        d = m['date']
        if d not in by_date:
            by_date[d] = []
        by_date[d].append(m)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({
            'total_matches': len(all_matches),
            'total_dates': len(by_date),
            'date_range': [min(by_date.keys()), max(by_date.keys())] if by_date else [],
            'by_date': by_date,
        }, f, ensure_ascii=False, indent=2)


def run(args):
    start = args.start
    end = args.end
    delay = args.delay
    force = args.force
    single = args.single

    if single:
        dates_to_fetch = [single]
    else:
        # 日期范围
        d = start
        dates_to_fetch = []
        while d <= end:
            dates_to_fetch.append(d.isoformat())
            d += timedelta(days=1)

    # 加载进度 (跳过已完成日期)
    done = set() if force else load_progress()
    if not single:
        dates_to_fetch = [d for d in dates_to_fetch if d not in done]
        if not dates_to_fetch:
            print(f'✅ 所有日期已处理 (共{len(done)}天)')
            return

    print(f'📡 开始抓取 {len(dates_to_fetch)} 天历史开奖数据...')
    all_matches = []
    success, fail, skip = 0, 0, 0

    for i, ds in enumerate(dates_to_fetch):
        if ds in done and not force:
            skip += 1
            continue

        url = BASE_URL.format(date=ds)
        html = _fetch(url)

        if not html:
            print(f'  ❌ [{i+1}/{len(dates_to_fetch)}] {ds} — 抓取失败')
            fail += 1
            time.sleep(delay * 2)
            continue

        matches = parse_kaijiang_page(html, ds)
        if matches:
            all_matches.extend(matches)
            done.add(ds)
            success += 1
            print(f'  ✅ [{i+1}/{len(dates_to_fetch)}] {ds} — {len(matches)} 场')
        else:
            # 无比赛也标记完成 (避免反复抓取)
            done.add(ds)
            skip += 1
            if i % 50 == 0:
                print(f'  ⏭️ [{i+1}/{len(dates_to_fetch)}] {ds} — 无赛事')

        # 保存进度
        if not single and (i + 1) % 10 == 0:
            save_progress(done)
            if all_matches:
                save_results_csv(all_matches)
                all_matches = []  # 清空已写CSV的内存

        time.sleep(delay)

    # 最终保存
    save_progress(done)
    if all_matches:
        save_results_csv(all_matches)

    # 重建JSON (从CSV读取全部)
    print(f'\n📊 重建 JSON 汇总...')
    all_data = []
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值字段
                for num_field in ['handicap', 'ht_h', 'ht_a', 'ft_h', 'ft_a', 'total_goals']:
                    if row.get(num_field):
                        row[num_field] = int(row[num_field])
                for sp_field in ['spf_sp', 'rqspf_sp', 'jqs_sp', 'bqc_sp']:
                    if row.get(sp_field):
                        row[sp_field] = float(row[sp_field])
                all_data.append(row)

    save_results_json(all_data)

    print(f'\n{"="*50}')
    print(f'  ✅ 完成!')
    print(f'  ✅ 成功: {success} 天')
    print(f'  ❌ 失败: {fail} 天')
    print(f'  ⏭️ 跳过: {skip} 天 (无赛事或已处理)')
    print(f'  📊 总场次: {len(all_data)}')
    print(f'  📁 CSV: {OUTPUT_CSV}')
    print(f'  📁 JSON: {OUTPUT_JSON}')
    print(f'  📁 进度: {PROGRESS_FILE}')
    print(f'{"="*50}')


def main():
    ap = argparse.ArgumentParser(description='500.com 竞彩历史开奖数据抓取器')
    ap.add_argument('--start', type=lambda s: date.fromisoformat(s),
                    default=date(2024, 1, 1),
                    help='开始日期 (默认 2024-01-01)')
    ap.add_argument('--end', type=lambda s: date.fromisoformat(s),
                    default=date.today() - timedelta(days=1),
                    help='结束日期 (默认昨日)')
    ap.add_argument('--delay', type=float, default=0.5,
                    help='请求间隔秒数 (默认 0.5)')
    ap.add_argument('--force', action='store_true',
                    help='强制重新抓取所有日期')
    ap.add_argument('--single', type=str, default=None,
                    help='抓取单日, 格式 YYYY-MM-DD')
    ap.add_argument('--test', action='store_true',
                    help='测试模式: 只抓取今天+昨天, 验证解析逻辑')

    args = ap.parse_args()

    if args.test:
        today = date.today()
        args.single = today.isoformat()

    run(args)


if __name__ == '__main__':
    main()
