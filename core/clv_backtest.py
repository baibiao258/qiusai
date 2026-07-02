#!/usr/bin/env python3
"""
clv_backtest.py — CLV (Closing Line Value) 回测系统
====================================================
用 500.com 真实历史收盘赔率验证预测管线的价值发现能力。

核心逻辑:
  CLV = (closing_odds / our_odds - 1) × 100%
  正CLV = 我们比市场更早发现价值 (好)
  负CLV = 市场赔率比我们更好 (差)

用法:
  python3 clv_backtest.py                  # 回测 predictions_log.csv
  python3 clv_backtest.py --fetch          # 先抓取历史赔率再回测
  python3 clv_backtest.py --report         # 只看已有数据的报告
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime, date, timedelta
from collections import defaultdict
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': 'https://odds.500.com/',
}
BASE_URL = 'https://odds.500.com'
HISTORICAL_ODDS_PATH = '/root/data/500_historical_odds.json'
PREDICTIONS_LOG = '/root/data/predictions_log.csv'


def _fetch(url, timeout=12):
    req = Request(url, headers=HEADERS)
    try:
        resp = urlopen(req, timeout=timeout)
        raw = resp.read()
        for enc in ['gb2312', 'gbk', 'gb18030', 'utf-8']:
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return raw.decode('utf-8', errors='replace')
    except Exception:
        return None


def load_historical_odds():
    """加载已缓存的历史赔率"""
    if os.path.exists(HISTORICAL_ODDS_PATH):
        with open(HISTORICAL_ODDS_PATH) as f:
            return json.load(f)
    return {}


def save_historical_odds(data):
    os.makedirs(os.path.dirname(HISTORICAL_ODDS_PATH), exist_ok=True)
    with open(HISTORICAL_ODDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def search_500_match_id(home_cn, away_cn):
    """
    在500.com搜索比赛的shuju ID。
    通过搜索页面找到匹配的比赛。
    """
    # 方法1: 通过500.com搜索接口
    search_url = f'https://live.500.com/search.php?keyword={home_cn}'
    html = _fetch(search_url, timeout=8)
    if html:
        # 搜索结果里找匹配的比赛链接
        pattern = r'shuju-(\d+)\.shtml'
        ids = re.findall(pattern, html)
        if ids:
            return ids[0]

    # 方法2: 通过联赛列表页查找 (需要知道联赛)
    return None


def fetch_match_closing_odds(shuju_id):
    """
    从500.com抓取一场比赛的收盘赔率。
    对于已结束的比赛: 从 bmatch 行提取收盘赔率和比分。
    返回 dict 或 None。
    """
    url = f'{BASE_URL}/fenxi/shuju-{shuju_id}.shtml'
    html = _fetch(url)
    if not html:
        return None

    result = {'shuju_id': shuju_id}

    # 比赛信息
    title_m = re.search(r'<title>(.*?)</title>', html)
    if title_m:
        result['title'] = title_m.group(1)

    # 比分 (从页面头部)
    score_m = re.search(r'<p class="odds_hd_bf"><strong>(\d+:\d+)</strong>', html)
    if score_m:
        result['score'] = score_m.group(1)
        hg, ag = map(int, result['score'].split(':'))
        result['hda'] = 'H' if hg > ag else ('D' if hg == ag else 'A')

    # 从 bmatch 行提取: 找 fid=shuju_id 的行
    # 已结束比赛: td[5] = 欧赔, td[6] = 亚盘
    # 未开始比赛: td[3] = VS, td[5] = 欧赔, td[6] = 亚盘
    row_m = re.search(
        r'<tr fid="%s"[^>]*>(.*?)</tr>' % re.escape(shuju_id),
        html, re.DOTALL
    )
    if row_m:
        row = row_m.group(1)
        tds = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)

        # 解析欧赔 (td[5] 或包含 pub_table_pl 的 td)
        for td in tds:
            if 'pub_table_pl' in td:
                odds_vals = re.findall(r'<span[^>]*>([\d.]+)</span>', td)
                if len(odds_vals) >= 3:
                    result['euro_odds'] = {
                        'home': float(odds_vals[0]),
                        'draw': float(odds_vals[1]),
                        'away': float(odds_vals[2]),
                    }
                break

        # 解析亚盘 (table_pl_center)
        for td in tds:
            if 'table_pl_center' in td:
                ah_vals = re.findall(r'<span[^>]*>([\d.]+)</span>', td)
                line_m = re.search(r'table_pl_center">\s*([^<]+)\s*</span>', td)
                if ah_vals and line_m:
                    result['asian_handicap'] = {
                        'home_water': float(ah_vals[0]) if ah_vals else 0,
                        'line': line_m.group(1).strip(),
                        'away_water': float(ah_vals[1]) if len(ah_vals) > 1 else 0,
                    }
                break

    return result


def load_500_analysis_cache():
    """从 daily_jczq 的分析缓存中读取 shuju_id 和赔率"""
    path = '/root/data/500_analysis_cache.json'
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    data = raw.get('data', {})
    # 转换为 CLV 格式: code -> {shuju_id, euro_odds, score}
    result = {}
    for code, analysis in data.items():
        sid = analysis.get('shuju_id', '')
        odds = analysis.get('current_euro_odds', {})
        if sid and odds:
            result[code] = {
                'shuju_id': sid,
                'euro_odds': odds,
                'score': analysis.get('score', ''),
            }
    return result


def fetch_batch_historical_odds(predictions, delay=2.0):
    """
    批量抓取历史赔率。
    优先从 500_analysis_cache 获取 shuju_id，然后逐个抓取页面获取收盘赔率。
    """
    cache = load_historical_odds()
    analysis_cache = load_500_analysis_cache()
    total = len(predictions)
    fetched = 0
    skipped = 0

    print(f"\n📡 批量抓取历史收盘赔率 ({total}场)...")
    print(f"   分析缓存有 {len(analysis_cache)} 场可用")

    for i, pred in enumerate(predictions):
        code = pred.get('code', '')
        home = pred.get('home_cn', '')
        away = pred.get('away_cn', '')

        # 已有缓存则跳过
        if code in cache and cache[code].get('euro_odds'):
            skipped += 1
            continue

        # 从分析缓存获取 shuju_id
        ac = analysis_cache.get(code, {})
        shuju_id = ac.get('shuju_id', '')

        # 备用: 搜索500.com
        if not shuju_id:
            import re as _re
            clean_home = _re.sub(r'\[\d+\]', '', home).strip()
            shuju_id = search_500_match_id(clean_home, away)

        if not shuju_id:
            print(f"  [{i+1}/{total}] ❌ {code} {home}vs{away} — 无shuju_id")
            continue

        # 抓取收盘赔率
        odds_data = fetch_match_closing_odds(shuju_id)
        if odds_data and odds_data.get('euro_odds'):
            cache[code] = odds_data
            fetched += 1
            eo = odds_data['euro_odds']
            print(f"  [{i+1}/{total}] ✅ {code} {home}vs{away} 收盘欧赔: {eo['home']}/{eo['draw']}/{eo['away']}")
        else:
            print(f"  [{i+1}/{total}] ⚠️ {code} {home}vs{away} — 页面无赔率数据 (ID:{shuju_id})")

        time.sleep(delay)

    save_historical_odds(cache)
    print(f"\n  抓取完成: 新增{fetched}场, 跳过{skipped}场(已有缓存), 共{len(cache)}场")
    return cache


def compute_clv(our_odds, closing_odds):
    """
    计算单场CLV。
    CLV = (closing_odds / our_odds - 1)
    正值 = 我们拿到的赔率比收盘更好 (正EV)
    """
    if our_odds <= 0 or closing_odds <= 0:
        return None
    return closing_odds / our_odds - 1


def compute_ev(prob, odds):
    """计算单场EV"""
    if odds <= 0 or prob <= 0:
        return None
    return prob * (odds - 1) - (1 - prob)


def run_backtest(predictions, historical_odds):
    """
    执行CLV回测。
    返回结构化结果。
    """
    results = []
    clv_values = []
    ev_values = []

    for pred in predictions:
        code = pred.get('code', '')
        home = pred.get('home_cn', '')
        away = pred.get('away_cn', '')

        # 我们的赔率
        try:
            our_h = float(pred.get('odds_h', 0) or 0)
            our_d = float(pred.get('odds_d', 0) or 0)
            our_a = float(pred.get('odds_a', 0) or 0)
        except (ValueError, TypeError):
            continue

        if our_h <= 1 or our_d <= 1 or our_a <= 1:
            continue

        # 我们的概率
        try:
            pred_h = float(pred.get('pred_h', 0) or 0) / 100
            pred_d = float(pred.get('pred_d', 0) or 0) / 100
            pred_a = float(pred.get('pred_a', 0) or 0) / 100
        except (ValueError, TypeError):
            continue

        # 收盘赔率
        hist = historical_odds.get(code, {})
        closing = hist.get('euro_odds', {})
        if not closing:
            continue

        closing_h = closing.get('home', 0)
        closing_d = closing.get('draw', 0)
        closing_a = closing.get('away', 0)

        if closing_h <= 1 or closing_d <= 1 or closing_a <= 1:
            continue

        # CLV (每个选项)
        clv_h = compute_clv(our_h, closing_h)
        clv_d = compute_clv(our_d, closing_d)
        clv_a = compute_clv(our_a, closing_a)

        # EV (用我们的概率 vs 我们的赔率)
        ev_h = compute_ev(pred_h, our_h)
        ev_d = compute_ev(pred_d, our_d)
        ev_a = compute_ev(pred_a, our_a)

        # EV (用我们的概率 vs 收盘赔率)
        ev_h_close = compute_ev(pred_h, closing_h)
        ev_d_close = compute_ev(pred_d, closing_d)
        ev_a_close = compute_ev(pred_a, closing_a)

        # 实际结果
        actual_hda = pred.get('actual_hda', '')
        checked = pred.get('checked', '0') == '1'

        row = {
            'code': code,
            'home': home,
            'away': away,
            'our_odds': {'H': our_h, 'D': our_d, 'A': our_a},
            'closing_odds': {'H': closing_h, 'D': closing_d, 'A': closing_a},
            'clv': {'H': clv_h, 'D': clv_d, 'A': clv_a},
            'pred_prob': {'H': pred_h, 'D': pred_d, 'A': pred_a},
            'ev_ours': {'H': ev_h, 'D': ev_d, 'A': ev_a},
            'ev_closing': {'H': ev_h_close, 'D': ev_d_close, 'A': ev_a_close},
            'actual_hda': actual_hda,
            'checked': checked,
            'score': hist.get('score', ''),
        }
        results.append(row)

        # 汇总 CLV (只统计我们推荐的选项)
        pick = pred.get('pred_spf_pick', '')
        pick_key = {'主胜': 'H', '平': 'D', '客胜': 'A'}.get(pick, '')
        if pick_key and row['clv'][pick_key] is not None:
            clv_values.append(row['clv'][pick_key])
        if pick_key and row['ev_ours'][pick_key] is not None:
            ev_values.append(row['ev_ours'][pick_key])

    return results, clv_values, ev_values


def print_report(results, clv_values, ev_values):
    """打印CLV回测报告"""
    print(f"\n{'='*70}")
    print(f"  📊 CLV 回测报告")
    print(f"{'='*70}")

    if not results:
        print("  ❌ 无可用数据 (需要 predictions_log + 历史收盘赔率)")
        return

    # 汇总统计
    n = len(results)
    print(f"\n  样本量: {n} 场")

    # CLV 统计
    if clv_values:
        avg_clv = sum(clv_values) / len(clv_values)
        positive_clv = sum(1 for c in clv_values if c > 0)
        print(f"\n  ── CLV (Closing Line Value) ──")
        print(f"  推荐选项平均CLV: {avg_clv:+.2%}")
        print(f"  正CLV场次: {positive_clv}/{len(clv_values)} ({positive_clv/len(clv_values)*100:.1f}%)")
        if avg_clv > 0:
            print(f"  ✅ 正CLV = 我们的赔率比市场收盘更好 (模型有预测价值)")
        else:
            print(f"  ⚠️ 负CLV = 市场收盘赔率比我们更好 (模型可能高估)")

    # EV 统计
    if ev_values:
        avg_ev = sum(ev_values) / len(ev_values)
        positive_ev = sum(1 for e in ev_values if e > 0)
        print(f"\n  ── EV (Expected Value) ──")
        print(f"  推荐选项平均EV: {avg_ev:+.2%}")
        print(f"  正EV场次: {positive_ev}/{len(ev_values)} ({positive_ev/len(ev_values)*100:.1f}%)")

    # 逐场明细
    print(f"\n  ── 逐场明细 ──")
    print(f"  {'比赛':<25} {'推荐':<5} {'我们的赔率':>8} {'收盘赔率':>8} {'CLV':>8} {'EV':>8} {'实际'}")
    print(f"  {'─'*80}")

    for r in results:
        pick_map = {'主胜': 'H', '平': 'D', '客胜': 'A'}
        pick = ''
        for label, key in pick_map.items():
            if r['pred_prob'].get(key, 0) == max(r['pred_prob'].values()):
                pick = key
                break

        our = r['our_odds'].get(pick, 0)
        closing = r['closing_odds'].get(pick, 0)
        clv = r['clv'].get(pick)
        ev = r['ev_ours'].get(pick)
        actual = r['actual_hda'] or '?'

        name = f"{r['home'][:8]}vs{r['away'][:8]}"
        clv_str = f"{clv:+.1%}" if clv is not None else '-'
        ev_str = f"{ev:+.1%}" if ev is not None else '-'

        print(f"  {name:<25} {pick:<5} {our:>8.2f} {closing:>8.2f} {clv_str:>8} {ev_str:>8} {actual}")

    # 赔率差异分析
    print(f"\n  ── 赔率差异分析 ──")
    for outcome in ['H', 'D', 'A']:
        diffs = []
        for r in results:
            our = r['our_odds'].get(outcome, 0)
            closing = r['closing_odds'].get(outcome, 0)
            if our > 0 and closing > 0:
                diffs.append(closing / our - 1)
        if diffs:
            avg = sum(diffs) / len(diffs)
            label = {'H': '主胜', 'D': '平局', 'A': '客胜'}[outcome]
            emoji = '✅' if avg > 0 else '⚠️'
            print(f"  {emoji} {label}: 平均CLV = {avg:+.2%} (收盘/我方 - 1)")

    print(f"\n{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CLV回测系统')
    parser.add_argument('--fetch', action='store_true', help='先抓取历史赔率再回测')
    parser.add_argument('--report', action='store_true', help='只看已有数据报告')
    parser.add_argument('--delay', type=float, default=2.0, help='抓取间隔秒数')
    args = parser.parse_args()

    # 加载预测
    if not os.path.exists(PREDICTIONS_LOG):
        print("❌ 无 predictions_log.csv")
        return

    with open(PREDICTIONS_LOG, encoding='utf-8') as f:
        predictions = list(csv.DictReader(f))

    print(f"📋 加载 {len(predictions)} 条预测记录")

    # 加载历史赔率
    historical_odds = load_historical_odds()
    print(f"📦 已有历史赔率缓存: {len(historical_odds)} 场")

    # 抓取模式
    if args.fetch:
        historical_odds = fetch_batch_historical_odds(predictions, delay=args.delay)

    # 回测
    results, clv_values, ev_values = run_backtest(predictions, historical_odds)

    if not results:
        print("\n⚠️ 无匹配数据。需要:")
        print("  1. predictions_log.csv 中有预测记录")
        print("  2. 500.com 历史收盘赔率缓存")
        print("\n  运行: python3 clv_backtest.py --fetch  抓取历史赔率")
        return

    print_report(results, clv_values, ev_values)


if __name__ == '__main__':
    main()
