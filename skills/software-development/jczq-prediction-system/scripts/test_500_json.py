#!/usr/bin/env python3
"""
test_500_json.py — 500.com 数据提取测试 (正则 vs BeautifulSoup)
=============================================================

结论: 500.com 没有嵌入 JS 变量/JSON。数据在 HTML data-sp/data-type/data-value 属性中。
正则提取 57x faster than BeautifulSoup (2ms vs 121ms per 26 fixtures).

用法: python3 scripts/test_500_json.py
"""
import re, sys, time, urllib.request

def extract_odds_regex(html):
    fixtures = []
    rows = re.findall(r'<tr[^>]*data-fixtureid=["\'](\d+)["\'][^>]*>(.*?)</tr>', html, re.DOTALL)
    for fid, content in rows:
        league_m = re.search(r'class=["\'][^"\']*evt[^"\']*["\'][^>]*>\s*<a[^>]*>([^<]+)</a>', content)
        teams = re.findall(r'class=["\']team-[lr]["\'][^>]*>([^<]+)<', content)
        handicap_m = re.search(r'itm-rangA2[^>]*>\s*([-+]?\d+)', content)
        odds_entries = re.findall(
            r'data-type=["\'](\w+)["\'][^>]*data-value=["\'](\d+)["\'][^>]*data-sp=["\']([0-9.]+)["\']', content)
        odds = {}
        for pt, val, sp in odds_entries:
            odds.setdefault(pt, {})[val] = float(sp)
        fixtures.append({
            'fixtureid': fid, 'league': league_m.group(1).strip() if league_m else '',
            'home': teams[0].strip() if teams else '', 'away': teams[1].strip() if len(teams)>1 else '',
            'handicap': int(handicap_m.group(1)) if handicap_m else 0, 'odds': odds,
        })
    return fixtures

def extract_odds_bs4(html):
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    soup = BeautifulSoup(html, 'html.parser')
    fixtures = []
    for tr in soup.find_all('tr', attrs={'data-fixtureid': True}):
        fid = tr['data-fixtureid']
        odds = {}
        for node in tr.find_all(attrs={'data-sp': True}):
            pt, val = node.get('data-type',''), node.get('data-value','')
            sp = float(node.get('data-sp', 0))
            if pt and val: odds.setdefault(pt, {})[val] = sp
        fixtures.append({'fixtureid': fid, 'odds': odds})
    return fixtures

if __name__ == '__main__':
    url = "https://trade.500.com/jczq/?playid=269&g=2&date=2026-06-10"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://trade.500.com/'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('gbk', errors='replace')
    print(f"Page: {len(html)} bytes")

    t0 = time.time()
    r1 = [extract_odds_regex(html) for _ in range(100)]
    t1 = (time.time()-t0)/100*1000

    t0 = time.time()
    r2 = [extract_odds_bs4(html) for _ in range(100)]
    t2 = (time.time()-t0)/100*1000

    print(f"Regex: {t1:.1f}ms | BS4: {t2:.1f}ms | Speedup: {t2/t1:.0f}x")
    print(f"Fixtures: regex={len(r1[0])}, bs4={len(r2[0])}")
