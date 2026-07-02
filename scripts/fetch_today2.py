import requests
from bs4 import BeautifulSoup
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://zx.500.com/zqdc/'
}

url = "https://zx.500.com/zqdc/kaijiang.php"
r = requests.get(url, params={"d": "2026-06-07"}, headers=headers, timeout=20)
r.encoding = "gb2312"
soup = BeautifulSoup(r.text, "html.parser")

tables = soup.find_all("table", class_="ld_table")
if tables:
    rows = tables[0].find_all("tr")
    print(f"总行数: {len(rows)} (含表头)")
    # 行 0=表头, 1-36=已完赛(06-05~06-07), 37-51=未开赛(06-07晚~06-09)
    print()
    print("=== 全部赛程 ===")
    for i, r2 in enumerate(rows):
        cells = r2.find_all("td")
        if len(cells) < 8:
            if i == 0:
                print(f"表头: {[c.get_text(strip=True) for c in cells]}")
            continue
        try:
            mid = cells[0].get_text(strip=True)
            league = cells[1].get_text(strip=True)
            mt = cells[2].get_text(strip=True)
            home = cells[3].get_text(strip=True)
            rq_span = cells[4].find("span")
            rq = rq_span.get_text(strip=True) if rq_span else cells[4].get_text(strip=True)
            away = cells[5].get_text(strip=True)
            score = cells[6].get_text(strip=True)
            status = "已完赛" if score != "-" else "未开赛"
            print(f"{i:>2}. [{mid:<4}] {league:<8} {mt:<12} {home:<10} {rq:<4} {away:<10} {score:<10} [{status}]")
        except Exception as e:
            print(f"  Row {i} ERROR: {e}")
