import requests
from bs4 import BeautifulSoup
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# 抓友谊赛欧赔
url = "https://odds.500.com/fenxi/ouzhi-19472.shtml"
r = requests.get(url, headers=headers, timeout=15)
r.encoding = "gb2312"
soup = BeautifulSoup(r.text, "html.parser")

# 找 6/7~6/8 比赛
text = soup.get_text()

# 看页面里的比赛列表
tables = soup.find_all("table")
print(f"tables: {len(tables)}")
for i, t in enumerate(tables):
    rows = t.find_all("tr")
    if len(rows) < 2:
        continue
    first = " ".join(c.get_text(strip=True) for c in rows[0].find_all(['th', 'td']))
    if '主队' in first or '球队' in first or 'VS' in first or '胜' in first:
        print(f"\nTable {i} ({len(rows)} rows): {first[:200]}")
        for r2 in rows[:20]:
            cells = r2.find_all(['th', 'td'])
            cell_text = [c.get_text(strip=True) for c in cells]
            if any('VS' in c or '06-' in c for c in cell_text):
                print(f"  {cell_text[:12]}")
        break
