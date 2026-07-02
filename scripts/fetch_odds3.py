import requests
from bs4 import BeautifulSoup
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# 500.com 友谊赛页 (liansai) - 找近期赛程
url = "https://liansai.500.com/zuqiu-19472/"
r = requests.get(url, headers=headers, timeout=15)
r.encoding = "gb2312"
soup = BeautifulSoup(r.text, "html.parser")

# 找比赛列表 - liansai 通常用 .saishi_list 或 .game_list
# 看 HTML 结构
content = soup.get_text()
# 找 "2026-06-07" "2026-06-08"
for date in ['2026-06-07', '2026-06-08', '06-07', '06-08', '06月07', '06月08']:
    idx = content.find(date)
    if idx >= 0:
        print(f"\n=== 找到 {date} ===")
        print(content[max(0,idx-200):idx+500])
        print()

# 找所有表格
tables = soup.find_all("table")
print(f"\n\ntables: {len(tables)}")
for i, t in enumerate(tables[:5]):
    rows = t.find_all("tr")
    if len(rows) > 0:
        first = " ".join(c.get_text(strip=True) for c in rows[0].find_all(['th', 'td']))
        print(f"Table {i}: {first[:200]}")
