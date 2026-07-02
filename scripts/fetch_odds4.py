import requests
from bs4 import BeautifulSoup
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# 500.com 友谊赛页
url = "https://liansai.500.com/zuqiu-19472/"
r = requests.get(url, headers=headers, timeout=15)
r.encoding = "gb2312"
soup = BeautifulSoup(r.text, "html.parser")

# Table 1 是赛程表
tables = soup.find_all("table")
t = tables[1]
rows = t.find_all("tr")
print(f"赛程表行数: {len(rows)}")
print()
for r2 in rows[:50]:
    cells = r2.find_all(['th', 'td'])
    cell_text = [c.get_text(strip=True) for c in cells]
    # 只显示有内容的行
    if any('2026-06-07' in c or '2026-06-08' in c or '06-07' in c or '06-08' in c for c in cell_text):
        print(cell_text)
    elif any(c for c in cell_text if c and c not in ['VS', '查看', '析', '平均欧指']):
        # 显示有队伍名的行
        if len(cell_text) >= 5:
            h, a = cell_text[1], cell_text[3] if len(cell_text) > 3 else ''
            if h and a and not any('球' in c for c in cell_text):
                print(f"  {cell_text[0]:<14} {h:<15} VS {a:<15} 赛果: {cell_text[2] if len(cell_text) > 2 else ''}")
