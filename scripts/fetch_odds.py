import requests
from bs4 import BeautifulSoup
import re

# 500.com zqdc 6/7 抓到的赛程 (37-46 场是未完赛)
# 关键场次: 6/8 凌晨友谊赛
# 用 liansai.500.com 抓赔率 (赔率页)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# 6/8 凌晨 4 场友谊赛赔率 (欧赔初盘/终盘)
# 用 liansai.500.com 的具体场次 ID
# 友谊赛联赛 ID: 19472 (从 6/6 kaijiang HTML 看到的)

# 先看 liansai 友谊赛页
url = "https://liansai.500.com/zuqiu-19472/"
r = requests.get(url, headers=headers, timeout=15)
r.encoding = "gb2312"
soup = BeautifulSoup(r.text, "html.parser")
text = soup.get_text()

# 找赛程
print("=== 友谊赛 6/7~6/8 赛程 ===")
# 找日期
dates = re.findall(r'(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2})?\s*([一-龥]+)\s*VS\s*([一-龥]+)', text)
for d, t, h, a in dates:
    if '2026-06-07' in d or '2026-06-08' in d or '2026-06-06' in d:
        print(f"  {d} {t or ''} {h} VS {a}")

# 也看 odds 页面
print("\n=== 赔率页 odds.500.com ===")
odds_urls = [
    "https://odds.500.com/fenxi/ouzhi-19472.shtml",
    "https://odds.500.com/fenxi/yrz.php?id=19472",
    "https://odds.500.com/fenxi/yazhi-19472.shtml",
]
for u in odds_urls:
    try:
        r = requests.get(u, headers=headers, timeout=5)
        r.encoding = "gb2312"
        print(f"{u}: {r.status_code} - {len(r.text)}")
    except Exception as e:
        print(f"{u}: ERROR {e}")
