import requests
from bs4 import BeautifulSoup
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://zx.500.com/zqdc/'
}

# zqdc 6/7 抓到的赛程 (字段含"平均赔率")
url = "https://zx.500.com/zqdc/kaijiang.php"
r = requests.get(url, params={"d": "2026-06-07"}, headers=headers, timeout=20)
r.encoding = "gb2312"
soup = BeautifulSoup(r.text, "html.parser")

tables = soup.find_all("table", class_="ld_table")
if tables:
    rows = tables[0].find_all("tr")
    print(f"总行数: {len(rows)}")
    print()
    # 6/8 凌晨未开赛 (41-46) - 重点关注
    target_indices = list(range(37, 47))  # 37~46 行
    for i in target_indices:
        if i >= len(rows):
            break
        r2 = rows[i]
        cells = r2.find_all("td")
        if len(cells) < 10:
            print(f"  Row {i}: 列数 {len(cells)}, 跳过")
            continue
        # zqdc 表格列: 编号, 赛事类型, 比赛时间, 主队, 让球, 客队, 比分,  让球彩果, 让球SP, 胜平负彩果, 胜平负SP, 总进球彩果, 总进球SP, 比分彩果, 比分SP, 上下单双彩果, 上下单双SP, 半全场彩果, 半全场SP
        # 因为未开赛, 比分列和彩果列都是 -
        try:
            mid = cells[0].get_text(strip=True)
            league = cells[1].get_text(strip=True)
            mt = cells[2].get_text(strip=True)
            home = cells[3].get_text(strip=True)
            rq_span = cells[4].find("span")
            rq = rq_span.get_text(strip=True) if rq_span else cells[4].get_text(strip=True)
            away = cells[5].get_text(strip=True)
            score = cells[6].get_text(strip=True)
            # 找平均赔率 - 在 "数据" 列?
            # zqdc 列: 0:编号 1:类型 2:时间 3:主 4:让 5:客 6:比分 7-9:让球彩果+SP 10-12:SPF 13-15:总进球 16-18:比分 19-21:上下 22-24:半全场
            # 赔率不在这里, 在 jczq 主页
            print(f"行 {i}: {mid:<4} {league:<8} {mt:<12} {home:<10} {rq:<4} {away:<10} 比分:{score}")
            # 找平均赔率 - 在 zqdc 表格中, 也许在 "数据" 列
            for c_idx, c in enumerate(cells):
                t = c.get_text(strip=True)
                if t and t not in ['-', '', '胜', '平', '负', '0', '1', '2', '3', '4', '5', '6', '7+', '胜胜', '胜平', '胜负', '平胜', '平平', '平负', '负胜', '负平', '负负']:
                    pass  # 可能是赔率
        except Exception as e:
            print(f"  Row {i} ERROR: {e}")
