#!/usr/bin/env python3
"""
fetch_500_wanchang.py
抓取 live.500.com/wanchang.php 历史完场数据

用法:
  # 单天
  python3 fetch_500_wanchang.py --date 2026-06-12

  # 批量回填
  python3 fetch_500_wanchang.py --start 2026-01-01 --end 2026-06-13

  # 只抓指定联赛(league_id)
  python3 fetch_500_wanchang.py --start 2026-01-01 --end 2026-06-13 --leagues 110 1829 65
"""
import re, csv, time, argparse, os, subprocess
from datetime import date, timedelta

OUTPUT_CSV = "/root/data/500_history_backfill.csv"
CURL_HEADERS = [
    '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    '-H', 'Referer: https://live.500.com/',
    '-H', 'Accept-Language: zh-CN,zh;q=0.9',
]

# ── 联赛白名单（从 paste.txt 的 checkbox 提取，只保留主流足球）
# league_id → 联赛名（颜色只是显示用，不影响筛选）
# 不填 --leagues 则抓全部
KNOWN_FOOTBALL_LEAGUES = {
    "110":  "世界杯",
    "1829": "世界杯预选赛-亚洲",
    "2241": "世界杯预选赛-欧洲",
    "65":   "世界杯预选赛-北美",
    "1011": "世界杯预选赛-U23",
    "2446": "世界杯预选赛-南美",
    "574":  "欧冠",
    "609":  "欧联",
    "558":  "英超",
    "557":  "西甲",
    "629":  "法甲",
    "611":  "德甲",
    "634":  "意甲",
    "50":   "荷甲",
    "707":  "葡超",
    "619":  "土超",
    "137":  "中超",
    "52":   "日职",
    "330":  "韩职",
    # 可在此追加
}


def fetch_one_day(date_str: str, allowed_leagues: set = None) -> list:
    """
    抓单天数据，返回 list of dict
    allowed_leagues: set of str league_id, None = 不过滤
    """
    url = f"https://live.500.com/wanchang.php?e={date_str}"
    try:
        r = subprocess.run(
            ['curl', '-s', url, '--max-time', '30'] + CURL_HEADERS,
            capture_output=True, timeout=35
        )
        # ★ 必须 GBK 解码，curl 返回 bytes
        html = r.stdout.decode("gbk", errors="replace")
    except Exception as e:
        print(f"  [{date_str}] 请求失败: {e}")
        return []

    # ── 1. 解析半场比分 parentid 行
    # 格式: <tr parentid="a{fid}" ...>...<font color="A52A2A">45</font> 1-0...
    ht_map = {}
    for fid, block in re.findall(
        r'<tr\s+parentid="a(\d+)"[^>]*>(.*?)</tr>', html, re.DOTALL
    ):
        # 找 45分钟那行
        m = re.search(r'<font[^>]*>45</font>\s*([\d]+-[\d]+)', block)
        if m:
            ht_map[fid] = m.group(1)  # "1-0"

    # ── 2. 解析主行
    results = []
    for m in re.finditer(
        r'<tr\s+id="a(\d+)"[^>]*\blid="(\d+)"[^>]*>(.*?)</tr>',
        html, re.DOTALL
    ):
        fid, lid, row = m.group(1), m.group(2), m.group(3)

        # 联赛过滤
        if allowed_leagues and lid not in allowed_leagues:
            continue

        # 日期时间  "06-12 10:00"
        dt_m = re.search(r'<td[^>]*>(\d{2}-\d{2}\s+[\d:]+)</td>', row)
        if not dt_m:
            continue
        dt_str = dt_m.group(1).strip()          # "06-12 10:00"
        month_day = dt_str[:5]                   # "06-12"
        time_str  = dt_str[6:].strip()           # "10:00"
        full_date = f"{date_str[:4]}-{month_day}"  # "2026-06-12"

        # 主队
        home_m = re.search(r'<span class="mainName[^"]*">([^<]+)</span>', row)
        # 客队
        away_m = re.search(r'<span class="clientName[^"]*">([^<]+)</span>', row)
        # 全场比分  class="red">2 - 1<
        score_m = re.search(r'<td[^>]*class="red">([^<]+)</td>', row)

        if not (home_m and away_m and score_m):
            continue

        score_full = score_m.group(1).strip()    # "2 - 1"
        # 跳过未完场（比分含 ":" 是进行中，或为空）
        if ":" in score_full or not re.search(r'\d', score_full):
            continue

        # 期号（竞彩期号，没有就空）
        no_m = re.search(r'<td[^>]*>(\d+)</td>', row)
        match_no = no_m.group(1) if no_m else ""

        results.append({
            "date":       full_date,
            "time":       time_str,
            "fid":        fid,
            "league_id":  lid,
            "league_name": KNOWN_FOOTBALL_LEAGUES.get(lid, lid),
            "match_no":   match_no,
            "home":       home_m.group(1).strip(),
            "away":       away_m.group(1).strip(),
            "score_full": score_full,            # "2 - 1"
            "score_ht":   ht_map.get(fid, ""),   # "1-0" 或 ""
            # 方便后续解析
            "home_goals": score_full.split("-")[0].strip() if "-" in score_full.replace(" ","") else "",
            "away_goals": score_full.split("-")[-1].strip() if "-" in score_full.replace(" ","") else "",
        })

    return results


def backfill(start: str, end: str, allowed_leagues: set = None,
             output_csv: str = OUTPUT_CSV, sleep: float = 0.5):
    """
    批量回填 [start, end] 每天数据到 CSV
    已存在的 fid 自动去重（append 模式，不会重复写）
    """
    # 读已有 fid
    existing_fids = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_fids.add(row.get("fid", ""))

    FIELDS = ["date","time","fid","league_id","league_name","match_no",
              "home","away","score_full","score_ht","home_goals","away_goals"]

    write_header = not os.path.exists(output_csv)
    total_new = 0

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
        while d <= end_d:
            ds = d.isoformat()
            rows = fetch_one_day(ds, allowed_leagues)
            new_rows = [r for r in rows if r["fid"] not in existing_fids]
            writer.writerows(new_rows)
            existing_fids.update(r["fid"] for r in new_rows)
            total_new += len(new_rows)
            print(f"  {ds}: {len(rows)} 场, 新写入 {len(new_rows)} 条")
            d += timedelta(days=1)
            time.sleep(sleep)

    print(f"\n完成！共写入 {total_new} 条新记录 → {output_csv}")


def merge_to_intl(csv_path: str, intl_path: str = "/root/data/international_results.json"):
    """
    将 500 数据合并到 international_results.json
    以 (date, home_lower, away_lower) 做去重键
    """
    if not os.path.exists(csv_path):
        print(f"❌ 找不到 {csv_path}")
        return 0
    import json
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not os.path.exists(intl_path):
        print(f"❌ 找不到 {intl_path}")
        return 0
    with open(intl_path, "r", encoding="utf-8") as f:
        intl = json.load(f)

    seen = set()
    for m in intl:
        seen.add((m.get("date",""), m.get("home","").lower().strip(),
                  m.get("away","").lower().strip()))

    added = 0
    for r in rows:
        sf = r.get("score_full","")
        if not sf or "-" not in sf:
            continue
        try:
            hs, ha = int(sf.split("-")[0].strip()), int(sf.split("-")[1].strip())
        except (ValueError, IndexError):
            continue
        key = (r["date"], r["home"].lower(), r["away"].lower())
        if key in seen:
            continue
        intl.append({
            "date": r["date"],
            "home": r["home"],
            "away": r["away"],
            "tournament": r.get("league_name","500.cn"),
            "h_score": hs,
            "a_score": ha,
            "neutral": False,
        })
        added += 1

    with open(intl_path, "w", encoding="utf-8") as f:
        json.dump(intl, f, ensure_ascii=False)
    print(f"✅ {intl_path}: +{added} 新场 (总计 {len(intl)} 场)")
    return added


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",  help="单天 YYYY-MM-DD")
    parser.add_argument("--start", help="批量开始日期")
    parser.add_argument("--end",   help="批量结束日期")
    parser.add_argument("--leagues", nargs="*",
                        help="只抓指定 league_id，不填=全部  例: --leagues 110 558 557")
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--merge", action="store_true",
                        help="完成后合并到 international_results.json")
    args = parser.parse_args()

    allowed = set(args.leagues) if args.leagues else None

    if args.date:
        rows = fetch_one_day(args.date, allowed)
        for r in rows:
            print(f"{r['date']} {r['home']:20s} {r['score_full']:5s} {r['away']:20s} "
                  f"[{r['league_name']}] HT={r['score_ht']}")
        print(f"\n共 {len(rows)} 场")
    elif args.start and args.end:
        backfill(args.start, args.end, allowed, args.output)
        if args.merge:
            merge_to_intl(args.output)
    else:
        parser.print_help()
