#!/usr/bin/env python3
"""
build_training_from_500.py
从 500.com 构建含赔率的训练数据

步骤:
  1. 遍历日期范围，从 trade.500.com 拉历史赔率
  2. 用 team_name_normalizer 标准化队名
  3. 匹配 500.com wanchang 赛果 (已回填的 500_history_backfill.csv)
  4. 输出 training_data_with_odds.json

用法:
  python3 scripts/build_training_from_500.py --start 2026-01-01 --end 2026-06-13
  python3 scripts/build_training_from_500.py --start 2026-06-01 --end 2026-06-13 --quick
"""
import csv, json, os, re, sys, subprocess, time
from datetime import date, timedelta

DATA_DIR = "/root/data"
WANCHANG_CSV = f"{DATA_DIR}/500_history_backfill.csv"
OUTPUT_JSON  = f"{DATA_DIR}/training_data_with_odds.json"

# trade.500.com 玩法ID
PLAYIDS = {
    '269': 'nspf',   # 标准胜平负
    '270': 'spf',    # 让球胜平负
    # '271': 'bf',   # 比分 (可选)
    # '272': 'jqs',  # 总进球 (可选)
}

CURL_HEADERS = [
    '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    '-H', 'Referer: https://trade.500.com/jczq/',
    '-H', 'Accept-Language: zh-CN,zh;q=0.9',
]

# ── 加载队名映射 ──
TEAM_NAME_MAP = {}
MAPPING_PATH = os.path.join(DATA_DIR, "team_name_mapping.json")
if os.path.exists(MAPPING_PATH):
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        TEAM_NAME_MAP = json.load(f)

def normalize_team(name: str) -> str:
    """统一标准化: 去空格, 转小写"""
    return name.strip().replace('\u3000', '').replace('\xa0', '').lower()


# ── 加载赛果数据 ──
def load_results(csv_path: str) -> dict:
    """
    从 500_history_backfill.csv 加载赛果
    返回 {(date, home_norm, away_norm): {score_full, score_ht, ...}}
    """
    results = {}
    if not os.path.exists(csv_path):
        print(f"⚠️ 未找到 {csv_path}，先回填 500.com wanchang 数据")
        return results
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["date"], normalize_team(row["home"]), normalize_team(row["away"]))
            if key not in results:
                results[key] = row
    print(f"📥 加载 {len(results)} 条赛果记录 (来自 {csv_path})")
    return results


# ── 从 trade.500.com 抓赔率 ──
_TR_ATTR_RE = re.compile(
    r'<tr[^>]*data-fixtureid="(\d+)"[^>]*>'
)
_SPF_ODDS_RE = re.compile(
    r'data-type="(nspf|spf)"\s*data-value="(\d)"\s*data-sp="([\d.]+)"'
)
_ATTR_RE = re.compile(r'([\w-]+)="([^"]*)"')

def fetch_odds_for_date(date_str: str) -> dict:
    """
    抓取单天 trade.500.com 赔率
    返回 dict[fixtureid_str] -> { fixtureid, match_num, league, home, away, 
                                    handicap, nspf, spf }
    """
    all_matches = {}
    for pid, pname in PLAYIDS.items():
        url = f"https://trade.500.com/jczq/?playid={pid}&g=2&date={date_str}&_t=1234"
        try:
            r = subprocess.run(
                ['curl', '-s', url, '--max-time', '20'] + CURL_HEADERS,
                capture_output=True, timeout=25
            )
            html = r.stdout.decode("gbk", errors="replace")
        except Exception as e:
            print(f"    [{date_str}] playid={pid} 请求失败: {e}")
            continue

        # 按 <tr> 解析
        for m in _TR_ATTR_RE.finditer(html):
            tr_html = m.group(0)
            # 提取属性
            attrs = dict(_ATTR_RE.findall(tr_html))
            fid = attrs.get("data-fixtureid", "")
            if not fid:
                continue

            if fid not in all_matches:
                # 使用 data-matchdate 而非请求日期
                match_date = attrs.get("data-matchdate", date_str)
                all_matches[fid] = {
                    "fixtureid": fid,
                    "match_num": attrs.get("data-matchnum", ""),
                    "match_date": match_date,
                    "match_time": attrs.get("data-matchtime", ""),
                    "league": attrs.get("data-simpleleague", ""),
                    "home": attrs.get("data-homesxname", ""),
                    "away": attrs.get("data-awaysxname", ""),
                    "handicap": attrs.get("data-rangqiu", "0"),
                    "nspf": {},
                    "spf": {},
                }

            # 从 <tr ...> 和 </tr> 之间提取赔率 (跨行)
            after_open = html.find('>', m.start()) + 1
            if after_open > m.start():
                close_tr = html.find('</tr>', after_open)
                between = html[after_open:close_tr] if close_tr > 0 else html[after_open:after_open + 3000]
            else:
                between = html[m.end():m.end() + 2000]

            for ptype, val, sp in _SPF_ODDS_RE.findall(between):
                try:
                    all_matches[fid][ptype][val] = float(sp)
                except ValueError:
                    pass

    return all_matches


def match_and_build(date_str: str, results: dict, odds: dict) -> list:
    """
    将赔率数据与赛果配对
    返回 [training_sample, ...]
    """
    samples = []
    matched = set()

    for fid, m in odds.items():
        home_norm = normalize_team(m["home"])
        away_norm = normalize_team(m["away"])
        match_date = m.get("match_date", date_str)  # 用比赛实际日期
        key = (match_date, home_norm, away_norm)

        # 尝试正向匹配
        row = results.get(key)
        # 尝试反向匹配 (主客交换)
        if not row:
            rev_key = (match_date, away_norm, home_norm)
            row = results.get(rev_key)

        # 宽容 ±1 天
        if not row:
            for offset in [1, -1]:
                try:
                    adj = date.fromisoformat(match_date) + timedelta(days=offset)
                    adj_key = (adj.isoformat(), home_norm, away_norm)
                    row = results.get(adj_key)
                    if row:
                        break
                    adj_rev = (adj.isoformat(), away_norm, home_norm)
                    row = results.get(adj_rev)
                    if row:
                        break
                except ValueError:
                    continue

        if not row:
            continue

        if key in matched or (date_str, away_norm, home_norm) in matched:
            continue
        matched.add(key)

        # 解析赛果
        sf = row.get("score_full", "0-0")
        try:
            ft_h, ft_a = int(sf.split("-")[0].strip()), int(sf.split("-")[1].strip())
        except (ValueError, IndexError):
            continue

        if ft_h > ft_a:
            spf_result = 3  # 主胜
        elif ft_h == ft_a:
            spf_result = 1  # 平
        else:
            spf_result = 0  # 客胜

        nspf = m.get("nspf", {})
        spf_odds = m.get("spf", {})

        # 让球胜平负结果 (需要 handicap)
        try:
            handicap = float(m.get("handicap", "0"))
        except ValueError:
            handicap = 0.0
        adj_h = ft_h + handicap
        adj_a = ft_a
        if adj_h > adj_a:
            rq_result = 3
        elif adj_h == adj_a:
            rq_result = 1
        else:
            rq_result = 0

        sample = {
            "date": date_str,
            "home_en": TEAM_NAME_MAP.get(m["home"], m["home"]),
            "away_en": TEAM_NAME_MAP.get(m["away"], m["away"]),
            "tournament": m["league"],
            "spf_result": spf_result,
            "nspf_3": nspf.get("3", 0),  # 主胜赔率
            "nspf_1": nspf.get("1", 0),  # 平赔率
            "nspf_0": nspf.get("0", 0),  # 客胜赔率
            "spf_3": spf_odds.get("3", 0),  # 让胜赔率
            "spf_1": spf_odds.get("1", 0),  # 让平赔率
            "spf_0": spf_odds.get("0", 0),  # 让负赔率
            "handicap": handicap,
            "ft_h": ft_h,
            "ft_a": ft_a,
            "score_ht": row.get("score_ht", ""),
            # 从赔率计算 implied probability (去水)
            "market_implied_prob": _implied_prob(nspf),
        }

        # 让球结果
        sample["rq_result"] = rq_result

        samples.append(sample)

    return samples


def _implied_prob(nspf: dict) -> float:
    """从 SPF 赔率计算市场隐含主胜概率 (简单去水)"""
    odds = [nspf.get("3", 0), nspf.get("1", 0), nspf.get("0", 0)]
    if any(o <= 0 for o in odds):
        return 0.0
    imp = [1.0 / o for o in odds]
    s = sum(imp)
    if s > 0:
        return imp[0] / s
    return 0.0


def rebuild(start: str, end: str, quick: bool = False):
    """
    主流程: 遍历日期, 抓赔率, 配对赛果, 输出训练数据
    quick=True: 只跑有赛果的日期（先读 CSV 中的日期集合）
    """
    results = load_results(WANCHANG_CSV)
    if not results:
        return

    # 收集有赛果的日期
    if quick:
        result_dates = sorted(set(k[0] for k in results.keys()))
        print(f"⚡ Quick 模式: {len(result_dates)} 天有赛果数据")
        date_range = [d for d in result_dates if start <= d <= end]
    else:
        d_start = date.fromisoformat(start)
        d_end = date.fromisoformat(end)
        date_range = []
        d = d_start
        while d <= d_end:
            date_range.append(d.isoformat())
            d += timedelta(days=1)

    all_samples = []
    for i, ds in enumerate(date_range):
        # 先检查这一天是否有赛果
        day_results = {k: v for k, v in results.items() if k[0] == ds}
        if not day_results:
            continue

        odds = fetch_odds_for_date(ds)
        if not odds:
            continue

        samples = match_and_build(ds, results, odds)
        if samples:
            all_samples.extend(samples)
            print(f"  [{i+1}/{len(date_range)}] {ds}: {len(odds)} 场赔率 → {len(samples)} 配对")

        time.sleep(0.3)  # 礼貌延迟

    print(f"\n总计: {len(all_samples)} 条有赔率的训练样本")

    if all_samples:
        # 按日期排序
        all_samples.sort(key=lambda x: x["date"])
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        print(f"✅ 写入 {OUTPUT_JSON}: {len(all_samples)} 条")

        # 统计
        years = set(s["date"][:4] for s in all_samples)
        print(f"   年份分布: {sorted(years)}")
        print(f"   有让球数据: {sum(1 for s in all_samples if s['spf_3'] > 0)}/{len(all_samples)}")
        print(f"   有 SPF 赔率: {sum(1 for s in all_samples if s['nspf_3'] > 0)}/{len(all_samples)}")
    else:
        print("❌ 没有生成任何样本")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-13")
    ap.add_argument("--quick", action="store_true", help="只跑有赛果的日期")
    args = ap.parse_args()
    rebuild(args.start, args.end, args.quick)
