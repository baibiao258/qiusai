#!/usr/bin/env python3
"""
pull_standings_cache.py — 拉取 7 大联赛积分榜，缓存本地供 _try_club_predict 使用
====================================================================
数据流:
  TheStatsAPI → /competitions/{id}/seasons/{sid}/standings → standings_cache.json

用法:
  python3 pull_standings_cache.py              # 全量拉取
  python3 pull_standings_cache.py --dry-run    # 预览不保存
  python3 pull_standings_cache.py --stats      # 查看缓存统计
"""
import json, os, sys, time
from datetime import date

KEY = os.environ.get("THE_STATS_KEY", "fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww")
BASE = "https://api.thestatsapi.com/api"
HDR = {"Authorization": f"Bearer {KEY}"}

CACHE_PATH = "/root/data/standings_cache.json"

# 俱乐部联赛: 当前 season_id 由昨日验证
CLUB_LEAGUES = [
    ("comp_3039", "Premier League",       "sn_6125938", 20),
    ("comp_4643", "Bundesliga",           "sn_5789634", 18),
    ("comp_8814", "LaLiga",               "sn_7246390", 20),
    ("comp_0256", "Ligue 1",              "sn_6120181", 18),
    ("comp_8385", "Liga Portugal Betclic", "sn_6120591", 18),
    ("comp_3809", "Eredivisie",           "sn_9674249", 18),
    ("comp_8321", "Championship",         "sn_3064530", 24),
]


def pull_standings(comp_id, comp_name, season_id, dry_run=False):
    """拉取一个联赛的 standings，返回 [{team, position, points, gd, gf, ga, form}]"""
    import requests
    url = f"{BASE}/football/competitions/{comp_id}/seasons/{season_id}/standings"
    r = requests.get(url, headers=HDR, timeout=20)
    if r.status_code != 200:
        print(f"  ❌ {comp_name} — HTTP {r.status_code}")
        return []
    data = r.json().get("data", [])
    if not data:
        print(f"  ⚠️ {comp_name} — 空数据")
        return []

    rows = []
    for t in data:
        team = t.get("team", {})
        rows.append({
            "team": team.get("name", ""),
            "team_id": team.get("id", ""),
            "position": t.get("position", 0),
            "points": t.get("points", 0),
            "goal_difference": t.get("goal_difference", 0),
            "goals_for": t.get("goals_for", 0),
            "goals_against": t.get("goals_against", 0),
            "matches_played": t.get("matches_played", 0),
            "wins": t.get("wins", 0),
            "draws": t.get("draws", 0),
            "losses": t.get("losses", 0),
            "form": t.get("form", ""),
        })

    if dry_run:
        print(f"\n  📋 {comp_name} ({len(rows)}队) — Top/Bottom 3:")
        for r in rows[:3]:
            print(f"    #{r['position']:>2}  {r['team']:<25}  "
                  f"P{r['matches_played']:>2}  GD={r['goal_difference']:+d}  {r['points']}pts")
        print(f"    ...")
        for r in rows[-3:]:
            print(f"    #{r['position']:>2}  {r['team']:<25}  "
                  f"P{r['matches_played']:>2}  GD={r['goal_difference']:+d}  {r['points']}pts")
    else:
        print(f"  ✅ {comp_name}: {len(rows)} 队")

    return rows


def build_cache(dry_run=False):
    """全量拉取 → 按 comp_id 建索引"""
    cache = {
        "meta": {
            "generated_at": date.today().isoformat(),
            "total_leagues": len(CLUB_LEAGUES),
            "note": "Club league standings for _try_club_predict",
        },
        "standings": {},
    }

    total = 0
    for comp_id, comp_name, season_id, n_teams in CLUB_LEAGUES:
        rows = pull_standings(comp_id, comp_name, season_id, dry_run=dry_run)
        if rows:
            cache["standings"][comp_id] = rows
            total += len(rows)
        if not dry_run:
            time.sleep(0.3)  # 限速保护

    if not dry_run:
        cache["meta"]["total_teams"] = total
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        sz = os.path.getsize(CACHE_PATH) / 1024
        print(f"\n  💾 已保存: {CACHE_PATH} ({sz:.1f} KB, {total} 队, {len(cache['standings'])} 联赛)")
    else:
        print(f"\n  🔍 Dry-run: {total} 队将保存")

    return cache


def show_stats():
    """显示缓存统计"""
    if not os.path.exists(CACHE_PATH):
        print("❌ 缓存不存在，先运行 pull_standings_cache.py")
        return
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    meta = cache.get("meta", {})
    print(f"📊 Standings 缓存统计")
    print(f"={'='*50}")
    print(f"  生成时间: {meta.get('generated_at', 'N/A')}")
    print(f"  联赛数:   {len(cache.get('standings', {}))}")
    print(f"  球队总数: {meta.get('total_teams', 0)}")
    for comp_id, rows in cache.get("standings", {}).items():
        comp_name = next((n for c, n, *_ in CLUB_LEAGUES if c == comp_id), comp_id)
        print(f"\n  {comp_name} ({comp_id}):")
        for r in rows:
            print(f"    #{r['position']:>2}  {r['team']:<25}  "
                  f"P{r['matches_played']:>2}  GD={r['goal_difference']:+d}  {r['points']}pts  "
                  f"GF={r['goals_for']} GA={r['goals_against']}")


if __name__ == "__main__":
    if "--stats" in sys.argv:
        show_stats()
    elif "--dry-run" in sys.argv:
        build_cache(dry_run=True)
    else:
        build_cache()
