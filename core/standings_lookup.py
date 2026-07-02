#!/usr/bin/env python3
"""
standings_lookup.py — 积分榜查询模块
====================================
供 _try_club_predict 调用，缓存 standings_cache.json，
返回球队排名信息。

用法:
    from standings_lookup import lookup_team, load_standings_cache
    rows = load_standings_cache()
    info = lookup_team("Liverpool FC", rows)
    # -> {"comp_id":"comp_3039","position":5,"points":60,"goal_difference":10,...}
"""
import json, os, re

CACHE_PATH = "/root/data/standings_cache.json"

_cache = None  # lazy load


def load_standings_cache():
    global _cache
    if _cache is not None:
        return _cache
    if not os.path.exists(CACHE_PATH):
        _cache = {}
        return _cache
    with open(CACHE_PATH) as f:
        raw = json.load(f)
    # 构建 {comp_id: [(team_name_norm, row_dict)]} 索引
    idx = {}
    for comp_id, rows in raw.get("standings", {}).items():
        entries = []
        for r in rows:
            name = r["team"]
            entries.append((_norm(name), r))
            # FC 变体
            entries.append((_norm(f"{name} FC"), r))
        idx[comp_id] = entries
    # 全量扁平索引 (跨联赛)
    flat = {}
    for comp_id, entries in idx.items():
        for nkey, r in entries:
            flat[nkey] = (comp_id, r)
    _cache = {
        "raw": raw,
        "by_comp": idx,
        "flat": flat,
    }
    return _cache


def _norm(s):
    """归一化队名用于模糊匹配"""
    s = s.lower().strip()
    s = s.replace("'", "").replace("&", "and")
    s = s.replace("-", " ").replace("/", " ")
    s = re.sub(r"\b(fc|afc|sc|ec|ac|cf|cc|usa)$", "", s)
    s = re.sub(r"\bfc$", "", s)
    s = re.sub(r"\s+", "", s)
    return s


# 常见短名→全名映射 (football-data.org shortName → standings)
_SHORT_NAME_MAP = {
    "man city": "Manchester City",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "psg": "Paris Saint-Germain",
    "spurs": "Tottenham Hotspur",
    "newcastle": "Newcastle United",
    "west ham": "West Ham United",
    "leeds": "Leeds United",
    "wolves": "Wolverhampton",
    "brighton": "Brighton & Hove Albion",
    "barca": "Barcelona",
    "real": "Real Madrid",
    "atletico": "Atlético Madrid",
    "atletico madrid": "Atlético Madrid",
    "atlético": "Atlético Madrid",
    "atlético madrid": "Atlético Madrid",
    "leverkusen": "Bayer 04 Leverkusen",
    "bayer leverkusen": "Bayer 04 Leverkusen",
    "gladbach": "Borussia M'gladbach",
    "mgladbach": "Borussia M'gladbach",
    "mönchengladbach": "Borussia M'gladbach",
    "moenchengladbach": "Borussia M'gladbach",
    "dortmund": "Borussia Dortmund",
    "bayern": "FC Bayern München",
    "lens": "RC Lens",
    "sporting": "Sporting CP",
    "sporting braga": "Sporting Braga",
    "braga": "Sporting Braga",
    "benfica": "Benfica",
    "porto": "FC Porto",
    "celtic": "Celtic FC",
    "rangers": "Rangers FC",
    "ajax": "Ajax",
    "feyenoord": "Feyenoord",
    "psv": "PSV Eindhoven",
    "twente": "FC Twente",
    "utrecht": "FC Utrecht",
    "groningen": "FC Groningen",
}


def lookup_team(name, cache=None):
    """查找球队排名信息
    
    Args:
        name: 队名 (支持 "Arsenal FC", "Liverpool", "FC Bayern München" 等格式)
        cache: load_standings_cache() 的返回值 (可选)
    
    Returns:
        {"comp_id": "comp_3039", "position": 1, "points": 85, ...} or None
    """
    if cache is None:
        cache = load_standings_cache()
    if not cache:
        return None

    # 短名→全名映射 (处理 "Man City" → "Manchester City" 等)
    name_lower = name.lower().strip()
    if name_lower in _SHORT_NAME_MAP:
        name = _SHORT_NAME_MAP[name_lower]

    n = _norm(name)
    flat = cache.get("flat", {})
    
    # 精确匹配
    if n in flat:
        cid, r = flat[n]
        return {**r, "comp_id": cid}
    
    # 子串匹配
    for nkey, (cid, r) in flat.items():
        if n in nkey or nkey in n:
            return {**r, "comp_id": cid}
    
    return None


def lookup_both(home, away, cache=None):
    """查找主客队排名，返回 (home_info, away_info, features)
    
    features: [rank_diff/max_n, point_diff/max_p, gd_diff/50]
    任一球队未找到则 features 全 0
    """
    if cache is None:
        cache = load_standings_cache()
    
    hi = lookup_team(home, cache)
    ai = lookup_team(away, cache)
    
    if hi and ai and hi.get("comp_id") == ai.get("comp_id"):
        max_pos = float(hi.get("matches_played", 38))
        max_pts = float(hi.get("points", 100))
        rank_diff = hi["position"] - ai["position"]
        pt_diff = hi["points"] - ai["points"]
        gd_diff = hi["goal_difference"] - ai["goal_difference"]
        features = [
            rank_diff / max_pos,
            pt_diff / max_pts,
            gd_diff / 50.0,
        ]
    else:
        features = [0.0, 0.0, 0.0]
    
    return hi, ai, features


def show_matchup(home, away):
    """打印两队排名对比 (调试用)"""
    cache = load_standings_cache()
    hi, ai, feats = lookup_both(home, away, cache)
    
    print(f"{'='*50}")
    print(f"  {home:25s} vs {away}")
    print(f"{'='*50}")
    if hi:
        print(f"  {hi['team']:25s}  #{hi['position']:>2}  {hi['points']}pts  GD={hi['goal_difference']:+d}")
    else:
        print(f"  {home:25s}  ❌ NOT IN STANDINGS")
    if ai:
        print(f"  {ai['team']:25s}  #{ai['position']:>2}  {ai['points']}pts  GD={ai['goal_difference']:+d}")
    else:
        print(f"  {away:25s}  ❌ NOT IN STANDINGS")
    print(f"  归一化特征: rank_diff/38={feats[0]:.4f}  pt_diff/85={feats[1]:.4f}  gd_diff/50={feats[2]:.4f}")
    return hi is not None and ai is not None


if __name__ == "__main__":
    # 测试常见的俱乐部队名
    test_pairs = [
        ("Arsenal FC", "Manchester City FC"),
        ("Liverpool FC", "Chelsea FC"),
        ("FC Bayern München", "Borussia Dortmund"),
        ("Barcelona", "Real Madrid"),
        ("PSV Eindhoven", "Feyenoord"),
        ("FC Porto", "Benfica"),
        ("Arsenal", "Liverpool"),
    ]
    ok = 0
    total = 0
    for h, a in test_pairs:
        both = show_matchup(h, a)
        ok += 1 if both else 0
        total += 1
        print()
    print(f"匹配率: {ok}/{total}")
