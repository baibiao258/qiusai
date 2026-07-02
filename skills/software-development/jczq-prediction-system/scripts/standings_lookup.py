#!/usr/bin/env python3
"""
standings_lookup.py — 积分榜查询模块
====================================
供 _try_club_predict 调用，缓存 standings_cache.json，
返回球队排名信息。

用法:
    from standings_lookup import lookup_team, load_standings_cache, lookup_both
    rows = load_standings_cache()
    info = lookup_team("Liverpool FC", rows)
    # -> {"comp_id":"comp_3039","position":5,"points":60,"goal_difference":10,...}
    hi, ai, feats = lookup_both("Arsenal", "Liverpool")
    # -> features = [rank_diff/38, pt_diff/85, gd_diff/50]
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
    idx = {}
    for comp_id, rows in raw.get("standings", {}).items():
        entries = []
        for r in rows:
            name = r["team"]
            entries.append((_norm(name), r))
            entries.append((_norm(f"{name} FC"), r))
        idx[comp_id] = entries
    flat = {}
    for comp_id, entries in idx.items():
        for nkey, r in entries:
            flat[nkey] = (comp_id, r)
    _cache = {"raw": raw, "by_comp": idx, "flat": flat}
    return _cache


def _norm(s):
    s = s.lower().strip()
    s = s.replace("'", "").replace("&", "and")
    s = s.replace("-", " ").replace("/", " ")
    s = re.sub(r"\b(fc|afc|sc|ec|ac|cf|cc|usa)$", "", s)
    s = re.sub(r"\bfc$", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def lookup_team(name, cache=None):
    if cache is None:
        cache = load_standings_cache()
    if not cache:
        return None
    n = _norm(name)
    flat = cache.get("flat", {})
    if n in flat:
        cid, r = flat[n]
        return {**r, "comp_id": cid}
    for nkey, (cid, r) in flat.items():
        if n in nkey or nkey in n:
            return {**r, "comp_id": cid}
    return None


def lookup_both(home, away, cache=None):
    """查找主客队排名，返回 (home_info, away_info, features)
    features: [rank_diff/max_n, point_diff/max_p, gd_diff/50] or zeros"""
    if cache is None:
        cache = load_standings_cache()
    hi = lookup_team(home, cache)
    ai = lookup_team(away, cache)
    if hi and ai and hi.get("comp_id") == ai.get("comp_id"):
        max_pos = float(hi.get("matches_played", 38))
        max_pts = float(hi.get("points", 100))
        features = [
            (hi["position"] - ai["position"]) / max_pos,
            (hi["points"] - ai["points"]) / max_pts,
            (hi["goal_difference"] - ai["goal_difference"]) / 50.0,
        ]
    else:
        features = [0.0, 0.0, 0.0]
    return hi, ai, features


def show_matchup(home, away):
    cache = load_standings_cache()
    hi, ai, feats = lookup_both(home, away, cache)
    print(f"--- {home:25s} vs {away} ---")
    if hi:
        print(f"  HOME: #{hi['position']:>2} {hi['team']:25s} {hi['points']}pts GD={hi['goal_difference']:+d}")
    else:
        print(f"  HOME: ❌ NOT FOUND")
    if ai:
        print(f"  AWAY: #{ai['position']:>2} {ai['team']:25s} {ai['points']}pts GD={ai['goal_difference']:+d}")
    else:
        print(f"  AWAY: ❌ NOT FOUND")
    print(f"  FEATURES: rd={feats[0]:.4f}  pd={feats[1]:.4f}  gd={feats[2]:.4f}")
    return hi is not None and ai is not None


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 2:
        show_matchup(args[0], args[1])
    else:
        for h, a in [("Arsenal FC", "Liverpool FC"), ("FC Bayern München", "Borussia Dortmund"),
                     ("Barcelona", "Real Madrid"), ("PSV Eindhoven", "Feyenoord")]:
            show_matchup(h, a)
            print()
