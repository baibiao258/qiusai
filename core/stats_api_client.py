"""TheStatsAPI client — team ID resolution, xG fetching, form data.

API base: https://api.thestatsapi.com/api/football
Requires THE_STATS_KEY env var (or THE_KEY fallback).

Team name resolution uses three layers:
  1. Exact match in locally cached team_id_cache.json (from API matches)
  2. Fuzzy match (SequenceMatcher ratio >= 0.85) against cached names
  3. team_name_mapping.json: Chinese → English → API lookup
"""

import json
import os
import time
from difflib import SequenceMatcher
from typing import Optional

import requests

# ── API config ──
THE_STATS_KEY = (
    os.environ.get("THE_KEY", "")
    or os.environ.get("THE_STATS_KEY", "")
)
if not THE_STATS_KEY:
    THE_STATS_KEY = "fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww"  # fallback from codebase

_HEADERS = {"Authorization": f"Bearer {THE_STATS_KEY}"}
_BASE = "https://api.thestatsapi.com/api/football"
_DATA_DIR = os.environ.get("DATA_DIR", "/root/data")

# ── Local caches ──
_team_cache: dict[str, str] = {}  # name → team_id (e.g. "Brazil" → "tm_86500")
_name_map: dict[str, str] = {}     # cn → en from team_name_mapping.json
_cache_loaded = False


def _load_caches():
    global _cache_loaded, _team_cache, _name_map
    if _cache_loaded:
        return

    # 1. Team ID cache (from API matches)
    cache_path = f"{_DATA_DIR}/team_id_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            _team_cache = json.load(f)
        print(f"  📡 team_id_cache: {len(_team_cache)} teams loaded")

    # 2. Team name mapping (Chinese ↔ English)
    mapping_path = f"{_DATA_DIR}/team_name_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, encoding="utf-8") as f:
            raw = json.load(f)
        # raw: {chinese_name: english_name} — store as cn→en
        for cn_name, en_name in raw.items():
            if cn_name not in _name_map:
                _name_map[cn_name] = en_name
        print(f"  📖 team_name_mapping: {len(_name_map)} cn→en entries")

    _cache_loaded = True


def resolve_team_id(team_name: str) -> Optional[str]:
    """Resolve a team name (English or Chinese) to a TheStatsAPI team ID.

    Resolution order:
      1. Exact match in team_id_cache
      2. Fuzzy match (ratio >= 0.85)
      3. If Chinese: look up English via team_name_mapping → API cache
      4. If Chinese: also check if the English name fuzzy-matches
    """
    _load_caches()
    name = team_name.strip()

    # Layer 1: exact match
    if name in _team_cache:
        return _team_cache[name]

    # Layer 2: Chinese → English via name mapping → exact
    if name in _name_map:
        en = _name_map[name]
        if en in _team_cache:
            return _team_cache[en]

    # Layer 3: fuzzy match against all cached names
    best_ratio = 0.0
    best_id = None
    for cached_name, tid in _team_cache.items():
        ratio = SequenceMatcher(None, name.lower(), cached_name.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = tid

    if best_ratio >= 0.85:
        return best_id

    # Layer 4: Chinese → English → fuzzy
    if name in _name_map:
        en = _name_map[name]
        for cached_name, tid in _team_cache.items():
            ratio = SequenceMatcher(None, en.lower(), cached_name.lower()).ratio()
            if ratio >= 0.85:
                return tid

    return None


def add_team_to_cache(team_name: str, team_id: str):
    """Manually add a team name→ID mapping to the local cache."""
    _load_caches()
    _team_cache[team_name.strip()] = team_id.strip()
    # Persist
    cache_path = f"{_DATA_DIR}/team_id_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(_team_cache, f, ensure_ascii=False, indent=2)


def batch_resolve(team_names: list[str]) -> dict[str, Optional[str]]:
    """Resolve multiple team names, returning {name: team_id or None}."""
    result = {}
    unresolved = []
    for name in team_names:
        tid = resolve_team_id(name)
        result[name] = tid
        if not tid:
            unresolved.append(name)
    return result, unresolved


def get_team_stats(team_id: str, season_id: Optional[str] = None) -> Optional[dict]:
    """Fetch /teams/{id}/stats for a team.

    If season_id is None, tries to find the current season for the team's
    most relevant competition.
    """
    if season_id:
        url = f"{_BASE}/teams/{team_id}/stats?season_id={season_id}"
    else:
        # Try common current competitions
        # World Cup 2026 season
        url = f"{_BASE}/teams/{team_id}/stats?season_id=sn_118868"

    try:
        r = requests.get(url, headers=_HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json().get("data", r.json())
            if isinstance(data, dict) and data.get("matches_played", 0) > 0:
                return data
        return None
    except Exception:
        return None


def fetch_recent_xg(team_id: str, n_matches: int = 10) -> Optional[dict]:
    """Fetch recent xG data for a team from the matches API.

    Returns dict with:
      - xg_recent_avg: average xG for the team in recent n_matches
      - xg_against_avg: average xGA for the team
      - xg_diff_avg: xG - xGA
      - n: number of matches used
      - form: recent form string like "WWDLW"
    """
    url = f"{_BASE}/matches?team_id={team_id}&status=finished&per_page={n_matches}&sort=-date"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        matches = r.json().get("data", [])
        if not matches:
            return None
    except Exception:
        return None

    total_xg = 0.0
    total_xga = 0.0
    form_chars = []
    match_count = 0

    for m in matches:
        stats = m.get("stats", {}) or {}
        team_stats_list = stats.get("team_stats", []) if isinstance(stats, dict) else []

        # Find the team's stats in the match
        for ts in team_stats_list:
            if ts.get("team_id") == team_id or ts.get("team", {}).get("id") == team_id:
                xg = ts.get("expected_goals")
                if xg is not None:
                    total_xg += float(xg)
                    total_xga += float(ts.get("expected_goals_against", 0))
                    match_count += 1

                    # Determine result for form string
                    gf = ts.get("goals_for", 0)
                    ga = ts.get("goals_against", 0)
                    if gf > ga:
                        form_chars.append("W")
                    elif gf == ga:
                        form_chars.append("D")
                    else:
                        form_chars.append("L")
                break

    if match_count == 0:
        return None

    return {
        "xg_recent_avg": round(total_xg / match_count, 4),
        "xg_against_avg": round(total_xga / match_count, 4),
        "xg_diff_avg": round((total_xg - total_xga) / match_count, 4),
        "n": match_count,
        "form": "".join(form_chars[-5:]),
    }


def batch_fetch_xg(team_ids: dict[str, str], n_matches: int = 10) -> dict:
    """Fetch xG for multiple teams.  team_ids = {team_name: team_id}

    Returns {team_name: xg_dict or None}
    Rate-limited internally (0.3s between requests).
    """
    result = {}
    for name, tid in team_ids.items():
        if not tid:
            result[name] = None
            continue
        result[name] = fetch_recent_xg(tid, n_matches=n_matches)
        time.sleep(0.3)
    return result
