"""External data I/O layer – football-data.org, 365scores.

Replaces 5 functions previously embedded in daily_jczq.py:
    api_get()               → shared HTTP helper
    fetch_league_history()  → football-data.org historical results
    get_today_matches()     → today's JCZQ fixtures
    load_365scores_today()  → local CSV cache or live fetch
    build_365_map()         → (home, away) → 365scores row lookup

Public API
----------
api_get(path) -> dict
fetch_league_history(code, months_back=10) -> list[dict]
get_today_matches() -> list[dict]
load_365scores_today() -> list[dict]
build_365_map(games) -> dict
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from typing import Optional

from config.settings import API_KEY, DATA_DIR, JCZQ_LEAGUES

# ── 常量 ─────────────────────────────────────────────────────────────────────

_HDR = {'X-Auth-Token': API_KEY, 'Accept': 'application/json'}
_SCORES365_DIR = os.path.join(DATA_DIR, '365scores')
_API_BASE = 'https://api.football-data.org/v4'
_API_TIMEOUT = 15
_SEGMENT_DAYS = 150  # ~5 months per segment; dynamic: months_back*15


# ── 公开 API ──────────────────────────────────────────────────────────────────

def api_get(path: str) -> dict:
    """Thin wrapper around football-data.org v4 REST API.

    Parameters
    ----------
    path : str
        API path including leading slash, e.g. '/competitions/PL/matches'.

    Raises
    ------
    urllib.error.HTTPError
        Propagated for callers that handle 429 rate-limit retry logic.
    """
    url = f'{_API_BASE}{path}'
    req = urllib.request.Request(url, headers=_HDR)
    with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
        return json.loads(resp.read().decode('utf-8'))


def fetch_league_history(code: str, months_back: int = 10) -> list[dict]:
    """Pull finished matches for one competition over a rolling window.

    Segments the date range to stay within API page limits, with a
    1.5 s polite delay between segments and one 429 retry (15 s back-off).

    Parameters
    ----------
    code : str
        football-data.org competition code, e.g. 'PL'.
    months_back : int
        How many months of history to fetch (default 10).

    Returns
    -------
    list[dict]
        Each dict has keys: date, home, away, h_score, a_score.
    """
    end = date.today()
    start = end - timedelta(days=months_back * 30)
    seg_size = months_back * 15

    segments: list[tuple[date, date]] = []
    s = start
    while s < end:
        e = min(s + timedelta(days=seg_size), end)
        segments.append((s, e))
        s = e + timedelta(days=1)

    all_matches: list[dict] = []
    for i, (seg_start, seg_end) in enumerate(segments):
        ss, ee = seg_start.isoformat(), seg_end.isoformat()
        for attempt in range(2):
            try:
                data = api_get(f'/competitions/{code}/matches?dateFrom={ss}&dateTo={ee}')
                for m in data.get('matches', []):
                    if m['status'] != 'FINISHED':
                        continue
                    sc = m['score']['fullTime']
                    if sc['home'] is None:
                        continue
                    all_matches.append({
                        'date':    m['utcDate'][:10],
                        'home':    m['homeTeam']['shortName'],
                        'away':    m['awayTeam']['shortName'],
                        'h_score': sc['home'],
                        'a_score': sc['away'],
                    })
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt == 0:
                    time.sleep(15)
                    continue
                break
            except Exception:
                break
        if i < len(segments) - 1:
            time.sleep(1.5)

    return all_matches


def get_today_matches() -> list[dict]:
    """Fetch today's JCZQ fixtures from all covered leagues.

    Deduplicates within each competition to avoid double-counting
    fixtures that appear in multiple API pages.

    Returns
    -------
    list[dict]
        Raw football-data.org match dicts with status SCHEDULED or TIMED.
    """
    today = date.today().isoformat()
    all_matches: list[dict] = []
    seen: set[tuple] = set()

    for code, _league_name in JCZQ_LEAGUES:
        try:
            data = api_get(f'/competitions/{code}/matches?dateFrom={today}&dateTo={today}')
            for m in data.get('matches', []):
                if m['status'] not in ('SCHEDULED', 'TIMED'):
                    continue
                key = (code, m['homeTeam']['shortName'], m['awayTeam']['shortName'])
                if key not in seen:
                    seen.add(key)
                    all_matches.append(m)
        except Exception:
            pass

    return all_matches


def load_365scores_today() -> list[dict]:
    """Load today's 365scores enrichment data.

    Checks a pre-fetched CSV cache first; falls back to live fetch via
    ``fetch_365scores`` module if cache is absent or empty.

    Returns
    -------
    list[dict]
        One dict per fixture with keys: home, away, competition, time,
        votes (h/d/a/total), pop_rank_home/away, fifa_rank_home/away,
        trend_home, trend_away, trend_win_rate_home/away.
    """
    date_str = date.today().isoformat()
    csv_path = os.path.join(_SCORES365_DIR, f'{date_str}.csv')
    games: list[dict] = []

    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    games.append(_parse_365_row(row))
        except Exception:
            games = []

    if not games:
        try:
            _ensure_365_on_path()
            from fetch_365scores import fetch_365scores_data, extract_games
            raw = fetch_365scores_data()
            games = extract_games(raw)
        except Exception:
            games = []

    return games


def build_365_map(games: list[dict]) -> dict:
    """Build a bi-directional (home, away) → game lookup from 365scores data.

    Team names are normalised via ``team_name_normalizer`` before keying,
    so lookups succeed regardless of variant spellings.

    Returns
    -------
    dict[(str, str), dict]
        Maps both (home, away) and (away, home) to the same game dict.
    """
    from team_name_normalizer import normalize_match_pair
    mapping: dict = {}
    for g in games:
        try:
            h, a = normalize_match_pair(g.get('home', ''), g.get('away', ''))
            if h and a:
                mapping[(h, a)] = g
                mapping[(a, h)] = g
        except Exception:
            continue
    return mapping


# ── 内部辅助 ──────────────────────────────────────────────────────────────────

def _ensure_365_on_path() -> None:
    """Make sure the /root directory is importable (fallback path for fetch_365scores)."""
    if '/root' not in sys.path:
        sys.path.insert(0, '/root')


def _safe_float(value, default=None):
    if value in ('', None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    v = _safe_float(value)
    return default if v is None else int(v)


def _parse_365_row(row: dict) -> dict:
    """Convert a raw CSV DictReader row into a normalised 365scores game dict."""
    return {
        'home':        row.get('home', ''),
        'away':        row.get('away', ''),
        'competition': row.get('competition', ''),
        'time':        row.get('time', ''),
        'votes': {
            'home':  _safe_float(row.get('vote_home')),
            'draw':  _safe_float(row.get('vote_draw')),
            'away':  _safe_float(row.get('vote_away')),
            'total': _safe_int(row.get('vote_count')),
        },
        'pop_rank_home':       _safe_int(row.get('pop_rank_home')),
        'pop_rank_away':       _safe_int(row.get('pop_rank_away')),
        'fifa_rank_home':      _safe_int(row.get('fifa_rank_home')),
        'fifa_rank_away':      _safe_int(row.get('fifa_rank_away')),
        'trend_home': [
            _safe_int(row.get('trend_home_w'), 0),
            _safe_int(row.get('trend_home_d'), 0),
            _safe_int(row.get('trend_home_l'), 0),
        ],
        'trend_away': [
            _safe_int(row.get('trend_away_w'), 0),
            _safe_int(row.get('trend_away_d'), 0),
            _safe_int(row.get('trend_away_l'), 0),
        ],
        'trend_win_rate_home': _safe_float(row.get('trend_win_rate_home')),
        'trend_win_rate_away': _safe_float(row.get('trend_win_rate_away')),
    }
