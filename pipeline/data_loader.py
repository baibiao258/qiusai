"""External data I/O layer — football-data.org v4 + 365scores.

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
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from typing import Optional

from config.settings import API_KEY, DATA_DIR, JCZQ_LEAGUES

# ── Constants ─────────────────────────────────────────────────────────────────

_HDR = {'X-Auth-Token': API_KEY, 'Accept': 'application/json'}
_SCORES365_DIR = os.path.join(DATA_DIR, '365scores')
_API_BASE = 'https://api.football-data.org/v4'
_API_TIMEOUT = 15


# ── Public API ────────────────────────────────────────────────────────────────

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
                data = api_get(
                    f'/competitions/{code}/matches'
                    f'?dateFrom={ss}&dateTo={ee}&status=FINISHED'
                )
                for m in data.get('matches', []):
                    hs = m['score']['fullTime'].get('home')
                    as_ = m['score']['fullTime'].get('away')
                    if hs is None or as_ is None:
                        continue
                    all_matches.append({
                        'date':    m['utcDate'][:10],
                        'home':    m['homeTeam']['shortName'],
                        'away':    m['awayTeam']['shortName'],
                        'h_score': int(hs),
                        'a_score': int(as_),
                    })
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt == 0:
                    time.sleep(15)
                else:
                    break
            except Exception:
                break
        if i < len(segments) - 1:
            time.sleep(1.5)

    return all_matches


def get_today_matches() -> list[dict]:
    """Fetch today's JCZQ-covered fixtures from football-data.org.

    Returns an empty list when the API is unreachable or returns no
    scheduled/live matches for today.
    """
    today = date.today().isoformat()
    league_map = {code: name for code, name in JCZQ_LEAGUES}
    matches: list[dict] = []

    for code, _ in JCZQ_LEAGUES:
        try:
            data = api_get(
                f'/competitions/{code}/matches'
                f'?dateFrom={today}&dateTo={today}'
            )
            for m in data.get('matches', []):
                if m.get('status') not in ('SCHEDULED', 'TIMED', 'IN_PLAY', 'LIVE'):
                    continue
                m['competition'] = {'code': code, 'name': league_map.get(code, code)}
                matches.append(m)
        except Exception:
            continue

    return matches


def load_365scores_today() -> list[dict]:
    """Load today's 365scores enrichment data from local CSV cache.

    Falls back to an empty list when the cache file is absent or
    unreadable.  The cache is populated by a separate scrape job.
    """
    today = date.today().isoformat()
    cache_path = os.path.join(_SCORES365_DIR, f'{today}.csv')
    if not os.path.exists(cache_path):
        return []
    try:
        with open(cache_path, encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def build_365_map(games: list[dict]) -> dict:
    """Build a (home_norm, away_norm) → row lookup from 365scores rows.

    Team names are normalised via team_name_normalizer when available;
    falls back to lowercased raw names on import failure.

    Parameters
    ----------
    games : list[dict]
        Rows returned by :func:`load_365scores_today`.

    Returns
    -------
    dict
        Keys are 2-tuples of normalised team name strings.
    """
    try:
        from team_name_normalizer import normalize_match_pair
    except ImportError:
        def normalize_match_pair(h, a):  # type: ignore[misc]
            return h.lower(), a.lower()

    result: dict = {}
    for row in games:
        home_raw = row.get('home_team', row.get('home', ''))
        away_raw = row.get('away_team', row.get('away', ''))
        if not home_raw or not away_raw:
            continue
        try:
            h_norm, a_norm = normalize_match_pair(home_raw, away_raw)
        except Exception:
            h_norm, a_norm = home_raw.lower(), away_raw.lower()

        # Parse votes sub-dict when stored as JSON string
        if 'votes' in row and isinstance(row['votes'], str):
            try:
                row = dict(row)
                row['votes'] = json.loads(row['votes'])
            except Exception:
                pass

        result[(h_norm, a_norm)] = row
    return result
