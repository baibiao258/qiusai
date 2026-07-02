"""League model trainer — Poisson attack/defence + Elo ratings.

Replaces train() previously in daily_jczq.py.
Zero I/O beyond the match list passed in.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime

from pipeline.probability import elo_expected


def train(all_matches: list[dict]) -> tuple[dict, float, dict]:
    """Fit time-decayed Poisson strength table and Elo ratings.

    Parameters
    ----------
    all_matches : list[dict]
        Each dict must have keys: date, home, away, h_score, a_score.

    Returns
    -------
    ts : dict
        {team: {attack, defense, m}}
    ga : float
        Global average goals (weighted).
    elo : dict
        {team: rating (float)}
    """
    cutoff = date.today().isoformat()
    stats: dict = defaultdict(lambda: {'wg': 0, 'wc': 0, 'ws': 0, 'm': 0})

    for m in all_matches:
        if m['date'] >= cutoff:
            continue
        days = (
            datetime.strptime(cutoff, '%Y-%m-%d')
            - datetime.strptime(m['date'], '%Y-%m-%d')
        ).days
        w = 0.5 ** (max(days, 0) / 180)
        for team, gf, ga in [
            (m['home'], m['h_score'], m['a_score']),
            (m['away'], m['a_score'], m['h_score']),
        ]:
            s = stats[team]
            s['wg'] += gf * w
            s['wc'] += ga * w
            s['ws'] += w
            s['m'] += 1

    total_ws = sum(s['ws'] for s in stats.values())
    ga_global = sum(s['wg'] for s in stats.values()) / max(total_ws, 1)

    ts: dict = {}
    for team, s in stats.items():
        avg_gf = s['wg'] / max(s['ws'], 0.001)
        avg_ga = s['wc'] / max(s['ws'], 0.001)
        ts[team] = {
            'attack':  avg_gf / max(ga_global, 0.01),
            'defense': avg_ga / max(ga_global, 0.01),
            'm':       s['m'],
        }

    elo: dict = defaultdict(lambda: 1500.0)
    for m in all_matches:
        if m['date'] >= cutoff:
            continue
        h, a = m['home'], m['away']
        e_h = elo_expected(elo[h], elo[a])
        sh, sa = (
            (1.0, 0.0) if m['h_score'] > m['a_score']
            else (0.5, 0.5) if m['h_score'] == m['a_score']
            else (0.0, 1.0)
        )
        elo[h] += 32 * (sh - e_h)
        elo[a] += 32 * (sa - (1 - e_h))

    return ts, ga_global, dict(elo)