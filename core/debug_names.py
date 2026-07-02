#!/usr/bin/env python3
"""Debug the simulation"""
import json, math, os, sys, urllib.request, csv, random
from datetime import datetime
from collections import defaultdict

MAX_GOALS = 6
DATA_DIR = 'data'

# ── 2026 世界杯参赛队
TEAMS_2026 = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia-Herzegovina", "Brazil", "Canada", "Cape Verde Islands", "Colombia",
    "Croatia", "Curacao", "Czech Republic", "DR Congo", "Ecuador",
    "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan",
    "Jordan", "Mexico", "Morocco", "Netherlands", "New Zealand",
    "Norway", "Panama", "Paraguay", "Portugal", "Qatar",
    "Saudi Arabia", "Scotland", "Senegal", "South Africa", "South Korea",
    "Spain", "Sweden", "Switzerland", "Tunisia", "Turkey",
    "United States", "Uruguay", "Uzbekistan"
]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def elo_expected(ra, rb):
    return 1.0 / (1 + 10 ** ((rb - ra) / 400))

def load_data(cache_path):
    with open(cache_path) as f:
        return json.load(f)

def compute_team_strengths(matches, half_life=180):
    cutoff = '2026-06-11'
    stats = defaultdict(lambda: {'wg': 0.0, 'wc': 0.0, 'weight_sum': 0.0, 'matches': 0})
    for m in matches:
        if m['date'] >= cutoff: continue
        days_ago = (datetime.strptime(cutoff, '%Y-%m-%d') - datetime.strptime(m['date'], '%Y-%m-%d')).days
        w = 0.5 ** (max(days_ago, 0) / half_life)
        for team, gf, ga in [(m['home'], m['h_score'], m['a_score']), (m['away'], m['a_score'], m['h_score'])]:
            s = stats[team]
            s['wg'] += gf * w; s['wc'] += ga * w
            s['weight_sum'] += w; s['matches'] += 1
    total_wg = sum(s['wg'] for s in stats.values())
    total_ws = sum(s['weight_sum'] for s in stats.values())
    global_avg = total_wg / max(total_ws, 1)
    team_data = {}
    for team, s in stats.items():
        avg_gf = s['wg'] / max(s['weight_sum'], 0.001)
        avg_ga = s['wc'] / max(s['weight_sum'], 0.001)
        team_data[team] = {'attack': avg_gf / max(global_avg, 0.01), 'defense': avg_ga / max(global_avg, 0.01), 'matches': s['matches']}
    return team_data, global_avg

def compute_elo_ratings(matches, cutoff='2026-06-11'):
    elo = defaultdict(lambda: 1500.0)
    for m in matches:
        if m['date'] >= cutoff: continue
        h, a = m['home'], m['away']
        hs, az = m['h_score'], m['a_score']
        e_h = elo_expected(elo[h], elo[a])
        sh, sa = (1.0, 0.0) if hs > az else ((0.5, 0.5) if hs == az else (0.0, 1.0))
        elo[h] += 32 * (sh - e_h)
        elo[a] += 32 * (sa - (1 - e_h))
    return dict(elo)

# Check team name alignment
matches = load_data(os.path.join(DATA_DIR, 'international_results.json'))
all_team_names = set(m['home'] for m in matches) | set(m['away'] for m in matches)

print("=== Team name check ===")
for t in TEAMS_2026:
    if t not in all_team_names:
        print(f"  ⚠️  '{t}' NOT found in dataset!")
        # Find closest
        for nt in sorted(all_team_names):
            if t[:3].lower() in nt.lower() or nt[:3].lower() in t.lower():
                print(f"     → Maybe '{nt}'?")
    else:
        elo = compute_elo_ratings(matches).get(t, 1500)
        print(f"  ✅ '{t}' found, Elo={elo:.0f}")

print(f"\n=== Random team names from dataset ===")
for t in sorted(all_team_names):
    if 'Bosnia' in t or 'Cape' in t or 'Curac' in t or 'Curaç' in t:
        print(f"  '{t}'")
