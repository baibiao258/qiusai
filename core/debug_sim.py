#!/usr/bin/env python3
"""Debug simulation"""
import json, math, os, sys, random
from collections import defaultdict

MAX_GOALS = 6

TEAMS_2026 = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia",
    "Croatia", "Curaçao", "Czech Republic", "DR Congo", "Ecuador",
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

DATA_DIR = 'data'
with open(os.path.join(DATA_DIR, 'international_results.json')) as f:
    matches = json.load(f)

# Team strength
cutoff = '2026-06-11'
from datetime import datetime
stats = defaultdict(lambda: {'wg': 0.0, 'wc': 0.0, 'weight_sum': 0.0})
for m in matches:
    if m['date'] >= cutoff: continue
    days_ago = (datetime.strptime(cutoff, '%Y-%m-%d') - datetime.strptime(m['date'], '%Y-%m-%d')).days
    w = 0.5 ** (max(days_ago, 0) / 180)
    for team, gf, ga in [(m['home'], m['h_score'], m['a_score']), (m['away'], m['a_score'], m['h_score'])]:
        s = stats[team]
        s['wg'] += gf * w; s['wc'] += ga * w; s['weight_sum'] += w
total_wg = sum(s['wg'] for s in stats.values())
total_ws = sum(s['weight_sum'] for s in stats.values())
global_avg = total_wg / max(total_ws, 1)

team_data = {}
for team, s in stats.items():
    avg_gf = s['wg'] / max(s['weight_sum'], 0.001)
    avg_ga = s['wc'] / max(s['weight_sum'], 0.001)
    team_data[team] = {'attack': avg_gf / max(global_avg, 0.01), 'defense': avg_ga / max(global_avg, 0.01)}

# Elo
elo = defaultdict(lambda: 1500.0)
for m in matches:
    if m['date'] >= cutoff: continue
    h, a = m['home'], m['away']
    hs, az = m['h_score'], m['a_score']
    e_h = elo_expected(elo[h], elo[a])
    sh, sa = (1.0, 0.0) if hs > az else ((0.5, 0.5) if hs == az else (0.0, 1.0))
    elo[h] += 32 * (sh - e_h)
    elo[a] += 32 * (sa - (1 - e_h))
elo_ratings = dict(elo)

# Check missing teams
print("=== Missing team data check ===")
for t in TEAMS_2026:
    if t not in team_data:
        print(f"  ⚠️ {t} not in team_data! Using defaults.")
for t in TEAMS_2026:
    if t not in elo_ratings:
        print(f"  ⚠️ {t} not in elo_ratings! Using 1500.")

# Test single simulation
print(f"\n=== Test single simulation ===")

# Setup groups
sorted_teams = sorted(TEAMS_2026, key=lambda t: elo_ratings.get(t, 1500), reverse=True)
pots = [sorted_teams[i:i+12] for i in range(0, 48, 12)]
print(f"Pot 0 (strongest): {pots[0]}")
print(f"Pot 3 (weakest): {pots[3]}")

groups = {}
for pot_idx, pot in enumerate(pots):
    random.shuffle(pot)
    for g_idx, team in enumerate(pot):
        gl = chr(ord('A')+g_idx)
        if gl not in groups: groups[gl] = []
        groups[gl].append(team)

print(f"\nGroups ({len(groups)}):")
for g in sorted(groups.keys()):
    print(f"  Group {g}: {groups[g]}")

# Group stage
print(f"\n=== Group stage ===")
qualifiers = []
for g_name in sorted(groups.keys()):
    g_teams = groups[g_name]
    if len(g_teams) != 3:
        print(f"  ⚠️ Group {g_name} has {len(g_teams)} teams!")
        continue
    
    points = {t: 0 for t in g_teams}
    gd = {t: 0 for t in g_teams}
    gf = {t: 0 for t in g_teams}
    
    fixtures = [(g_teams[0], g_teams[1]), (g_teams[0], g_teams[2]), (g_teams[1], g_teams[2])]
    
    for t1, t2 in fixtures:
        ts1 = team_data.get(t1, {'attack': 1.0, 'defense': 1.0})
        ts2 = team_data.get(t2, {'attack': 1.0, 'defense': 1.0})
        
        lam_h = global_avg * ts1['attack'] * ts2['defense']
        lam_a = global_avg * ts2['attack'] * ts1['defense']
        lam_h = max(0.1, min(5.0, lam_h)); lam_a = max(0.1, min(5.0, lam_a))
        
        h_probs = [poisson_pmf(k, lam_h) for k in range(MAX_GOALS+1)]
        a_probs = [poisson_pmf(k, lam_a) for k in range(MAX_GOALS+1)]
        h_cum = [sum(h_probs[:i+1]) for i in range(MAX_GOALS+1)]
        a_cum = [sum(a_probs[:i+1]) for i in range(MAX_GOALS+1)]
        
        r = random.random()
        hg = next((i for i, c in enumerate(h_cum) if r <= c), MAX_GOALS)
        r = random.random()
        ag = next((i for i, c in enumerate(a_cum) if r <= c), MAX_GOALS)
        
        gf[t1] += hg; gf[t2] += ag
        gd[t1] += hg - ag; gd[t2] += ag - hg
        if hg > ag: points[t1] += 3
        elif hg == ag: points[t1] += 1; points[t2] += 1
        else: points[t2] += 3
        
        print(f"    {t1:<25s} {hg}-{ag} {t2:<25s}")
    
    ranked = sorted(g_teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True)
    print(f"  Group {g_name}: {[f'{t}({points[t]}pts, gd={gd[t]})' for t in ranked]}")
    qualifiers.append(ranked[:2])

print(f"\nQualifiers: {len(qualifiers)} groups")
print(f"First 3 qualifiers: {qualifiers[:3]}")
