#!/usr/bin/env python3
"""下载最新国际赛数据 + 2026世界杯信息"""
import urllib.request, csv, json, os, sys
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 1. 国际赛历史数据
url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
print("📡 下载国际赛数据...")
req = urllib.request.Request(url, headers={'User-Agent': 'wc_predictor/1.0'})
raw = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')

matches = []
for row in csv.DictReader(raw.splitlines()):
    try:
        matches.append({
            'date': row['date'],
            'home': row['home_team'],
            'away': row['away_team'],
            'tournament': row['tournament'],
            'h_score': int(row['home_score']),
            'a_score': int(row['away_score']),
        })
    except:
        continue

cache_path = os.path.join(DATA_DIR, 'international_results.json')
with open(cache_path, 'w') as f:
    json.dump(matches, f)

# 统计
print(f"  ✅ {len(matches)} 条比赛记录")
dates = sorted(set(m['date'] for m in matches))
print(f"  📅 日期范围: {dates[0]} ~ {dates[-1]}")
print(f"  🏟️  球队数: {len(set(m['home'] for m in matches) | set(m['away'] for m in matches))}")

# 最新比赛
recent = [m for m in matches if m['date'] >= '2026-01-01']
print(f"\n  📰 2026年比赛: {len(recent)} 场")
for m in recent[-20:]:
    print(f"    {m['date']} {m['home']} {m['h_score']}-{m['a_score']} {m['away']} [{m['tournament']}]")

# 2. 热门球队近期表现 (2025-2026)
top_teams = ['Argentina', 'France', 'Brazil', 'England', 'Spain', 'Portugal', 
             'Germany', 'Netherlands', 'Belgium', 'Croatia', 'Italy', 'Uruguay',
             'Morocco', 'USA', 'Mexico', 'Japan', 'Senegal', 'Switzerland']
print(f"\n{'='*60}")
print(f"  热门球队 2025-2026 战绩")
print(f"{'='*60}")
for team in top_teams:
    team_matches = [m for m in matches if m['date'] >= '2025-01-01' and 
                    (m['home'] == team or m['away'] == team)]
    if not team_matches:
        continue
    w, d, l = 0, 0, 0
    for m in team_matches:
        if m['home'] == team:
            if m['h_score'] > m['a_score']: w += 1
            elif m['h_score'] == m['a_score']: d += 1
            else: l += 1
        else:
            if m['a_score'] > m['h_score']: w += 1
            elif m['h_score'] == m['a_score']: d += 1
            else: l += 1
    pct = w/max(w+d+l,1)*100
    print(f"  {team:<15s} {w:>2d}W {d:>2d}D {l:>2d}L ({pct:.0f}%) — {len(team_matches)}场")

# 3. 2026世界杯分组 (最新信息)
print(f"\n{'='*60}")
print(f"  2026 世界杯 (48队, 6月11日-7月19日)")
print(f"{'='*60}")
print("  主办国: 美国/加拿大/墨西哥")
print("  赛制: 16组×3队 → 小组前2 → 32强淘汰赛")
