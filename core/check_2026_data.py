#!/usr/bin/env python3
"""检查 2026 世界杯可用数据"""
import urllib.request, json

# Try openfootball for 2026 data
url = 'https://raw.githubusercontent.com/openfootball/world-cup.json/master/2026/worldcup.json'
req = urllib.request.Request(url, headers={'User-Agent': 'wc_predictor/1.0'})
try:
    data = urllib.request.urlopen(req, timeout=15).read().decode('utf-8')
    wc = json.loads(data)
    if 'rounds' in wc:
        for rnd in wc.get('rounds', []):
            name = rnd.get('name', '?')
            matches_count = len(rnd.get('matches', []))
            print(f"  {name}: {matches_count} matches")
        teams = set()
        for rnd in wc.get('rounds', []):
            for m in rnd.get('matches', []):
                teams.add(m.get('team1', ''))
                teams.add(m.get('team2', ''))
        print(f"\nTeams ({len(teams)}):")
        for t in sorted(teams):
            print(f"  {t}")
    elif 'matches' in wc:
        teams = set()
        for m in wc['matches']:
            teams.add(m.get('team1', ''))
            teams.add(m.get('team2', ''))
        print(f"Teams ({len(teams)}):")
        for t in sorted(teams):
            print(f"  {t}")
except Exception as e:
    print(f"No 2026 data from openfootball: {e}")

# Try Wikipedia for group info
print("\n--- Searching for 2026 WC info via web ---")
try:
    url2 = "https://en.wikipedia.org/api/rest_v1/page/summary/2026_FIFA_World_Cup"
    req2 = urllib.request.Request(url2, headers={'User-Agent': 'wc_predictor/1.0'})
    resp = urllib.request.urlopen(req2, timeout=15)
    data = json.loads(resp.read().decode('utf-8'))
    extract = data.get('extract', '')
    print(extract[:2000])
except Exception as e:
    print(f"Wikipedia error: {e}")
