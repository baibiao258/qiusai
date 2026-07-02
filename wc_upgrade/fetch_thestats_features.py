"""
fetch_thestats_features.py — 世界杯球队状态+首发阵容抓取
=========================================================
每日调度:
  赛前 1h: 拉所有今日WC比赛的 Lineups
  每日 06:00: 拉所有48支WC球队的 Team Stats (从各自赛事)

输出:
  /root/data/thestats_team_stats.json   — 球队赛季统计
  /root/data/thestats_lineups.json     — 赛前首发+阵型
"""

import requests, json, os, sys, time
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

THE_KEY = os.environ.get('THE_KEY', '') or os.environ.get('THE_STATS_KEY', 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww')
if not THE_KEY:
    print("NO API KEY")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {THE_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"
DATA_DIR = "/root/data"

# === 1. WC 2026 team roster (48 teams) ===
WC_COMP = "comp_6107"
WC_SEASON = "sn_118868"

def fetch_wc_teams():
    """Get all 48 World Cup teams with their IDs"""
    url = f"{BASE}/competitions/{WC_COMP}/seasons/{WC_SEASON}/standings"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"Standings error: {r.status_code}")
        return {}
    
    teams = {}
    for row in r.json().get('data', []):
        team = row.get('team', {})
        tid = team.get('id', '')
        tname = team.get('name', '')
        group = row.get('group_label', '')
        if tid and tname:
            teams[tid] = {'name': tname, 'group': group}
    
    print(f"  WC teams: {len(teams)}")
    return teams

# === 2. Team Stats from ALL available competitions ===
def get_team_competitions(team_id):
    """Find what competitions a team has stats for"""
    # Check common national team competitions
    comps_to_try = [
        ("comp_6107", WC_SEASON, "World Cup 2026"),
        ("comp_8973", None, "WCQ AFC"),
        ("comp_5720", None, "WCQ CAF"),
        ("comp_0836", None, "WCQ CONCACAF"),
        ("comp_4682", None, "WCQ CONMEBOL"),
        ("comp_7363", None, "WCQ OFC"),
        ("comp_2954", None, "WCQ UEFA"),
        ("comp_574977", None, "UEFA Nations League"),
        ("comp_29967", None, "Intl Friendly"),
    ]
    
    results = []
    for cid, sid_hint, label in comps_to_try:
        # Find the current season for this competition
        if sid_hint:
            season_ids = [sid_hint]
        else:
            try:
                sr = requests.get(f"{BASE}/competitions/{cid}/seasons", headers=HEADERS, timeout=15)
                if sr.status_code == 200:
                    seasons = sr.json().get('data', [])
                    season_ids = [s['id'] for s in seasons if s.get('is_current')]
                else:
                    season_ids = []
            except:
                season_ids = []
        
        for sid in season_ids:
            try:
                sr = requests.get(f"{BASE}/teams/{team_id}/stats?season_id={sid}", headers=HEADERS, timeout=15)
                if sr.status_code == 200:
                    stats = sr.json().get('data', sr.json())
                    if isinstance(stats, dict) and stats.get('matches_played', 0) > 0:
                        stats['competition_label'] = label
                        stats['competition_id'] = cid
                        results.append(stats)
            except:
                pass
            time.sleep(0.05)
    
    return results

def fetch_all_team_stats(teams):
    """Fetch stats for all teams in parallel"""
    print("\n=== Fetching Team Stats ===")
    
    cache_path = f"{DATA_DIR}/thestats_team_stats.json"
    
    team_stats = {}
    lock = Lock()
    
    def fetch_one(tid):
        stats_list = get_team_competitions(tid)
        with lock:
            if stats_list:
                team_stats[tid] = stats_list
    
    with ThreadPoolExecutor(max_workers=10) as ex:
        ex.map(fetch_one, list(teams.keys()))
    
    # Pick best stats for each team (most matches_played)
    best_stats = {}
    for tid, stats_list in team_stats.items():
        if stats_list:
            best = max(stats_list, key=lambda s: s.get('matches_played', 0))
            best_stats[tid] = best
    
    print(f"  Teams with stats: {len(best_stats)}/{len(teams)}")
    
    # Build output
    output = {}
    for tid, info in teams.items():
        name = info['name']
        entry = {'name': name, 'group': info['group']}
        if tid in best_stats:
            s = best_stats[tid]
            entry['form'] = s.get('form', '')
            entry['mp'] = s.get('matches_played', 0)
            entry['w'] = s.get('wins', 0)
            entry['d'] = s.get('draws', 0)
            entry['l'] = s.get('losses', 0)
            entry['gf'] = s.get('goals_for', 0)
            entry['ga'] = s.get('goals_against', 0)
            entry['gd'] = s.get('goal_difference', 0)
            entry['pts'] = s.get('points', 0)
            entry['pos'] = s.get('position', 0)
            entry['comp'] = s.get('competition_label', '')
            entry['comp_id'] = s.get('competition_id', '')
        else:
            entry['form'] = ''
            entry['mp'] = 0
        
        # Latest results from matches API (actual recent form)
        try:
            url = f"{BASE}/matches?team_id={tid}&status=finished&per_page=5"
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                recent = []
                for m in r.json().get('data', []):
                    sc = m.get('score', {})
                    hsc = sc.get('home', '') if isinstance(sc, dict) else ''
                    asc = sc.get('away', '') if isinstance(sc, dict) else ''
                    is_home = m.get('home_team', {}).get('id') == tid
                    if is_home:
                        recent.append(f"{hsc}-{asc}")
                    else:
                        recent.append(f"{asc}-{hsc}")
                entry['recent_results'] = list(reversed(recent))
        except:
            entry['recent_results'] = []
        
        output[tid] = entry
    
    with open(cache_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {cache_path}")
    return output

# === 3. Lineups for today's matches ===
def fetch_today_lineups():
    """Fetch lineups for today's World Cup matches"""
    print("\n=== Fetching Today's Lineups ===")
    
    today = date.today().isoformat()
    cache_path = f"{DATA_DIR}/thestats_lineups.json"
    
    # Get today's WC matches
    url = f"{BASE}/matches?date_from={today}&date_to={today}&competition_id={WC_COMP}&per_page=50"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"  Matches query error: {r.status_code}")
        return {}
    
    matches = r.json().get('data', [])
    print(f"  Today's WC matches: {len(matches)}")
    
    if not matches:
        return {}
    
    lineups_data = {}
    lock = Lock()
    
    def get_lineup(m):
        mid = m.get('id', '')
        ht = m.get('home_team', {}).get('name', '?')
        at = m.get('away_team', {}).get('name', '?')
        
        try:
            r = requests.get(f"{BASE}/matches/{mid}/lineups", headers=HEADERS, timeout=30)
            if r.status_code == 200:
                lu = r.json().get('data', r.json())
                if isinstance(lu, dict) and lu.get('confirmed'):
                    lineup_info = {
                        'match_id': mid,
                        'confirmed': True,
                        'home_team': ht,
                        'away_team': at,
                    }
                    for side in ['home', 'away']:
                        s = lu.get(side, {})
                        if s:
                            lineup_info[f'{side}_formation'] = s.get('formation', '')
                            lineup_info[f'{side}_starters'] = [p.get('name', '') for p in s.get('starting_xi', [])]
                            lineup_info[f'{side}_subs'] = len(s.get('substitutes', []))
                    with lock:
                        lineups_data[mid] = lineup_info
                    print(f"  ✅ {ht} vs {at}: formation {lu.get('home',{}).get('formation','?')} vs {lu.get('away',{}).get('formation','?')}")
                else:
                    print(f"  ⏳ {ht} vs {at}: lineup not yet confirmed")
            else:
                print(f"  ⏳ {ht} vs {at}: lineup not available yet")
        except Exception as e:
            print(f"  ❌ {ht} vs {at}: {e}")
        
        time.sleep(0.2)
    
    for m in matches:
        get_lineup(m)
    
    # Save
    if lineups_data:
        with open(cache_path, 'w') as f:
            json.dump(lineups_data, f, indent=2)
        print(f"  Saved: {cache_path}")
    
    return lineups_data

# === MAIN ===
if __name__ == '__main__':
    import sys
    lineups_only = '--lineups-only' in sys.argv
    
    print("=" * 60)
    print(f"  TheStatsAPI Feature Fetcher  |  {datetime.now().isoformat()}")
    if lineups_only:
        print(f"  MODE: lineups only")
    print("=" * 60)
    
    if not lineups_only:
        # Step 1: Get WC teams
        teams = fetch_wc_teams()
        
        # Step 2: Get team stats
        team_stats = fetch_all_team_stats(teams)
        
        # Summary
        with_stats = sum(1 for v in team_stats.values() if v.get('mp', 0) > 0)
        print(f"\n=== Summary ===")
        print(f"  Teams: {len(teams)} | With stats: {with_stats}")
        if team_stats:
            ranked = sorted(team_stats.values(), key=lambda x: x.get('mp', 0), reverse=True)
            for t in ranked[:5]:
                name = t.get('name', '?')
                mp = t.get('mp', 0)
                w = t.get('w', 0)
                form = t.get('form', '')
                comp = t.get('comp', '')
                recent = t.get('recent_results', [])
                print(f"  {name:20s} | MP={mp:3d} W={w:2d} | form={form:10s} | recent={recent}")
    else:
        teams = fetch_wc_teams()
        print(f"  Teams: {len(teams)} (loaded for lineup context)")
    
    # Step 3: Today's lineups  
    lineups = fetch_today_lineups()
    print(f"  Lineups loaded: {len(lineups)}")
    
    print("\n  Done.")
