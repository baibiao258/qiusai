import requests, json, os

KEY = os.environ.get('THE_KEY', 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww')
H = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

# 1. Lineups for Sweden vs Tunisia
print("=== Lineups for Sweden vs Tunisia ===")
r = requests.get(f"{BASE}/matches/mt_209798753/lineups", headers=H, timeout=15)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    d = r.json().get('data', r.json())
    if isinstance(d, dict):
        print(f"confirmed: {d.get('confirmed')}")
        for side in ['home', 'away']:
            s = d.get(side, {})
            if s:
                print(f"  {side}: formation={s.get('formation')}, starters={len(s.get('starting_xi',[]))}")
                for p in s.get('starting_xi', [])[:5]:
                    print(f"    {p.get('name','')} ({p.get('position','')})")
                print(f"  subs: {len(s.get('substitutes',[]))}")
else:
    print(f"Error: {r.status_code} {r.text[:300]}")

# 2. Team Info for Mexico  
print("\n=== Mexico Team Info ===")
r = requests.get(f"{BASE}/teams/tm_28735", headers=H, timeout=15)
if r.status_code == 200:
    print(json.dumps(r.json(), indent=2)[:500])

# 3. Check WCQ CONCACAF seasons for Mexico stats
print("\n=== WCQ CONCACAF Seasons ===")
r = requests.get(f"{BASE}/competitions/comp_0836/seasons", headers=H, timeout=15)
if r.status_code == 200:
    seasons = r.json().get('data', [])
    for s in seasons:
        print(f"  {s.get('id')}: {s.get('name')} is_current={s.get('is_current')}")
        if s.get('is_current'):
            sid = s['id']
            # Try Mexico stats in this competition
            print(f"\n=== Mexico stats in WCQ CONCACAF (season {sid}) ===")
            r2 = requests.get(f"{BASE}/teams/tm_28735/stats?season_id={sid}", headers=H, timeout=15)
            if r2.status_code == 200:
                print(json.dumps(r2.json().get('data', r2.json()), indent=2))
            else:
                print(f"  Error: {r2.status_code} {r2.text[:200]}")
