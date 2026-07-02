import requests, json, os

KEY = os.environ.get('THE_KEY', 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww')
H = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

# Probe players endpoint for a known WC team
print("=== Mexico (tm_28735) Players ===")
r = requests.get(f"{BASE}/teams/tm_28735/players", headers=H, timeout=30)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    d = data.get('data', data)
    if isinstance(d, list):
        print(f"Players: {len(d)}")
        for p in d[:5]:
            print(f"  {json.dumps(p, indent=2)[:300]}")
        if len(d) > 5:
            print(f"  ... ({len(d)-5} more)")
            # Show last 3
            for p in d[-3:]:
                print(f"  {json.dumps(p, indent=2)[:200]}")
    elif isinstance(d, dict):
        print(f"Keys: {list(d.keys())}")
        print(json.dumps(d, indent=2)[:500])
else:
    print(r.text[:300])

# Also check if there's a competition-level players endpoint
print("\n=== WC Standings for team IDs ===")
r2 = requests.get(f"{BASE}/competitions/comp_6107/seasons/sn_118868/standings", headers=H, timeout=15)
if r2.status_code == 200:
    rows = r2.json().get('data', [])
    teams = []
    for row in rows:
        team = row.get('team', {})
        tid = team.get('id')
        tname = team.get('name')
        group = row.get('group_label', '')
        if tid and (tid, tname) not in [(t[0], t[1]) for t in teams]:
            teams.append((tid, tname, group))
    print(f"Unique WC teams: {len(teams)}")
    for tid, tname, group in teams[:10]:
        print(f"  {tname:25s} ({tid:15s}) Group {group}")
