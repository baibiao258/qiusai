import requests, json, os

API_KEY = os.environ.get('THESTATSAPI_KEY', '')
if not API_KEY:
    API_KEY = 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww'
    
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

# Step 1: World Cup finished matches
print("=== World Cup finished matches (June 2026) ===")
url = f"{BASE}/matches?date_from=2026-06-01&date_to=2026-06-16&competition_ids=comp_6107&status=finished"
resp = requests.get(url, headers=HEADERS, timeout=30)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Keys: {list(data.keys())}")
    matches = data.get('data', [])
    print(f"Match count: {len(matches)}")
    meta = data.get('meta', {})
    print(f"Meta: {json.dumps(meta, indent=2)}")
    
    if matches:
        m = matches[0]
        print(f"\n=== First match structure ===")
        print(f"ID: {m.get('id')}")
        for k, v in m.items():
            val_str = json.dumps(v) if not isinstance(v, str) else str(v)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            print(f"  {k}: {val_str}")
        
        match_id = m.get('id', '')
        
        print(f"\n=== Stats for {match_id} ===")
        sr = requests.get(f"{BASE}/matches/{match_id}/stats", headers=HEADERS, timeout=30)
        print(f"Status: {sr.status_code}")
        if sr.status_code == 200:
            print(json.dumps(sr.json(), indent=2)[:1500])
        else:
            print(sr.text[:500])
        
        print(f"\n=== Odds for {match_id} ===")
        or_ = requests.get(f"{BASE}/matches/{match_id}/odds", headers=HEADERS, timeout=30)
        print(f"Status: {or_.status_code}")
        if or_.status_code == 200:
            print(json.dumps(or_.json(), indent=2)[:1500])
        else:
            print(or_.text[:500])
    else:
        # Try without competition filter
        print("\nNo WC matches. Checking all matches 06-14~15...")
        url2 = f"{BASE}/matches?date_from=2026-06-14&date_to=2026-06-15&status=finished"
        r2 = requests.get(url2, headers=HEADERS, timeout=30)
        if r2.status_code == 200:
            d2 = r2.json()
            ms = d2.get('data', [])
            print(f"All finished matches: {len(ms)}")
            for m2 in ms[:5]:
                home = '?'
                away = '?'
                if isinstance(m2.get('home_team'), dict):
                    home = m2['home_team'].get('name', '?')
                elif isinstance(m2.get('home_team'), str):
                    home = m2['home_team']
                if isinstance(m2.get('away_team'), dict):
                    away = m2['away_team'].get('name', '?')
                elif isinstance(m2.get('away_team'), str):
                    away = m2['away_team']
                print(f"  {m2.get('id')} | {home} vs {away} | comp={m2.get('competition_id')} | score={m2.get('home_score','?')}-{m2.get('away_score','?')}")
        else:
            print(f"Error: {r2.text[:300]}")
else:
    print(f"Error: {resp.text[:500]}")
