import requests, json, os, sys

THE_KEY = os.environ.get('THE_KEY', '') or os.environ.get('THE_STATS_KEY', '')
if not THE_KEY:
    print("NO KEY")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {THE_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

# Test if competition_ids filter works
print("=== Test 1: World Cup only ===")
url = f"{BASE}/matches?date_from=2026-06-01&date_to=2026-06-16&status=finished&competition_ids=comp_6107&per_page=5"
r = requests.get(url, headers=HEADERS, timeout=30)
if r.status_code == 200:
    data = r.json()
    matches = data.get("data", [])
    print(f"Returned {len(matches)} matches")
    for m in matches:
        cid = m.get("competition_id", "?")
        ht = m.get("home_team", {}).get("name", "?") if isinstance(m.get("home_team"), dict) else m.get("home_team", "?")
        at = m.get("away_team", {}).get("name", "?") if isinstance(m.get("away_team"), dict) else m.get("away_team", "?")
        print(f"  {m.get('id')} | comp={cid} | {ht} vs {at}")
else:
    print(f"Error: {r.status_code}: {r.text[:300]}")

print("\n=== Test 2: No filter (all comps) ===")
url2 = f"{BASE}/matches?date_from=2026-06-01&date_to=2026-06-16&status=finished&per_page=5"
r2 = requests.get(url2, headers=HEADERS, timeout=30)
if r2.status_code == 200:
    data2 = r2.json()
    matches2 = data2.get("data", [])
    print(f"Returned {len(matches2)} matches")
    for m in matches2:
        cid = m.get("competition_id", "?")
        ht = m.get("home_team", {}).get("name", "?") if isinstance(m.get("home_team"), dict) else m.get("home_team", "?")
        at = m.get("away_team", {}).get("name", "?") if isinstance(m.get("away_team"), dict) else m.get("away_team", "?")
        print(f"  {m.get('id')} | comp={cid} | {ht} vs {at}")
else:
    print(f"Error: {r2.status_code}: {r2.text[:300]}")

print("\n=== Test 3: UEFA Nations League only ===")
url3 = f"{BASE}/matches?date_from=2026-06-01&date_to=2026-06-16&status=finished&competition_ids=comp_574977&per_page=5"
r3 = requests.get(url3, headers=HEADERS, timeout=30)
if r3.status_code == 200:
    data3 = r3.json()
    matches3 = data3.get("data", [])
    print(f"Returned {len(matches3)} matches")
    for m in matches3:
        cid = m.get("competition_id", "?")
        ht = m.get("home_team", {}).get("name", "?") if isinstance(m.get("home_team"), dict) else m.get("home_team", "?")
        at = m.get("away_team", {}).get("name", "?") if isinstance(m.get("away_team"), dict) else m.get("away_team", "?")
        print(f"  {m.get('id')} | comp={cid} | {ht} vs {at}")
else:
    print(f"Error: {r3.status_code}: {r3.text[:300]}")
