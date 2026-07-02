import requests, json, os, sys

THE_KEY = os.environ.get('THE_KEY', '') or os.environ.get('THE_STATS_KEY', '')
if not THE_KEY:
    print("NO KEY")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {THE_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

# Target competition IDs (the ones we want)
TARGET_IDS = {
    "comp_6107", "comp_8973", "comp_5720", "comp_0836", "comp_4682",
    "comp_7363", "comp_2954", "comp_2949", "comp_3759", "comp_5749",
    "comp_574977", "comp_193547", "comp_1376", "comp_1554", "comp_83579",
    "comp_29967", "comp_920080",
}

# Quick test: how many matches in a single day?
print("=== Test: Single day (2026-06-14) ===")
url = f"{BASE}/matches?date_from=2026-06-14&date_to=2026-06-14&status=finished&per_page=100"
r = requests.get(url, headers=HEADERS, timeout=30)
if r.status_code == 200:
    data = r.json()
    meta = data.get("meta", {})
    total_all = meta.get("total", 0)
    matches = data.get("data", [])
    target_matches = [m for m in matches if m.get("competition_id") in TARGET_IDS]
    print(f"All matches on 06-14: {total_all}")
    print(f"Target competition matches in page: {len(target_matches)}/{len(matches)}")
    for m in target_matches[:3]:
        cid = m.get("competition_id", "")
        ht = m.get("home_team", {}).get("name", "?") if isinstance(m.get("home_team"), dict) else "?"
        at = m.get("away_team", {}).get("name", "?") if isinstance(m.get("away_team"), dict) else "?"
        sc = m.get("score", {})
        hsc = sc.get("home", "?") if isinstance(sc, dict) else "?"
        asc = sc.get("away", "?") if isinstance(sc, dict) else "?"
        print(f"  {cid} | {ht} {hsc}-{asc} {at}")
    print(f"\nNon-target competitions in page:")
    comps_seen = set()
    for m in matches[:20]:
        cid = m.get("competition_id", "")
        if cid not in TARGET_IDS and cid not in comps_seen:
            comps_seen.add(cid)
            name = ""
            print(f"  {cid}")
else:
    print(f"Error: {r.status_code}: {r.text[:300]}")

# Test: Week in June
print("\n=== Test: Week (2026-06-08 ~ 2026-06-15) ===")
url2 = f"{BASE}/matches?date_from=2026-06-08&date_to=2026-06-15&status=finished&per_page=100"
r2 = requests.get(url2, headers=HEADERS, timeout=30)
if r2.status_code == 200:
    data2 = r2.json()
    meta2 = data2.get("meta", {})
    total2 = meta2.get("total", 0)
    print(f"Total matches in week: {total2}")
else:
    print(f"Error: {r2.status_code}")
