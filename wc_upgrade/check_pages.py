import requests, json, os, sys

THE_KEY = os.environ.get('THE_KEY', '') or os.environ.get('THE_STATS_KEY', '')
if not THE_KEY:
    print("NO KEY")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {THE_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

COMP_IDS = [
    ("comp_6107", "FIFA World Cup"),
    ("comp_8973", "WCQ AFC"),
    ("comp_5720", "WCQ CAF"),
    ("comp_0836", "WCQ CONCACAF"),
    ("comp_4682", "WCQ CONMEBOL"),
    ("comp_7363", "WCQ OFC"),
    ("comp_2954", "WCQ UEFA"),
    ("comp_2949", "EURO"),
    ("comp_3759", "EURO Qual"),
    ("comp_5749", "Copa America"),
    ("comp_574977", "UEFA Nations League"),
    ("comp_193547", "CONCACAF Nations League"),
    ("comp_1376", "CONCACAF Gold Cup"),
    ("comp_1554", "Africa Cup of Nations"),
    ("comp_83579", "Africa Cup of Nations Qual."),
    ("comp_29967", "International Friendly Games"),
    ("comp_920080", "FIFA Series"),
]

print("=== Page counts per competition (2024-01-01 ~ 2026-06-16) ===")
total_all = 0
for cid, name in COMP_IDS:
    url = f"{BASE}/matches?date_from=2024-01-01&date_to=2026-06-16&status=finished&competition_ids={cid}&page=1&per_page=1"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 200:
        meta = r.json().get("meta", {})
        total = meta.get("total", 0)
        pages = meta.get("total_pages", 0) or ((total + 19) // 20)
        total_all += total
        print(f"  {name:35s} ({cid:15s}) : {total:6d} matches, {pages} pages")
    else:
        print(f"  {name:35s} ({cid:15s}) : ERROR {r.status_code}")

print(f"\n  {'TOTAL':35s} {'':15s} : {total_all:6d} matches")
