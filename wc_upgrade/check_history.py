import requests, json, os, sys

API_KEY = os.getenv('THE_STATS_KEY', os.getenv('THE_KEY', ''))
if not API_KEY:
    print("ERROR: need API key in env THE_KEY or THE_STATS_KEY")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

INTL_IDS = "comp_6107,comp_8973,comp_5720,comp_0836,comp_4682,comp_7363,comp_2954,comp_2949,comp_3759,comp_5749,comp_574977,comp_193547,comp_1376,comp_1554,comp_83579,comp_29967,comp_920080"

print("=== Historical Coverage Check ===\n")
check_periods = [
    ("2024-06~07 (Euro+Copa)", "2024-06-01", "2024-07-31"),
    ("2024-09~11 (WCQ+NationsLg+Friendly)", "2024-09-01", "2024-11-30"),
    ("2025-03~06 (WCQ)", "2025-03-01", "2025-06-30"),
    ("2025-09~11 (WCQ)", "2025-09-01", "2025-11-30"),
    ("2026-01~05 (WCQ+Friendly)", "2026-01-01", "2026-05-31"),
    ("2026-06 (World Cup)", "2026-06-01", "2026-06-16"),
]

for label, dfrom, dto in check_periods:
    url = f"{BASE}/matches?date_from={dfrom}&date_to={dto}&competition_ids={INTL_IDS}&status=finished"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 200:
        total = resp.json().get('meta', {}).get('total', 0)
        print(f"  {label}: {total} finished matches")
    else:
        print(f"  {label}: ERROR {resp.status_code}")
