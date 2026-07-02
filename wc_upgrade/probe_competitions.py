import requests, json, os
from collections import Counter

API_KEY = os.environ.get('THESTATS_API_KEY', 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww')
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

# Fetch all pages
all_comps = []
for page in range(1, 9):
    url = f"{BASE}/competitions?page={page}&per_page=20"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"Page {page} failed: {resp.status_code} {resp.text[:200]}")
        continue
    data = resp.json().get('data', [])
    all_comps.extend(data)
    print(f"Page {page}: {len(data)} comps (total: {len(all_comps)})")

print(f"\n=== TOTAL: {len(all_comps)} competitions ===\n")

# Type distribution
type_counts = Counter(c.get('type', 'unknown') for c in all_comps)
print("Type distribution:")
for t, c in type_counts.most_common():
    print(f"  {t}: {c}")

# National team / cup keywords
keywords = ['world cup', 'euro', 'copa america', 'friendly', 'nations league',
            'european championship', 'qualif', 'asian cup', 'concacaf',
            'ofc', 'conmebol', 'fifa', 'international', 'friendlies',
            'cup', 'finalissima', 'super cup', 'intercontinental']

national_team_keywords = ['world cup', 'euro ', 'copa america', 'friendly', 'nations league',
                          'european championship', 'asian cup', 'concacaf', 'ofc', 'conmebol',
                          'fifa', 'international', 'friendlies', 'africa cup',
                          'arab cup', 'asean', 'championship']

intl_comps = []
for c in all_comps:
    name = str(c.get('name', '')).lower()
    if any(k in name for k in national_team_keywords):
        intl_comps.append(c)

print(f"\n=== International/National Team Competitions ({len(intl_comps)}) ===\n")
intl_comps.sort(key=lambda x: str(x.get('name', '')))
for c in intl_comps:
    cid = c.get('id', '')
    name = c.get('name', '')
    ctype = c.get('type', '')
    country = c.get('country', c.get('country_name', ''))
    season = c.get('current_season', c.get('season', ''))
    print(f"  ID: {str(cid):20s} | {str(name):45s} | {str(ctype):12s} | {str(country):10s} | season={season}")

# Print ALL names to see what we're missing
print("\n\n=== ALL COMPETITION NAMES ===")
for c in sorted(all_comps, key=lambda x: str(x.get('name', ''))):
    cid = c.get('id', '')
    name = c.get('name', '')
    ctype = c.get('type', '')
    season = c.get('current_season', c.get('season', ''))
    print(f"  {str(cid):25s} | {str(name):50s} | {str(ctype):12s} | season={season}")
