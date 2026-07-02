import requests, json
from datetime import datetime, timezone

K = 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww'
H = {'Authorization': f'Bearer {K}'}

today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
print(f"Today (UTC): {today}")

# Get ALL matches for WC 2026, ordered by date
r = requests.get(f"https://api.thestatsapi.com/api/football/matches?competition_id=comp_6107&per_page=50&order=asc", headers=H, timeout=15)
data = r.json().get('data', [])

print(f"WC matches found: {len(data)}")

# Group by utc_date
from collections import defaultdict
by_date = defaultdict(list)
for m in data:
    dt = m.get('utc_date', '')[:10]
    if dt:
        by_date[dt].append(m)

# Show all dates with match counts
for dt in sorted(by_date.keys()):
    matches = by_date[dt]
    statuses = set(m.get('status','?') for m in matches)
    print(f"  {dt}: {len(matches)} matches | statuses: {statuses}")
    for m in matches[:3]:
        ht = m.get('home_team',{}).get('name','?')
        at = m.get('away_team',{}).get('name','?')
        st = m.get('status','?')
        sc = m.get('score',{})
        sc_str = f" {sc.get('home','')}-{sc.get('away','')}" if st == 'finished' else ''
        print(f"    {ht:25s} vs {at:25s} | {st}{sc_str}")
    if len(matches) > 3:
        print(f"    ... ({len(matches)-3} more)")

print(f"\nToday ({today}): {len(by_date.get(today,[]))} matches")
print(f"Tomorrow (next day): {sorted(by_date.keys())[:5]}")
