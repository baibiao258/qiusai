import requests, json, os, sys
from datetime import datetime

KEY = os.environ.get('THE_KEY', '') or os.environ.get('THE_STATS_KEY', 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww')
H = {"Authorization": f"Bearer {KEY}"}

# Check recent WC matches
r = requests.get("https://api.thestatsapi.com/api/football/matches?competition_id=comp_6107&per_page=20", headers=H, timeout=15)
data = r.json().get('data', [])
print(f"WC matches returned: {len(data)}")
dates = set()
for m in data:
    ht = m.get('home_team',{}).get('name','?')
    at = m.get('away_team',{}).get('name','?')
    st = m.get('status','?')
    sd = m.get('start_date','')[:19]
    dates.add(sd[:10])
    print(f"  {sd} | {ht:25s} vs {at:25s} | {st}")
print(f"\nMatch dates: {sorted(dates)}")
print(f"Latest: {max(dates) if dates else 'none'}")

# Also check finished matches
print("\n--- Finished ---")
r2 = requests.get("https://api.thestatsapi.com/api/football/matches?competition_id=comp_6107&status=finished&per_page=5", headers=H, timeout=15)
d2 = r2.json().get('data', [])
for m in d2:
    ht = m.get('home_team',{}).get('name','?')
    at = m.get('away_team',{}).get('name','?')
    sd = m.get('start_date','')[:10]
    sc = m.get('score',{})
    print(f"  {sd} | {ht:25s} vs {at:25s} | FT {sc}")
