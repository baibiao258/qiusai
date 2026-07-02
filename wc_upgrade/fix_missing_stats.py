import requests, json, os, csv, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

THE_KEY = os.environ.get('THE_KEY', '') or os.environ.get('THE_STATS_KEY', '')
if not THE_KEY:
    print("NO KEY")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {THE_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"
CSV_PATH = "/root/wc_2026_upgrade/training_data_thestats.csv"

# Read existing CSV, find matches with empty stats
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

FIELDS = reader.fieldnames
stats_cols = ['home_xg','away_xg','possession_h','possession_a','shots_ot_h','shots_ot_a','total_shots_h','total_shots_a']
odds_cols = ['market_h','market_d','market_a']

to_fix = [r for r in rows if not r.get('home_xg','').strip() and r.get('home_score','').strip()]
print(f"Total CSV rows: {len(rows)}")
print(f"Matches with empty stats but finished: {len(to_fix)}", flush=True)

if not to_fix:
    print("Nothing to fix!")
    sys.exit(0)

lock = Lock()
fixed = [0]

def fix_one(row):
    mid = row['match_id']
    rec = {}
    
    # Stats - use simpler path
    try:
        r = requests.get(f"{BASE}/matches/{mid}/stats", headers=HEADERS, timeout=30)
        if r.status_code == 200:
            d = r.json()
            d2 = d.get('data', {}) if isinstance(d, dict) else d
            if isinstance(d2, dict):
                ov = d2.get('overview', {})
                if isinstance(ov, dict):
                    eg = ov.get('expected_goals', {})
                    if isinstance(eg, dict):
                        a = eg.get('all', {})
                        if isinstance(a, dict):
                            rec['home_xg'] = a.get('home', '')
                            rec['away_xg'] = a.get('away', '')
                    bp = ov.get('ball_possession', {})
                    if isinstance(bp, dict):
                        a = bp.get('all', {})
                        if isinstance(a, dict):
                            rec['possession_h'] = a.get('home', '')
                            rec['possession_a'] = a.get('away', '')
                    sot = ov.get('shots_on_target', {})
                    if isinstance(sot, dict):
                        a = sot.get('all', {})
                        if isinstance(a, dict):
                            rec['shots_ot_h'] = a.get('home', '')
                            rec['shots_ot_a'] = a.get('away', '')
                    ts = ov.get('total_shots', {})
                    if isinstance(ts, dict):
                        a = ts.get('all', {})
                        if isinstance(a, dict):
                            rec['total_shots_h'] = a.get('home', '')
                            rec['total_shots_a'] = a.get('away', '')
    except:
        pass
    
    time.sleep(0.05)
    
    # Odds
    try:
        r = requests.get(f"{BASE}/matches/{mid}/odds", headers=HEADERS, timeout=30)
        if r.status_code == 200:
            d = r.json()
            d2 = d.get('data', {}) if isinstance(d, dict) else d
            if isinstance(d2, dict):
                for bm in d2.get('bookmakers', []):
                    if bm.get('bookmaker') in ['Betfair Exchange', 'Pinnacle', 'Bet365']:
                        mo = bm.get('markets', {}).get('match_odds', {})
                        if isinstance(mo, dict):
                            for side, key in [('home','market_h'),('draw','market_d'),('away','market_a')]:
                                sd = mo.get(side, {})
                                if isinstance(sd, dict):
                                    val = sd.get('last_seen') or sd.get('opening')
                                    if val:
                                        try: rec[key] = float(val)
                                        except: pass
                        break
    except:
        pass
    
    with lock:
        fixed[0] += 1
        if fixed[0] % 20 == 0:
            print(f"  Fixed {fixed[0]}/{len(to_fix)}", flush=True)
    
    # Update the row
    for k, v in rec.items():
        if v is not None and v != '':
            row[k] = v
    
    return row

# Fix in parallel
with ThreadPoolExecutor(max_workers=20) as ex:
    ex.map(fix_one, to_fix)

# Write back
with open(CSV_PATH, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

print(f"\nFixed: {fixed[0]} rows", flush=True)

# Verify Sweden vs Tunisia
for r in rows:
    if r['match_id'] == 'mt_209798753':
        print(f"\nSweden vs Tunisia now:")
        print(f"  xG={r['home_xg']} vs {r['away_xg']}")
        print(f"  possession={r['possession_h']}% vs {r['possession_a']}%")
        print(f"  odds={r['market_h']} / {r['market_d']} / {r['market_a']}")
        break
