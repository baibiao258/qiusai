import requests, json, os, csv, time, sys
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# === CONFIG ===
THE_KEY = os.environ.get('THE_KEY', '')
if not THE_KEY:
    THE_KEY = os.environ.get('THE_STATS_KEY', '')
if not THE_KEY:
    print("ERROR: NEED API KEY in env THE_KEY or THE_STATS_KEY")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {THE_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

COMP_IDS = [
    "comp_6107",  # FIFA World Cup
    "comp_8973",  # WCQ AFC
    "comp_5720",  # WCQ CAF
    "comp_0836",  # WCQ CONCACAF
    "comp_4682",  # WCQ CONMEBOL
    "comp_7363",  # WCQ OFC
    "comp_2954",  # WCQ UEFA
    "comp_2949",  # EURO
    "comp_3759",  # EURO Qual
    "comp_5749",  # Copa America
    "comp_574977", # UEFA Nations League
    "comp_193547", # CONCACAF Nations League
    "comp_1376",  # CONCACAF Gold Cup
    "comp_1554",  # Africa Cup of Nations
    "comp_83579", # Africa Cup of Nations Qual.
    "comp_29967", # International Friendly Games
    "comp_920080", # FIFA Series
]

OUTPUT_CSV = "/root/wc_2026_upgrade/training_data_thestats.csv"
BASE_JSON = "/root/wc_2026_upgrade/base_matches_thestats.json"
MAX_PAGES = 500

# Stats
stats = {"total": 0, "stats_ok": 0, "odds_ok": 0, "written": 0, "skipped": 0, "errors": 0}
stats_lock = Lock()

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

# === PHASE 1: Fetch all base matches ===
def fetch_all_matches():
    log("=== PHASE 1: Fetching all base matches ===")
    
    if os.path.exists(BASE_JSON):
        with open(BASE_JSON) as f:
            all_matches = json.load(f)
        log(f"Loaded {len(all_matches)} matches from {BASE_JSON}")
        return all_matches
    
    all_matches = []
    for comp_id in COMP_IDS:
        page = 1
        while page <= MAX_PAGES:
            url = (f"{BASE}/matches?date_from=2024-01-01&date_to=2026-06-16"
                   f"&status=finished&competition_ids={comp_id}&page={page}&per_page=20")
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                if resp.status_code != 200:
                    log(f"  {comp_id} p{page}: HTTP {resp.status_code}")
                    break
                data = resp.json()
                matches = data.get('data', [])
                if not matches:
                    break
                all_matches.extend(matches)
                meta = data.get('meta', {})
                tp = meta.get('total_pages', 0)
                if page % 5 == 0 or page == 1:
                    log(f"  {comp_id}: p{page}/{tp} (total {len(all_matches)})")
                page += 1
                time.sleep(0.1)
            except Exception as e:
                log(f"  {comp_id} p{page}: {e}")
                time.sleep(2)
    
    log(f"Total base matches: {len(all_matches)}")
    with open(BASE_JSON, 'w') as f:
        json.dump(all_matches, f, indent=2)
    log(f"Saved to {BASE_JSON}")
    return all_matches

# === PHASE 2: Fetch details ===
def fetch_one(match_id, comp_id, utc_date):
    """Fetch stats + odds for one match, return flat dict"""
    rec = {"match_id": match_id, "competition_id": comp_id, "utc_date": utc_date}
    
    # Stats
    try:
        sr = requests.get(f"{BASE}/matches/{match_id}/stats", headers=HEADERS, timeout=30)
        if sr.status_code == 200:
            sdata = sr.json()
            if isinstance(sdata, dict):
                sd = sdata.get('data', sdata)
                ov = sd.get('overview', sd) if isinstance(sd, dict) else sd
                if isinstance(ov, dict):
                    # xG
                    xg_sec = ov.get('expected_goals', {})
                    if isinstance(xg_sec, dict):
                        xg_all = xg_sec.get('all', {})
                        if isinstance(xg_all, dict):
                            rec['home_xg'] = xg_all.get('home')
                            rec['away_xg'] = xg_all.get('away')
                    # possession
                    pos_sec = ov.get('ball_possession', {})
                    if isinstance(pos_sec, dict):
                        pos_all = pos_sec.get('all', {})
                        if isinstance(pos_all, dict):
                            rec['possession_h'] = pos_all.get('home')
                            rec['possession_a'] = pos_all.get('away')
                    # shots on target
                    sot_sec = ov.get('shots_on_target', {})
                    if isinstance(sot_sec, dict):
                        sot_all = sot_sec.get('all', {})
                        if isinstance(sot_all, dict):
                            rec['shots_ot_h'] = sot_all.get('home')
                            rec['shots_ot_a'] = sot_all.get('away')
                    # total shots
                    ts_sec = ov.get('total_shots', {})
                    if isinstance(ts_sec, dict):
                        ts_all = ts_sec.get('all', {})
                        if isinstance(ts_all, dict):
                            rec['total_shots_h'] = ts_all.get('home')
                            rec['total_shots_a'] = ts_all.get('away')
            with stats_lock:
                stats['stats_ok'] += 1
    except Exception as e:
        with stats_lock:
            stats['errors'] += 1
    
    time.sleep(0.05)
    
    # Odds
    try:
        or_ = requests.get(f"{BASE}/matches/{match_id}/odds", headers=HEADERS, timeout=30)
        if or_.status_code == 200:
            odata = or_.json()
            if isinstance(odata, dict):
                od = odata.get('data', odata)
                bms = od.get('bookmakers', []) if isinstance(od, dict) else []
                if bms:
                    # Prefer Betfair
                    target = None
                    for bm in bms:
                        if bm.get('bookmaker') == 'Betfair Exchange':
                            target = bm
                            break
                    if not target:
                        target = bms[0]
                    if target:
                        mkt = target.get('markets', {})
                        mo = mkt.get('match_odds', {})
                        if isinstance(mo, dict):
                            for side, key in [('home', 'market_h'), ('draw', 'market_d'), ('away', 'market_a')]:
                                sd = mo.get(side, {})
                                if isinstance(sd, dict):
                                    val = sd.get('last_seen') or sd.get('opening')
                                    if val:
                                        try:
                                            rec[key] = float(val)
                                        except:
                                            pass
            with stats_lock:
                stats['odds_ok'] += 1
    except Exception as e:
        with stats_lock:
            stats['errors'] += 1
    
    return rec

def process_matches(all_matches):
    log("=== PHASE 2: Fetching stats + odds ===")
    
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_ids.add(row.get('match_id', ''))
        log(f"Already processed: {len(processed_ids)}")
    
    to_do = [m for m in all_matches if m.get('id') not in processed_ids]
    log(f"Remaining: {len(to_do)}")
    
    fields = ['match_id', 'competition_id', 'utc_date',
              'home_team', 'away_team', 'home_score', 'away_score',
              'group_label', 'matchday', 'stage_name',
              'home_xg', 'away_xg', 'possession_h', 'possession_a',
              'shots_ot_h', 'shots_ot_a', 'total_shots_h', 'total_shots_a',
              'market_h', 'market_d', 'market_a']
    
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    
    total_batches = (len(to_do) + 9) // 10
    
    for i in range(0, len(to_do), 10):
        batch = to_do[i:i+10]
        batch_num = i // 10 + 1
        
        rows = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            fut_map = {}
            for m in batch:
                mid = m.get('id', '')
                cid = m.get('competition_id', '')
                ud = m.get('utc_date', '')
                f = ex.submit(fetch_one, mid, cid, ud)
                fut_map[f] = m
            
            for f in as_completed(fut_map):
                m = fut_map[f]
                rec = f.result()
                # Merge base
                ht = m.get('home_team', {})
                rec['home_team'] = ht.get('name', '') if isinstance(ht, dict) else str(ht)
                at = m.get('away_team', {})
                rec['away_team'] = at.get('name', '') if isinstance(at, dict) else str(at)
                sc = m.get('score', {})
                rec['home_score'] = sc.get('home', '') if isinstance(sc, dict) else str(m.get('home_score', ''))
                rec['away_score'] = sc.get('away', '') if isinstance(sc, dict) else str(m.get('away_score', ''))
                rec['group_label'] = m.get('group_label', '')
                rec['matchday'] = m.get('matchday', '')
                rec['stage_name'] = m.get('stage_name', '')
                rows.append(rec)
                
                with stats_lock:
                    stats['written'] += 1
        
        if rows:
            with open(OUTPUT_CSV, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                for r in rows:
                    clean = {k: r.get(k, '') for k in fields}
                    w.writerow(clean)
        
        if batch_num % 20 == 0 or batch_num == 1:
            with stats_lock:
                s = dict(stats)
            log(f"  Batch {batch_num}/{total_batches} | written={s['written']} "
                f"stats={s['stats_ok']} odds={s['odds_ok']} err={s['errors']}")
    
    log(f"\n=== COMPLETE ===")
    with stats_lock:
        s = dict(stats)
    log(f"Written: {s['written']} | Stats OK: {s['stats_ok']} | Odds OK: {s['odds_ok']} | Errors: {s['errors']}")
    
    # Sample
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            lines = f.readlines()
        log(f"\nCSV has {len(lines)-1} data rows")
        log(f"Columns: {lines[0].strip()}")
        for line in lines[1:4]:
            log(f"  {line.strip()}")

if __name__ == '__main__':
    all_matches = fetch_all_matches()
    process_matches(all_matches)
