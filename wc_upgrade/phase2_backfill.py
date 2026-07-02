import requests, json, os, csv, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

THE_KEY = os.environ.get('THE_KEY', '') or os.environ.get('THE_STATS_KEY', '')
if not THE_KEY:
    print("NO KEY")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {THE_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"
INPUT_JSON = "/root/wc_2026_upgrade/base_matches_thestats.json"
OUTPUT_CSV = "/root/wc_2026_upgrade/training_data_thestats.csv"

with open(INPUT_JSON) as f:
    all_matches = json.load(f)
print(f"Loaded {len(all_matches)} matches")

done_ids = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            done_ids.add(row.get("match_id", ""))
    print(f"Already in CSV: {len(done_ids)}")

to_do = [m for m in all_matches if m.get("id") not in done_ids]
print(f"Remaining: {len(to_do)} matches", flush=True)

FIELDS = ["match_id","competition_id","utc_date","home_team","away_team",
          "home_score","away_score","group_label","matchday","stage_name",
          "home_xg","away_xg","possession_h","possession_a",
          "shots_ot_h","shots_ot_a","total_shots_h","total_shots_a",
          "market_h","market_d","market_a"]

lock = Lock()
cnt = {"done":0,"stats_ok":0,"odds_ok":0,"errors":0,"written":0}

def detail(mid):
    rec = {}
    try:
        sr = requests.get(f"{BASE}/matches/{mid}/stats", headers=HEADERS, timeout=30)
        if sr.status_code == 200:
            sd = sr.json()
            od = sd.get("data", sd) if isinstance(sd, dict) else sd
            ov = od.get("overview", od) if isinstance(od, dict) else od
            if isinstance(ov, dict):
                xg = ov.get("expected_goals", {})
                if isinstance(xg, dict) and isinstance(xg.get("all"), dict):
                    rec["home_xg"] = xg["all"].get("home")
                    rec["away_xg"] = xg["all"].get("away")
                pos = ov.get("ball_possession", {})
                if isinstance(pos, dict) and isinstance(pos.get("all"), dict):
                    rec["possession_h"] = pos["all"].get("home")
                    rec["possession_a"] = pos["all"].get("away")
                sot = ov.get("shots_on_target", {})
                if isinstance(sot, dict) and isinstance(sot.get("all"), dict):
                    rec["shots_ot_h"] = sot["all"].get("home")
                    rec["shots_ot_a"] = sot["all"].get("away")
                ts = ov.get("total_shots", {})
                if isinstance(ts, dict) and isinstance(ts.get("all"), dict):
                    rec["total_shots_h"] = ts["all"].get("home")
                    rec["total_shots_a"] = ts["all"].get("away")
            with lock:
                cnt["stats_ok"] += 1
        else:
            with lock:
                cnt["errors"] += 1
    except:
        with lock:
            cnt["errors"] += 1
    time.sleep(0.05)
    try:
        or_ = requests.get(f"{BASE}/matches/{mid}/odds", headers=HEADERS, timeout=30)
        if or_.status_code == 200:
            od2 = or_.json()
            if isinstance(od2, dict):
                d2 = od2.get("data", od2)
                bms = d2.get("bookmakers", []) if isinstance(d2, dict) else []
                if bms:
                    target = None
                    for bm in bms:
                        if bm.get("bookmaker") == "Betfair Exchange":
                            target = bm; break
                    if not target and bms:
                        target = bms[0]
                    if target:
                        mo = target.get("markets", {}).get("match_odds", {})
                        if isinstance(mo, dict):
                            for side, key in [("home","market_h"),("draw","market_d"),("away","market_a")]:
                                sd = mo.get(side, {})
                                if isinstance(sd, dict):
                                    val = sd.get("last_seen") or sd.get("opening")
                                    if val:
                                        try:
                                            rec[key] = float(val)
                                        except:
                                            pass
            with lock:
                cnt["odds_ok"] += 1
        else:
            with lock:
                cnt["errors"] += 1
    except:
        with lock:
            cnt["errors"] += 1
    return rec

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

for i in range(0, len(to_do), 20):
    batch = to_do[i:i+20]
    results = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        fm = {ex.submit(detail, m["id"]): m for m in batch}
        for f in as_completed(fm):
            m = fm[f]
            r = f.result()
            r["match_id"] = m["id"]
            r["competition_id"] = m.get("competition_id","")
            r["utc_date"] = m.get("utc_date","")
            ht = m.get("home_team",{})
            r["home_team"] = ht.get("name","") if isinstance(ht,dict) else str(ht)
            at = m.get("away_team",{})
            r["away_team"] = at.get("name","") if isinstance(at,dict) else str(at)
            sc = m.get("score",{})
            r["home_score"] = sc.get("home","") if isinstance(sc,dict) else ""
            r["away_score"] = sc.get("away","") if isinstance(sc,dict) else ""
            r["group_label"] = m.get("group_label","")
            r["matchday"] = m.get("matchday","")
            r["stage_name"] = m.get("stage_name","")
            results.append(r)
            with lock:
                cnt["done"] += 1
    if results:
        with open(OUTPUT_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            for r in results:
                w.writerow({k: r.get(k,"") for k in FIELDS})
        with lock:
            cnt["written"] += len(results)
    if (i//20+1) % 10 == 0 or i == 0:
        with lock:
            print(f"Batch {i//20+1}/{(len(to_do)+19)//20} | done={cnt['done']} stats={cnt['stats_ok']} odds={cnt['odds_ok']} err={cnt['errors']}", flush=True)

with lock:
    print(f"\nDONE: {cnt['done']} matches | stats={cnt['stats_ok']} odds={cnt['odds_ok']} err={cnt['errors']}", flush=True)

# Show sample
with open(OUTPUT_CSV) as f:
    lines = f.readlines()
print(f"CSV: {len(lines)} rows", flush=True)
for line in lines[:4]:
    print(f"  {line.strip()}", flush=True)
print(f"  ...", flush=True)
for line in lines[-3:]:
    print(f"  {line.strip()}", flush=True)
