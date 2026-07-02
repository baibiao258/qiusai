import requests, json, os, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

THE_KEY = os.environ.get('THE_KEY', '') or os.environ.get('THE_STATS_KEY', '')
if not THE_KEY:
    print("NO KEY")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {THE_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

COMP_IDS = [
    ("comp_6107",  "FIFA World Cup"),
    ("comp_8973",  "WCQ AFC"),
    ("comp_5720",  "WCQ CAF"),
    ("comp_0836",  "WCQ CONCACAF"),
    ("comp_4682",  "WCQ CONMEBOL"),
    ("comp_7363",  "WCQ OFC"),
    ("comp_2954",  "WCQ UEFA"),
    ("comp_2949",  "EURO"),
    ("comp_3759",  "EURO Qual"),
    ("comp_5749",  "Copa America"),
    ("comp_574977","UEFA Nations League"),
    ("comp_193547","CONCACAF Nations League"),
    ("comp_1376",  "CONCACAF Gold Cup"),
    ("comp_1554",  "Africa Cup of Nations"),
    ("comp_83579", "Africa Cup of Nations Qual."),
    ("comp_29967", "International Friendly Games"),
    ("comp_920080","FIFA Series"),
]

lock = Lock()
all_matches = []
pages_done = 0

def fetch_page(cid, page):
    url = (f"{BASE}/matches?date_from=2024-01-01&date_to=2026-06-16"
           f"&status=finished&competition_id={cid}&page={page}&per_page=100")
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        matches = data.get("data", [])
        meta = data.get("meta", {})
        return {"matches": matches, "total_pages": meta.get("total_pages", 0),
                "total": meta.get("total", 0), "cid": cid, "page": page}
    except Exception as e:
        print(f"  ERROR {cid} p{page}: {e}")
        return None

# Phase 1: Get page 1 for all comps to discover counts
print("=== PHASE 1: Discovering match counts ===")
comp_info = {}
with ThreadPoolExecutor(max_workers=17) as ex:
    fut_map = {ex.submit(fetch_page, cid, 1): cid for cid, _ in COMP_IDS}
    for f in as_completed(fut_map):
        res = f.result()
        if res and res["matches"]:
            comp_info[res["cid"]] = {
                "total_pages": res["total_pages"],
                "total": res["total"],
                "page1": res["matches"]
            }
            with lock:
                all_matches.extend(res["matches"])
            total = res["total"]
            pages = res["total_pages"]
            print(f"  {res['cid']}: {total} matches, {pages} pages")

# Phase 2: Fetch remaining pages for all comps
print(f"\n=== PHASE 2: Fetching remaining ({len([1 for cid,_ in COMP_IDS if cid in comp_info])} comps) ===")
remaining = []
for cid, _ in COMP_IDS:
    if cid in comp_info:
        for p in range(2, comp_info[cid]["total_pages"] + 1):
            remaining.append((cid, p))

total_remaining = len(remaining)
print(f"Total remaining pages: {total_remaining}")

batch_size = 30
for i in range(0, len(remaining), batch_size):
    batch = remaining[i:i+batch_size]
    with ThreadPoolExecutor(max_workers=30) as ex:
        fut_map = {}
        for cid, p in batch:
            f = ex.submit(fetch_page, cid, p)
            fut_map[f] = (cid, p)
        for f in as_completed(fut_map):
            res = f.result()
            if res and res["matches"]:
                with lock:
                    all_matches.extend(res["matches"])
                    pages_done += 1
        time.sleep(0.05)
    
    if (i // batch_size) % 10 == 0:
        print(f"  {i+batch_size}/{total_remaining} pages | {len(all_matches)} matches")

print(f"\n=== DONE: {len(all_matches)} matches ===")

# Deduplicate by match_id
seen_ids = set()
unique = []
for m in all_matches:
    mid = m.get("id")
    if mid not in seen_ids:
        seen_ids.add(mid)
        unique.append(m)
print(f"After dedup: {len(unique)} unique matches")

# Save
outpath = "/root/wc_2026_upgrade/base_matches_thestats.json"
with open(outpath, "w") as f:
    json.dump(unique, f)
print(f"Saved to {outpath}")

# Comp breakdown
comp_counts = Counter(m.get("competition_id", "") for m in unique)
name_map = dict(COMP_IDS)
print("\nBreakdown:")
for cid, cnt in comp_counts.most_common():
    name = name_map.get(cid, cid)
    print(f"  {name:35s} ({cid:15s}): {cnt}")
