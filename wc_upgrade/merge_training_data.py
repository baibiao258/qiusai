import json, csv, os
from datetime import datetime

# === Load existing training data ===
OLD_PATH = "/root/data/training_data_with_odds.json"
NEW_CSV = "/root/wc_2026_upgrade/training_data_thestats.csv"
OUT_PATH = "/root/data/training_data_with_odds.json"
BACKUP_PATH = "/root/data/training_data_with_odds.json.bak"

# Backup old
if os.path.exists(OLD_PATH):
    os.system(f"cp {OLD_PATH} {BACKUP_PATH}")
    print(f"Backed up old to {BACKUP_PATH}")

with open(OLD_PATH) as f:
    old = json.load(f)
print(f"Old data: {len(old)} matches")

# === Load new data ===
rows = []
with open(NEW_CSV) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Filter to only finished matches (home_score and away_score not empty)
finished = [r for r in rows if r.get('home_score','').strip() and r.get('away_score','').strip()]
print(f"New CSV total: {len(rows)}, finished with score: {len(finished)}")

# === Convert to training_data format ===
def spf_result(h, a):
    try:
        hs, as_ = int(h), int(a)
    except:
        return None
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"

def comp_name(cid):
    names = {
        "comp_6107": "FIFA World Cup",
        "comp_8973": "WCQ AFC",
        "comp_5720": "WCQ CAF",
        "comp_0836": "WCQ CONCACAF",
        "comp_4682": "WCQ CONMEBOL",
        "comp_7363": "WCQ OFC",
        "comp_2954": "WCQ UEFA",
        "comp_2949": "UEFA Euro",
        "comp_3759": "Euro Qual",
        "comp_5749": "Copa America",
        "comp_574977": "UEFA Nations League",
        "comp_193547": "CONCACAF Nations League",
        "comp_1376": "CONCACAF Gold Cup",
        "comp_1554": "Africa Cup of Nations",
        "comp_83579": "Africa Cup of Nations Qual.",
        "comp_29967": "International Friendly",
        "comp_920080": "FIFA Series",
    }
    return names.get(cid, cid)

new_records = []
for r in rows:
    # Basic fields
    spf = spf_result(r.get('home_score',''), r.get('away_score',''))
    if spf is None:
        continue  # skip unfinished
    
    # Date: utc_date format 2026-06-15T02:00:00.000Z
    dt_str = r.get('utc_date', '')[:10]
    
    # Market odds
    market_h = r.get('market_h', '')
    market_d = r.get('market_d', '')
    market_a = r.get('market_a', '')
    
    # Compute market implied probabilities
    try:
        mh = float(market_h) if market_h else None
        md = float(market_d) if market_d else None
        ma = float(market_a) if market_a else None
    except:
        mh = md = ma = None
    
    if mh and md and ma:
        margin = 1/mh + 1/md + 1/ma
        implied_h = (1/mh) / margin
        implied_d = (1/md) / margin
        implied_a = (1/ma) / margin
    else:
        implied_h = implied_d = implied_a = None
    
    # xG
    try:
        hxg = float(r.get('home_xg', '') or 0)
        axg = float(r.get('away_xg', '') or 0)
    except:
        hxg = axg = 0
    
    record = {
        "date": dt_str,
        "home_en": r.get('home_team', ''),
        "away_en": r.get('away_team', ''),
        "tournament": comp_name(r.get('competition_id', '')),
        "spf_result": spf,
        "home_goals": int(r.get('home_score', 0)),
        "away_goals": int(r.get('away_score', 0)),
        "home_xg": hxg,
        "away_xg": axg,
        "possession_h": r.get('possession_h', ''),
        "possession_a": r.get('possession_a', ''),
        "shots_ot_h": r.get('shots_ot_h', ''),
        "shots_ot_a": r.get('shots_ot_a', ''),
        "total_shots_h": r.get('total_shots_h', ''),
        "total_shots_a": r.get('total_shots_a', ''),
        "market_h": mh,
        "market_d": md,
        "market_a": ma,
        "market_implied_h": implied_h,
        "market_implied_d": implied_d,
        "market_implied_a": implied_a,
        "stage": r.get('stage_name', ''),
        "group_label": r.get('group_label', ''),
        "matchday": r.get('matchday', ''),
        "source": "thestatsapi",
    }
    new_records.append(record)

print(f"Converted new records: {len(new_records)}")

# === Merge with old data ===
# Dedup by (date, home_en, away_en)
seen = set()
merged = []

# Add old records first (they have more features)
for rec in old:
    key = (rec.get('date','')[:10], rec.get('home_en',''), rec.get('away_en',''))
    if key not in seen:
        seen.add(key)
        merged.append(rec)

overlap_count = 0
new_count = 0
for rec in new_records:
    key = (rec['date'], rec['home_en'], rec['away_en'])
    if key not in seen:
        seen.add(key)
        merged.append(rec)
        new_count += 1
    else:
        overlap_count += 1
        
        # Merge in xG data if old record doesn't have it
        for existing in merged:
            ek = (existing.get('date','')[:10], existing.get('home_en',''), existing.get('away_en',''))
            if ek == key:
                # Add xG if old record doesn't have it
                if 'home_xg' not in existing or not existing['home_xg']:
                    existing['home_xg'] = rec.get('home_xg')
                    existing['away_xg'] = rec.get('away_xg')
                    existing['possession_h'] = rec.get('possession_h')
                    existing['possession_a'] = rec.get('possession_a')
                    existing['shots_ot_h'] = rec.get('shots_ot_h')
                    existing['shots_ot_a'] = rec.get('shots_ot_a')
                    existing['total_shots_h'] = rec.get('total_shots_h')
                    existing['total_shots_a'] = rec.get('total_shots_a')
                    existing['market_h'] = rec.get('market_h') or existing.get('market_h')
                    existing['market_d'] = rec.get('market_d') or existing.get('market_d')
                    existing['market_a'] = rec.get('market_a') or existing.get('market_a')
                    existing['market_implied_h'] = rec.get('market_implied_h') or existing.get('market_implied_h')
                    existing['market_implied_d'] = rec.get('market_implied_d') or existing.get('market_implied_d')
                    existing['market_implied_a'] = rec.get('market_implied_a') or existing.get('market_implied_a')
                break
        overlap_count -= 1  # already counted in the for loop above
        # Actually let me redo this properly
        continue

# Redo: properly merge
seen = set()
merged = []
new_from_thestats = 0
filled_xg = 0

for rec in old:
    key = (str(rec.get('date',''))[:10], str(rec.get('home_en','')), str(rec.get('away_en','')))
    seen.add(key)
    merged.append(rec)

# Build lookup
new_by_key = {}
for rec in new_records:
    key = (rec['date'], rec['home_en'], rec['away_en'])
    new_by_key[key] = rec

for rec in new_records:
    key = (rec['date'], rec['home_en'], rec['away_en'])
    if key not in [k for k in seen]:  # wrong way to check
        pass
    
# Cleaner approach
seen = set()
merged = []
for rec in old:
    key = (str(rec.get('date',''))[:10], str(rec.get('home_en','')), str(rec.get('away_en','')))
    seen.add(key)
    merged.append(rec)

thestats_added = 0
xg_merged = 0

for rec in new_records:
    key = (rec['date'], rec['home_en'], rec['away_en'])
    if key in seen:
        # Merge xG into existing record
        for existing in merged:
            ek = (str(existing.get('date',''))[:10], str(existing.get('home_en','')), str(existing.get('away_en','')))
            if ek == key:
                if rec.get('home_xg', 0) > 0:
                    existing['home_xg'] = rec['home_xg']
                    existing['away_xg'] = rec['away_xg']
                    xg_merged += 1
                # Merge possession/shots
                if rec.get('possession_h', ''):
                    existing['possession_h'] = rec['possession_h']
                    existing['possession_a'] = rec['possession_a']
                if rec.get('shots_ot_h', ''):
                    existing['shots_ot_h'] = rec['shots_ot_h']
                    existing['shots_ot_a'] = rec['shots_ot_a']
                break
    else:
        seen.add(key)
        merged.append(rec)
        thestats_added += 1

# Sort by date
merged.sort(key=lambda x: str(x.get('date', '')))

print(f"\n=== Merge Summary ===")
print(f"Old records: {len(old)}")
print(f"TheStatsAPI converted: {len(new_records)}")
print(f"  - New unique added: {thestats_added}")
print(f"  - Overlap (xG merged into old): {xg_merged}")
print(f"Merged total: {len(merged)}")
print(f"Date range: {merged[0]['date'][:10]} ~ {merged[-1]['date'][:10]}")

# Year breakdown
from collections import Counter
years = Counter()
for rec in merged:
    y = rec.get('date','')[:4]
    years[y] += 1
print(f"\nBy year:")
for y in sorted(years):
    print(f"  {y}: {years[y]}")

# xG coverage
has_xg = sum(1 for rec in merged if rec.get('home_xg', 0) > 0)
has_odds = sum(1 for rec in merged if rec.get('market_h'))
print(f"\nxG coverage: {has_xg}/{len(merged)} ({has_xg/len(merged)*100:.1f}%)")
print(f"Odds coverage: {has_odds}/{len(merged)} ({has_odds/len(merged)*100:.1f}%)")

# Save
with open(OUT_PATH, 'w') as f:
    json.dump(merged, f, indent=2)
print(f"\nSaved to {OUT_PATH}")

# Sample
print(f"\n=== Sample records ===")
for rec in merged[:3]:
    d = {k: v for k, v in rec.items() if v}
    print(f"  {json.dumps(d, ensure_ascii=False)[:200]}")
