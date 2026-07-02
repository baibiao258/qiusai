#!/usr/bin/env python3
"""Check which completed WC matches are missing from training data, and append them."""
import json, sys, os

# Load completed results
results = json.load(open('/root/data/wc_completed_results.json'))
print(f"✅ Completed results DB: {len(results)} total records")

# Load training data
training = json.load(open('/root/data/training_data_with_odds.json'))
print(f"📚 Training data: {len(training)} total records")

# Build seen set from training
seen = set()
for s in training:
    key = (s.get('date', ''), s.get('home_en', ''), s.get('away_en', ''))
    seen.add(key)

# Find missing completions
missing = []
for r in results:
    key = (r['date'], r['home'], r['away'])
    if key not in seen:
        missing.append(r)

print(f"\n🔍 Training has records for {len(seen)} matches")
print(f"🔍 Missing from training: {len(missing)} records")

if not missing:
    print("\n✅ All completed matches are already in training data — nothing to append.")
    sys.exit(0)

for r in missing:
    print(f"   📅 {r['date']}: {r['home']} {r['home_score']}-{r['away_score']} {r['away']}")

# Try to load odds from pred files to construct training records
print("\n--- Attempting to backfill missing records with odds from prediction files ---")
appended = 0
for r in missing:
    date = r['date']
    pred_path = f"/root/data/wc_pred_{date}.json"
    if not os.path.exists(pred_path):
        print(f"   ⚠ No pred file for {date}, skipping {r['home']} vs {r['away']}")
        continue
    
    try:
        preds = json.load(open(pred_path))
    except:
        print(f"   ⚠ Could not parse {pred_path}")
        continue
    
    # Find matching prediction
    match_pred = None
    for p in preds:
        if p.get('home', '') == r['home'] and p.get('away', '') == r['away']:
            match_pred = p
            break
    
    if match_pred is None:
        print(f"   ⚠ No prediction found for {r['home']} vs {r['away']} in {pred_path}")
        continue
    
    # Construct training record
    rec = {
        'date': r['date'],
        'home_en': r['home'],
        'away_en': r['away'],
        'tournament': 'FIFA World Cup',
        'spf_result': r['result'],
        'home_goals': r['home_score'],
        'away_goals': r['away_score'],
        'home_xg': 0.0,
        'away_xg': 0.0,
        'market_h': match_pred.get('odds_h', 0),
        'market_d': match_pred.get('odds_d', 0),
        'market_a': match_pred.get('odds_a', 0),
        'stage': 'group_stage',
        'source': 'theoddsapi',
    }
    
    # Compute implied probs
    oh, od, oa = rec['market_h'], rec['market_d'], rec['market_a']
    if all(x > 0 for x in [oh, od, oa]):
        margin = 1/oh + 1/od + 1/oa
        rec['market_implied_h'] = (1/oh) / margin
        rec['market_implied_d'] = (1/od) / margin
        rec['market_implied_a'] = (1/oa) / margin
    else:
        rec['market_implied_h'] = 0.0
        rec['market_implied_d'] = 0.0
        rec['market_implied_a'] = 0.0
    
    training.append(rec)
    seen.add((r['date'], r['home'], r['away']))
    appended += 1
    print(f"   ✅ Appended: {r['home']} {r['home_score']}-{r['away_score']} {r['away']} (odds: H={oh} D={od} A={oa})")

if appended > 0:
    json.dump(training, open('/root/data/training_data_with_odds.json', 'w'), indent=2)
    print(f"\n💾 Saved {appended} new records to training_data_with_odds.json")
    print(f"📚 Training data now has {len(training)} total records")
else:
    print("\n⚠ No new records appended (missing pred files or no matches found)")
