# WC Results Backfill Fallback Procedure

## Problem

`check_training_gap.py` only backfills matches that have a corresponding entry in the `wc_pred_YYYY-MM-DD.json` file for the exact match date. However, The Odds API's `/odds/` endpoint returns a different subset of matchups each day. A match that completed on date X may never have appeared in that date's pred file.

## Detection

After running `check_training_gap.py`, check for skipped records:

```bash
python3 -c "
import json
results = json.load(open('/root/data/wc_completed_results.json'))
training = json.load(open('/root/data/training_data_with_odds.json'))
seen = set((s.get('date',''), s.get('home_en',''), s.get('away_en','')) for s in training)
missing = [r for r in results if (r['date'], r['home'], r['away']) not in seen]
print(f'Still missing from training: {len(missing)}')
for m in missing:
    print(f'  ⚠ {m[\"date\"]} {m[\"home\"]} {m[\"home_score\"]}-{m[\"away_score\"]} {m[\"away\"]}')
"
```

## Procedure: Cross-Date Odds Search + Manual Append

For each missing match, search all historical `wc_odds_*.json` files for the most recent odds before match date:

```bash
python3 -c "
import json, glob, os

missing_home_away = [('TeamA', 'TeamB')]  # Replace with actual missing pairs
for f in sorted(glob.glob('/root/data/wc_odds_*.json')):
    odds = json.load(open(f))
    date = os.path.basename(f).replace('wc_odds_','').replace('.json','')
    for h, a in missing_home_away:
        for o in odds:
            if o['home'] == h and o['away'] == a:
                print(f'{date}: {o[\"home\"]}:{o[\"away\"]} odds={o[\"odds_h\"]}/{o[\"odds_d\"]}/{o[\"odds_a\"]}')
"
```

Pick the most recent odds entry **before** the match date. Then construct the training record:

```python
import json

# Load existing training data
path = '/root/data/training_data_with_odds.json'
data = json.load(open(path))

# For each missing match, construct record with the last-known odds
rec = {
    'date': 'YYYY-MM-DD',           # match completion date
    'home_en': 'HomeTeam',
    'away_en': 'AwayTeam',
    'tournament': 'FIFA World Cup',
    'spf_result': 'H/D/A',          # from completed result
    'home_goals': N, 'away_goals': M,
    'home_xg': 0.0, 'away_xg': 0.0,
    'market_h': ODDS_H, 'market_d': ODDS_D, 'market_a': ODDS_A,
    'stage': 'group_stage',
    'source': 'theoddsapi_backfill',
}

# Compute implied probs
oh, od, oa = rec['market_h'], rec['market_d'], rec['market_a']
if all(x > 0 for x in [oh, od, oa]):
    margin = 1/oh + 1/od + 1/oa
    rec['market_implied_h'] = round((1/oh) / margin, 4)
    rec['market_implied_d'] = round((1/od) / margin, 4)
    rec['market_implied_a'] = round((1/oa) / margin, 4)

# Check for duplicates
seen = set((s.get('date',''), s.get('home_en',''), s.get('away_en','')) for s in data)
key = (rec['date'], rec['home_en'], rec['away_en'])
if key not in seen:
    data.append(rec)
    json.dump(data, open(path, 'w'), indent=2)
    print(f'✅ Added: {rec[\"home_en\"]} vs {rec[\"away_en\"]}')
```

## Historical Example (2026-06-24 Session)

3 matches could not be auto-backfilled because they weren't in the pred files on their completion date. Found via cross-date odds search and manually appended:

| Match | Score | Last Odds Before Match | Odds Source |
|-------|-------|----------------------|-------------|
| Jordan vs Algeria | 1-2 (A) | 6.10 / 4.20 / 1.54 | wc_odds_2026-06-22.json |
| Norway vs Senegal | 3-2 (H) | 2.16 / 3.45 / 3.35 | wc_odds_2026-06-22.json |
| Colombia vs DR Congo | 1-0 (H) | 1.52 / 3.80 / 6.50 | wc_odds_2026-06-23.json |

After backfill, training_data_with_odds.json grew from 2,464 → 2,470 records (6 new: 3 from check_training_gap.py + 3 manual), and all 46 completed results were covered.

## Why This Happens

The Odds API `/odds/` endpoint returns different matchup subsets on different days. A match that completes on day X may have last appeared in the odds data on day X-2 or X-1. The `/scores/?daysFrom=3` endpoint returns all completed matches regardless. The gap arises because:

1. `daily_wc_pipeline.py` saves odds for whatever the API returns today
2. The API gradually drops matches from its odds response as they get closer to kickoff (or as new matches get added)
3. By match day, the odds file may already have moved on to other upcoming matchups
4. The pred file (derived from odds) reflects whatever was in the API at pipeline runtime
