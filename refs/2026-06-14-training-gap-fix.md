# Training Data Gap Fix (2026-06-14)

## Problem

User perceived the bottleneck as "365scores features not yet in model" (5/157 valid matches). Investigation showed the real bottleneck was different:

| Perceived Problem | Actual Root Cause |
|-------------------|-------------------|
| 365scores data has 5/157 valid matches | **training_data_with_odds.json stops at 2024-11**, 365scores data starts 2026-06 → 0 overlap |
| vote_count = 0? | 100% of finished matches have votes → not the issue |
| Need Playwright for 365scores? | SPA issue applies to web UI, not REST API endpoint |
| 76% non-football in CSV | ✅ Fixed with SID=1 filter |

## Discovery Method

1. **Check EVERY link in the data chain**: source → training data → model → predictions
2. **Timestamp each component**: training_data_with_odds.json (2024-11), international_results.json (2026-03-31), football_games.csv (2026-06-05)
3. **Identify the actual gap**: training_data_with_odds.json only had 263 matches, all from 2024. But `xgb_model_29.pkl` was trained by `wc_2026_final.py` which uses `international_results.json`, not `training_data_with_odds.json` — two different pipelines!
4. **Test the simplest fix first**: Re-download `international_results.json` from the martj42 GitHub repo

## Fix Pipeline

```bash
# Step 1: Update training data from GitHub source
python3 -c "
import requests, json, csv
url = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
resp = requests.get(url, headers={'User-Agent': 'wc_predictor/1.0'}, timeout=30)
rows = list(csv.DictReader(resp.text.splitlines()))
matches = [{'date': r['date'], 'home': r['home_team'], 'away': r['away_team'],
            'tournament': r['tournament'], 'h_score': int(r['home_score']),
            'a_score': int(r['away_score']),
            'neutral': r.get('neutral','').strip().lower() == 'true'}
           for r in rows if r.get('home_score','').isdigit() and r.get('away_score','').isdigit()]
with open('/root/data/international_results.json', 'w') as f:
    json.dump(matches, f)
"

# Step 2: Backup old model
cp /root/data/xgb_model_29.pkl /root/data/xgb_model_29.pkl.bak
cp /root/data/dc_model.pkl /root/data/dc_model.pkl.bak

# Step 3: Retrain (skip MC for speed)
python3 /root/wc_2026_final.py --no-mc --no-odds
```

## Validation

- Before: Brier ~0.2465 (from backfill report, 96 samples)
- After retrain: Val Brier=0.1453, 2022 WC backtest Brier=0.1795
- All models updated: xgb_model_29.pkl, dc_model.pkl, elo_ratings.pkl, calibrators.pkl

## Key Insight

**The training data bottleneck wasn't where it first appeared.** Multiple pipelines exist:
- `wc_2026_final.py` trains xgb_model_29.pkl from `international_results.json` (up to 2026-03-31) 
- `prepare_training_data.py` → `training_data_with_odds.json` (only 2024, used by `retrain_xgb_with_odds.py`)
- These are DIFFERENT pipelines! Don't assume a model file was trained by the retrain script.

Always check: what script actually wrote the `.pkl` file?
```bash
grep -rn 'xgb_model_29' . --include='*.py'
# → ./wc_2026_final.py:891: joblib.dump(xgb_model, ...)
```

## 365scores Feature Integration Setup

`/root/scripts/build_training_with_365scores.py` — reads football_games.csv, joins 10-dim features to training data. Current 0 matches (data windows don't overlap). Wait until ~June 28 then:
```bash
python3 scripts/build_training_with_365scores.py --min-overlap 200
```

## Daily Monitoring

```bash
python3 /root/backfill_results.py --report
```
Shows: Brier trend per day, drift detection, league performance grades, unfilled count.
