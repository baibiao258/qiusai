# Data Chain Audit Methodology

## When to Use

When a system component isn't performing as expected (calibration degradation, feature not working, data not flowing) and you need to find the **actual** bottleneck — not the perceived one.

## The Audit Protocol (5 Steps)

### Step 1: List ALL data files with their time ranges

```bash
# Data files
ls -la /root/data/training_data_with_odds.json  # 263 matches, 2024-01 to 2024-11
ls -la /root/data/international_results.json     # 49,409 matches, 1872-2026-06-12
ls -la /root/data/365scores/football_games.csv    # 565 rows, 2026-06-05 to 2026-06-07
ls -la /root/data/historical_kaijiang.csv         # 3248 matches, 2024-01 to 2026-06

# Model files
ls -la /root/data/xgb_model_29.pkl                # Production model
ls -la /root/data/xgb_model_30.pkl                # Shadow model (market odds)
ls -la /root/data/xgb_model_33.pkl                # Shadow model (market+stage)
```

Key insight: **check time windows overlap**. If data source A ends at 2024-11 and data source B starts at 2026-06, they have 0 matches in common regardless of code correctness.

### Step 2: Trace what script trains each model (never assume by name)

```bash
grep -rn 'xgb_model_29' . --include='*.py' | head -10
```

**Critical distinction found in June 2024 audit:**
- `wc_2026_final.py` → `xgb_model_29.pkl` (using `international_results.json`, 1872-2026)
- `retrain_xgb_with_odds.py` → `xgb_model_30/33.pkl` (using `training_data_with_odds.json`, 2024 only)

These are **two parallel pipelines** feeding different model files. Don't assume a model was trained by a particular script just because you found a script that references its filename.

Always print the training script path by grepping `joblib.dump('xgb_model_29'` across all `.py` files.

### Step 3: Box-check each data source independently

Test each source in isolation before diagnosing the join/cross-reference:

```bash
# Source 1: 365scores vote data
python3 -c "from collect_365scores_daily import *; tg=get_today_games(); print(f'Today games: {len(tg)}, with_votes={sum(1 for g in tg if g.get(\"vote_count\",0)>0)}')"

# Source 2: football_games.csv
python3 -c "import csv; rows=list(csv.DictReader(open('/root/data/365scores/football_games.csv'))); print(f'Football CSV: {len(rows)} rows, vote_count>0: {sum(1 for r in rows if int(r.get(\"vote_count\",\"0\"))>0)}')"

# Source 3: training_data_with_odds.json date range
python3 -c "import json; d=json.load(open('/root/data/training_data_with_odds.json')); dates=[m['date'] for m in d]; print(f'Training data: {min(dates)} to {max(dates)}, n={len(dates)}')"

# Source 4: International results
python3 -c "import json; d=json.load(open('/root/data/international_results.json')); dates=[m['date'] for m in d]; print(f'Intl results: {min(dates)} to {max(dates)}, n={len(dates)}')"
```

### Step 4: Cross-reference the time windows

The bottleneck is almost always a **time window gap**, not a code bug:

| Symptom | Common Root Cause | Fix |
|---------|-------------------|-----|
| "365scores features only 5/157 valid" | training_data ends 2024-11, 365scores starts 2026-06 → **0 overlap**, not a mapping bug | Wait for overlap or re-download older 365scores data |
| "vote_count always 0" | Only finished matches have votes, unfinished matches show 0 | Filter `IsFinished == True` before checking |
| "Backfilled 0 results" | target date data not yet available from any source | Expand source range or wait |
| "Brier drift detected" | Model trained on 2024 data predicting 2026 | Retrain with latest data (Step 5) |

### Step 5: Fix — simplest fix first

**Rule**: before building new infrastructure, try the simplest fix.

```
1. Re-download from original source (martj42/international_results, GitHub, etc.)
2. Run the original training script with `--no-mc --no-odds`
3. Only then consider building a new pipeline
```

**Example fix** (June 2026):
```bash
# Step 1: Download latest international results from GitHub
python3 -c "
import requests, json, csv
url = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
resp = requests.get(url, timeout=30)
# ... parse and save to international_results.json
"
# Step 2: Backup old models
cp /root/data/xgb_model_29.pkl /root/data/xgb_model_29.pkl.bak
# Step 3: Retrain
python3 /root/wc_2026_final.py --no-mc --no-odds
```

Result: 15-minute fix vs. building a new training data pipeline that would take days.

## Applied Examples

### Example 1: "365scores integration not working" (2026-06-14)

**Perceived problem**: 5/157 valid matches → "365scores mapping must be broken"
**Step 1 audit**: training_data_with_odds.json (2024-01 to 2024-11), football_games.csv (2026-06-05 to 2026-06-07)
**Actual root cause**: **Zero time window overlap** between training data and 365scores data
**Fix**: None needed — wait until ~June 28 for enough overlap, then join

### Example 2: "Model Brier degraded" (2026-06-14)

**Perceived problem**: Calibrator must be broken
**Step 1 audit**: xgb_model_29.pkl mtime vs training data date range
**Step 2 trace**: grep found wc_2026_final.py trains xgb_model_29.pkl → checked international_results.json → **stopped at 2026-03-31**
**Actual root cause**: Training data 3 months stale, missing 2026 matches
**Fix**: Re-download international_results.json → retrain → Brier back to 0.1453

## Checklist Format

For quick diagnostic runs:

```bash
echo "=== STEP 1: Data time ranges ==="
python3 -c "import json; d=json.load(open('/root/data/training_data_with_odds.json')); print(f'training_data_with_odds: {min(m[\"date\"])} to {max(m[\"date\"])}')"
python3 -c "import json; d=json.load(open('/root/data/international_results.json')); print(f'international_results: {min(m[\"date\"])} to {max(m[\"date\"])}')"
echo "=== STEP 2: Model training source ==="
grep -rn 'joblib.dump.*xgb_model_29' . --include='*.py'
echo "=== STEP 3: Model freshness ==="
ls -la /root/data/xgb_model_29.pkl /root/data/xgb_model_30.pkl /root/data/xgb_model_33.pkl
echo "=== STEP 4: Backfill status ==="
python3 /root/backfill_results.py --stats 2>/dev/null | head -20
```
