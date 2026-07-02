# System Architecture Audit (2026-06-10)

## Architecture Overview

```
Data Collection Layer:
  500.com ──────→ 5-play odds + analysis page (FIFA rank, future fixtures, form)
  football-data.org → 9 league schedules + historical results
  365scores ────→ Public vote, recent trend, FIFA rank, popularity

Model Layer (4-way cascade, step-down degradation):
  Club DC+XGB (37-dim) → Intl DC+XGB (29-dim) → Poisson+Elo → Market Fallback
                           ↑
                     Main production path (SPF/RQ/HTFT/Score/Goals 5-play)
                           ↓
  A/B shadow XGB30 (with market_implied) → pred30_h/d/a written to CSV

Fusion Layer:
  Entropy-based dynamic weight (α=0.30, β=0.50)
  Isotonic calibrator (3: home/draw/away)
  Market weight fusion (mc_market_weight_helper)
  365scores posterior adjustment (±5pp)
  Fatigue adjustment (±5pp)

Persistence Layer:
  predictions_log.csv (85 records as of 2026-06-10)
  backfill_results.py (multi-source result backfill + Brier auto-calc)
  500breaker.log (circuit breaker isolation)
```

## Critical Findings

### P0: Zero Draw Predictions
Out of 32 backfilled matches, **0 predicted draws**. The 3-class model degenerated to 2-class (H/A only). In real football ~25% of matches are draws. This is a structural defect:
- DC's independent Poisson assumption under-predicts draws (no correlation term)
- Elo's binary win/loss bias compresses draw probability
- Training data may have insufficient draw samples

**Diagnostic:**
```python
import csv
rows = list(csv.DictReader(open('/root/data/predictions_log.csv')))
draw_predicted = sum(1 for r in rows if float(r.get('pred_d',0)) > float(r.get('pred_h',0)) and float(r.get('pred_d',0)) > float(r.get('pred_a',0)))
total = len(rows)
print(f"Draw predictions: {draw_predicted}/{total} = {draw_predicted/max(total,1)*100:.1f}%")
```

### P0: Isotonic Calibrator Makes Production Worse
Severe miscalibration across all decile bins (calib_error of +20-75pp in most bins). The earlier diagnosis ("raw prob Brier=0.2053 better than any calibrator") was based on validation data, but the Isotonic calibrator is STILL applied in production:
- `_try_hybrid_predict()` applies `calibrators['away']['draw']['home'].predict()` after fusion
- `_try_club_predict()` does the same with `_calibrators_club`

**Production Brier by decile (32 matches, 2026-06-10):**
| Predicted H range | n | Actual H | Calib Error |
|---|---|---|---|
| 0-10% | 2 | 50% | +45pp |
| 10-20% | 5 | 60% | +45pp |
| 30-40% | 7 | 71% | +36pp |
| 40-50% | 9 | 33% | -12pp |
| 50-60% | 2 | 100% | +45pp |
| 60-70% | 4 | 75% | +10pp |
| 70-80% | 2 | 100% | +25pp |

### P1: Brier Degradation +14% (Train-Serve Skew)
- Training: 0.2053 (6046 matches, 2023-2025, mixed types)
- Production: 0.2341 (32 matches, 2026, all friendlies)
- Degradation: +0.0288 (+14.0%)

**When predicted A (away win, 11/32):** Accuracy only 18.2%. The model predicts away wins confidently but is wrong 82% of the time.

### P1: Club DC+XGB Pathway Permanently Dead
`_try_club_predict()` always returns None despite model files existing:
- `xgb_model_club.pkl` (1MB, 37-dim)
- `dc_model_club.pkl` (7KB)
- `elo_club.pkl` (7KB)

Likely root causes (needs investigation):
1. `form_club.json` may not exist or be empty
2. `xg_proxy_club.json` may not exist or be empty
3. `_load_club_models()` guard clause: `if not all(os.path.exists(p) for p in REQUIRED_PATHS): return`
4. H2H cache `h2h_cache_club.json` may be missing

### P2: 365scores Features Collected But Never Trained
14 features (FIFA rank, trend win rate, popularity diff, vote fusion, etc.) are collected from 365scores and stored in CSV under `s365_*` prefix, but never incorporated into the XGBoost feature vector. These have significant independent predictive value for international friendlies where DC/form data is weak.

### P2: Fatigue Adjustment = ±5pp Placebo
`fatigue_adjustment()` caps at 5pp max movement, only triggers when `rotation_diff >= 0.1`. For World Cup friendlies where rotation risk can be 95%, this is far too conservative. The adjustment doesn't match the actual impact magnitude.

### P2: Tiny Sample + No Confidence Intervals
32 backfilled matches → Brier=0.2341, but std=0.1261. A single outlier (Qatar vs Switzerland Brier=0.53) shifts the mean by 4pp. Without confidence intervals, "model improvement" vs "sampling luck" is indistinguishable.

### P3: model_route Field Empty for All Records
All 32 backfilled records have `model_route=''` empty. Cannot drill down Brier by model path (hybrid vs market_fallback). This is a CSV field-writing compatibility issue from when `record_prediction()` was extended.

### P3: Multiple Redundant Model Files
- `xgb_model_20_3.pkl` (1.2MB) — old, unused
- `xgb_model_29.pkl.bak` — backup, unneeded
- `xgb_model_simple.pkl` — 6-dim toy model, unclear if actively used
- Multiple backtest JSONs: `strict_backtest_2022.json`, `optuna_backtest.json`, etc.

## Improvement Roadmap

### Immediate (this week)
1. **Disable Isotonic calibration** — Bypass `_calibrators` in `_try_hybrid_predict()`, output raw DC+XGB fusion. Compare Brier against 0.2341 baseline.
2. **Fix club pathway** — Diagnose why `_try_club_predict()` returns None (check form_club.json, xg_proxy_club.json, h2h_cache_club.json existence).
3. **Write model_route** — Ensure `record_prediction()` actually writes the model route field.

### Short-term (2-4 weeks)
4. **Incorporate 365scores features into XGB** — Start with FIFA rank, trend win rate, vote deviation as 3 new features (29→32 dim). Train xgb_model_32.pkl, run as A/B shadow.
5. **Automated retraining pipeline** — When >100 backfilled records: auto-load all xgb_model_{N}.pkl, compare Brier on production data, promote the best.
6. **Draw correction** — Post-fusion step that boosts draw probability toward empirical baseline (~25%) using entropy-scaled allocation.

### Long-term (1-2 months)
7. **Lineup data integration** — Flashscore/SofaScore API for predicted lineups. The `has_lineups` field from 365scores is only reliable <24h before kickoff.
8. **Pure 2026 retraining** — At 200+ backfilled records: train from scratch on 2026-only data, no old calibrator, no old data skew.
9. **Rolling time-series validation** — On accumulated backfill data, do rolling-window validation per model version to measure degradation curve.

## Diagnostic Commands

### Full Brier Analysis
```bash
python3 -c '
import csv
rows = list(csv.DictReader(open("/root/data/predictions_log.csv")))
brier_matches = []
for r in rows:
    b = r.get("brier_spf","").strip()
    if not b: continue
    score = r.get("actual_score","").strip()
    if not score: continue
    parts = score.split(":")
    if len(parts) != 2: continue
    try: hg, ag = int(parts[0]), int(parts[1])
    except: continue
    actual = "H" if hg > ag else ("D" if hg == ag else "A")
    pred_h, pred_d, pred_a = float(r.get("pred_h",0)), float(r.get("pred_d",0)), float(r.get("pred_a",0))
    predicted = "H" if pred_h > pred_d and pred_h > pred_a else ("D" if pred_d > pred_a else "A")
    brier_matches.append((r["code"], r["home_cn"], r["away_cn"], float(b), actual, predicted, pred_h, pred_d, pred_a))
print(f"Total: {len(brier_matches)} Mean Brier: {sum(r[3] for r in brier_matches)/len(brier_matches):.4f}")
correct = sum(1 for r in brier_matches if r[4]==r[5])
print(f"Accuracy: {correct}/{len(brier_matches)} = {correct/len(brier_matches)*100:.1f}%")
for label in ["H","D","A"]:
    subset = [r for r in brier_matches if r[5]==label]
    if subset:
        avg = sum(r[3] for r in subset)/len(subset)
        corr = sum(1 for r in subset if r[4]==r[5])
        print(f"  Pred={label}: n={len(subset)} brier={avg:.4f} acc={corr/len(subset)*100:.1f}%")
'
```

### Calibration Decile Check
```bash
python3 -c '
import csv
rows = list(csv.DictReader(open("/root/data/predictions_log.csv")))
brier = [(float(r.get("brier_spf",0)), float(r.get("pred_h",0))/100, float(r.get("pred_d",0))/100, float(r.get("pred_a",0))/100, r.get("actual_score","")) for r in rows if r.get("brier_spf","").strip()]
results = []
for b, ph, pd_, pa, score in brier:
    parts = score.split(":")
    if len(parts)==2:
        hg, ag = int(parts[0]), int(parts[1])
        actual_h = 1 if hg>ag else 0
        results.append((ph, actual_h))
for dec in range(10):
    lo, hi = dec*0.1, (dec+1)*0.1
    sub = [r for r in results if lo <= r[0] < hi]
    if sub:
        actual_rate = sum(r[1] for r in sub)/len(sub)
        print(f"  [{lo:.0%}-{hi:.0%}] n={len(sub)} actual_H={actual_rate:.0%} calib_err={actual_rate-(lo+0.05):+.0%}")
'
```

### Model File Inspection
```bash
python3 -c "import joblib; m = joblib.load('/root/data/xgb_model_29.pkl'); print(f'n_estimators={m.n_estimators}, max_depth={m.max_depth}, features={m.get_booster().feature_names}')"
python3 -c "import joblib; m = joblib.load('/root/data/xgb_model_30.pkl'); print(f'n_estimators={m.n_estimators}, max_depth={m.max_depth}, features={m.get_booster().feature_names}')"
python3 -c "import joblib; c = joblib.load('/root/data/calibrators.pkl'); print(f'Calibrators: {list(c.keys())}')"
```

### Club Pathway Diagnostic
```bash
python3 -c "
import os
for f in ['/root/data/dc_model_club.pkl', '/root/data/xgb_model_club.pkl', '/root/data/elo_club.pkl',
           '/root/data/calibrators_club.pkl', '/root/data/form_club.json', '/root/data/xg_proxy_club.json',
           '/root/data/h2h_cache_club.json']:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f'{os.path.basename(f)}: exists={exists} size={size}')
"
```

### Redundant File Cleanup Candidates
```bash
ls -la /root/data/xgb_model_20_3.pkl /root/data/xgb_model_29.pkl.bak /root/data/optuna_backtest.json /root/data/strict_backtest_2022*.json
```
