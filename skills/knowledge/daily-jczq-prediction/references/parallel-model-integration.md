# Parallel Model Integration

## Overview

Running a second simpler XGB model alongside the main model provides:
1. Consensus signal (both agree = higher confidence)
2. Disagreement signal (models differ = caution)
3. Fallback validation (if main model fails, simple model provides backup)

## Model Architecture

### Main Model (existing)
- 29 features (b15 + gold + odds_feat + form_feat + xg_proxy)
- File: `xgb_model_29.pkl` + `calibrators.pkl`
- DC + XGB fusion with entropy-based dynamic weighting

### Simple Model (new)
- 7 features (market_odds + 6 form features)
- File: `xgb_model_simple.pkl` + `calibrators_simple.pkl`
- Direct market_odds + form prediction

## Feature Mapping

```python
# Simple model features (7-dim)
simple_feat = np.array([[
    market_odds,           # 1/op_h (Elo-implied odds)
    fh5[0], fh5[1], fh5[2],  # home form: win_rate, gf, ga
    fa5[0], fa5[1], fa5[2],  # away form: win_rate, gf, ga
]])
```

## Integration Points

### 1. Model Loading (`_load_shared_models()`)
```python
simple_model_path = os.path.join(DATA_DIR, 'xgb_model_simple.pkl')
simple_cal_path = os.path.join(DATA_DIR, 'calibrators_simple.pkl')
if os.path.exists(simple_model_path):
    _xgb_simple = joblib.load(simple_model_path)
if os.path.exists(simple_cal_path):
    _cal_simple = joblib.load(simple_cal_path)
```

### 2. Prediction (`_try_hybrid_predict()`)
```python
# After main model prediction
simple_pred = None
simple_conf = 0
if '_xgb_simple' in globals() and _xgb_simple is not None:
    try:
        market_odds_h = 1.0 / max(op_h, 0.01)
        simple_feat = np.array([[
            market_odds_h,
            fh5[0], fh5[1], fh5[2],
            fa5[0], fa5[1], fa5[2],
        ]])
        simple_proba = _xgb_simple.predict_proba(simple_feat)[0]
        # Apply calibration
        if '_cal_simple' in globals() and _cal_simple is not None:
            simple_cal = np.zeros(3)
            for j, key in enumerate(['home', 'draw', 'away']):
                if key in _cal_simple:
                    simple_cal[j] = _cal_simple[key].predict([simple_proba[j]])[0]
                else:
                    simple_cal[j] = simple_proba[j]
            s = simple_cal.sum()
            if s > 0: simple_cal /= s
            simple_proba = simple_cal
        simple_pred = ['H', 'D', 'A'][simple_proba.argmax()]
        simple_conf = simple_proba.max()
    except Exception as e:
        pass
```

### 3. Bundle Construction (`build_prediction_bundle()`)
```python
simple_pred = p.get('simple_pred', '')
simple_conf = p.get('simple_conf', 0)
# ... add to return dict
```

### 4. CSV Recording (`record_prediction()`)
```python
cmd += [
    '--simple-pred', str(bundle.get('simple_pred', '')),
    '--simple-conf', str(bundle.get('simple_conf', 0)),
]
```

### 5. Backtest Script (`backtest_jczq.py`)
- Add `simple_pred` and `simple_conf` to FIELDS list
- Add argument parsing in `cmd_record()`

## Consensus Analysis

```python
# After predictions are in CSV
main = 'H' if pred_h > pred_d and pred_h > pred_a else ('D' if pred_d > pred_h else 'A')
simple = row['simple_pred']
agree = main == simple

# Typical results:
# - Consensus rate: ~68%
# - Disagreement cases: ~32%
# - Simple model tends to favor away wins more than main model
```

## Performance (2026-06-09 test)

- Training data: 263 matches from `training_data_with_odds.json`
- CV accuracy: 59.4%
- Test accuracy (80/20 time-series split): 57.5%
- Test EV>5% bets: 44, Hit rate: 54.5%, ROI: +34.1%
- Feature importance: market_odds (0.289), form features (0.711 total)

## Pitfalls

1. **market_odds derivation**: Use `1/op_h` (Elo-implied), not raw 500.com odds, because `_try_hybrid_predict()` doesn't have access to market_row
2. **Calibration order**: Must calibrate AFTER main model calibration to ensure comparable probability scales
3. **CSV field naming**: Use `simple_pred`/`simple_conf` (with hyphens in CLI args, underscores in CSV)
4. **Empty values**: When simple_pred is None (model not loaded or prediction failed), CSV shows empty string
