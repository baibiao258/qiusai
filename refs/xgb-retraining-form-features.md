# XGB Retraining with Form Features

## Overview
XGBoost model retrained with market odds + form features from 365scores API.

## Training Data

### Source
`/root/data/training_data_with_odds.json` (263 matches)

### Fields
- `home_en`, `away_en` (team names)
- `spf_result` (3/1/0 format)
- `market_odds` (market odds)
- `form_home_*`, `form_away_*` (form features from `form_state.json`)

### Generation Script
`/root/build_training_data.py`

```bash
python3 /root/build_training_data.py
# Output: /root/data/training_data.csv (263 rows, 7 features + label)
```

## Features (7 dimensions)
1. `market_odds` — market odds (strongest predictor, 28.9% importance)
2. `form_home_win` — home team win rate (last 5 matches)
3. `form_home_gf` — home team avg goals for
4. `form_home_ga` — home team avg goals against
5. `form_away_win` — away team win rate
6. `form_away_gf` — away team avg goals for
7. `form_away_ga` — away team avg goals against

## Model Training

### Script
`/root/retrain_xgb_simple.py`

```bash
python3 /root/retrain_xgb_simple.py
```

### Parameters
- n_estimators: 200
- max_depth: 3
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8

### Output
- Model: `/root/data/xgb_model_simple.pkl`
- Calibrators: `/root/data/calibrators_simple.pkl`

## Performance

### Time-Series Holdout (80/20 split)
- Train: 187 rows, Test: 47 rows
- Test date range: 2024-10-11 ~ 2024-11-15

### Results
- CV accuracy: 59.4%
- Test accuracy: 57.5%
- Test EV>5% bets: 44
- Hit rate: 54.5%
- ROI: +34.1%

### Feature Importance
- `market_odds`: 0.289
- `form_home_gf`: 0.137
- `form_away_ga`: 0.133
- `form_home_ga`: 0.117
- `form_away_gf`: 0.111
- `form_home_win`: 0.109
- `form_away_win`: 0.105

## Label Mapping
```python
# Original: 3 (home win), 1 (draw), 0 (away win)
# Mapped:   0 (home),    1 (draw),  2 (away)
label_map = {'3': 0, '1': 1, '0': 2}
```

## Verification
```bash
# Check model exists
ls -la /root/data/xgb_model_simple.pkl

# Check calibrators
ls -la /root/data/calibrators_simple.pkl

# Quick test
python3 -c "import pickle; m=pickle.load(open('/root/data/xgb_model_simple.pkl','rb')); print(f'Features: {m.n_features_in_}')"
```

## Integration with daily_jczq.py
To use the new model in production, update `_load_shared_models()` to load `xgb_model_simple.pkl` instead of `xgb_model_29.pkl`.

## Limitations
- Small training set (263 matches)
- Only international matches (no club data)
- Form features depend on `form_state.json` quality
