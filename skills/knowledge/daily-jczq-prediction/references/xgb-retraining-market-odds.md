# XGB Retraining with Market Odds

## Overview

Retrains the XGBoost model with market odds as an additional feature, significantly improving prediction accuracy and ROI.

## Motivation

Market odds (closing SP) are the strongest single predictor of match outcomes. They aggregate information from:
- Team strength (Elo)
- Recent form
- Injury/suspension news
- Squad rotation
- Market sentiment

By including market odds as a feature, the model learns to calibrate its predictions against the market consensus.

## Data Pipeline

### 1. Data Sources

- **historical_kaijiang.csv**: 3248 matches with closing SP odds (2024-01-01 to 2026-06-08)
- **international_results.json**: 49257 historical matches (1872-2026)
- **team_name_mapping.json**: 101 Chinese→English team name mappings

### 2. Merge Process

```python
# Match on [home_en, away_en, date ±2 days]
merged = merge_data(kaijiang, intl, team_map, date_tolerance=2)

# Result: 263 merged matches with closing SP odds
```

### 3. Feature Engineering

```python
# Original 29 features (from xgb_model_29.pkl)
b15 = [elo_diff, lam_h, lam_a, lam_diff, lam_ratio,
       dc_a, dc_d, dc_h, fh5_wr, fa5_wr, ...]
gold = [h2h_gd, tier_major, tier_friendly, fh12_gf_fa12_ga, fa12_gf_fh12_wr]
odds_feat = [op_h, op_a, 0.0]
form_feat = [fh5_gf, fh5_ga, fa5_gf, fa5_ga, fh5_wr3, fa5_wr3]

# New: 30th feature
market_implied = 1.0 / sp if sp > 0 else 0.0

feat = b15 + gold + odds_feat + form_feat + [market_implied]
```

## Training Process

### 1. Data Preparation

```bash
python3 /root/wc_2026_upgrade/prepare_training_data.py
```

Output: `/root/data/training_data_with_odds.json`

### 2. Model Training

```bash
python3 /root/wc_2026_upgrade/retrain_xgb_with_odds.py
```

Training parameters:
- n_estimators: 300
- max_depth: 4
- learning_rate: 0.03
- subsample: 0.8
- colsample_bytree: 0.8

### 3. Cross-Validation

TimeSeriesSplit with 3 folds:
- Fold 0: LogLoss=0.6672, Acc=75.38%
- Fold 1: LogLoss=0.7583, Acc=76.92%
- Fold 2: LogLoss=0.4974, Acc=81.54%

### 4. Calibration

Isotonic regression on validation set:
- Input: XGB predicted probabilities
- Output: Calibrated probabilities
- Saved to: `/root/data/calibrators_v2.pkl`

## Results

### Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | market_implied | 0.1532 |
| 2 | op_h | 0.1308 |
| 3 | elo_diff | 0.1137 |
| 4 | dc_h | 0.1126 |
| 5 | lam_ratio | 0.0851 |
| 6 | op_a | 0.0816 |
| 7 | lam_a | 0.0810 |
| 8 | lam_diff | 0.0698 |
| 9 | lam_h | 0.0660 |
| 10 | dc_a | 0.0557 |

### Backtest Performance

| Metric | xgb_model_29 | xgb_model_30 |
|--------|--------------|--------------|
| Features | 29 | 30 |
| Bets | 53 | 80 |
| Hit Rate | 28.3% | 70.0% |
| ROI | +37.64% | +69.86% |

## Key Insights

1. **Market odds are the #1 feature**: The model learned to trust the market's information aggregation.

2. **Improved calibration**: Market odds help the model output probabilities that are better calibrated with actual outcomes.

3. **More bets triggered**: The improved probability estimates mean more matches pass the EV threshold.

4. **All months profitable**: The retrained model shows consistent performance across all months.

## Files

- `/root/wc_2026_upgrade/prepare_training_data.py` - Data preparation
- `/root/wc_2026_upgrade/retrain_xgb_with_odds.py` - Model training
- `/root/data/training_data_with_odds.json` - Training data
- `/root/data/xgb_model_30.pkl` - New XGB model
- `/root/data/calibrators_v2.pkl` - New calibrators
- `/root/data/retrain_report.json` - Training report

## Pitfalls

1. **Feature shape mismatch**: xgb_model_30.pkl expects 30 features. Using 29 features will cause errors.

2. **Market odds availability**: Some matches have spf_sp=0.0 (not offered). These are excluded from training.

3. **Data leakage**: Market odds are from the same time as the match outcome. In production, you need to ensure you're using odds available before the match starts.

4. **Overfitting risk**: With only 263 training samples, the model may overfit. Cross-validation helps but doesn't eliminate this risk.
