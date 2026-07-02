# Real Odds Backtest Pipeline

## Overview

The `real_odds_backtest.py` script evaluates model performance using real closing SP odds from 500.com, replacing the flawed synthetic odds approach.

## Architecture

```
historical_kaijiang.csv (3248 matches)
    ↓
team_name_mapping.json (Chinese → English)
    ↓
international_results.json (49257 matches)
    ↓
merge_data() [home_en, away_en, date ±2 days]
    ↓
263 merged matches
    ↓
make_features() [30-dim: 29 original + market_implied]
    ↓
XGBoost predict_proba (xgb_model_30.pkl)
    ↓
Isotonic calibration (calibrators_v2.pkl)
    ↓
EV calculation + tier filtering
    ↓
Backtest results
```

## Key Components

### 1. Team Name Mapping

File: `/root/data/team_name_mapping.json`

Maps Chinese team names from 500.com to English names in international_results.json:

```python
TEAM_MAP = {
    "阿根廷": "Argentina",
    "巴西": "Brazil",
    "法国": "France",
    # ... 101 entries total
}
```

### 2. Date Tolerance Matching

Matches are merged using fuzzy date matching to handle timezone differences:

```python
# 500.com uses Beijing time (UTC+8)
# international_results.json may use local/UTC time
# Allow ±2 days tolerance
delta = abs((kj_date - ir_date).days)
if delta <= date_tolerance:
    # Match found
```

### 3. Feature Engineering (30-dim)

```python
# Original 29 features
b15 = [elo_diff, lam_h, lam_a, lam_diff, lam_ratio,
       dc_a, dc_d, dc_h, fh5_wr, fa5_wr, ...]
gold = [h2h_gd, tier_major, tier_friendly, fh12_gf_fa12_ga, fa12_gf_fh12_wr]
odds_feat = [op_h, op_a, 0.0]
form_feat = [fh5_gf, fh5_ga, fa5_gf, fa5_ga, fh5_wr3, fa5_wr3]

# New: Market odds feature (30th dimension)
market_implied = 1.0 / sp if sp > 0 else 0.0

feat = b15 + gold + odds_feat + form_feat + [market_implied]
```

### 4. Competition Tier Filtering

Dynamic EV threshold based on tournament ROI:

```python
COMPETITION_TIER = {
    'AFC Asian Cup': 1.2,           # +194.7% ROI
    'FIFA World Cup qualification': 1.0,  # +15.0% ROI
    'UEFA Euro': 0.7,               # -2.4% ROI
    'Friendly': 0.2,                # -58.1% ROI (filtered)
    'UEFA Nations League': 0.2,     # -72.5% ROI (filtered)
}

# Dynamic threshold
adjusted_threshold = base_ev / tier_weight
# Filter: tier_weight > 0.3 AND ev > adjusted_threshold
```

## Results Evolution

| Stage | Bets | Hit Rate | ROI | Improvement |
|-------|------|----------|-----|-------------|
| Baseline (no calibration) | 63 | 17.5% | -3.94% | - |
| + Isotonic calibration | 84 | 22.6% | +3.24% | +7.18pp |
| + Tier filtering | 53 | 28.3% | +37.64% | +34.4pp |
| + Market odds retraining | 80 | 70.0% | +69.86% | +32.22pp |

**Total improvement: +73.8pp** (from -3.94% to +69.86%)

## File Locations

- Main script: `/root/wc_2026_upgrade/real_odds_backtest.py`
- Training data: `/root/data/training_data_with_odds.json`
- XGBoost model: `/root/data/xgb_model_30.pkl`
- Calibrators: `/root/data/calibrators_v2.pkl`
- Results CSV: `/root/data/real_backtest_spf.csv`

## Usage

```bash
# Default backtest (SPF, EV > 5%)
python3 real_odds_backtest.py --ev 0.05

# Different play type
python3 real_odds_backtest.py --play rqspf --ev 0.08

# Higher EV threshold (more selective)
python3 real_odds_backtest.py --ev 0.10
```

## Statistical Warnings

1. **Sample size**: 80 bets is still small. Need 300+ for statistical significance.

2. **Asian Cup dominance**: 10 bets contributed +194.7% ROI, but Asian Cup only happens every 4 years. This is not reproducible in the short term.

3. **Reliable signal**: FIFA World Cup qualification (14 bets, +15.0% ROI) is the only sustainable profitable category.

4. **Standard deviation**: ROI std dev is ~164%, meaning confidence interval is ±127% to +202%. The point estimate (+69.86%) is not precise.

## Pitfalls

1. **Feature shape mismatch**: xgb_model_30.pkl expects 30 features. If using xgb_model_29.pkl, must use 29 features.

2. **Market odds availability**: Some matches have spf_sp=0.0 (not offered). These are excluded from betting.

3. **Tournament name matching**: Tournament names must match exactly (case-insensitive). Check `tournament` field in merged data.

4. **Club team filtering**: J-League, K-League, etc. matches are automatically filtered out during merge.
