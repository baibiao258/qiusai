# HT/FT Model + Dynamic Market Weight (2026-06-08)

## HT/FT XGB Model
- **Training data**: 10,077 club matches with half-time scores (from football-data.org)
- **Model**: XGB 9-class classifier (HH/HD/HA/DH/DD/DA/AH/AD/AA)
- **Features**: 12-dim [lam_h, lam_a, elo_diff, elo_expected, form_wr×2, form_gf×2, form_ga×2, h2h_gd, neutral]
- **Calibration**: Per-class Isotonic (critical — raw acc 16.4% → calibrated 30.3%)
- **Baseline comparison**: r_ht=0.45 math derivation acc=25.5%, top3=63.0%, brier=0.8225
- **XGB calibrated**: acc=30.3%, top3=60.8%, brier=0.8168

### Category Distribution
```
HH: 26.6% (most common — home leads at HT and FT)
AA: 17.0%
DD: 15.1%
DH: 14.5%
DA: 11.0%
HD:  5.6%
AD:  5.1%
AH:  2.9%
HA:  2.3% (rarest — home leads at HT, away at FT)
```

### Files
- `/root/train_htft_club.py` — Training script
- `/root/htft_predictor.py` — Inference wrapper (lazy loads xgb_htft_club.pkl)
- `/root/data/xgb_htft_club.pkl` — Trained XGB model
- `/root/data/htft_calibrators.pkl` — Per-class Isotonic calibrators

### Integration
`compute_htft_topn()` in daily_jczq.py now tries XGB first, falls back to math:
```python
def compute_htft_topn(lambda_home, lambda_away, topn=6, home=None, away=None):
    try:
        from htft_predictor import predict_htft_probs
        probs = predict_htft_probs(lambda_home, lambda_away, home=home, away=away)
        # Convert HH/HD/... to 胜胜/胜平/...
    except:
        # Fallback to predict_half_full_probs(r_ht=0.45)
```

## Dynamic Market Weight
- **Location**: `build_prediction_bundle()` in daily_jczq.py
- **Logic**: Blends model probs with market implied probs using mc_market_weight_helper
- **Weight range**: 10%-42% based on:
  - Elo gap (wider → more market reliance)
  - Neutral flag (neutral → less market reliance)
  - Market strength (strong → modestly more weight)
- **Only applied when**: 500.com odds are available (odds_h/odds_d/odds_a all > 1)
- **Formula**: `pred = (1-mkt_w) * model + mkt_w * market_implied`

### Weight Schedule (from mc_market_weight_helper.py)
```
Elo gap ≥220: base=0.32
Elo gap 160-219: base=0.29
Elo gap 100-159: base=0.26
Elo gap 50-99: base=0.23
Elo gap <50: base=0.19
Neutral: -0.03
Non-neutral: +0.02
Strong market: +0.03
Weak market: -0.03
Final clamp: [0.10, 0.42]
```
