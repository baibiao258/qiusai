# Calibration Diagnostics (2026-06-10)

## Background

Isotonic regression (calibrators.pkl) trained on 263 matches (2024-01 to 2024-11) caused severe overconfidence on 2026 friendlies. Diagnosed via calibration curve analysis on 32 backfilled matches.

## Diagnostic Results (32 matches)

### By class (raw probabilities, no calibration)

| Class | Avg Predicted | Actual Rate | Gap | Brier | Status |
|-------|--------------|-------------|-----|-------|--------|
| Home(H) | 40.0% | 62.5% | +22.5pp | 0.2888 | CONSERV |
| Draw(D) | 24.2% | 18.8% | -5.4pp | 0.1868 | OK |
| Away(A) | 32.7% | 18.8% | -14.0pp | 0.2267 | OVERCONF |

### By bet_action (inferred from existing data)

| Action | n | Avg Max Prob | Hit Rate | Gap | Brier |
|--------|---|-------------|----------|-----|-------|
| WATCH_UNIFORM | 5 | 36.4% | 80.0% | +43.6pp | 0.2048 |
| WATCH_LOW | 25 | 53.2% | 48.0% | -5.2pp | 0.2254 |
| RECOMMEND | 2 | 70.2% | 0.0% | -70.2pp | 0.4166 |

### Key finding

RECOMMEND group: model gave 70% confidence, 0% hit rate. This is the smoking gun for Isotonic overfitting.

## Root Cause Analysis

1. Isotonic regression is non-parametric — fits every local fluctuation in training data
2. 263 matches is far too small for Isotonic (needs 1000+ per class)
3. 2024 international friendlies have different distribution than 2026 (World Cup prep year)
4. Isotonic "stretched" high XGB outputs even higher based on 2024 patterns → catastrophic overconfidence in 2026

## Failed Fix: Platt Scaling on Mixed Data

Attempted sigmoid (Platt Scaling) calibrator using LogisticRegression on combined 2024+2026 data.

**Result: Overall Brier worsened from 0.2053 to 0.2378.**

Root cause: 2024 data uses `market_implied = 1/spf_sp` as features, 2026 data uses actual `XGB pred_h/pred_d/pred_a`. Different probability sources → LR learns wrong sigmoid transform.

**Lesson: Never mix heterogeneous probability sources for calibration training.**

## Correct Fix Path

1. **Short term (done)**: `compute_bet_action()` → friendlies blanket `WATCH_FRIENDLY`
2. **Medium term**: Wait for 200+ 2026 backfilled matches, then train calibration on pure (XGB output, actual result) pairs
3. **Long term**: Consider per-league calibration (friendlies vs competitive matches have different distributions)

## Analysis Script

`/root/calibration_analysis.py` — generates calibration curve PNG + numeric diagnostics.

Usage:
```bash
python3 /root/calibration_analysis.py
# Output: /root/data/calibration_curve.png
```

## When to Re-run Calibration Analysis

- After every 50 new backfilled matches
- After switching from xgb_model_29 to xgb_model_30
- After retraining any calibrator
- When Brier Score in backfill stats exceeds 0.28
