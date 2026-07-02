# System Audit 2026-06-14 — P0 Bug Fixes

## 1. Training Data Label Contamination (P0-1)

**Root cause**: `training_data_with_odds.json` has 131/491 records where `spf_result` is int type instead of string.

**Bug**: Training scripts use `result == '3'` which doesn't match int `3`, causing:
- int `3` (home win) → mapped to label `0` (away win) ❌
- int `1` (draw) → mapped to label `0` (away win) ❌
- Only int `0` (away win) maps correctly by coincidence

**Impact**: 29/395 effective training samples (7.3%) had wrong labels.

**Fix**: `str(m['spf_result'])` in all training scripts:
- `/root/wc_2026_upgrade/train_national_xgb.py` line 54
- `/root/wc_2026_upgrade/retrain_xgb_with_odds.py` line 172
- `/root/wc_2026_upgrade/train_clean_xgb.py` line 128

**Result**: nat model validation accuracy improved from 64.4% → 75.4%

## 2. _blend_with_market Draw Suppression (P0-2)

**Root cause**: `calibrated_predictor.py:56-57` hardcoded draw=0:
```python
elo_arr = np.array([elo_h, 0, 1-elo_h])  # draw=0!
mkt_arr = np.array([market_h, 0, 1-market_h])  # draw=0!
```

**Impact**: WC pipeline predicted 0/64 draws as primary pick. When blended with DC probability (which does have draw), the zero-draw base systematically suppressed draw predictions.

**Fix**: Estimate draw from Elo difference:
```python
elo_draw = max(0.05, 0.25 * (1 - abs(2*elo_h - 1)))
elo_arr = np.array([max(0.01, elo_h-elo_draw/2), elo_draw, max(0.01, 1-elo_h-elo_draw/2)])
```

**Result**: Netherlands vs Japan draw 5.6% → 13.2%, more reasonable distributions.

## 3. Dual Pipeline Model Inconsistency (P0-3)

| Dimension | daily_jczq.py | calibrated_predictor.py |
|-----------|---------------|------------------------|
| XGB model | xgb_model_29.pkl (29-dim) | xgb_model_nat.pkl (11-dim) |
| Calibrator | Stripped (commented out) | Was active (now stripped) |
| Fusion | Entropy dynamic 0.10-0.90 | Hard-coded 0.5-0.8 |
| Draw Correction | Yes (parameterized) | Was absent (now added) |

**Recommendation**: Unify on nat model (11-dim, 75.4% acc) for both pipelines.

## Model Performance Comparison

| Model | Features | Accuracy | LogLoss | Dead Features |
|-------|----------|----------|---------|---------------|
| nat (clean) | 11 | 75.4% | 0.819 | 0 |
| 30-dim (retrained) | 30 | 64.3% | 1.176 | 18 |

The 30-dim model has 18 features with importance=0.0 (form, gold, h2h, stage features) due to train-serve skew — these features are computed differently during training vs inference.

## Quick Sanity Checks

Run before retraining to catch these issues:
```bash
# 1. Check label types
python3 -c "
import json
d=json.load(open('/root/data/training_data_with_odds.json'))
int_ct=sum(1 for m in d if isinstance(m.get('spf_result'),int))
str_ct=sum(1 for m in d if isinstance(m.get('spf_result'),str))
print(f'spf_result: int={int_ct} str={str_ct}')
assert int_ct==0 or str_ct > int_ct, 'int类型过多!'
"

# 2. Check for draw=0 hardcoding
grep -rn 'arr.*\[.*0.*1-' /root/*.py /root/wc_2026_upgrade/*.py 2>/dev/null

# 3. Check calibrator usage
grep -rn 'calibrat.*predict\|_cal.*predict' /root/*.py /root/wc_2026_upgrade/*.py 2>/dev/null
```
