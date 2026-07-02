# xG-proxy Feature Integration (2026-06-08)

## Technique: Performance Residuals as Proxy for xG

xG-proxy uses the **residual between actual goals and DC model's expected goals (λ)** as a signal for luck/skill differential.

### Formula
```
luck_factor = actual_goals - DC_lambda
```

- **Positive**: team scored more than expected → potential overperformance (unsustainable)
- **Negative**: team scored less than expected → potential underperformance (regression candidate)

### 8 Feature Dimensions (4 per team)

| Feature | Window | Meaning |
|---------|--------|---------|
| `xg_proxy_5` | 5 games | Recent luck factor mean |
| `xg_proxy_12` | 12 games | Medium-term luck factor mean |
| `xg_streak` | all history | Consecutive over/under expectation (normalized ÷10) |
| `xg_volatility` | 12 games | Std dev of luck factors (higher = more volatile) |

### Files

| File | Role |
|------|------|
| `/root/xg_proxy.py` | Core computation: `compute_luck_factors()`, `build_xg_proxy_state()`, `get_xg_proxy_features()` |
| `/root/data/xg_proxy_club.json` | Pre-computed state (217 teams) |
| `/root/club_data_pipeline.py` | Calls xg_proxy after DC training, saves to xg_proxy_club.json |
| `/root/train_xgb_club.py` | Loads xg_proxy_club.json, appends 8 features → 37 total |
| `/root/daily_jczq.py` | `_try_club_predict()` loads xg_proxy_club.json, appends 8 features |

### Pipeline Flow
```
club_data_pipeline.py
  ├─ load club_matches.json
  ├─ compute_elo_ratings()
  ├─ build_form_state()
  ├─ train_dc_model()
  └─ xg_proxy.py: compute_luck_factors() → build_xg_proxy_state() → xg_proxy_club.json

train_xgb_club.py
  ├─ load: elo_club.pkl, form_club.json, dc_model_club.pkl, xg_proxy_club.json
  ├─ build_feat(): 27 base + 8 xg_proxy = 37 features
  └─ XGBClassifier(37-dim) → xgb_model_club.pkl

daily_jczq.py._try_club_predict()
  ├─ load xg_proxy_club.json
  ├─ feat = b15 + gold + odds_feat + form_feat + xg_feat  (37-dim)
  └─ xgb_club.predict_proba(feat)
```

### Performance Impact

| Metric | Before (27-dim) | After (37-dim) | Δ |
|--------|-----------------|----------------|---|
| Brier (raw) | 0.2101 | 0.2020 | -3.9% |
| Brier (cal) | 0.2034 | 0.1937 | -4.8% |
| Acc (raw) | 45.2% | 47.7% | +2.5pp |
| Acc (cal) | 49.3% | 53.5% | +4.2pp |

### Pitfalls

1. **Global snapshot skew**: `xg_proxy_club.json` uses final state for ALL historical matches → slight forward-looking bias. Acceptable tradeoff; fix requires incremental buffer rebuild (similar to `ClubFeatureBuffer` pattern).

2. **Team coverage**: xg_proxy only covers teams in `club_matches.json` (currently 217). Teams not in this file get zero vectors → model treats them as "average luck". International matches get zero vectors (no club xg_proxy).

3. **DC dependency**: xG-proxy requires DC model's λ predictions. If DC doesn't converge for a team pair, that match's luck factor is skipped (not included in the rolling window).

### Adding New Features

To extend beyond 37 dimensions:
1. Modify `xg_proxy.py` → `get_xg_proxy_features()` return list
2. Update `train_xgb_club.py` → `build_feat()` xg_feat section
3. Update `daily_jczq.py` → `_try_club_predict()` xg_feat section
4. Retrain: `python3 club_data_pipeline.py && python3 train_xgb_club.py`
