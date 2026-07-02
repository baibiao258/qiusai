# 33-Dim XGB Model Training & Post-Processing Rules

## 33-Dim Model Training (2026-06-11)

### Training Data
- Source: `/root/data/training_data_with_odds.json` (263 matches)
- Added 4 tournament stage features: `points_diff`, `rank_diff`, `is_knockout`, `round_num`

### Feature Vector (34 dims)
```
29 base: elo_diff, lam_h, lam_a, lam_diff, lam_ratio, dc_a/d/h, fh5_wr, fa5_wr,
         fh5_gf-fa5_ga, fa5_gf-fh5_ga, fh5_gf-fa5_gf, fh5_wr-fa5_wr, bias,
         h2h_gd, tier_major, tier_friendly, fh12_gf-fa12_ga, fa12_gf-fh12_wr,
         op_h, op_a, op_0, fh5_gf, fh5_ga, fa5_gf, fa5_ga, fh5_wr3, fa5_wr3
1 market: market_implied
4 stage: points_diff, rank_diff, is_knockout, round_num
```

### Training Results
- Model: XGBClassifier (n_estimators=300, max_depth=4, lr=0.03)
- Cross-validation: TimeSeriesSplit(n_splits=3)
- Mean LogLoss: 0.6419
- Mean Accuracy: 78.5%

### Feature Importance (Top 10)
1. op_a: 0.1465
2. market_implied: 0.1417
3. op_h: 0.1139
4. dc_h: 0.1127
5. elo_diff: 0.1018
6. lam_ratio: 0.0788
7. lam_a: 0.0750
8. lam_h: 0.0644
9. lam_diff: 0.0625
10. dc_a: 0.0545

**Note**: Tournament stage features have 0% importance because training data uses placeholder values (0.0, 0.333). Real values would only be available for actual World Cup matches. The features are "pre-embedded" (预埋不入模) — stored in bundle+CSV but not yet influencing XGB predictions.

### Files
- Model: `/root/data/xgb_model_33.pkl`
- Calibrators: `/root/data/calibrators_v2.pkl`
- Report: `/root/data/retrain_report.json`
- Training script: `/root/wc_2026_upgrade/retrain_xgb_with_odds.py`
- Data prep: `/root/wc_2026_upgrade/prepare_training_data.py`

## Post-Processing: Motivation Drop Rule

### Scenario
World Cup group stage round 3: strong team has already qualified (6+ points), opponent eliminated (0-1 points). Strong team rotates squads → lower win probability.

### Implementation Location
`_try_hybrid_predict()` in `/root/daily_jczq.py`, after Draw Correction Layer (~line 480).

### Trigger Conditions
1. `round_num_normalized ∈ [0.33, 0.53]` (round 3 of 7)
2. `|points_diff| >= 0.5` (≥3 points gap)
3. `'世界杯' in league` AND `is_knockout == False`

### Adjustment
- Strong team probability reduced by 15%
- Half goes to draw, half to weak team win
- Re-normalized after adjustment

### Test Results
```
场景: 巴西 vs 摩洛哥 (世界杯, 第三轮, 巴西6分/摩洛哥0分)
原始: 主70.0% / 平20.0% / 客10.0%
调整: 主59.5% / 平25.2% / 客15.2%
```

All 5 test cases pass (round 1 no trigger, round 2 no trigger, round 3 +6pts trigger, round 3 +3pts trigger, round 4 knockout no trigger).

## Pre-Match Odds Monitoring

### Architecture
- Cron: `*/30 * * * *` (every 30 minutes)
- Script: `/root/scripts/pre_match_odds_refresh.py`
- Alert threshold: >10% odds change
- Logs: `/root/data/odds_alerts.log`, `/root/data/odds_refresh.log`
- Snapshot: `/root/data/odds_history.json`

### Usage
```bash
# Manual check before placing bets
python3 /root/scripts/pre_match_odds_refresh.py

# Check for alerts
cat /root/data/odds_alerts.log

# If alerts found, re-run predictions
python3 /root/daily_jczq.py
```
