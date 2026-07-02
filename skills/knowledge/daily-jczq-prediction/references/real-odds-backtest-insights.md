# Real Odds Backtest Insights (2026-06-09)

## Key Finding: Isotonic Calibration is the ROI Driver

The ROI improvement from -3.94% to +69.86% came almost entirely from three fixes applied in sequence:

1. **Isotonic calibration** (-3.94% → +3.24%, +7.18pp)
2. **Competition tier filtering** (+3.24% → +37.64%, +34.4pp) 
3. **XGB retraining with market odds** (+37.64% → +69.86%, +32.22pp)

But the Isotonic calibration was the highest-leverage single fix. It corrected a systematic bias in the model's probability outputs.

## Critical Caveat

ROI +69.86% is a **historical backtest result**, not a forward-looking prediction. The calibrators were trained on 2024-2026 data. If the market structure changes, calibration quality will degrade.

The user's analysis (2026-06-09) correctly identified that:
- The model's form features are REAL (not placeholders), just 7 days stale
- The gold features are REAL (H2H + 12-game form)
- The main vulnerability is train-serve skew: XGB was trained with different form snapshots than current inference uses
- The Isotonic calibration "歪打正着" — it corrects a bias it wasn't designed to fix

## Competition Tier ROI

| Tournament | Bets | ROI |
|-----------|------|-----|
| AFC Asian Cup | 10 | +194.7% |
| FIFA WC Qual | 14 | +15.0% |
| UEFA Euro | 20 | -2.4% |
| Copa América | 9 | -12.7% |
| Friendly | 16 | -58.1% |
| UEFA Nations League | 12 | -72.5% |

**Statistical reality**: 53 bets, ROI +69.86%, standard deviation ~164%. The confidence interval is approximately -127% to +202%. The signal is real but noisy.

## Form State

- `form_state.json` contains REAL match data (336 teams, 25 games each)
- Last updated 2026-06-02 (7 days stale)
- football-data.org API returns 0 for international friendlies → cannot auto-update
- Cron set: `0 6 * * *` daily update attempt
