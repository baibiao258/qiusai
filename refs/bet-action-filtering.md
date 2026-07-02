# Bet Action Filtering

## Overview

Added 2026-06-10 to address the gap between backtest COMPETITION_TIER filtering (which existed in `real_odds_backtest.py` but not in the daily pipeline). Three-tier label system applied per-match in `build_prediction_bundle()`.

## Rules

```python
def compute_bet_action(league, model_type, bet_analysis, htft_top6, handicap, rq_probs):
    """
    Returns: 'RECOMMEND' | 'WATCH' | 'SKIP_LEAGUE'

    Rule 1: UEFA Nations League → SKIP_LEAGUE (历史ROI -72.5%)
    Rule 2: 友谊赛 + 最大margin < 20pp → WATCH (历史ROI -58.1%)
    """
    # Rule 1
    if league == 'UEFA Nations League':
        return 'SKIP_LEAGUE'

    # margin_pp = 最大模型概率优势(edge) × 100
    margin_pp = 0
    if bet_analysis and bet_analysis.scenarios:
        for s in bet_analysis.scenarios:
            edge_pp = (s.prob - 1.0 / s.odds) * 100 if s.odds > 1 else 0
            if edge_pp > margin_pp:
                margin_pp = edge_pp

    # Rule 2
    if league == '友谊赛' and margin_pp < 20:
        return 'WATCH'

    return 'RECOMMEND'
```

## Output Display

Per-match:
```
👀 bet_action: WATCH（友谊赛 margin<20pp，仅观察不推荐）
🚫 bet_action: SKIP_LEAGUE（UEFA Nations League 历史ROI -72.5%，跳过）
```

Global summary (auto-filtered):
```
ℹ️ 已过滤 N 场赛事类型不推荐场次 (SKIP_LEAGUE/WATCH)
```

## Integration Points

- **Per-match display**: `print_match_bundle()` checks `bundle.get('bet_action')` → prints label after model_note
- **Global summary**: `main()` filters bundles before passing to `format_value_summary()`:
  ```python
  all_analyses = [b.get('bet_analysis') for b in bundles if b.get('bet_action') not in ('SKIP_LEAGUE', 'WATCH')]
  ```
- **CSV**: bet_action field NOT currently written to predictions_log.csv (only used for real-time display)

## Future Expansion

The `compute_bet_action()` function is designed to be extended:
- Add more league rules (e.g. 美洲杯 → SKIP if ROI bad)
- Adjust margin_pp threshold per competition tier
- Integrate with COMPETITION_TIER dict from `real_odds_backtest.py` for unified config
