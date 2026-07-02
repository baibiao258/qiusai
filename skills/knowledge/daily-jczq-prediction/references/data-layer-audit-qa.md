# Data Layer Audit Q&A Methodology

## Overview

When the user asks probing questions about data quality, answer each by looking up the actual filesystem/data state — not from memory or mental model. This reference documents the 7-question audit framework used on 2026-06-10.

## The 7 Questions

### Q1: form 数据实时性

What to check:
```bash
crontab -l | grep form          # cron schedule
stat /root/data/form_state.json  # last update time
head -3 /root/update_form_from_365.py  # data source URL
wc -l /root/data/form_state.json; tail -5 /root/data/form_update.log  # team count + last run stats
```

Key signals:
- Cron schedule (should be daily, e.g. `0 6 * * *`)
- Data source: 365scores API or football-data.org?
- Days parameter: `--days 2` means last 48 hours covered
- form_state.json ONLY records match scores (goals), NOT red cards, substitutions, manager changes, weather, etc.

### Q2: 球队名映射覆盖率

What to check:
```bash
python3 -c "import json; m=json.load(open('/root/data/team_name_mapping.json')); print(len(m))"
python3 -c "import csv; rows=list(csv.DictReader(open('/root/data/predictions_log.csv'))); fb=[r for r in rows if 'market_fallback' in r.get('model_version','')]; print(len(fb), len(rows))"
```

Key signals:
- Mapping count (currently 101)
- Today's fallback count vs total matches
- If fallback > 20%, the team name mapping IS NOT the bottleneck — it's the training data coverage (football-data.org only covers 9 leagues + UCL)

### Q3: 路由统计 + Brier Score

What to check:
```bash
python3 /root/backfill_results.py --stats
python3 -c "import csv; rows=list(csv.DictReader(open('/root/data/predictions_log.csv'))); mvs={}; [mvs.update({r.get('model_route',''): mvs.get(r.get('model_route',''),0)+1}) for r in rows]; print(mvs)"
```

Key signals:
- `brier_spf` field now exists in CSV (2026-06-10 added)
- `model_route` field now exists (hybrid/market_fallback/club)
- Use `backfill_results.py --stats` for aggregated Brier by model_route and bet_action
- Per-match Brier in CSV: `brier_spf` column (4 decimal places)
- Route-level Brier available via pandas groupby on model_route

### Q4: 并行模型分歧 (simple vs main)

What to check:
```bash
python3 << 'PYEOF'
import csv
rows = list(csv.DictReader(open('/root/data/predictions_log.csv')))
latest = [r for r in rows if '06-10' in r.get('code','')][-26:]
conflicts = 0
for r in latest:
    sp, sc, mp = r.get('simple_pred',''), r.get('simple_conf',''), r.get('pred_spf_pick','')
    if sp and sc and mp:
        sp_label = {'H':'主胜','D':'平局','A':'客胜'}.get(sp, sp)
        if sp_label != mp:
            conflicts += 1
            print(f"{r.get('home_cn','')} vs {r.get('away_cn','')}: main={mp} simple={sp_label}({float(sc)*100:.1f}%) odds={r.get('odds_h','')}/{r.get('odds_d','')}/{r.get('odds_a','')}")
print(f"\n总分歧: {conflicts}")
PYEOF
```

Key signals:
- Count of divergence matches (was 8 on 2026-06-10, user guessed 6)
- Highest simple confidence often does NOT match market odds direction
- Divergence may indicate model overfitting or data anomaly — not automatically actionable but worth flagging

### Q5: 校准器时效性

What to check:
```bash
stat -c '%y' /root/data/calibrators.pkl  # modification time
stat -c '%y' /root/data/xgb_model_30.pkl
python3 -c "import json; d=json.load(open('/root/data/training_data_with_odds.json')); dates=[x.get('date','')[:10] for x in d if x.get('date')]; dates.sort(); print(f'{dates[0]} → {dates[-1]}')"  # training data range
```

Key signals:
- calibrators.pkl age (may be 0 days = regenerated today)
- Training data range: currently 2024-01-13 → 2024-11-15 (263 matches)
- Even if pkl file is fresh, the training data may be 1.5 years stale
- Distribution shift from 2024 data applied to 2026 predictions is real but currently unmitigated

### Q6: EV 数字真实性检查

For any suspiciously high EV value:
1. Check model type: `python3 -c "from daily_jczq import predict_match_wrapper; p=predict_match_wrapper('TeamA','TeamB'); print(p.get('model',''))"`
2. Check if it's hybrid (trustworthy) or market_fallback (EV is from reverse-engineered market odds)
3. Check if it's a high-variance play (比分/半全场 with prob < 20%)
4. Even hybrid models produce unreliable EV for 半全场胜胜 when prob < 20% — the Poisson-extrapolated probability is not well-calibrated for extreme scenarios

Reference: `is_sane_bet()` in bet_math.py — filters odds>30, prob<15%, fallback比分&半全场.

### Q7: 赛事过滤规则生效检查

What to check:
```bash
grep -n 'discount\|友谊\|Friendly\|filter\|skip\|exclude' /root/daily_jczq.py | head -10
```

Key signals:
- COMPETITION_TIER filtering only exists in `real_odds_backtest.py`, NOT in `daily_jczq.py`
- All today's matches = 友谊赛 (100%) if no league matches available
- CSV has `league` field but no filtering logic on it
- Fix: `compute_bet_action()` in daily_jczq.py now handles Rule 1 (UEFA Nations League → SKIP) and Rule 2 (友谊赛 + margin<20pp → WATCH)

## General Methodology

1. **Never answer from mental model** — always check filesystem: file timestamps, CSV contents, pkl metadata
2. **Cross-reference model files with training data** — a fresh model file might be trained on stale data
3. **pkl modification time ≠ training time** — loading a model updates access time; check training script timestamps
4. **Check both pkl age AND training data date range** — both matter
5. **Distinguish calibration file freshness vs training data staleness** — they're separate concerns

### Q8: 校准曲线分析 (2026-06-10 新增)

When to ask: after any calibrator retrain, after every 50 new backfilled matches, when Brier exceeds 0.28.

```bash
python3 /root/calibration_analysis.py
# Generates /root/data/calibration_curve.png + terminal diagnostics
```

Key signals:
- Per-class gap > ±10pp → calibration failure
- RECOMMEND group gap < -20pp → severe overconfidence (Isotonic overfitting)
- WATCH_UNIFORM gap > +30pp → model is too conservative for uniform predictions
- If calibrated Brier > raw Brier → do NOT use that calibrator (mixed-distribution problem)

Diagnostic dimensions:
1. By class (H/D/A): avg predicted vs actual hit rate → gap + Brier
2. By bet_action (inferred): WATCH_UNIFORM / WATCH_LOW / RECOMMEND
3. By model_route: hybrid / market_fallback / club

Reference: `references/calibration-diagnostics.md`
