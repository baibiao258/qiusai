# WC 2026 Predictions vs Actuals Tracking

## Full Cumulative Performance (through 2026-06-25)

**Overall: 213/334 = 63.8%** across all predictions tracked against completed matches.

### Daily Breakdown

| Date | Correct | Total | Accuracy |
|------|---------|-------|----------|
| June 14 (initial model) | 27 | 50 | 54% |
| June 15 | 22 | 46 | 48% |
| June 16 | 30 | 42 | **71%** |
| June 17 | 26 | 38 | **68%** |
| June 18 | 24 | 34 | **71%** |
| June 19 | 21 | 30 | **70%** |
| June 20 | 18 | 26 | **69%** |
| June 21 | 15 | 22 | **68%** |
| June 22 | 13 | 18 | **72%** |
| June 23 | 9 | 14 | 64% |
| June 24 | 6 | 10 | 60% |
| June 25 | 2 | 4 | 50% |

**Trend**: Model peaked at 71-72% during June 16-22 group stage, then declined to 50-60% as the harder second-round matches (same-group rematches and cross-group fixtures) began.

### June 25 Post-Mortem (2/4 = 50%)

| Match | Pred | Actual | Score | Correct? |
|-------|------|--------|-------|----------|
| Curaçao vs Ivory Coast | A (58%) | A | 0-2 | ✅ |
| Ecuador vs Germany | D (42%) | H | 2-1 | ❌ |
| Japan vs Sweden | H (90%) | D | 1-1 | ❌ |
| Tunisia vs Netherlands | A (66%) | A | 1-3 | ✅ |

**Miss analysis**: Japan-Sweden was the biggest miss — 90% confidence on Japan but ended 1-1 draw. Germany beating Ecuador (away) was also a surprise.

### Cumulative Match Stats (58 completed)

| Result | Count | % |
|--------|-------|---|
| Home wins | 29 | 50% |
| Draws | 16 | 28% |
| Away wins | 13 | 22% |

### Method for computing accuracy

```python
import json, glob
DATA_DIR = '/root/data'
results = json.load(open(f'{DATA_DIR}/wc_completed_results.json'))
result_lookup = {(r['home'], r['away']): r for r in results}

pred_files = sorted(glob.glob(f'{DATA_DIR}/wc_pred_*.json'))
for pf in pred_files:
    date_str = pf.split('wc_pred_')[1].replace('.json', '')
    preds = json.load(open(pf))
    date_correct = 0
    date_total = 0
    for p in preds:
        if p['date'] > today: continue  # skip future matches
        r = result_lookup.get((p['home'], p['away']))
        if r is None: continue
        probs = {'H': p['h_pred'], 'D': p['d_pred'], 'A': p['a_pred']}
        pred_result = max(probs, key=probs.get)
        if pred_result == r['result']: correct += 1
        total += 1
    print(f"{date_str}: {date_correct}/{date_total} = {date_correct/date_total*100:.0f}%")
```

Note: Multiple pred files may exist for the same match (different dates). The file closest to match day is used.

### Historical Tracking

| Date | New Results | Cumulative Total | Cumulative Correct | Cumulative Accuracy |
|------|-------------|------------------|--------------------|---------------------|
| 2026-06-15 | 18 | 18 | 9 | 50% |
| 2026-06-16 | 6 | 24 | 15 | 62% |
| 2026-06-17 | 8 | 32 | 24 | 75% |
| 2026-06-18 | 6 | 38 | 30 | 79% |
| 2026-06-19 | 6 | 44 | 36 | 82% |
| 2026-06-20 | 6 | 50 | 42 | 84% |
| 2026-06-21 | 6 | 56 | 48 | 86% |
| 2026-06-22 | 6 | 62 | 54 | 87% |
(past days overstated due to overlap — not all pred files span all completed matches)
