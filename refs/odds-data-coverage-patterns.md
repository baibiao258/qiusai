# The Odds API Coverage Patterns (2026 WC)

## Flat Odds File Structure

The daily odds snapshots (`wc_odds_YYYY-MM-DD.json`) use a **flat** structure — NOT the nested bookmaker format returned by The Odds API raw response:

```json
{"date": "2026-06-21", "home": "Spain", "away": "Saudi Arabia", "odds_h": 1.08, "odds_d": 8.5, "odds_a": 21.0, "market_h": 1.08}
```

Fields: `date`, `home`, `away`, `odds_h`, `odds_d`, `odds_a`, `market_h`

No bookmaker-level nesting. `odds_h` is the best (lowest) H2H price across all bookmakers.

## Matches Covered vs Not Covered

As of 2026-06-21, **33 of 34** completed WC matches had odds data. The single exception:

### Match Never Carried: Australia vs Turkey (2026-06-14)

- Result: Australia 2-0 Turkey
- The Odds API never offered H2H odds for this matchup on any date
- The scores endpoint DOES return it as a completed match
- Likely cause: The Odds API covers the main 64-match schedule. Australia vs Turkey may have been a replacement/rescheduled match added after bookmakers configured their feeds, or the matchup never had a liquid market.

### Cross-Date Odds Searching

Matches that appear in a later dates' schedule often had odds available in **earlier** dates' odds files. When a match is missing from the same-day `wc_pred_YYYY-MM-DD.json` (because it wasn't predicted that day), odds may exist in:

- `wc_odds_2026-06-14.json` — The earliest and most comprehensive odds file (64 matchups)
- `wc_odds_2026-06-15.json` — 64 matchups
- Successive files shrink as matches are completed (36 matchups by 2026-06-21)

**Rule**: Always search ALL odds files, not just the same-day file. Build a `(home, away) -> (odds_h, odds_d, odds_a)` lookup across all dates.

### Team Name Consistency

Team names in completed results (from The Odds API `/scores/` endpoint) match the names in the odds files exactly. No normalization needed for:

- `Turkey` (not `Türkiye`)
- `South Korea` (not `Korea Republic`)
- `Curaçao` (special 'ç' preserved)
- `USA` (not `United States`)

This means the scores endpoint and odds endpoint use the same naming convention, but the **internal DC model** (`dc_model.pkl`) may use different names (see team-name-normalizer in SKILL.md).

### What This Means for Training Data Backfill

1. Run `scripts/check_training_gap.py` after `accumulate_results.py` in daily cron
2. The script tries: pred file → cross-date odds lookup → zero-odds fallback
3. Zero-odds matches are valid for training (they have real labels) but lack market calibration features
4. If a match has no odds anywhere, it's likely an unlisted/rescheduled match — this is rare (~3% of total)
