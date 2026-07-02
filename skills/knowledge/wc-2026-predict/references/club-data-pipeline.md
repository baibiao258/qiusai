# Club Data Pipeline (2026-06-08)

## Overview
Separate DC+Elo+form system for club matches (domestic leagues), isolated from national team data.

## Data Source
- football-data.org API (free tier: ~3 seasons per league)
- **Seasons endpoint returns 404** — use `/competitions/{code}/matches?season=YYYY` instead
- Rate limit: 10 req/min (6.5s interval)

## Files
| File | Path | Purpose |
|------|------|---------|
| Raw matches | `/root/data/club_matches.json` | 10,077 matches (9 leagues × 3 seasons) |
| Club Elo | `/root/data/elo_club.pkl` | 119 teams, half_life=150d |
| Club Form | `/root/data/form_club.json` | 119 teams × 25 recent matches |
| Club Form 12 | `/root/data/form_12_club.json` | Pre-computed 12-game form |
| Club DC | `/root/data/dc_model_club.pkl` | ρ=0.25, γ=0.27 |

## Key Parameters
- **Elo half-life**: 150 days (vs 540 for national teams)
- **Promoted team initial Elo**: 1400 (not 1500)
- **DC rho**: 0.25 (similar to national teams)
- **DC gamma**: 0.27 (home advantage)

## Scripts
- `/root/fetch_league_data.py` — Fetch from football-data.org (incremental save)
- `/root/club_data_pipeline.py` — Build Elo + form + train DC

## Integration with daily_jczq.py (已完成 2026-06-08)
`predict_match_wrapper()` routes through two tracks:
```python
def predict_match_wrapper(home, away):
    r = _try_club_predict(home, away)   # club: elo_club + dc_club + xgb_club
    if r is None:
        r = _try_hybrid_predict(home, away)  # intl: elo_intl + dc_intl + xgb_intl
    # + 365scores adjustment
```
- Club track: Brier 0.21, 119 teams, XGB 29-dim
- International track: Brier 0.46, 336 teams, XGB 29-dim
- Model files: `xgb_model_club.pkl`, `calibrators_club.pkl`

## Top 10 Club Elo (2026-06-08)
1. FC Bayern München: 1677
2. FC Barcelona: 1657
3. FC Internazionale Milano: 1645
4. Arsenal FC: 1636
5. Real Madrid CF: 1629
6. Manchester City FC: 1612
7. Paris Saint-Germain FC: 1611
8. Manchester United FC: 1602
9. AS Roma: 1602
10. Borussia Dortmund: 1599

## Pitfalls
1. `fetch_league_data.py` MUST save incrementally (after each league) to avoid data loss on process kill
2. `club_data_pipeline.py` must include `neutral` column in DataFrame for DC.fit()
3. Free tier only allows ~3 seasons (2023-2025). 2022+ returns 403.
4. Club and national team Elo must stay separate — no cross-contamination
