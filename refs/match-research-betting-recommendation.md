# Match Research & Betting Recommendation Workflow

## When to use

User asks for match-specific betting advice beyond what the standard pipeline outputs — e.g. "结合全网信息再给出推荐的买法" / "今天应该怎么买".

## Quick-start: single-match ad-hoc query

When the user asks about a specific match (e.g. "美国 vs 波黑"), run in parallel:

```bash
# 1. Quick pipeline summary (fastest — H/D/A + odds only)
cat /root/data/wc_pred_$(date +%Y-%m-%d).json 2>/dev/null | python3 -m json.tool
# or for any date:
cat /root/data/wc_pred_*.json | python3 -c "import json,sys; d=json.load(sys.stdin); [print(f\"{m['date']} {m['home']} vs {m['away']}: H={m['h_pred']}% D={m['d_pred']}% A={m['a_pred']}% odds={m['odds_h']}/{m['odds_d']}/{m['odds_a']}\") for m in d if 'USA' in m['home'] or '波黑' in m['home']]"

# 2. Detailed 5-play (single match)
python3 /root/scripts/run_single_match.py "美国" "波黑" 2>&1

# 3. CSV grep for full field data (EV, bet_action, model_route, distribution JSONs)
grep "波黑" /root/data/predictions_log.csv | grep "$(date +%Y-%m-%d)"

# 4. Team form
python3 -c "import json; d=json.load(open('/root/data/wc_completed_results.json')); [print(f\"{m['home']} {m.get('home_score','?')}-{m.get('away_score','?')} {m['away']} ({m.get('result','?')})\") for m in d if 'USA' in m['home'] or 'USA' in m['away']]"

# 5. Session search for prior cron pipeline outputs
# (Use session_search tool with query about the match)
```

### ⚠️ Match date timezone ambiguity

`daily_jczq.py` and `daily_wc_pipeline.py` may disagree on match date by ±1 day (UTC vs local timezone). Always cross-check:
- `wc_pred_*.json` has a `date` field
- `predictions_log.csv` has both `date` (prediction run day) and `match_date` (actual match day via `time` column)
- When date differs between sources, use `match_date` from CSV as ground truth.

### Model cross-referencing: daily_jczq (XGB) vs WC pipeline (nat)

Two independent model pipelines may produce different H/D/A probabilities for the same match. This is expected — they use different feature sets (XGBoost 11-dim vs nat model):

| Source | Model | Coverage | Strength |
|--------|-------|----------|----------|
| daily_jczq.py | XGB + Poisson/Elo | Full CSV, 5-play, EV | Deep per-match detail |
| daily_wc_pipeline.py | nat model (simple) | Compact JSON | Fast summary, high-level |

When both agree on direction (both pick H or both pick A) → increased confidence.
When they disagree on magnitude (e.g. 66% vs 89%) → the XGB model is more conservative/calibrated.

## Data sources (in order of preference)

### 1. Match metadata (football-data.org API)

```bash
curl -s "https://api.football-data.org/v4/competitions/WC/matches?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD" \
  -H 'X-Auth-Token: {FOOTBALL_API_KEY}' | python3 -m json.tool
```

Key fields to extract:
- `stage`: `GROUP_STAGE` vs `LAST_32` vs `LAST_16` vs `QUARTER_FINALS` etc → knockout has **extra time/penalties**
- `group`: group letter (e.g. `GROUP_J`)
- `status`: `TIMED` (upcoming) vs `FINISHED` vs `IN_PLAY`
- `score.fullTime.home/away`: actual result (only for FINISHED)
- `matchday`: round number

### 2. System prediction (predictions_log.csv + daily_jczq.py output)

Read the latest entry from daily_jczq.py terminal output or grep the CSV:

```bash
grep "南非" /root/data/predictions_log.csv | grep "2026-06-28"  # most recent run
```

Key extraction:
- `pred_h/d/a`: SPF probabilities (%)
- `odds_h/d/a`: market odds (500.com)
- `pred_rq_win/draw/loss`: handicap probabilities
- `rq`: handicap value (e.g. `1` = 受让1, `-1` = 让1)
- `kelly_pct`: Quarter-Kelly position (0 = no EV)
- `bet_action`: RECOMMEND / SKIP_WORLD_CUP_FALLBACK / WATCH / etc
- `model_route`: xgb_dc_nat_11d / market_fallback / etc
- `ev_h/d/a`: expected value for each outcome
- `pred_spf_pick`: model's SPF main pick
- `pred_rq_pick`: model's handicap main pick
- `score_full`: JSON score distribution
- `goals_full`: JSON goals distribution
- `htft_full`: JSON half-time/full-time distribution

### 3. 365scores cached data

From `/root/data/365scores/football_games.csv`:

```bash
grep "South Africa\|加拿大" /root/data/365scores/football_games.csv | grep "2026-06-28"
```

Key columns:
- vote_h/d/a: crowd voting percentages
- vote_count: number of voters (sample size check)
- pop_rank_home/away: popularity rank
- trend_win_rate_home/away: recent win rate
- s365_home_fifa/away_fifa: FIFA rank (lower = better)
- has_lineups, has_news: lineup/team news flags

### 4. Prediction evolution over days

The same match may have entries on multiple dates. Read them to see trend:

```python
rows = list(csv.DictReader(open('/root/data/predictions_log.csv')))
for r in rows:
    if '南非' in r.get('home_cn','') and '加拿大' in r.get('away_cn',''):
        print(f"{r['date']}: H={r['pred_h']}% D={r['pred_d']}% A={r['pred_a']}%")
```

Trend tells you which way the model probabilities are moving.

### 5. Historical actual results for both teams

#### From predictions_log.csv (all matches, including backfilled results)

```python
for r in rows:
    if r.get('result_status','') == 'filled' and ('南非' in r['home_cn'] or '南非' in r['away_cn']):
        print(f"{r['match_date']} {r['home_cn']} vs {r['away_cn']}: {r['actual_score']} (Brier={r['brier_spf']})")
```

Also check 365scores data for non-World-Cup friendlies/fixtures.

#### From wc_completed_results.json (World Cup match results — faster, tournament-wide)

This file accumulates all completed World Cup 2026 match results. More comprehensive than CSV for WC:
```bash
# Quick form check for a team
python3 -c "
import json
d = json.load(open('/root/data/wc_completed_results.json'))
for m in d:
    if 'USA' in m['home'] or 'USA' in m['away']:
        print(f\"{m['home']} {m.get('home_score','?')}-{m.get('away_score','?')} {m['away']} → {m.get('result','?')}\")
"

# Filter by both home/away team names
python3 -c "
import json
d = json.load(open('/root/data/wc_completed_results.json'))
for m in d:
    hs, aw = m['home'], m['away']
    # team names appear in both English and format as stored
    if 'Bosnia' in hs or 'Bosnia' in aw or 'Canada' in hs or 'Canada' in aw:
        print(f\"{hs} {m.get('home_score','?')}-{m.get('away_score','?')} {aw}\")
"
```
Limitation: only covers World Cup matches, not friendlies or other competitions.

### 6. TheStatsAPI (when needed)

For deeper stats (injuries, lineups, xG):

```python
import requests
headers = {"Authorization": f"Bearer {THE_STATS_KEY}"}
r = requests.get(f"{BASE}/matches", params={"date": "2026-06-28", "page": 1, "limit": 50}, headers=headers)
```

Note: THE_STATS_KEY from backfill_results.py: `fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww`

## Research synthesis checklist

After collecting the above, answer these questions:

| Question | Data source | Implication |
|----------|------------|-------------|
| Is it a knockout match? | football-data.org stage | Regulation draw leads to ET/penalties → draw probability matters more |
| What's the model preference vs market? | pred_h/d/a vs odds_h/d/a | Big divergence = disagreement signal |
| Is there positive EV? | ev_h/d/a, kelly_pct | EV<0 across all = no quantifiable edge |
| Is the crowd confident? | 365scores vote | >60% one side = popular pick |
| Are the teams in form? | trend_win_rate, actual results | Recent results tell more than FIFA rank |
| How has the prediction evolved? | Multi-day entries | Upward trend in H/D/A = model gaining/losing confidence |
| Is model using XGBoost or just odds? | model_route | market_fallback = no ML, just market odds |
| Is there market disagreement? | RQ market vs model | Directional split = uncertainty |

## Recommendation structure

Structure the response as:

1. **Match context** — stage, format (knockout/group), time, venue
2. **Model vs market** — brief comparison table showing model prob vs market implied prob
3. **Key signals** — bullet points of what the data says (EV, form, crowd, trends)
4. **Recommendation tiers**:
   - **Conservative**: safe/reliable option (maybe cost-adjusted)
   - **Value**: best EV/risk ratio (the system's primary recommendation if any)
   - **Entertainment**: high-odds fun bets (scores, HTFT)
   - **Avoid**: what NOT to bet and why
5. **Optional**: alternative scenarios (e.g. what changes if a key player is out)

## Pitfalls

- **Knockout != group**: The model outputs 90-min SPF (includes regulation draw). A 25% draw probability in knockout is NOT just "possible draw" — it means extra time is likely.
- **no positive EV ≠ no bet possible**: It means no quantifiable edge on straight SPF. Handicap, goals, or double-chance might still have value.
- **市场倾向 ≠ model pick**: The market lean (from RQ/score direction) is a separate signal, not a replacement for the model.
- **Model_route = market_fallback**: The probabilities are just market odds normalized. They do NOT add independent insight.
- **世界杯爆冷高风险 (SKIP_WORLD_CUP_FALLBACK)**: Models use simple market odds without XGBoost for these. The skip flag is about data quality, not about this specific match being more or less risky.
- **365scores vote sample size**: vote_count < 1000 = noise. > 10,000 = meaningful crowd signal.
