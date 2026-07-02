# 365scores Enrichment Data Pipeline

## Purpose

Extract supplementary data from 365scores for today's buyable matches to enhance prediction transparency. Use this data for display enrichment only — do NOT feed into model core probability (that would create data leakage across different competition types).

## Available Data Categories

### 1. Public Vote / Popularity / Trend
- **Endpoint**: `webws.365scores.com/web/games/current/` (same as form update)
- **Fields**: `game.homeCompetitor.vote` / `game.awayCompetitor.vote` (only when populated)
- **Field**: `game.trend` — 1=home trending, -1=away trending, 0=neutral/stable
- **Limitation**: Not all games have vote data. Missing vote data → adjuster stays at 0pp.
- **Display format**: `365公众投票: 主XX% / 客XX% (样本N)` or `365趋势: 主↑`

### 2. Lineups / Starting XI
- **Endpoint**: `webws.365scores.com/web/game/?gameId={gameId}&sports=1`
- **Fields**: `game.homeCompetitor.lineup` / `game.awayCompetitor.lineup`
  - Each lineup entry: `{name, shirt, position, formationPlace}` 
- **Field**: `game.homeCompetitor.formation` / `game.awayCompetitor.formation`
- **Availability window**: `has_lineups=true` only when match <~24h from kickoff
- ⚠️ **Pitfall**: 500.com opening lists span 8 days (e.g. Jun 10 listing Jun 11-18). Only the closest 1-2 matches have lineup data. For the rest, `has_lineups=false` — this is not a bug.

### 3. Missing Players / Injuries
- **Endpoint**: Same as lineups: `webws.365scores.com/web/game/?gameId={gameId}&sports=1`
- **Fields**: `game.missingPlayers` array — `{name, reason, type, returnDate}`
  - `reason`: "Injured", "Suspended", "Illness", "Coach decision"
  - `type`: likely severity indicator
- **Availability**: `has_missing_players=true/false` — same timing constraint as lineups (>24h out = false)
- **Display format**: `伤病: 梅西(肌肉伤) / C罗(休息)`

### 4. Team News / Press Conference Updates
- **Endpoint**: Same as above
- **Fields**: `game.teamNews` — array of text summaries
- **Availability**: `has_news=true/false` — same timing constraint
- **Display format**: `赛前消息: 巴萨确认梅西轮休`

### 5. Head-to-Head (H2H)
- **Endpoint**: `webws.365scores.com/web/game/?gameId={gameId}&sports=1`
- **Fields**: `game.h2h.games` — array of past meetings with `{homeCompetitor.name, awayCompetitor.name, homeScore, awayScore, date}`
- **Availability**: Usually available for most matchups, even without lineups
- **Limitation**: May not cover all competition types (friendly H2H sparser than league)
- **Display format**: `历史交锋: 主4胜2平3负 (近6场)`

### 6. xG Data
- **Endpoint**: Same as above
- **Fields**: `game.homeCompetitor.xg` / `game.awayCompetitor.xg` (only when populated)
- **Availability**: Only for major leagues (PL/BL1/SA/PD) with advanced stats. Not available for friendlies.
- **Display format**: `xG: 主1.72 / 客0.89`

## Data Fetching Script

Located at: `/root/fetch_365scores.py`

```python
# Usage
python3 /root/fetch_365scores.py  # fetches today's game list
python3 /root/fetch_365scores.py --game-id 12345  # fetches single game detail
```

The script:
1. Fetches game list from `/web/games/current/`
2. Builds name mapping (365scores English name → 500.com Chinese name via `normalize_match_pair()`)
3. For matched games, fetches game detail from `/web/game/?gameId=N`
4. Returns structured dict with all available fields

## Daily Collection Cron (2026-06-10)

- Script: `/root/collect_365scores_daily.py` — fetches all available data every night
- Cron: `3fee9087ae2c` — 02:00 UTC daily
- Output: `/root/data/365scores/*.json` — one file per date
- Data retention: 30 days

## Display Integration

When displaying predictions, include a `365scores` section per match:

```
📡 365scores 数据:
  公众投票: 主58% / 客42% (样本2,341) | 趋势: 主↑
  H2H: 主3胜2平1负 (近6场)
  阵容: ✅已发布 (4-3-3) | ❌未公布 (>24h)
  伤病: 无 | 缺阵: 梅西(轮休)
  xG: 1.72 / 0.89 (仅主流联赛)
```

If a category is unavailable (>24h out match), note it concisely:
```
  📡 365scores 数据 (开赛>24h, 阵容/伤病数据不释放)
```

## Pitfalls

1. **Lineup availability ≠ prediction quality**: Missing lineup data doesn't mean model is wrong — it means the information just isn't public yet. Do NOT add uncertainty disclaimers unless the match is within 24h and still has no data.

2. **Multi-sport filtering**: The 365scores API (`sports=1` filter) still returns non-football sports. The `normalize_match_pair()` function in `fetch_365scores.py` filters by team name matching — if a match has unrecognizable team names, it won't be mapped. This is not a bug.

3. **Data freshness**: The daily cron runs at 02:00 UTC. For matches the same day, data may be 12-18h stale. For matches 7 days out, it's about as fresh as it gets (lineups data won't appear until <24h before anyway).

4. **GameId persistence**: The 365scores gameId changes between match listing and match day. The `gameId` from today's list may differ from tomorrow's detail endpoint. Always use the most recent fetch's gameId.
