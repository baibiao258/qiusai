# 365scores API Probe Technique

## SID Field (Sports Identifier)

The 365scores API (`webws.365scores.com/data/games/`) returns a `SID` field per game:

| SID | Sport |
|-----|-------|
| 1 | Football (soccer) |
| 2 | Basketball |
| 3 | Tennis |
| 7 | Baseball |
| 8 | Volleyball |

**Usage**: `extract_games(data, filter_sid=1)` to only get football.

## Key Endpoints Probed

| Endpoint | Status | Notes |
|----------|--------|-------|
| `webws.365scores.com/data/games/` | ✅ Working | Returns all current games. Parameters: `lang=1&app-type=1&cid=2&sport-type=1`. Also accepts `teamId=N` and `compId=N` as filters. |
| `webws.365scores.com/web/trends/?gameId=N` | ❌ 500 | Always returns 500, regardless of parameters. Dead endpoint. |
| `momentumsr.365scores.com/api/SportRadarMomentum/GetMomentum?partnerId=46927441` | ❌ HTML Widget | Returns SportRadar momentum HTML widget, not an API. Used for iframe embedding, not data extraction. |
| `365scores.com/football/match/...` | ❌ Pure SPA | No SSR (`__NEXT_DATA__` not found). Playwright needed for SPA data. |

## Fields extracted from `data/games/` API

### Already extracted (before 2026-06-14):
- `WhoWillWinReults` (vote1/X/2, total) ✅
- `Trend` (5-element per team) ✅
- `PopularityRank` ✅
- `FIFA ranking` (from Rankings array) ✅
- `HasLineups`, `HasStatistics`, `HasNews`, `HasBuzz` ✅
- `SocialStats` (Comments) ✅
- `Scrs[0:2]` (final score) ✅
- `HasDoubtful`, `HasMissingPlayers` ✅
- `Events` (goals, cards) ✅
- `Venue`, `Attendance` ✅

### Added 2026-06-14:
- `Scrs[2:4]` (half-time score → `score_ht`) ✅
- `Winner` (-1=draw, 1=home, 2=away → `winner`) ✅

### Present but NOT extracted (low value for prediction):
- `HasBets` — always False (0/425 checked)
- `Bookmakers` — empty array
- `OnTV`, `ShowTracker`, `HasFieldPositions`
- `Group`, `Stage`, `Round`, `Season`

## Competition Metadata (available in API response)

- `HasTbl` / `HasLiveTable` — league table exists (85 competitions have it)
- `HasSquads` — squad data exists (31 competitions, 258 teams)
- No squad/player/table REST endpoint discovered (all `data/members/`, `data/players/` etc. return 404)

## API Reliability

- No auth required, no special headers needed
- No rate limiting observed
- `app-type` variations (0/1/2/3) return same data + `Notifications`
- No historical archive — API always returns current/upcoming matches only

## Key Discovery: SID Filtering

The most impactful discovery: using `filter_sid=1` replaces keyword-based competition name filtering which had 76% false positive rate (non-football data). SID is 100% accurate and set by 365scores on their end.
