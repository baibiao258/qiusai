# 365scores API for Form Data Update

## Overview
365scores provides a public API for match results that covers international friendlies, unlike football-data.org.

## API Endpoint
```
https://webws.365scores.com/web/games/current/?sports=1&date=YYYY-MM-DD&games=1&startIndex=0&count=200&withTop=true
```

## Required Headers
```python
HEADERS = {
    "Accept": "application/json",
    "Referer": "https://www.365scores.com/",
    "User-Agent": "Mozilla/5.0 (compatible; FormUpdater/1.0)",
}
```

## Response Structure
```json
{
  "games": [
    {
      "homeCompetitor": {
        "name": "Japan",
        "score": 2
      },
      "awayCompetitor": {
        "name": "Thailand",
        "score": 0
      },
      "statusGroup": 4,
      "statusText": "Finished"
    }
  ]
}
```

## Key Fields
- `game.homeCompetitor.name` → team name
- `game.homeCompetitor.score` → home goals
- `game.awayCompetitor.score` → away goals
- `game.statusGroup` → 4 = finished

## Filter Rules
- Skip if `statusGroup != 4` (not finished)
- Skip if `statusText` contains "cancelled", "postponed", "abandoned"
- Skip if team name contains "(W)" (women's matches)

## Deduplication
Old format: `[home_goals, away_goals]`
New format: `[home_goals, away_goals, "YYYY-MM-DD"]`

```python
if not any(x[0]==gh and x[1]==ga and (len(x)<3 or x[2]==date_str)
           for x in form_state[team]):
    form_state[team].append([gh, ga, date_str])
```

## Rate Limiting
- Max 2 requests per second
- Use `time.sleep(0.5)` between requests

## Script
`/root/update_form_from_365.py`

## Usage
```bash
# Update last 7 days
python3 /root/update_form_from_365.py --days 7

# Update last 2 days (for cron)
python3 /root/update_form_from_365.py --days 2
```

## Cron Setup
```bash
0 6 * * * cd /root && python3 /root/update_form_from_365.py --days 2 >> /root/data/form_update.log 2>&1
```

## Verification
```bash
ls -la /root/data/form_state.json  # check mtime
python3 -c "import json; fs=json.load(open('/root/data/form_state.json')); print(len(fs))"
python3 -c "import json; fs=json.load(open('/root/data/form_state.json')); print(fs.get('Japan', [])[-3:])"
```

## Limitations
- Some teams may not be in the response
- Historical data limited to recent matches
- May not cover all leagues/competitions
