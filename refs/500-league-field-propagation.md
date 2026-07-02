# 500.com League Field Propagation Fix (2026-06-11)

## Problem

All international matches (including World Cup) were incorrectly classified as "友谊赛" (friendlies), causing `compute_bet_action()` to return `WATCH_FRIENDLY` instead of `RECOMMEND`.

## Root Cause

The data field existed in the scraper output but was dropped in the intermediate function:

```
async_500_scraper.py  →  scrape_500_odds_today()  →  build_prediction_bundle()  →  compute_bet_action()
       ↓                           ↓                          ↓                          ↓
  league: "世界杯"          (field MISSING)           hard-coded '友谊赛'        → WATCH_FRIENDLY
```

## Data Flow (Before Fix)

1. `async_500_scraper.py` extracts `simpleleague` attribute from HTML:
   ```python
   league = attrs.get('simpleleague', '')  # e.g., "世界杯", "英超"
   ```

2. Returns dict with `'league': league` field included

3. `scrape_500_odds_today()` in `daily_jczq.py` receives this data but **does NOT include `league` in its return dict**:
   ```python
   result.append({
       'code': code,
       'home_cn': home_cn,
       'away_cn': away_cn,
       'time': row.get('endtime', ''),
       # 'league' field MISSING here!
       'odds_h': std_h,
       ...
   })
   ```

4. `build_prediction_bundle()` receives `league` parameter from caller, which hard-codes `'友谊赛'`:
   ```python
   bundle = build_prediction_bundle(m5['code'], home_cn, away_cn, m5['time'], '友谊赛', p, m5, score_meta)
   ```

5. `compute_bet_action()` checks: `if '友谊赛' in league: return 'WATCH_FRIENDLY'`

## Fix (4 locations in `/root/daily_jczq.py`)

### 1. `scrape_500_odds_today()` return dict (line ~598)
```python
result.append({
    'code': code,
    'home_cn': home_cn,
    'away_cn': away_cn,
    'time': row.get('endtime', ''),
    'league': row.get('league', ''),  # ADDED: propagate league from scraper
    'odds_h': std_h,
    ...
})
```

### 2. `build_prediction_bundle()` call (line ~1828)
```python
# Before:
bundle = build_prediction_bundle(m5['code'], home_cn, away_cn, m5['time'], '友谊赛', p, m5, score_meta)

# After:
league_name = m5.get('league', '') or '友谊赛'  # Use league from 500.com, fallback to '友谊赛'
bundle = build_prediction_bundle(m5['code'], home_cn, away_cn, m5['time'], league_name, p, m5, score_meta)
```

### 3. `compute_fatigue_features()` call (line ~1842)
```python
# Before:
fatigue = compute_fatigue_features(
    clean_home, clean_away, m5.get('time', ''), '友谊赛', a_data['future_fixtures']
)

# After:
fatigue = compute_fatigue_features(
    clean_home, clean_away, m5.get('time', ''), league_name, a_data['future_fixtures']
)
```

### 4. Header print (line ~1816)
```python
# Before:
print(f"\n  📋 国际友谊赛 ({len(_500_odds)}场)")

# After:
print(f"\n  📋 国际赛事（来自500.com，{len(_500_odds)}场）")
```

## Verification

```bash
# 1. Check scraper output includes league
python3 /root/wc_2026_upgrade/async_500_scraper.py 2026-06-11 269 | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result'][0]['league'])"
# Output: "世界杯"

# 2. Check output shows correct league
python3 /root/daily_jczq.py 2>&1 | grep -E "(世界杯|bet_action)"
# Output: 周四001 墨西哥 vs 南非  ()  [世界杯]
#         bet_action: RECOMMEND
```

## Lesson

When adding new fields to scraper output, trace the ENTIRE data path:
1. Scraper → `scrape_500_odds_today()` return dict ← **MUST include field**
2. Return dict → `build_prediction_bundle()` → `compute_bet_action()`
3. Any missing link = field silently lost = hard-coded fallback kicks in

## Related Pitfalls

- See `references/500-api-spf-nspf-quirk.md` for spf/nspf field mapping issues
- See `references/async-scraper-architecture.md` for scraper architecture
- Common Pitfall #0 in SKILL.md documents this exact issue
