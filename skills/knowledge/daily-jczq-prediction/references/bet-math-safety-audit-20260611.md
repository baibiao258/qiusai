# bet_math.py Safety Audit — 2026-06-11

## Context
User requested audit of Kelly/EV/market_fallback for World Cup extreme upset risk. 7 vulnerabilities identified, 6 fixed in session.

## Fixes Applied

### Fix 1: Kelly Upper Bound Clamp
**File**: `bet_math.py` `compute_kelly()`
```python
MAX_SINGLE_BET = 0.05  # Single bet cap at 5% of bankroll
f_star = (prob * b - q) / b
return max(0.0, min(f_star, MAX_SINGLE_BET))
```
**Before**: f_star could reach 70% for strong edges → 17.5% quarter-Kelly exceeds 15% daily cap
**After**: Hard cap at 5% regardless of edge size

### Fix 2: is_sane_bet — 2 New Filters
**File**: `bet_math.py` `is_sane_bet()`
```python
# NEW Filter 4: High odds + low prob = World Cup upset zone
if s.odds > 5.0 and s.prob < 0.25:
    return False
# NEW Filter 5: market_fallback ALL plays blocked (not just score/htft)
if s.model_type == 'market_fallback' and s.play in ('胜平负', '让球'):
    return False
```
**Rationale**: market_fallback EV is circular (probability derived from same odds). Previously only blocked score/htft; SPF/RQ still passed.

### Fix 3: Correlation Discount in Total Position
**File**: `bet_math.py` `format_value_summary()`
```python
# OLD: total_quarter = sum(s.kelly_quarter for ...)
# NEW: Group by match, take max per group
match_groups = {}
for home, away, s in value_bets:
    key = f"{home}_{away}"
    match_groups.setdefault(key, []).append(s)
independent_total = sum(max(s.kelly_quarter for s in group) for group in match_groups.values())
```
**Why**: 5 bets on same match (SPF + RQ + score + goals + HTFT) are 95%+ correlated. Old sum double/triple-counted risk.

### Fix 4: Per-Match Correlation Warning
**File**: `bet_math.py` `format_ev_table()`
```python
if value_count > 1:
    lines.append(f"  ⚠️  同场 {value_count} 注高度正相关, 实际风险≈单注最大仓位")
```

### Fix 5: World Cup + market_fallback → SKIP
**File**: `daily_jczq.py` `compute_bet_action()`
```python
# NEW Rule 4
if model_type == 'market_fallback':
    if '世界杯' in league or 'World Cup' in league:
        return 'SKIP_WORLD_CUP_FALLBACK'
    return 'WATCH'
```
**Why**: market_fallback is circular EV. In World Cup with extreme volatility, even WATCH is too permissive.

## Fix 6: Dynamic Lambda by Tournament Stage (2026-06-11)
**File**: `daily_jczq.py` `fallback_market_predict()`
```python
STAGE_LAM = {
    'group': 2.55, 'last_16': 2.30, 'quarter': 2.10,
    'semi': 2.00, 'final': 1.90, 'third': 2.20,
}
stage = _detect_stage(market_row.get('league', ''))
lam_total = STAGE_LAM.get(stage, 2.55)
```
**Why**: Fixed 2.55 overestimated goals in knockout rounds. World Cup semi/finals average ~2.0 total goals.
**Pitfall**: `_detect_stage` keyword order matters — "1/8决赛" contains "决赛" but means round-of-16. Must check `last_16` before `final`. Fallback log added for unrecognized World Cup stage names.

## Test Results
All `bet_math.py` unit tests pass after fixes. Key change: Kelly f* for prob=55%, odds=2.10 now returns 0.050 (clamped) instead of 0.141 (unclamped).
