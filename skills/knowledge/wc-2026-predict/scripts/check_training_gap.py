#!/usr/bin/env python3
"""
Post-cron backfill: append new WC completed results to training_data_with_odds.json.

Run AFTER accumulate_results.py in the daily cron sequence:
    cd /root/wc_2026_upgrade
    python3 daily_wc_pipeline.py
    python3 accumulate_results.py
    python3 /root/.hermes/skills/knowledge/wc-2026-predict/scripts/check_training_gap.py

Step 1: Load wc_completed_results.json and training_data_with_odds.json
Step 2: Find completed matches missing from training data
Step 3: For each missing match, look up market odds:
   a) First try wc_pred_{date}.json (same-day prediction file)
   b) Fall back to searching ALL wc_odds_*.json files (match may be in earlier dates)
Step 4: Construct the training record with market_h/d/a and implied probs
Step 5: Save updated training_data_with_odds.json

Known gaps:
- Some WC matches are never carried by The Odds API (e.g. Australia vs Turkey, 2026-06-14).
  These are appended with market_h=0 and without implied probs.
- Team naming is consistent between scores endpoint and odds files (both use the same
  display names like 'Turkey' not 'Türkiye', 'South Korea' not 'Korea Republic').
- The odds files use flat structure (home/away/odds_h/odds_d/odds_a) — NOT nested
  bookmaker > markets > outcomes format.
"""

import json, os, sys, glob

# Paths
COMPLETED_PATH = '/root/data/wc_completed_results.json'
TRAINING_PATH = '/root/data/training_data_with_odds.json'
ODDS_PATTERN = '/root/data/wc_odds_*.json'
PRED_PATTERN = '/root/data/wc_pred_*.json'


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_odds_lookup():
    """Build (home, away) -> (odds_h, odds_d, odds_a) from ALL odds files."""
    lookup = {}
    for fpath in sorted(glob.glob(ODDS_PATTERN)):
        try:
            odds = load_json(fpath)
        except (json.JSONDecodeError, IOError):
            continue
        date_str = os.path.basename(fpath).replace('wc_odds_', '').replace('.json', '')
        for m in odds:
            h = m.get('home', '')
            a = m.get('away', '')
            oh = m.get('odds_h', 0)
            od = m.get('odds_d', 0)
            oa = m.get('odds_a', 0)
            if h and a and oh > 0 and od > 0 and oa > 0:
                key = (h, a)
                # Prefer earlier date's odds (closer to match time)
                if key not in lookup:
                    lookup[key] = (oh, od, oa, date_str)
    return lookup


def find_prediction(date, home, away):
    """Look up a match's prediction/odds from the same-day pred file."""
    pred_path = f"/root/data/wc_pred_{date}.json"
    if not os.path.exists(pred_path):
        return None
    try:
        preds = load_json(pred_path)
    except (json.JSONDecodeError, IOError):
        return None
    for p in preds:
        if p.get('home', '') == home and p.get('away', '') == away:
            return p
    return None


def construct_training_record(r, odds):
    """Build a training record dict from a completed result and optional odds tuple."""
    oh, od, oa, odds_source = odds if odds else (0, 0, 0, None)
    rec = {
        'date': r['date'],
        'home_en': r['home'],
        'away_en': r['away'],
        'tournament': 'FIFA World Cup',
        'spf_result': r['result'],
        'home_goals': r['home_score'],
        'away_goals': r['away_score'],
        'home_xg': 0.0,
        'away_xg': 0.0,
        'market_h': oh,
        'market_d': od,
        'market_a': oa,
        'stage': 'group_stage',
        'source': 'theoddsapi',
    }
    if all(x > 0 for x in [oh, od, oa]):
        margin = 1/oh + 1/od + 1/oa
        rec['market_implied_h'] = (1/oh) / margin
        rec['market_implied_d'] = (1/od) / margin
        rec['market_implied_a'] = (1/oa) / margin
    else:
        rec['market_implied_h'] = 0.0
        rec['market_implied_d'] = 0.0
        rec['market_implied_a'] = 0.0
    return rec


def main():
    results = load_json(COMPLETED_PATH)
    training = load_json(TRAINING_PATH)
    odds_lookup = build_odds_lookup()

    print(f"✅ Completed results DB: {len(results)} total records")
    print(f"📚 Training data: {len(training)} total records")
    print(f"🔍 Odds lookup: {len(odds_lookup)} unique matchups across all dates")

    # Build seen set from existing training
    seen = set()
    for s in training:
        key = (s.get('date', ''), s.get('home_en', ''), s.get('away_en', ''))
        seen.add(key)

    # Find missing matches
    missing = [r for r in results if (r['date'], r['home'], r['away']) not in seen]

    if not missing:
        print("✅ All completed matches already in training data — nothing to append.")
        return

    print(f"\n🔍 Missing from training: {len(missing)} records")

    appended = 0
    no_odds_matches = []

    for r in missing:
        home, away, date = r['home'], r['away'], r['date']

        # Strategy A: try same-day pred file
        p = find_prediction(date, home, away)
        if p:
            oh = p.get('odds_h', 0)
            od = p.get('odds_d', 0)
            oa = p.get('odds_a', 0)
            if oh > 0 and od > 0 and oa > 0:
                rec = construct_training_record(r, (oh, od, oa, f"pred_{date}"))
                training.append(rec)
                appended += 1
                print(f"   ✅ (pred) {date}: {home} {r['home_score']}-{r['away_score']} {away}")
                continue

        # Strategy B: search across ALL odds files (match may be in earlier date)
        entry = odds_lookup.get((home, away))
        if not entry:
            entry = odds_lookup.get((away, home))  # home/away may be swapped
        if entry:
            rec = construct_training_record(r, entry)
            training.append(rec)
            appended += 1
            print(f"   ✅ (odds@{entry[3]}) {date}: {home} {r['home_score']}-{r['away_score']} {away}")
            continue

        # Strategy C: no odds available, append with zero odds
        rec = construct_training_record(r, None)
        training.append(rec)
        appended += 1
        no_odds_matches.append(f"   ⚠ (no odds) {date}: {home} {r['home_score']}-{r['away_score']} {away}")
        print(no_odds_matches[-1])

    # Save
    json.dump(training, open(TRAINING_PATH, 'w'), indent=2)
    print(f"\n💾 Saved {appended} new records → {TRAINING_PATH}")
    print(f"📚 Training data now has {len(training)} records")

    if no_odds_matches:
        print(f"\n⚠ {len(no_odds_matches)} match(es) without market odds (not covered by The Odds API):")
        for m in no_odds_matches:
            print(m)


if __name__ == '__main__':
    main()
