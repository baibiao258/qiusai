#!/usr/bin/env python3
"""
WC 2026 Results Accumulator — save completed match scores for training data.

Call this after daily_wc_pipeline.py to pull completed matches from the
Odds API and accumulate them in /root/data/wc_completed_results.json.

Uses daysFrom=3 to catch matches that finished since the last run.
"""
import json, urllib.request, os
from datetime import date

DATA_DIR = '/root/data'
API_KEY = '425a7cb6604fe89fcbd46a524ac08a11'
TRAINING_FILE = os.path.join(DATA_DIR, 'wc_completed_results.json')

def log(msg):
    print(f'[results] {msg}', flush=True)

def main():
    url = ('https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/scores/'
           '?apiKey=' + API_KEY + '&daysFrom=3')
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=20) as resp:
        all_matches = json.loads(resp.read())
    remaining = resp.headers.get('x-requests-remaining', '?')
    completed = [e for e in all_matches if e.get('completed')]
    log(f'API: {len(all_matches)} total, {len(completed)} completed, quota={remaining}')

    existing = {}
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE) as f:
            for item in json.load(f):
                key = '|'.join([item['home'], item['away'], item.get('commence_time', '')])
                existing[key] = item
        log(f'Loaded {len(existing)} existing results')

    new_count = 0
    for c in completed:
        home, away = c['home_team'], c['away_team']
        commence = c.get('commence_time', '')
        scores = {s['name']: s['score'] for s in c.get('scores', [])}
        hs, aws = scores.get(home), scores.get(away)
        if hs is None or aws is None:
            continue

        result = 'H' if int(hs) > int(aws) else ('A' if int(hs) < int(aws) else 'D')
        key = '|'.join([home, away, commence])

        if key not in existing:
            existing[key] = {
                'home': home, 'away': away,
                'commence_time': commence, 'date': commence[:10],
                'home_score': int(hs), 'away_score': int(aws),
                'result': result, 'saved_at': str(date.today()),
            }
            new_count += 1
            log(f'  NEW: {home} vs {away} -> {hs}-{aws} ({result})')

    all_results = sorted(existing.values(), key=lambda x: (x['date'], x['home']))
    with open(TRAINING_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f'Saved {len(all_results)} total ({new_count} new)')

    # Summary
    print()
    print('=' * 70)
    print('  Completed Matches')
    print('=' * 70)
    for r in all_results:
        score = '{}-{}'.format(r['home_score'], r['away_score'])
        print('  {:<22} {:<22} {:>5} {}'.format(r['home'], r['away'], score, r['result']))

if __name__ == '__main__':
    main()
