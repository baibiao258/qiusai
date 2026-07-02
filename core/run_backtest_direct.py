#!/usr/bin/env python3
from historical_backtest import fetch_history, JCZQ_LEAGUES, LOOKBACK_DAYS
from team_name_normalizer import normalize_match_pair
from predict_match import predict_match
import sys, os, json, math
from datetime import date, timedelta
import numpy as np

def run_it():
    all_matches = []
    for code, lname in JCZQ_LEAGUES:
        print(f"Fetching {lname}...")
        try:
            matches = fetch_history(code, LOOKBACK_DAYS)
            for m in matches:
                if m['status'] != 'FINISHED': continue
                sc = m['score']['fullTime']
                if sc['home'] is None: continue
                all_matches.append({
                    'date': m['utcDate'][:10],
                    'league': lname,
                    'home': m['homeTeam']['shortName'],
                    'away': m['awayTeam']['shortName'],
                    'h_score': sc['home'],
                    'a_score': sc['away'],
                })
        except Exception as e:
            print(f"  Error: {e}")
            
    print(f"\nTotal matches: {len(all_matches)}")
    all_matches.sort(key=lambda x: x['date'])
    
    results = []
    model_probs = []
    actual_results = []
    
    count = 0
    for m in all_matches:
        try:
            h, a = normalize_match_pair(m['home'], m['away'])
            res = predict_match(h, a, host_bonus=0.0, match_type='competitive')
            if isinstance(res, tuple) or not res:
                # 打印出错或跳过的队伍，以了解为何全是0个。
                print(f"Skipped {h} vs {a}: {res}")
                continue
            r = res
            mp = {'H': r['fin_h']/100, 'D': r['fin_d']/100, 'A': r['fin_a']/100}
            model_probs.append(mp)
            
            if m['h_score'] > m['a_score']: actual = 'H'
            elif m['h_score'] == m['a_score']: actual = 'D'
            else: actual = 'A'
            actual_results.append(actual)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} matches...")
        except Exception as e:
            print(f"Error on {m['home']} - {m['away']}: {e}")
            
    print(f"Done. Processed {count} matches.")
    
run_it()
