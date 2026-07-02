#!/usr/bin/env python3
"""
历史模拟回测：用过去已决赛事跑预测管线，对比实际结果
"""
import sys, os, json, math
from datetime import date, timedelta
from collections import defaultdict

sys.path.extend([
    '/usr/local/lib/hermes-agent/models',
    '/usr/local/lib/hermes-agent/strategy',
])
import numpy as np
from scipy.stats import poisson as sp_poisson
from scipy.special import softmax, logit
import joblib

from predict_match import predict_match, _dc as dc_model, _xgb_model as xgb_model, _elo as elo_dict
from half_full_model import predict_half_full_probs

# 加载 form_state
DATA_DIR = '/root/data'
LOOKBACK_DAYS = 180
with open(os.path.join(DATA_DIR, 'form_state.json')) as f:
    form_state = json.load(f)
from predict_match import _dc as dc_model, _xgb_model as xgb_model, _elo as elo_dict

# 加载 form_state
with open(os.path.join(DATA_DIR, 'form_state.json')) as f:
    form_state = json.load(f)

def recent_form(team, n=5):
    if team not in form_state or len(form_state[team]) < 1:
        return [0.5, 0.0, 0.0, 0.0]
    games = form_state[team][-n:]
    if not games:
        return [0.5, 0.0, 0.0, 0.0]
    w = sum(1 for g in games if g[0] > g[1]) + sum(0.5 for g in games if g[0] == g[1])
    gf = sum(g[0] for g in games) / len(games)
    ga = sum(g[1] for g in games) / len(games)
    return [w / len(games), gf, ga, gf - ga]

def rps_score(y_true, y_proba):
    cdf_true = np.cumsum(y_true, axis=1)
    cdf_pred = np.cumsum(y_proba, axis=1)
    n_cat = y_proba.shape[1] - 1
    return np.mean(np.sum((cdf_true - cdf_pred) ** 2, axis=1) / n_cat)

def quick_validate(model_probs, actual_results):
    y_true = np.zeros((len(actual_results), 3))
    y_proba = np.zeros((len(actual_results), 3))
    label_to_idx = {'A': 0, 'D': 1, 'H': 2}
    for i, (mp, act) in enumerate(zip(model_probs, actual_results)):
        y_proba[i] = [mp['A'], mp['D'], mp['H']]
        y_true[i, label_to_idx[act]] = 1
    brier = np.mean(np.sum((y_true - y_proba) ** 2, axis=1))
    rps = rps_score(y_true, y_proba)
    return {'brier': brier, 'rps': rps, 'n': len(actual_results)}

def fetch_history(code, days_back):
    """从 football-data.org 拉取已完赛比赛"""
    import urllib.request
    API_KEY = os.environ.get('FOOTBALL_API_KEY', '5d07c80baa2645d0809b6ec96d6b49c6')
    HDR = {'X-Auth-Token': API_KEY, 'Accept': 'application/json'}
    
    end = date.today()
    start = end - timedelta(days=days_back)
    
    url = f"https://api.football-data.org/v4/competitions/{code}/matches?dateFrom={start}&dateTo={end}&status=FINISHED"
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode('utf-8')).get('matches', [])

# 竞彩联赛
JCZQ_LEAGUES = [
    ('PL','英超'), ('BL1','德甲'), ('PD','西甲'),
    ('SA','意甲'), ('FL1','法甲'), ('DED','荷甲'),
    ('PPL','葡超'), ('ELC','英冠'),
]

# 球队名标准化
from team_name_normalizer import normalize_match_pair

def run_backtest():
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
    
    # 跑预测
    results = []
    model_probs = []
    actual_results = []
    
    for m in all_matches:
        try:
            h, a = normalize_match_pair(m['home'], m['away'])
            res = predict_match(h, a, host_bonus=0.0, match_type='competitive')
            if isinstance(res, tuple) or not res:
                continue
            r = res
            
            # 模型概率
            mp = {'H': r['fin_h']/100, 'D': r['fin_d']/100, 'A': r['fin_a']/100}
            model_probs.append(mp)
            
            # 实际结果
            if m['h_score'] > m['a_score']: actual = 'H'
            elif m['h_score'] == m['a_score']: actual = 'D'
            else: actual = 'A'
            actual_results.append(actual)
            
            # 记录详情
            results.append({
                'date': m['date'],
                'league': m['league'],
                'match': f"{m['home']} vs {m['away']}",
                'score': f"{m['h_score']}:{m['a_score']}",
                'actual': actual,
                'pred_H': round(mp['H'], 4),
                'pred_D': round(mp['D'], 4),
                'pred_A': round(mp['A'], 4),
                'best_pick': r['bet_recommendation']['best_pick'],
                'margin_pp': r['bet_recommendation']['margin_pp'],
                'action': r['bet_recommendation']['action'],
            })
        except Exception as e:
            print(f"Error on {m}: {e}")
    
    # 统计
    print(f"\n{'='*60}")
    print(f"VALIDATION ({len(model_probs)} matches)")
    print(f"{'='*60}")
    
    if model_probs:
        metrics = quick_validate(model_probs, actual_results)
        print(f"Brier: {metrics['brier']:.4f}")
        print(f"RPS:   {metrics['rps']:.4f}")
        print(f"Count: {metrics['n']}")
        
        # 准确率
        correct = sum(1 for mp, act in zip(model_probs, actual_results) 
                     if ['A','D','H'][[mp['A'],mp['D'],mp['H']].index(max(mp.values()))] == act)
        print(f"Accuracy: {correct}/{len(model_probs)} = {correct/len(model_probs)*100:.1f}%")
        
        # 按联赛分组
        by_league = defaultdict(lambda: {'probs':[], 'actuals':[]})
        for res in results:
            by_league[res['league']]['probs'].append(
                {'H':res['pred_H'], 'D':res['pred_D'], 'A':res['pred_A']})
            by_league[res['league']]['actuals'].append(res['actual'])
        
        for lg, data in by_league.items():
            m = quick_validate(data['probs'], data['actuals'])
            print(f"  {lg}: Brier={m['brier']:.4f} RPS={m['rps']:.4f} N={m['n']}")
    
    # 保存详细结果
    with open('/root/data/backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved details to /root/data/backtest_results.json")

if __name__ == '__main__':
    run_backtest()