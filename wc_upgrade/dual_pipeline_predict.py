#!/usr/bin/env python3
"""
dual_pipeline_predict.py — 双轨预测管线
=======================================
Pipeline A (国家队): xgb_model_nat + dc_model (226队)
Pipeline B (俱乐部): dc_club + Elo + Market 融合 (2174队)
Fallback: Elo + Market 纯融合
"""
import json, os, sys, math, numpy as np, joblib
sys.path.insert(0, '/root')
DATA_DIR = '/root/data'

# 加载模型
def _lazy_load():
    global _xgb_nat, _dc, _dc_club, _elo, _cal_nat
    if '_xgb_nat' in globals() and _xgb_nat is not None:
        return
    _xgb_nat = joblib.load(f'{DATA_DIR}/xgb_model_nat.pkl')
    _dc = joblib.load(f'{DATA_DIR}/dc_model.pkl')
    _dc_club = joblib.load(f'{DATA_DIR}/dc_club.pkl')
    _elo = joblib.load(f'{DATA_DIR}/elo_ratings.pkl')
    cal_path = f'{DATA_DIR}/calibrators_nat.pkl'
    _cal_nat = joblib.load(cal_path) if os.path.exists(cal_path) else None

def _normalize_team(name):
    """标准化队名以匹配 dc_model 命名"""
    mapping = {
        'USA': 'United States',
        'Türkiye': 'Turkey',
        'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
        'Côte d\'Ivoire': 'Ivory Coast',
    }
    return mapping.get(name, name)

def predict_one_match(home, away, market_implied=0.0):
    """单场比赛预测, 自动路由到合适管线"""
    _lazy_load()
    home = _normalize_team(home)
    away = _normalize_team(away)
    eh = _elo.get(home, 1500)
    ea = _elo.get(away, 1500)
    op_h = 1/(1+10**((ea-eh)/400))
    op_a = 1/(1+10**((eh-ea)/400))
    
    # ── Pipeline A: 两队都在 dc_model 中 ──
    if home in _dc.team_idx_ and away in _dc.team_idx_:
        try:
            pr = _dc.predict_proba(home, away, neutral=True)
            if pr is not None:
                lam_h, lam_a = _dc.predict_lambda(home, away, neutral=True)
                lam_h = max(0.1, min(5.0, lam_h))
                lam_a = max(0.1, min(5.0, lam_a))
                dc_probs = np.clip([pr[2], pr[1], pr[0]], 0.01, 0.99)
                
                f28 = np.array([[(eh-ea)/400, lam_h, lam_a, lam_h-lam_a,
                    math.log(max(lam_h,0.01)/max(lam_a,0.01)),
                    dc_probs[0], dc_probs[1], dc_probs[2], op_h, op_a, market_implied]])
                
                xgb_p = _xgb_nat.predict_proba(f28)[0]  # [A, D, H]
                # Isotonic calibration
                if _cal_nat:
                    for j, key in enumerate(['away', 'draw', 'home']):
                        cal = _cal_nat.get(key)
                        if cal: xgb_p[j] = cal.predict([[xgb_p[j]]])[0]
                    xgb_p /= xgb_p.sum()
                
                hy = 0.3 * dc_probs + 0.7 * xgb_p
                hy /= hy.sum()
                return hy * 100, 'A:nat'
        except: pass
    
    # ── Pipeline B: 两队都在 dc_club 中 ──
    if home in _dc_club.team_idx_ and away in _dc_club.team_idx_:
        try:
            pr = _dc_club.predict_proba(home, away, neutral=False)
            if pr is not None:
                dc_probs = np.clip(pr, 0.01, 0.99)  # [H, D, A]
                
                # 融合: 0.5 dc_club + 0.2 elo + 0.3 market
                elo_arr = np.array([op_h, 0, op_a])
                market_arr = np.array([market_implied, 0, 1-market_implied])
                hy = 0.5 * dc_probs + 0.2 * elo_arr + 0.3 * market_arr
                hy /= hy.sum()
                return hy * 100, 'B:club'
        except: pass
    
    # ── Fallback: 纯 Elo + Market ──
    fallback = np.array([op_h, 0, op_a]) * 0.4 + np.array([market_implied, 0, 1-market_implied]) * 0.6
    fallback /= fallback.sum()
    return fallback * 100, 'C:elo+market'

def format_probs(p):
    """概率格式化 [H%, D%, A%]"""
    return p[0], p[1], p[2]

if __name__ == '__main__':
    # 跑今天世界杯64场
    wc = json.load(open(f'{DATA_DIR}/wc_2026_odds_today.json'))
    
    seen = set()
    results = []
    for m in wc:
        k = (m['home_en'], m['away_en'])
        if k in seen: continue
        seen.add(k)
        hy, pipe = predict_one_match(m['home_en'], m['away_en'], m['market_implied_prob'])
        results.append({
            'date': m['date'], 'home': m['home_en'], 'away': m['away_en'],
            'h': hy[0], 'd': hy[1], 'a': hy[2],
            'oh': m['odds_h'], 'od': m['odds_d'], 'oa': m['odds_a'],
            'mh': m['market_implied_prob']*100, 'pipe': pipe,
        })
    
    results.sort(key=lambda r: (r['date'], -r['h']))
    
    print(f'{"日期":>12} {"主队":>22} {"客队":>22} {"H":>6} {"D":>6} {"A":>6} {"管线":>12} {"市H":>6}')
    print('-' * 90)
    for r in results:
        print(f'{r["date"]:>12} {r["home"][:20]:>22} {r["away"][:20]:<22} '
              f'{r["h"]:>5.1f}% {r["d"]:>5.1f}% {r["a"]:>5.1f}% '
              f'{r["pipe"]:>12} {r["mh"]:>5.1f}%')
    
    # 统计管线分布
    from collections import Counter
    pipes = Counter(r['pipe'] for r in results)
    print(f'\n管线分布: {dict(pipes)}')
