#!/usr/bin/env python3
"""
calibrated_predictor.py — 带置信度权重的双轨预测器
===================================================
K1 + K2 合并: 用 Elo 分歧度做 DC 置信度权重
"""
import json, os, sys, math, numpy as np, joblib
sys.path.insert(0, '/root')
DATA_DIR = '/root/data'

_loaded = False
_xgb_nat = _dc = _dc_club = _elo = _cal_nat = _team_matches = None

def _load():
    global _loaded, _xgb_nat, _dc, _dc_club, _elo, _cal_nat, _team_matches
    if _loaded: return
    _xgb_nat = joblib.load(f'{DATA_DIR}/xgb_model_nat.pkl')
    _dc = joblib.load(f'{DATA_DIR}/dc_model.pkl')
    _dc_club = joblib.load(f'{DATA_DIR}/dc_club.pkl')
    _elo = joblib.load(f'{DATA_DIR}/elo_ratings.pkl')
    cal_path = f'{DATA_DIR}/calibrators_nat.pkl'
    _cal_nat = joblib.load(cal_path) if os.path.exists(cal_path) else None
    
    # 从 international_results.json 计算每队出场数
    intl = json.load(open(f'{DATA_DIR}/international_results.json'))
    from collections import Counter
    _team_matches = Counter()
    for m in intl:
        _team_matches[m['home']] += 1
        _team_matches[m['away']] += 1
    _loaded = True

def _normalize(name):
    mapping = {
        'USA': 'United States',
        'Türkiye': 'Turkey',
        'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
        "Côte d'Ivoire": 'Ivory Coast',
    }
    return mapping.get(name, name)

def _dc_confidence(team):
    """DC 置信度: 基于训练数据中该队出场数"""
    n = _team_matches.get(team, 0)
    if n >= 200: return 1.0
    if n >= 100: return 0.9
    if n >= 50: return 0.8
    if n >= 20: return 0.7
    if n >= 10: return 0.5
    if n >= 5: return 0.3
    return 0.1

def _blend_with_market(dc_prob, elo_h, market_h, dc_conf):
    """带置信度权重的融合（v2: 修复平局灭绝bug）"""
    # dc_conf 控制 DC 权重, 低置信度时让 Elo + Market 主导
    
    # 从 Elo 差估算平局概率: 实力越接近, 平局概率越高
    elo_draw = max(0.05, 0.25 * (1 - abs(2*elo_h - 1)))
    elo_arr = np.array([
        max(0.01, elo_h - elo_draw/2),
        elo_draw,
        max(0.01, 1-elo_h - elo_draw/2)
    ])
    elo_arr /= elo_arr.sum()
    
    # 市场隐含概率同理
    mkt_draw = max(0.05, 0.25 * (1 - abs(2*market_h - 1)))
    mkt_arr = np.array([
        max(0.01, market_h - mkt_draw/2),
        mkt_draw,
        max(0.01, 1-market_h - mkt_draw/2)
    ])
    mkt_arr /= mkt_arr.sum()
    
    # 基底: Elo + Market 融合 (给市场赔率更高权重)
    base = 0.3 * elo_arr + 0.7 * mkt_arr
    base /= base.sum()
    
    # 最终: 按置信度融合 DC 和基底
    final = dc_conf * dc_prob + (1 - dc_conf) * base
    final /= final.sum()
    return final

def predict(home, away, market_h=0.0):
    """单场预测, 带 DC 置信度权重"""
    _load()
    home = _normalize(home)
    away = _normalize(away)
    
    eh = _elo.get(home, 1500)
    ea = _elo.get(away, 1500)
    elo_h = 1/(1+10**((ea-eh)/400))
    
    # ── Pipeline A: 国家队列 ──
    if home in _dc.team_idx_ and away in _dc.team_idx_:
        try:
            pr = _dc.predict_proba(home, away, neutral=True)
            if pr is not None:  # [H, D, A]
                dc_probs = np.clip([pr[2], pr[1], pr[0]], 0.01, 0.99)  # [A, D, H]
                lam_h, lam_a = _dc.predict_lambda(home, away, neutral=True)
                lam_h = max(0.1, min(5.0, lam_h))
                lam_a = max(0.1, min(5.0, lam_a))
                
                # DC 置信度 (取两队最小)
                conf = min(_dc_confidence(home), _dc_confidence(away))
                
                # XGBoost
                f28 = np.array([[(eh-ea)/400, lam_h, lam_a, lam_h-lam_a,
                    math.log(max(lam_h,0.01)/max(lam_a,0.01)),
                    dc_probs[0], dc_probs[1], dc_probs[2],
                    elo_h, 1-elo_h, market_h]])
                xgb_p = _xgb_nat.predict_proba(f28)[0]  # [A, D, H]
                
                # ── Isotonic 校准器已剥离 (同 daily_jczq.py, 2026-06-10 诊断) ──
                # 保留变量以防外部引用, 但不再调用 predict()
                # if _cal_nat:
                #     for j, key in enumerate(['away', 'draw', 'home']):
                #         c = _cal_nat.get(key)
                #         if c: xgb_p[j] = c.predict([[xgb_p[j]]])[0]
                #     xgb_p /= xgb_p.sum()
                
                # 混合: XGB 权重受 DC 置信度影响
                xgb_weight = 0.5 + 0.3 * conf  # 0.5~0.8
                dc_weight = 1 - xgb_weight
                hy = dc_weight * dc_probs + xgb_weight * xgb_p
                
                # 再用 Elo+Market 矫正极端值
                hy = _blend_with_market(hy, elo_h, market_h, conf)
                
                # ── Draw Correction Layer (参数化, 同 daily_jczq.py) ──
                if hy[1] < 0.15:  # p_draw < 15%
                    confidence = max(hy[2], hy[0])
                    draw_boost = 0.05 * (1.0 - confidence)
                    hy[1] += draw_boost
                    denom = hy[2] + hy[0] + 1e-10
                    hy[2] -= draw_boost * (hy[2] / denom)
                    hy[0] -= draw_boost * (hy[0] / denom)
                    s = hy.sum()
                    if s > 0: hy /= s
                
                return hy * 100, f'A:nat(conf={conf:.1f})'
        except: pass
    
    # ── Pipeline B: 俱乐部列 ──
    if home in _dc_club.team_idx_ and away in _dc_club.team_idx_:
        try:
            pr = _dc_club.predict_proba(home, away, neutral=False)
            if pr is not None:
                dc_probs = np.clip(pr, 0.01, 0.99)  # [H, D, A]
                hy = _blend_with_market(dc_probs, elo_h, market_h, 0.5)
                return hy * 100, 'B:club'
        except: pass
    
    # ── Fallback ──
    elo_draw = max(0.05, 0.25 * (1 - abs(2*elo_h - 1)))
    mkt_draw = max(0.05, 0.25 * (1 - abs(2*market_h - 1)))
    elo_3 = np.array([max(0.01, elo_h-elo_draw/2), elo_draw, max(0.01, 1-elo_h-elo_draw/2)])
    mkt_3 = np.array([max(0.01, market_h-mkt_draw/2), mkt_draw, max(0.01, 1-market_h-mkt_draw/2)])
    fb = 0.3 * elo_3 + 0.7 * mkt_3
    fb /= fb.sum()
    return fb * 100, 'C:elo+market'

if __name__ == '__main__':
    wc = json.load(open(f'{DATA_DIR}/wc_2026_odds_today.json'))
    seen = set()
    results = []
    for m in wc:
        k = (m['home_en'], m['away_en'])
        if k in seen: continue
        seen.add(k)
        hy, pipe = predict(m['home_en'], m['away_en'], m['market_implied_prob'])
        results.append({'date': m['date'], 'home': m['home_en'], 'away': m['away_en'],
                        'h': hy[2], 'd': hy[1], 'a': hy[0],  # hy=[A,D,H] → [H,D,A]
                        'oh': m['odds_h'], 'od': m['odds_d'], 'oa': m['odds_a'],
                        'mh': m['market_implied_prob']*100, 'pipe': pipe})
    
    results.sort(key=lambda r: (r['date'], -r['h']))
    
    print(f'{"日期":>12} {"主队":>22} {"客队":>22} {"H":>6} {"D":>6} {"A":>6} {"管线":>18} {"市H":>6}')
    print('-' * 100)
    for r in results:
        print(f'{r["date"]:>12} {r["home"][:20]:>22} {r["away"][:20]:<22} '
              f'{r["h"]:>5.1f}% {r["d"]:>5.1f}% {r["a"]:>5.1f}% '
              f'{r["pipe"]:>18} {r["mh"]:>5.1f}%')
    
    # 对比旧版极端值
    print(f'\n📊 置信度矫正效果 (对比旧版极端预测):')
    old_extremes = ['Jordan vs Argentina', 'Germany vs Curaçao', 'Spain vs Cape Verde', 'USA vs Australia']
    for match_str in old_extremes:
        parts = match_str.split(' vs ')
        home, away = parts[0], parts[1]
        m_info = [m for m in wc if m['home_en'] == home and m['away_en'] == away]
        if not m_info: continue
        m = m_info[0]
        hy, pipe = predict(home, away, m['market_implied_prob'])
        diff = abs(hy[2] - m['market_implied_prob']*100)  # hy[2]=H
        print(f'  {home:20s} vs {away:20s} → H={hy[2]:.1f}% 市场={m["market_implied_prob"]*100:.1f}% 偏差={diff:.1f}pp {pipe}')
