#!/usr/bin/env python3
"""
daily_jczq.py — 每日竞彩足球预测 (v4)
=========================================
数据流:
  1) football-data.org ← 当日赛程
  2) 500.com ← 今日可买场次 + 5玩法赔率
  3) 365scores ← 投票/趋势/人气增强
  4) 优先 DC+XGBoost 混合模型 (国际赛)
     回退 泊松+Elo (联赛)
  5) 输出并统一落盘到 predictions_log.csv

直接运行:   python3 daily_jczq.py
Cron自动:  cronjob create ...
"""
import csv
import json
import math
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, date, timedelta
from collections import defaultdict

import numpy as np
from scipy.stats import poisson as sp_poisson
import bet_math
from scraper_500_analysis import scrape_500_analysis, enrich_bundle_with_500, format_500_analysis_lines
from fatigue_features import compute_fatigue_features, fatigue_adjustment, format_fatigue_lines
from pipeline.probability import (
    poisson_pmf,
    elo_expected,
    dc_tau,
    compute_rq_probs,
    implied_probs_from_odds,
    compute_goals_distribution,
    compute_score_topn,
    compute_htft_topn_math,
    compute_dynamic_xgb_weight,
    rps_score,
    brier_decomposition_multiclass,
    quick_validate,
    format_pct,
)
from pipeline.scraper import (
    apply_euro_fallback,
    fetch_live_odds_map,
    scrape_500_odds_today,
)
from pipeline.data_loader import (
    api_get,
    fetch_league_history,
    get_today_matches,
    load_365scores_today,
    build_365_map,
)


# ── 常量 ──
API_KEY = os.environ.get('FOOTBALL_API_KEY', '5d07c80baa2645d0809b6ec96d6b49c6')
HDR = {'X-Auth-Token': API_KEY, 'Accept': 'application/json'}
MAX_GOALS = 6
BACKTEST_SCRIPT = '/root/.hermes/scripts/backtest_jczq.py'
PREDICTIONS_LOG = '/root/data/predictions_log.csv'
MODEL_VERSION = 'daily_jczq_v3'

# 竞彩足球覆盖联赛 (football-data.org codes)
JCZQ_LEAGUES = [
    ('PL','英超'), ('BL1','德甲'), ('PD','西甲'),
    ('SA','意甲'), ('FL1','法甲'), ('DED','荷甲'),
    ('PPL','葡超'), ('ELC','英冠'),
]

HTFT_ORDER = ['胜胜','胜平','胜负','平胜','平平','平负','负胜','负平','负负']
HTFT_SHORT_MAP = {
    '胜胜': 'HH', '胜平': 'HD', '胜负': 'HA',
    '平胜': 'DH', '平平': 'DD', '平负': 'DA',
    '负胜': 'AH', '负平': 'AD', '负负': 'AA',
}
HTFT_DISPLAY_MAP = {
    'HH': '胜胜', 'HD': '胜平', 'HA': '胜负',
    'DH': '平胜', 'DD': '平平', 'DA': '平负',
    'AH': '负胜', 'AD': '负平', 'AA': '负负',
    'H/H': '胜胜', 'H/D': '胜平', 'H/A': '胜负',
    'D/H': '平胜', 'D/D': '平平', 'D/A': '平负',
    'A/H': '负胜', 'A/D': '负平', 'A/A': '负负',
}




# ── 共享混合模型加载 (lazy) ──

_dc_model = None
_xgb_model = None
_elo_dict = None
_calibrators = None  # Isotonic校准器 (home/draw/away)
_xgb_model_30 = None  # A/B测试: 30维模型 (含market_implied)

# ── 俱乐部模型 (lazy) ──
_dc_club = None
_xgb_club = None
_elo_club = None
_calibrators_club = None
_form_club = None
_xg_club = None

def _load_shared_models():
    global _dc_model, _xgb_model, _elo_dict
    global _xgb_simple, _cal_simple
    global _xgb_model_30
    if _dc_model is not None:
        return
    import joblib
    DATA_DIR = '/root/data'
    _dc_model = joblib.load(os.path.join(DATA_DIR, 'dc_model.pkl'))
    _xgb_model = joblib.load(os.path.join(DATA_DIR, 'xgb_model_29.pkl'))
    _elo_dict = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))
    # A/B测试: 加载30维模型 (不覆盖主路由)
    m30_path = os.path.join(DATA_DIR, 'xgb_model_30.pkl')
    if os.path.exists(m30_path):
        _xgb_model_30 = joblib.load(m30_path)
    cal_path = os.path.join(DATA_DIR, 'calibrators.pkl')
    if os.path.exists(cal_path):
        global _calibrators
        _calibrators = joblib.load(cal_path)
    simple_model_path = os.path.join(DATA_DIR, 'xgb_model_simple.pkl')
    simple_cal_path = os.path.join(DATA_DIR, 'calibrators_simple.pkl')
    if os.path.exists(simple_model_path):
        _xgb_simple = joblib.load(simple_model_path)
    if os.path.exists(simple_cal_path):
        _cal_simple = joblib.load(simple_cal_path)


def _load_club_models():
    global _dc_club, _xgb_club, _elo_club, _calibrators_club, _form_club, _xg_club
    if _dc_club is not None:
        return
    import joblib
    DATA_DIR = '/root/data'
    club_dc_path = os.path.join(DATA_DIR, 'dc_model_club.pkl')
    club_xgb_path = os.path.join(DATA_DIR, 'xgb_model_club.pkl')
    club_elo_path = os.path.join(DATA_DIR, 'elo_club.pkl')
    club_cal_path = os.path.join(DATA_DIR, 'calibrators_club.pkl')
    club_form_path = os.path.join(DATA_DIR, 'form_club.json')

    if not all(os.path.exists(p) for p in [club_dc_path, club_xgb_path, club_elo_path]):
        return

    _dc_club = joblib.load(club_dc_path)
    _xgb_club = joblib.load(club_xgb_path)
    _elo_club = joblib.load(club_elo_path)
    if os.path.exists(club_cal_path):
        _calibrators_club = joblib.load(club_cal_path)
    if os.path.exists(club_form_path):
        with open(club_form_path) as f:
            _form_club = json.load(f)
    club_xg_path = os.path.join(DATA_DIR, 'xg_proxy_club.json')
    if os.path.exists(club_xg_path):
        with open(club_xg_path) as f:
            _xg_club = json.load(f)


def _try_hybrid_predict(home, away):
    """尝试 DC+XGBoost 混合预测 (仅国际赛球队). 成功返回 dict, 失败返回 None."""
    try:
        from team_name_normalizer import normalize_match_pair
        h, a = normalize_match_pair(home, away)
        _load_shared_models()

        from predict_match import _load_form_state as _p_load_fs
        fs = _p_load_fs()

        lam_h, lam_a = _dc_model.predict_lambda(h, a, neutral=True)
        if lam_h is None or lam_a is None:
            return None

        dc_p = _dc_model.predict_proba(h, a, neutral=True)
        dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])

        eh = _elo_dict.get(h, 1500)
        ea = _elo_dict.get(a, 1500)

        from predict_match import recent_form as pm_recent_form
        fh5 = pm_recent_form(h, 5)
        fa5 = pm_recent_form(a, 5)

        op_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        op_a = 1 / (1 + 10 ** ((eh - ea) / 400))

        b15 = [
            (eh - ea) / 400, lam_h, lam_a, lam_h - lam_a,
            math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
            dc_p[0], dc_p[1], dc_p[2],
            fh5[0], fa5[0],
            fh5[1] - fa5[2], fa5[1] - fh5[2],
            fh5[1] - fa5[1], fh5[0] - fa5[0],
            1,
        ]
        from feature_helper import build_gold_features
        gold = build_gold_features(home, away, match_type='competitive')
        odds_feat = [op_h, op_a, 0.0]
        form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]
        feat = np.array([b15 + gold + odds_feat + form_feat])

        xgb_p = _xgb_model.predict_proba(feat)[0]

        # ── A/B测试: 30维模型并行推理 (不影响主路由) ──
        xgb30_p = None
        if _xgb_model_30 is not None and op_h > 0:
            try:
                market_implied = 1.0 / op_h
                feat_30 = np.array([b15 + gold + odds_feat + form_feat + [market_implied]])
                xgb30_raw = _xgb_model_30.predict_proba(feat_30)[0]
                # 30维也做DC融合
                xgb_w30, dc_w30, _ = compute_dynamic_xgb_weight(xgb30_raw)
                xgb30_hybrid = dc_w30 * dc_ado + xgb_w30 * xgb30_raw
                s30 = xgb30_hybrid.sum()
                if s30 > 0: xgb30_hybrid /= s30
                xgb30_p = xgb30_hybrid
            except Exception:
                pass
        
        # ── 基于熵的动态融合权重 (替代 min_games 硬阈值) ──
        xgb_w, dc_w, _conf = compute_dynamic_xgb_weight(xgb_p)
        hybrid = dc_w * dc_ado + xgb_w * xgb_p
        s = hybrid.sum()
        if s > 0: hybrid = hybrid / s

        # ── Isotonic 校准 (国际赛) ──
        if _calibrators:
            calibrated = np.zeros(3)
            for j, key in enumerate(['away', 'draw', 'home']):
                if key in _calibrators:
                    calibrated[j] = _calibrators[key].predict([hybrid[j]])[0]
                else:
                    calibrated[j] = hybrid[j]
            s = calibrated.sum()
            if s > 0: calibrated = calibrated / s
            hybrid = calibrated

        # ── 并行模型预测 (simple_model) ──
        simple_pred = None
        simple_conf = 0
        if '_xgb_simple' in globals() and _xgb_simple is not None:
            try:
                # 使用 market_odds 作为特征 (从 op_h 推算)
                market_odds_h = 1.0 / max(op_h, 0.01)
                simple_feat = np.array([[
                    market_odds_h,
                    fh5[0], fh5[1], fh5[2],
                    fa5[0], fa5[1], fa5[2],
                ]])
                simple_proba = _xgb_simple.predict_proba(simple_feat)[0]
                if '_cal_simple' in globals() and _cal_simple is not None:
                    simple_cal = np.zeros(3)
                    for j, key in enumerate(['home', 'draw', 'away']):
                        if key in _cal_simple:
                            simple_cal[j] = _cal_simple[key].predict([simple_proba[j]])[0]
                        else:
                            simple_cal[j] = simple_proba[j]
                    s = simple_cal.sum()
                    if s > 0: simple_cal /= s
                    simple_proba = simple_cal
                simple_pred = ['H', 'D', 'A'][simple_proba.argmax()]
                simple_conf = simple_proba.max()
            except Exception as e:
                pass

        hp = [sp_poisson.pmf(k, lam_h) for k in range(MAX_GOALS + 1)]
        ap = [sp_poisson.pmf(k, lam_a) for k in range(MAX_GOALS + 1)]
        bp, bh, ba = 0, 0, 0
        for hg in range(MAX_GOALS + 1):
            for ag in range(MAX_GOALS + 1):
                p = hp[hg] * ap[ag]
                if p > bp:
                    bp, bh, ba = p, hg, ag

        hw, dr, aw = float(hybrid[2]), float(hybrid[1]), float(hybrid[0])
        result = 'H' if hw > dr and hw > aw else ('D' if dr > hw and dr > aw else 'A')

        probs_sorted = sorted([hw, dr, aw], reverse=True)
        margin_pp = (probs_sorted[0] - probs_sorted[1]) * 100
        best_label = ['H', 'D', 'A'][[hw, dr, aw].index(probs_sorted[0])]

        from predict_match import _load_form_state
        fs = _load_form_state()
        home_has = home in fs and len(fs[home]) >= 1
        away_has = away in fs and len(fs[away]) >= 1
        form_gap = (not home_has) or (not away_has)

        if form_gap:
            bet_action = 'SKIP_DATA'
        elif margin_pp >= 10:
            bet_action = 'BET'
        else:
            bet_action = 'SKIP'

        return {
            'probs': {'H': round(hw, 4), 'D': round(dr, 4), 'A': round(aw, 4)},
            'score': f"{bh}-{ba}",
            'result': result,
            'min_odds': {k: round(1.02 / max(v, 1e-6), 2) for k, v in [('H', hw), ('D', dr), ('A', aw)]},
            'matches_data': (0, 0),
            'lambda_ft': {'home': float(lam_h), 'away': float(lam_a)},
            'model': 'hybrid',
            'form': {
                'home_gf': round(fh5[1], 2), 'home_ga': round(fh5[2], 2),
                'away_gf': round(fa5[1], 2), 'away_ga': round(fa5[2], 2),
            },
            'simple_pred': simple_pred,
            'simple_conf': round(simple_conf, 4) if simple_conf else 0,
            # A/B测试: 30维模型概率 (DC+XGB融合后, 未校准)
            'pred30_h': round(float(xgb30_p[2]), 4) if xgb30_p is not None else None,
            'pred30_d': round(float(xgb30_p[1]), 4) if xgb30_p is not None else None,
            'pred30_a': round(float(xgb30_p[0]), 4) if xgb30_p is not None else None,
            'bet_recommendation': {
                'action': bet_action,
                'margin_pp': round(margin_pp, 1),
                'best_pick': best_label,
                'best_prob_pct': round(probs_sorted[0] * 100, 1),
            },
        }
    except Exception:
        return None

# ── 500.com 赔率抓取 (市场校准用) ──



# ── 数据 ──


def train(all_matches):
    """训练泊松+Elo (后备模型, 用于联赛)"""
    cutoff = date.today().isoformat()
    stats = defaultdict(lambda: {'wg':0,'wc':0,'ws':0,'m':0})
    for m in all_matches:
        if m['date']>=cutoff: continue
        days = (datetime.strptime(cutoff,'%Y-%m-%d')-datetime.strptime(m['date'],'%Y-%m-%d')).days
        w = 0.5**(max(days,0)/180)
        for team,gf,ga in [(m['home'],m['h_score'],m['a_score']),(m['away'],m['a_score'],m['h_score'])]:
            s=stats[team]; s['wg']+=gf*w; s['wc']+=ga*w; s['ws']+=w; s['m']+=1
    total_wg=sum(s['wg'] for s in stats.values())
    ga=total_wg/max(sum(s['ws'] for s in stats.values()),1)
    ts={}
    for team,s in stats.items():
        avg_gf=s['wg']/max(s['ws'],0.001); avg_ga=s['wc']/max(s['ws'],0.001)
        ts[team]={'attack':avg_gf/max(ga,0.01),'defense':avg_ga/max(ga,0.01),'m':s['m']}
    elo=defaultdict(lambda:1500.0)
    for m in all_matches:
        if m['date']>=cutoff: continue
        h,a=m['home'],m['away']
        e_h=elo_expected(elo[h],elo[a])
        sh,sa=(1.0,0.0) if m['h_score']>m['a_score'] else((0.5,0.5)if m['h_score']==m['a_score'] else(0.0,1.0))
        elo[h]+=32*(sh-e_h); elo[a]+=32*(sa-(1-e_h))
    return ts,ga,dict(elo)



def predict_match_legacy(home,away,ts,ga,elo_r):
    """后备: 泊松+Elo 预测 (联赛)"""
    h_ts=ts.get(home,{'attack':1.0,'defense':1.0})
    a_ts=ts.get(away,{'attack':1.0,'defense':1.0})
    lam_h=ga*h_ts['attack']*a_ts['defense']*1.05
    lam_a=ga*a_ts['attack']*h_ts['defense']*0.95
    lam_h=max(0.1,min(5.0,lam_h)); lam_a=max(0.1,min(5.0,lam_a))
    hw,dr,aw=0.0,0.0,0.0
    for hg in range(MAX_GOALS+1):
        for ag in range(MAX_GOALS+1):
            p=poisson_pmf(hg,lam_h)*poisson_pmf(ag,lam_a)
            if hg>ag: hw+=p
            elif hg==ag: dr+=p
            else: aw+=p
    t=hw+dr+aw; hw,dr,aw=hw/t,dr/t,aw/t
    eh=elo_r.get(home,1500); ea=elo_r.get(away,1500)
    ep=elo_expected(eh,ea); w=0.55
    hw=hw*w+ep*(1-w); aw=aw*w+(1-ep)*(1-w); dr=dr*w+0.2*(1-w)
    t=hw+dr+aw; hw,dr,aw=hw/t,dr/t,aw/t
    hp=[poisson_pmf(k,lam_h) for k in range(MAX_GOALS+1)]
    ap=[poisson_pmf(k,lam_a) for k in range(MAX_GOALS+1)]
    bp,bh,ba=0,0,0
    for hg in range(MAX_GOALS+1):
        for ag in range(MAX_GOALS+1):
            p=hp[hg]*ap[ag]
            if p>bp: bp,bh,ba=p,hg,ag
    result='H' if hw>dr and hw>aw else('D' if dr>hw and dr>aw else'A')
    return {
        'probs':{'H':round(hw,4),'D':round(dr,4),'A':round(aw,4)},
        'score':f"{bh}-{ba}",'result':result,
        'min_odds':{k:round(1.02/v,2) for k,v in [('H',hw),('D',dr),('A',aw)]},
        'matches_data':(h_ts.get('m',0),a_ts.get('m',0)),
        'lambda_ft': {'home': float(lam_h), 'away': float(lam_a)},
        'model': 'legacy_poisson',
    }


def _try_club_predict(home, away):
    """俱乐部 DC+XGB 混合预测. 成功返回 dict, 失败返回 None."""
    try:
        from team_name_normalizer import normalize_match_pair
        h, a = normalize_match_pair(home, away)
        _load_club_models()

        if _dc_club is None or _xgb_club is None or _elo_club is None:
            return None

        if _form_club is None:
            return None

        if h not in _form_club or a not in _form_club:
            return None
        if len(_form_club.get(h, [])) < 1 or len(_form_club.get(a, [])) < 1:
            return None

        lam_h, lam_a = _dc_club.predict_lambda(h, a, neutral=True)
        if lam_h is None or lam_a is None:
            return None

        dc_p = _dc_club.predict_proba(h, a, neutral=True)
        dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])  # [A, D, H]

        eh = _elo_club.get(h, 1400)
        ea = _elo_club.get(a, 1400)

        # 俱乐部 form
        def _recent_form_club(team, n=5):
            games = _form_club.get(team, [])
            recent = games[-n:] if len(games) >= n else games
            if not recent:
                return [0.5, 0.0, 0.0, 0.0]
            wins = sum(1 for g in recent if g[0] > g[1]) + \
                   sum(0.5 for g in recent if g[0] == g[1])
            gf = sum(g[0] for g in recent) / len(recent)
            ga = sum(g[1] for g in recent) / len(recent)
            return [wins / len(recent), gf, ga, gf - ga]

        fh5 = _recent_form_club(h, 5)
        fa5 = _recent_form_club(a, 5)
        fh12 = _recent_form_club(h, 12)
        fa12 = _recent_form_club(a, 12)

        # H2H
        h2h_gd = 0.0
        try:
            key = tuple(sorted([h, a]))
            import json as _json
            h2h_path = os.path.join('/root/data', 'h2h_cache_club.json')
            if os.path.exists(h2h_path):
                with open(h2h_path) as _f:
                    h2h_cache = _json.load(_f)
                cache_key = f"{key[0]}||{key[1]}"
                entry = h2h_cache.get(cache_key)
                if entry:
                    h2h_gd = entry[1] - entry[2] if h == key[0] else entry[2] - entry[1]
        except:
            pass

        op_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        op_a = 1 / (1 + 10 ** ((eh - ea) / 400))

        b15 = [
            (eh - ea) / 400, lam_h, lam_a, lam_h - lam_a,
            math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
            dc_p[0], dc_p[1], dc_p[2],
            fh5[0], fa5[0], fh5[1] - fa5[2], fa5[1] - fh5[2],
            fh5[1] - fa5[1], fh5[0] - fa5[0], 1,
        ]
        gold = [h2h_gd, 0, 0, fh12[1] - fa12[2], fa12[1] - fh12[0]]
        odds_feat = [op_h, op_a, 0.0]
        form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]
        # xG-proxy (8维: 主客各4)
        xg_feat = []
        for team in [h, a]:
            s = (_xg_club or {}).get(team, {})
            xg_feat.extend([
                s.get('xg_proxy_5', 0.0),
                s.get('xg_proxy_12', 0.0),
                s.get('xg_streak', 0) / 10.0,
                s.get('xg_volatility', 0.0),
            ])
        feat = np.array([b15 + gold + odds_feat + form_feat + xg_feat])

        xgb_p = _xgb_club.predict_proba(feat)[0]

        # Dynamic weight
        p = np.clip(xgb_p, 1e-10, 1.0)
        p = p / p.sum()
        e = -np.sum(p * np.log2(p))
        conf = 1.0 - e / math.log2(3)
        xgb_w = max(0.10, min(0.90, 0.30 + 0.50 * conf))
        dc_w = 1.0 - xgb_w

        hybrid = dc_w * dc_ado + xgb_w * xgb_p
        s = hybrid.sum()
        if s > 0: hybrid = hybrid / s

        # Isotonic 校准
        if _calibrators_club:
            calibrated = np.zeros(3)
            for j, key in enumerate(['away', 'draw', 'home']):
                if key in _calibrators_club:
                    calibrated[j] = _calibrators_club[key].predict([hybrid[j]])[0]
                else:
                    calibrated[j] = hybrid[j]
            s = calibrated.sum()
            if s > 0: calibrated = calibrated / s
            hybrid = calibrated

        hw, dr, aw = float(hybrid[2]), float(hybrid[1]), float(hybrid[0])
        result = 'H' if hw > dr and hw > aw else ('D' if dr > hw and dr > aw else 'A')

        probs_sorted = sorted([hw, dr, aw], reverse=True)
        margin_pp = (probs_sorted[0] - probs_sorted[1]) * 100

        hp = [poisson_pmf(k, lam_h) for k in range(MAX_GOALS + 1)]
        ap = [poisson_pmf(k, lam_a) for k in range(MAX_GOALS + 1)]
        bp, bh, ba = 0, 0, 0
        for hg in range(MAX_GOALS + 1):
            for ag in range(MAX_GOALS + 1):
                p = hp[hg] * ap[ag]
                if p > bp: bp, bh, ba = p, hg, ag

        # ── Standings 联赛排名信息 ──
        standings_info = None
        try:
            from standings_lookup import load_standings_cache, lookup_both
            _sl_cache = load_standings_cache()
            hi, ai, _sfeats = lookup_both(h, a, _sl_cache)
            if hi and ai and hi.get('comp_id') == ai.get('comp_id'):
                standings_info = {
                    'home': f"#{hi['position']} {hi['points']}pts GD{hi['goal_difference']:+d}",
                    'away': f"#{ai['position']} {ai['points']}pts GD{ai['goal_difference']:+d}",
                    'rank_diff': hi['position'] - ai['position'],
                    'pt_diff': hi['points'] - ai['points'],
                    'gd_diff': hi['goal_difference'] - ai['goal_difference'],
                    'comp_id': hi['comp_id'],
                }
        except Exception:
            pass

        return {
            'probs': {'H': round(hw, 4), 'D': round(dr, 4), 'A': round(aw, 4)},
            'score': f"{bh}-{ba}", 'result': result,
            'lambda_ft': {'home': float(lam_h), 'away': float(lam_a)},
            'model': 'club_hybrid',
            'form': {
                'home_gf': round(fh5[1], 2), 'home_ga': round(fh5[2], 2),
                'away_gf': round(fa5[1], 2), 'away_ga': round(fa5[2], 2),
            },
            'margin_pp': round(margin_pp, 1),
            'standings': standings_info,
        }
    except Exception as e:
        return None


def predict_match_wrapper(home, away):
    """主入口: 俱乐部 → 国际赛 → 泊松, 叠加 365scores 调整"""

    # ── 优先: 俱乐部 DC+XGB ──
    r = _try_club_predict(home, away)
    source = 'club'

    # ── 回退: 国际赛 DC+XGB ──
    if r is None:
        r = _try_hybrid_predict(home, away)
        source = 'intl'

    if r is None:
        return None

    r['source'] = source

    # ── 365scores 后验调整 ──
    try:
        from scores365_adjuster import adjust_with_365scores
        today_str = date.today().isoformat()
        model_probs = r.get('probs', {})
        if model_probs:
            adjusted = adjust_with_365scores(home, away, model_probs, today_str)
            if adjusted != model_probs:
                r['probs'] = adjusted
                r['scores365_adjusted'] = True
    except Exception:
        pass

    return r





def fallback_market_predict(market_row):
    odds_h = market_row.get('odds_h', 0)
    odds_d = market_row.get('odds_d', 0)
    odds_a = market_row.get('odds_a', 0)
    probs = implied_probs_from_odds(odds_h, odds_d, odds_a)
    lam_total = 2.55
    lam_home = lam_total * (probs['H'] + 0.5 * probs['D'])
    lam_away = max(0.2, lam_total - lam_home)
    lam_home = max(0.2, lam_home)
    hp = [poisson_pmf(k, lam_home) for k in range(MAX_GOALS + 1)]
    ap = [poisson_pmf(k, lam_away) for k in range(MAX_GOALS + 1)]
    bp, bh, ba = 0, 0, 0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = hp[hg] * ap[ag]
            if p > bp:
                bp, bh, ba = p, hg, ag
    result = 'H' if probs['H'] >= probs['D'] and probs['H'] >= probs['A'] else ('D' if probs['D'] >= probs['A'] else 'A')
    return {
        'probs': {'H': round(probs['H'], 4), 'D': round(probs['D'], 4), 'A': round(probs['A'], 4)},
        'score': f"{bh}-{ba}",
        'result': result,
        'min_odds': {'H': round(1.02 / max(probs['H'], 1e-6), 2), 'D': round(1.02 / max(probs['D'], 1e-6), 2), 'A': round(1.02 / max(probs['A'], 1e-6), 2)},
        'matches_data': (0, 0),
        'lambda_ft': {'home': float(lam_home), 'away': float(lam_away)},
        'model': 'market_fallback',
    }



def compute_htft_topn(lambda_home, lambda_away, topn=6, home=None, away=None):
    """半全场预测: 优先 XGB 模型, 回退数学推导."""
    try:
        from htft_predictor import predict_htft_probs
        probs = predict_htft_probs(
            lambda_home, lambda_away,
            home=home, away=away,
        )
        # 确保使用中文标签映射
        label_map = {'胜胜':'胜胜','胜平':'胜平','胜负':'胜负','平胜':'平胜','平平':'平平','平负':'平负','负胜':'负胜','负平':'负平','负负':'负负',
                     'HH':'胜胜','HD':'胜平','HA':'胜负','DH':'平胜','DD':'平平','DA':'平负','AH':'负胜','AD':'负平','AA':'负负'}
        probs_cn = {}
        for k, v in probs.items():
            cn = label_map.get(k, k)
            probs_cn[cn] = probs_cn.get(cn, 0) + v
    except Exception:
        _, probs_cn = compute_htft_topn_math(lambda_home, lambda_away, topn=9)

    rows = sorted(probs_cn.items(), key=lambda kv: kv[1], reverse=True)
    return rows[:topn], probs_cn


def pick_best_htft(htft_probs, market_htft_odds=None):
    if market_htft_odds:
        available = [(label, prob) for label, prob in htft_probs.items() if market_htft_odds.get(label, 0) > 0]
        if available:
            return max(available, key=lambda x: x[1])[0]
    return max(htft_probs.items(), key=lambda x: x[1])[0]


def top_market_label(odds_map, fallback_label):
    if odds_map:
        available = [(label, odd) for label, odd in odds_map.items() if odd and odd > 0]
        if available:
            return min(available, key=lambda x: x[1])[0]
    return fallback_label


def estimate_vote_fusion_alpha(votes):
    if not votes:
        return ''
    total = votes.get('total') or 0
    if total >= 5000:
        return '0.30'
    if total >= 1000:
        return '0.20'
    if total >= 200:
        return '0.10'
    return '0.05'




def print_match_bundle(bundle):
    def _fmt_htft(label):
        return HTFT_DISPLAY_MAP.get(label, label)

    spf_prob_map = {
        '主胜': bundle['pred_h'],
        '平': bundle['pred_d'],
        '客胜': bundle['pred_a'],
    }
    rq_prob_map = {
        '让胜': bundle['pred_rq_win'],
        '让平': bundle['pred_rq_draw'],
        '让负': bundle['pred_rq_loss'],
    }
    score_prob_map = {k: v * 100 for k, v in bundle.get('score_prob_map', {}).items()}
    htft_prob_map = {k: v * 100 for k, v in bundle.get('htft_prob_map', {}).items()}
    top_goals_prob = bundle['goals_top5'][0][1] * 100 if bundle['goals_top5'] else None
    spf_pick_prob = spf_prob_map.get(bundle['spf_pick'])
    rq_pick_prob = rq_prob_map.get(bundle['rq_pick'])
    score_pick_prob = score_prob_map.get(bundle['pred_top_score'])
    htft_pick_prob = htft_prob_map.get(bundle['pred_top_htft'])

    # ── helper: 概率列表格式化 ──
    def _fmt_prob_list(full_data, top_data, min_prob=0.001):
        src = full_data if full_data else (top_data if top_data else [])
        if not src:
            return ''
        items = [(label, prob * 100) for label, prob in src if prob > min_prob]
        items.sort(key=lambda x: -x[1])
        return ' '.join(f"{label}({p:.1f}%)" for label, p in items)

    def _fmt_htft_list(full_data, top_data):
        src = full_data if full_data else (top_data if top_data else [])
        if not src:
            return ''
        items = [(_fmt_htft(label), prob * 100) for label, prob in src if prob > 0.001]
        items.sort(key=lambda x: -x[1])
        return ' '.join(f"{label}({p:.1f}%)" for label, p in items)

    # ── 500.com 分析数据展示 ──
    _500_lines = format_500_analysis_lines(bundle)
    for line in _500_lines:
        print(line)
    # ── 疲劳度特征展示 ──
    _fatigue_lines = format_fatigue_lines(bundle.get('fatigue', {}))
    for line in _fatigue_lines:
        print(line)

    # ── 新格式打印 ──
    league_str = bundle.get('league', '')
    bet_action_str = bundle.get('bet_action', '')
    rq_text = bundle.get('rq_text', '')

    print()
    print('=' * 70)
    print(f"  {bundle.get('code', '')} {bundle['home']} vs {bundle['away']}  ({bundle['time']})  [{league_str}]")
    print(f"  bet_action: {bet_action_str}")
    print('=' * 70)

    # ── 胜平负 ──
    print(f"  【胜平负】主{bundle['pred_h']:.1f}% / 平{bundle['pred_d']:.1f}% / 客{bundle['pred_a']:.1f}%")
    print(f"  → 推荐: {bundle['spf_pick']}({spf_pick_prob:.1f}%)")
    odds_h_s = bundle.get('odds_h_str', '')
    odds_d_s = bundle.get('odds_d_str', '')
    odds_a_s = bundle.get('odds_a_str', '')
    if odds_h_s or odds_d_s or odds_a_s:
        print(f"  SPF市场赔率: {odds_h_s} / {odds_d_s} / {odds_a_s}")

    # ── 竞彩让球 ──
    print(f"  【竞彩让球({rq_text})】让胜{bundle['pred_rq_win']:.1f}% / 让平{bundle['pred_rq_draw']:.1f}% / 让负{bundle['pred_rq_loss']:.1f}%")
    print(f"  → 推荐: {bundle['rq_pick']}({rq_pick_prob:.1f}%)")

    # ── 比分 ──
    score_str = _fmt_prob_list(bundle.get('score_all'), bundle.get('score_top8'))
    if score_str:
        print(f"  【比分】{score_str}")
        print(f"  → 推荐: {bundle['pred_top_score']}({score_pick_prob:.1f}%)")

    # ── 总进球 ──
    goals_str = _fmt_prob_list(bundle.get('goals_all'), bundle.get('goals_top5'))
    if goals_str:
        print(f"  【总进球】{goals_str}")
        if top_goals_prob is not None:
            print(f"  → 推荐: {bundle['goals_pick']}球({top_goals_prob:.1f}%)")

    # ── 半全场 ──
    htft_str = _fmt_htft_list(bundle.get('htft_all'), bundle.get('htft_top6'))
    if htft_str:
        print(f"  【半全场】{htft_str}")
        print(f"  → 推荐: {_fmt_htft(bundle['pred_top_htft'])}({htft_pick_prob:.1f}%)")

    # ── 365scores投票 ──
    vh = bundle.get('vote_h_str', '')
    vd = bundle.get('vote_d_str', '')
    va = bundle.get('vote_a_str', '')
    vc = bundle.get('vote_count_str', '')
    if vh or vd or va:
        print(f"  365scores公众投票: 主{vh}% / 平{vd}% / 客{va}% (n={vc})")

    # ── 365基本面 (Trend/FIFA/人气) ──
    home_wr = bundle.get('s365_home_winrate')
    away_wr = bundle.get('s365_away_winrate')
    home_fifa = bundle.get('s365_home_fifa')
    away_fifa = bundle.get('s365_away_fifa')
    if home_wr is not None or home_fifa is not None:
        parts = []
        if home_wr is not None and away_wr is not None:
            parts.append(f"胜率(主{home_wr:.0%} vs 客{away_wr:.0%})")
        if home_fifa is not None and away_fifa is not None:
            parts.append(f"FIFA({home_fifa} vs {away_fifa})")
        rank_diff = bundle.get('s365_rank_diff')
        if rank_diff is not None:
            parts.append(f"差距{rank_diff:+d}")
        if parts:
            print(f"  📊 365基本面: {' | '.join(parts)}")

    # ── 联赛积分榜排名 ──
    si = bundle.get('standings')
    if si:
        h_rank = si.get('home', '')
        a_rank = si.get('away', '')
        rd = si.get('rank_diff', 0)
        pd = si.get('pt_diff', 0)
        gd = si.get('gd_diff', 0)
        print(f"  🏆 联赛排名: 主{h_rank} | 客{a_rank}  (差: {rd:+d}位, {pd:+d}分, GD{gd:+d})")

    # ── 模型版本 ──
    model_ver = bundle.get('model_version', '')
    print(f"  模型: {model_ver} (生产) / xgb_model_30 (影子后台运行)")

    # ── EV/Kelly 价值投注 ──
    ba = bundle.get('bet_analysis')
    if ba and ba.scenarios:
        value_bets = [s for s in ba.scenarios if s.ev > 0.02 and bet_math.is_sane_bet(s)]
        if value_bets:
            value_bets.sort(key=lambda s: -s.ev)
            parts = [f"{s.play}{s.pick}(EV={s.ev:+.1%}, Kelly½={s.kelly_half:.1%})" for s in value_bets[:3]]
            print(f"  💰 价值投注: {' | '.join(parts)}")

    # ── 市场分歧 ──
    if bundle.get('market_conflicts'):
        print(f"  市场分歧: {' | '.join(bundle['market_conflicts'])}")



def ensure_log_has_source_fields():
    if not os.path.exists(PREDICTIONS_LOG):
        return
    with open(PREDICTIONS_LOG, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
        if not fieldnames:
            return
    extras = []
    if 'source_tag' not in fieldnames:
        extras.append('source_tag')
    if 'model_version' not in fieldnames:
        extras.append('model_version')
    for col in ('pred_spf_pick', 'pred_rq_pick', 'pred_htft_pick', 'pred_goals_pick', 'pred_score_pick'):
        if col not in fieldnames:
            extras.append(col)
    for col in ('s365_home_winrate', 's365_away_winrate', 's365_home_fifa', 's365_away_fifa', 's365_rank_diff', 's365_popularity_diff'):
        if col not in fieldnames:
            extras.append(col)
    if not extras:
        return
    new_fields = fieldnames + extras
    with open(PREDICTIONS_LOG, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for row in rows:
            for ex in extras:
                row.setdefault(ex, '')
            writer.writerow(row)


def patch_logged_metadata(code, source_tag, model_version):
    if not os.path.exists(PREDICTIONS_LOG):
        return
    with open(PREDICTIONS_LOG, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []
    if not rows:
        return
    changed = False
    for row in rows:
        if row.get('code') == code and row.get('date') == date.today().isoformat():
            row['source_tag'] = source_tag
            row['model_version'] = model_version
            changed = True
    if not changed:
        return
    with open(PREDICTIONS_LOG, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def record_prediction(bundle):
    cmd = [
        'python3', BACKTEST_SCRIPT, 'record',
        '--code', bundle['code'],
        '--home', bundle['home_cn'],
        '--away', bundle['away_cn'],
        '--league', bundle['league'],
        '--time', bundle['time'],
        '--rq', str(bundle['handicap']),
        '--pred-h', f"{bundle['pred_h']:.1f}",
        '--pred-d', f"{bundle['pred_d']:.1f}",
        '--pred-a', f"{bundle['pred_a']:.1f}",
        '--pred-rq-win', f"{bundle['pred_rq_win']:.1f}",
        '--pred-rq-draw', f"{bundle['pred_rq_draw']:.1f}",
        '--pred-rq-loss', f"{bundle['pred_rq_loss']:.1f}",
        '--pred-score', bundle['pred_top_score'],
        '--pred-goals', str(bundle['pred_top_goals']),
        '--pred-htft', HTFT_SHORT_MAP.get(bundle['pred_top_htft'], bundle['pred_top_htft']),
        '--pred-spf-pick', bundle['pred_spf_pick'],
        '--pred-rq-pick', bundle['pred_rq_pick'],
        '--pred-htft-pick', bundle['pred_htft_pick'],
        '--pred-goals-pick', str(bundle['pred_goals_pick']),
        '--pred-score-pick', bundle['pred_score_pick'],
        '--odds-h', bundle['odds_h_str'],
        '--odds-d', bundle['odds_d_str'],
        '--odds-a', bundle['odds_a_str'],
        '--ev-h', bundle['ev_h_str'],
        '--ev-d', bundle['ev_d_str'],
        '--ev-a', bundle['ev_a_str'],
        '--dir', bundle['direction'],
        '--vote-h', bundle['vote_h_str'],
        '--vote-d', bundle['vote_d_str'],
        '--vote-a', bundle['vote_a_str'],
        '--vote-count', bundle['vote_count_str'],
        '--vote-fusion-alpha', bundle['vote_fusion_alpha'],
        '--pop-rank-home', bundle['pop_rank_home_str'],
        '--pop-rank-away', bundle['pop_rank_away_str'],
        '--pop-rank-diff', bundle['pop_rank_diff_str'],
        '--trend-win-rate-home', bundle['trend_win_rate_home_str'],
        '--trend-win-rate-away', bundle['trend_win_rate_away_str'],
        '--trend-win-rate-diff', bundle['trend_win_rate_diff_str'],
        '--simple-pred', str(bundle.get('simple_pred', '')),
        '--simple-conf', str(bundle.get('simple_conf', 0)),
        '--bet-action', str(bundle.get('bet_action', '')),
        '--model-route', str(bundle.get('model', '')),
        '--match-key', f"{bundle.get('date','')}|{bundle.get('league','')}|{bundle.get('home_cn','')}|{bundle.get('away_cn','')}|{bundle.get('time','')}",
        '--pred30-h', f"{bundle.get('pred30_h', '')}" if bundle.get('pred30_h') is not None else '',
        '--pred30-d', f"{bundle.get('pred30_d', '')}" if bundle.get('pred30_d') is not None else '',
        '--pred30-a', f"{bundle.get('pred30_a', '')}" if bundle.get('pred30_a') is not None else '',
        '--s365-home-winrate', f"{bundle.get('s365_home_winrate', '')}" if bundle.get('s365_home_winrate') is not None else '',
        '--s365-away-winrate', f"{bundle.get('s365_away_winrate', '')}" if bundle.get('s365_away_winrate') is not None else '',
        '--s365-home-fifa', str(bundle.get('s365_home_fifa', '')) if bundle.get('s365_home_fifa') is not None else '',
        '--s365-away-fifa', str(bundle.get('s365_away_fifa', '')) if bundle.get('s365_away_fifa') is not None else '',
        '--s365-rank-diff', str(bundle.get('s365_rank_diff', '')) if bundle.get('s365_rank_diff') is not None else '',
        '--s365-popularity-diff', str(bundle.get('s365_popularity_diff', '')) if bundle.get('s365_popularity_diff') is not None else '',
    ]
    # 序列化完整概率分布 (score_top8 / htft_top6 / goals_top5)
    score_full = {}
    for s, pr, _hg, _ag in bundle.get('score_top8', []):
        score_full[s] = round(pr, 4)
    htft_full = {}
    for label, pr in bundle.get('htft_top6', []):
        htft_full[HTFT_SHORT_MAP.get(label, label)] = round(pr, 4)
    goals_full = {}
    for g, pr in bundle.get('goals_all', []):
        goals_full[str(g)] = round(pr, 4)

    import json as _json
    cmd += [
        '--score-full', _json.dumps(score_full, ensure_ascii=False),
        '--htft-full', _json.dumps(htft_full, ensure_ascii=False),
        '--goals-full', _json.dumps(goals_full, ensure_ascii=False),
    ]

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"     ⚠ 落盘失败: {bundle['code']} | rc={proc.returncode} | stderr={proc.stderr.strip()}")
        return
    if proc.stdout.strip():
        print(f"     💾 {proc.stdout.strip()}")
    ensure_log_has_source_fields()
    patch_logged_metadata(bundle['code'], bundle['source_tag'], bundle['model_version'])


def build_prediction_bundle(code, home, away, utc, league, p, market_row=None, score_meta=None):
    lambda_home = p['lambda_ft']['home']
    lambda_away = p['lambda_ft']['away']
    market_row = market_row or {}
    votes = score_meta.get('votes') if score_meta else None

    # ── 动态市场权重融合 ──
    pred_h = p['probs']['H']
    pred_d = p['probs']['D']
    pred_a = p['probs']['A']

    odds_h = market_row.get('odds_h', 0)
    odds_d = market_row.get('odds_d', 0)
    odds_a = market_row.get('odds_a', 0)

    if odds_h and odds_d and odds_a and odds_h > 1 and odds_d > 1 and odds_a > 1:
        try:
            from mc_market_weight_helper import market_weight_for_match
            mkt_h, mkt_d, mkt_a = 1/odds_h, 1/odds_d, 1/odds_a
            mkt_total = mkt_h + mkt_d + mkt_a
            mkt_h /= mkt_total; mkt_d /= mkt_total; mkt_a /= mkt_total

            elo_h = p.get('elo_h', 1500)
            elo_a = p.get('elo_a', 1500)
            neutral = p.get('neutral', True)
            mkt_w = market_weight_for_match(elo_h, elo_a, neutral=neutral)

            pred_h = (1 - mkt_w) * pred_h + mkt_w * mkt_h
            pred_d = (1 - mkt_w) * pred_d + mkt_w * mkt_d
            pred_a = (1 - mkt_w) * pred_a + mkt_w * mkt_a
            s = pred_h + pred_d + pred_a
            if s > 0: pred_h /= s; pred_d /= s; pred_a /= s
        except Exception:
            pass

    pred_h *= 100
    pred_d *= 100
    pred_a *= 100

    spf_pick = max([('主胜', pred_h), ('平', pred_d), ('客胜', pred_a)], key=lambda x: x[1])[0]
    market_spf_pick = top_market_label(
        {'主胜': market_row.get('odds_h', 0), '平': market_row.get('odds_d', 0), '客胜': market_row.get('odds_a', 0)},
        spf_pick,
    )

    handicap = int(market_row.get('handicap', 0) or 0)
    dc_rho = p.get('rho', 0.0)
    # NOTE 2026-07-04: DC rho 修正仅对 handicap=0 (SPF) 场景生效。
    # handicap≠0 时 tau 修正的低比分格 (0,0)/(1,0)/(0,1)/(1,1)
    # 与"让平"判定区间不重合，会系统性压低让平概率 ~1%。
    # A/B 验证: 181场中 145/181 负向。详见 commit 26814f2。
    rq_probs = compute_rq_probs(lambda_home, lambda_away, handicap,
                                rho=(dc_rho if handicap == 0 else 0.0))
    rq_pick = max(rq_probs.items(), key=lambda x: x[1])[0]
    market_rq_pick = top_market_label(
        {'让胜': market_row.get('rq_h', 0), '让平': market_row.get('rq_d', 0), '让负': market_row.get('rq_a', 0)},
        rq_pick,
    )

    goals_dist = compute_goals_distribution(lambda_home, lambda_away, rho=dc_rho)
    goals_items = sorted(goals_dist.items(), key=lambda kv: kv[1], reverse=True)
    goals_pick = int(goals_items[0][0]) if goals_items else int(sum(int(k) * v for k, v in goals_dist.items()))
    goals_top5 = goals_items[:5]
    # ── 全部总进球 (0~12球, 按概率降序) ──
    goals_all = goals_items

    score_top8 = compute_score_topn(lambda_home, lambda_away, 8, rho=dc_rho)
    pred_top_score = score_top8[0][0] if score_top8 else p['score'].replace('-', ':')
    market_score_pick = top_market_label(market_row.get('bf_odds', {}), pred_top_score)
    score_prob_map = {score: prob for score, prob, _hg, _ag in score_top8}
    # ── 全部比分 (按概率降序, 不含概率为0) ──
    score_all = compute_score_topn(lambda_home, lambda_away, topn=999, rho=dc_rho)
    score_all = [(s, prob) for s, prob, _hg, _ag in score_all if prob > 0]

    htft_top6, htft_probs = compute_htft_topn(lambda_home, lambda_away, 6, home=home, away=away)
    pred_top_htft = max(htft_probs.items(), key=lambda x: x[1])[0]
    market_htft_pick = pick_best_htft(htft_probs, market_row.get('htft_odds'))
    # ── 全部半全场 (9种全部, 按概率降序) ──
    htft_all = sorted(htft_probs.items(), key=lambda kv: kv[1], reverse=True)

    odds_h = market_row.get('odds_h', 0)
    odds_d = market_row.get('odds_d', 0)
    odds_a = market_row.get('odds_a', 0)

    # ── 并行模型结果 ──
    simple_pred = p.get('simple_pred', '')
    simple_conf = p.get('simple_conf', 0)

    # ── EV 计算 (SPF 玩法) ──
    def _ev(prob_pct, odds):
        p = prob_pct / 100.0
        if odds > 1 and p > 0:
            return p * (odds - 1) - (1 - p)
        return ''
    ev_h = _ev(pred_h, odds_h)
    ev_d = _ev(pred_d, odds_d)
    ev_a = _ev(pred_a, odds_a)

    # ── bet_math: 全玩法 EV + Kelly 分析 ──
    _predictions = {
        'spf': {'h': pred_h / 100, 'd': pred_d / 100, 'a': pred_a / 100},
        'rq': {
            'rq_win': rq_probs.get('让胜', 0),
            'rq_draw': rq_probs.get('让平', 0),
            'rq_lose': rq_probs.get('让负', 0),
        },
        'score': [{'score': s, 'prob': pr} for s, pr, _hg, _ag in score_top8[:5]],
        'total_goals': [{'goals': g, 'prob': pr} for g, pr in goals_top5],
        'half_full': [{'hf': label, 'prob': pr} for label, pr in htft_top6[:4]],
    }
    _odds = {
        'spf': {'h': odds_h, 'd': odds_d, 'a': odds_a} if odds_h and odds_d and odds_a else {},
        'rq': {
            'rq_win': market_row.get('rq_h', 0),
            'rq_draw': market_row.get('rq_d', 0),
            'rq_lose': market_row.get('rq_a', 0),
        },
        'score': market_row.get('bf_odds', {}),
        'total_goals': {str(k).replace('球', ''): v for k, v in market_row.get('zjq_odds', {}).items()} if market_row.get('zjq_odds') else {},
        'half_full': market_row.get('htft_odds', {}),
    }
    model_type = p.get('model', '')
    bet_analysis = bet_math.analyze_match(home, away, _predictions, _odds, model_type)

    # 兼容旧格式 (spf_value_tips)
    spf_value_tips = []
    for s in bet_analysis.scenarios:
        if s.play == '胜平负' and s.is_value:
            spf_value_tips.append({
                'label': s.pick,
                'prob': s.prob * 100,
                'odd': s.odds,
                'ev': s.ev,
            })
    spf_value_tips.sort(key=lambda x: x['ev'], reverse=True)

    market_conflicts = []
    if market_spf_pick != spf_pick:
        market_conflicts.append(f"SPF市场倾向={market_spf_pick}")
    if market_rq_pick != rq_pick:
        market_conflicts.append(f"RQ市场倾向={market_rq_pick}")
    if market_score_pick != pred_top_score:
        market_conflicts.append(f"比分市场倾向={market_score_pick}")
    if market_htft_pick != pred_top_htft:
        market_conflicts.append(f"半全场市场倾向={HTFT_DISPLAY_MAP.get(market_htft_pick, market_htft_pick)}")

    # 总进球市场倾向（zjq_odds）
    zjq_odds_map = market_row.get('zjq_odds', {})
    market_zjq_pick = top_market_label(zjq_odds_map, f"{goals_pick}球") if zjq_odds_map else None
    if market_zjq_pick and market_zjq_pick != f"{goals_pick}球":
        market_conflicts.append(f"总进球市场倾向={market_zjq_pick}")

    vote_h = votes.get('home') if votes else None
    vote_d = votes.get('draw') if votes else None
    vote_a = votes.get('away') if votes else None
    vote_count = votes.get('total') if votes else None
    vote_text = ''
    if votes and vote_h is not None and vote_d is not None and vote_a is not None:
        vote_text = f"公众{vote_h:.1f}/{vote_d:.1f}/{vote_a:.1f}% n={vote_count or 0}"
        if score_meta.get('trend_home') and score_meta.get('trend_away'):
            th = score_meta['trend_home']
            ta = score_meta['trend_away']
            vote_text += f" 近况{home}{th[0]}-{th[1]}-{th[2]} {away}{ta[0]}-{ta[1]}-{ta[2]}"

    rq_text = f"{handicap:+d}" if handicap else '0'
    if handicap > 0:
        rq_text = f"受让{handicap}"
    elif handicap < 0:
        rq_text = f"让{abs(handicap)}"

    model_note = p.get('model', '')

    # ── 亚盘价值 (AH) 公平赔率 ──
    ah_fair_odds = {}
    try:
        from asian_handicap import ah_probs
        # 扫描以竞彩让球为中心的 AH 线
        base = float(handicap) if handicap != 0 else 0.5
        # 扫描范围: base±2范围, 0.25步进, 仅非负盘口
        scan_range = [h * 0.25 for h in range(int(max(base - 2, 0) * 4), int((base + 2) * 4) + 1)]
        for h in scan_range:
            p = ah_probs(lambda_home, lambda_away, h, max_goals=15)
            ah_fair_odds[h] = round(p['fair_odds'], 2)
    except Exception:
        pass

    # ── bet_action: 赛事类型过滤标签 ──
    bet_action = compute_bet_action(league, model_type, bet_analysis, htft_top6, handicap, rq_probs)
    # ── P3 Shadow Mode: 低置信度标记 (2026-06-30) ──
    # 不修改原有 action 逻辑，仅追加 [LOW_CONF] 标签用于观察
    max_hda_prob = max(pred_h, pred_d, pred_a)
    if max_hda_prob < 60 and bet_action.startswith(('RECOMMEND', 'WATCH')):
        bet_action = f'{bet_action} [LOW_CONF]'
    # ── 半全场胜胜低概率外推标记 ──
    htft_warning = False
    if bet_analysis and bet_analysis.scenarios:
        for s in bet_analysis.scenarios:
            if s.play == '半全场' and s.pick == '胜胜' and s.prob < 0.20 and model_type == 'hybrid':
                htft_warning = True
                break

    return {
        'code': code,
        'league': league,
        'time': utc,
        'home': home,
        'away': away,
        'home_cn': market_row.get('home_cn', home),
        'away_cn': market_row.get('away_cn', away),
        'handicap': handicap,
        'rq_text': rq_text,
        'pred_h': pred_h,
        'pred_d': pred_d,
        'pred_a': pred_a,
        'spf_pick': spf_pick,
        'market_spf_pick': market_spf_pick,
        'pred_rq_win': rq_probs['让胜'] * 100,
        'pred_rq_draw': rq_probs['让平'] * 100,
        'pred_rq_loss': rq_probs['让负'] * 100,
        'rq_pick': rq_pick,
        'market_rq_pick': market_rq_pick,
        'pred_top_score': pred_top_score,
        'market_score_pick': market_score_pick,
        'score_prob_map': score_prob_map,
        'pred_top_goals': goals_pick,
        'pred_top_htft': pred_top_htft,
        'market_htft_pick': market_htft_pick,
        'pred_spf_pick': spf_pick,
        'pred_rq_pick': rq_pick,
        'pred_htft_pick': HTFT_DISPLAY_MAP.get(pred_top_htft, pred_top_htft),
        'pred_goals_pick': goals_pick,
        'pred_score_pick': pred_top_score,
        'score_top8': score_top8,
        'score_all': score_all,
        'goals_top5': goals_top5,
        'goals_all': goals_all,
        'goals_pick': goals_pick,
        'htft_top6': htft_top6,
        'htft_all': htft_all,
        'htft_prob_map': htft_probs,
        'market_spf': f"{odds_h:.2f}-{odds_d:.2f}-{odds_a:.2f}" if odds_h and odds_d and odds_a else '',
        'zjq_odds_str': (lambda z: (lambda d: f"{int(d[0][0])}球{d[0][1]:.2f}-{int(d[1][0])}球{d[1][1]:.2f}-{int(d[2][0])}球{d[2][1]:.2f}-{int(d[3][0])}球{d[3][1]:.2f}" \
                            if len(d) >= 4 else '')(sorted([(k.replace('球',''), v) for k, v in z.items() if '球' in k])) if z else '')(market_row.get('zjq_odds', {})),
        'spf_value_tips': spf_value_tips,
        'bet_analysis': bet_analysis,
        'market_conflicts': market_conflicts,
        'votes_text': vote_text,
        'model_note': model_note,
        'direction': f"SPF:{spf_pick} | RQ:{rq_pick} | HTFT:{pred_top_htft} | Goals:{goals_pick} | Score:{pred_top_score}",
        'source_tag': '500+365',
        'model_version': MODEL_VERSION,
        'simple_pred': simple_pred,
        'simple_conf': simple_conf,
        'pred30_h': p.get('pred30_h'),
        'pred30_d': p.get('pred30_d'),
        'pred30_a': p.get('pred30_a'),
        'odds_h_str': f'{odds_h:.2f}' if odds_h else '',
        'odds_d_str': f'{odds_d:.2f}' if odds_d else '',
        'odds_a_str': f'{odds_a:.2f}' if odds_a else '',
        'ev_h_str': f'{ev_h:.4f}' if ev_h != '' else '',
        'ev_d_str': f'{ev_d:.4f}' if ev_d != '' else '',
        'ev_a_str': f'{ev_a:.4f}' if ev_a != '' else '',
        'vote_h_str': f'{vote_h:.1f}' if vote_h is not None else '',
        'vote_d_str': f'{vote_d:.1f}' if vote_d is not None else '',
        'vote_a_str': f'{vote_a:.1f}' if vote_a is not None else '',
        'vote_count_str': str(vote_count) if vote_count is not None else '',
        'vote_fusion_alpha': estimate_vote_fusion_alpha(votes),
        'pop_rank_home_str': str(score_meta.get('pop_rank_home')) if score_meta and score_meta.get('pop_rank_home') is not None else '',
        'pop_rank_away_str': str(score_meta.get('pop_rank_away')) if score_meta and score_meta.get('pop_rank_away') is not None else '',
        'pop_rank_diff_str': str((score_meta.get('pop_rank_away') - score_meta.get('pop_rank_home'))) if score_meta and score_meta.get('pop_rank_home') is not None and score_meta.get('pop_rank_away') is not None else '',
        'trend_win_rate_home_str': f"{score_meta.get('trend_win_rate_home'):.4f}" if score_meta and score_meta.get('trend_win_rate_home') is not None else '',
        'trend_win_rate_away_str': f"{score_meta.get('trend_win_rate_away'):.4f}" if score_meta and score_meta.get('trend_win_rate_away') is not None else '',
        'trend_win_rate_diff_str': f"{(score_meta.get('trend_win_rate_home') - score_meta.get('trend_win_rate_away')):.4f}" if score_meta and score_meta.get('trend_win_rate_home') is not None and score_meta.get('trend_win_rate_away') is not None else '',
        # ── 365基本面特征 (预埋, 不入 XGB 特征向量) ──
        's365_home_winrate': score_meta.get('trend_win_rate_home') if score_meta else None,
        's365_away_winrate': score_meta.get('trend_win_rate_away') if score_meta else None,
        's365_home_fifa': score_meta.get('fifa_rank_home') if score_meta else None,
        's365_away_fifa': score_meta.get('fifa_rank_away') if score_meta else None,
        's365_rank_diff': (score_meta.get('fifa_rank_away', 100) - score_meta.get('fifa_rank_home', 100)) if score_meta and score_meta.get('fifa_rank_home') is not None and score_meta.get('fifa_rank_away') is not None else None,
        's365_popularity_diff': (score_meta.get('pop_rank_home', 50000) - score_meta.get('pop_rank_away', 50000)) if score_meta and score_meta.get('pop_rank_home') is not None and score_meta.get('pop_rank_away') is not None else None,
        'ah_fair_odds': ah_fair_odds,
        'bet_action': bet_action,
        'htft_warning': htft_warning,
        'standings': p.get('standings'),
    }


def compute_bet_action(league, model_type, bet_analysis, htft_top6, handicap, rq_probs):
    """赛事类型过滤标签: RECOMMEND / WATCH / SKIP_LEAGUE

    Rule 1: UEFA Nations League → SKIP_LEAGUE (历史ROI -72.5%)
    Rule 2: 友谊赛 → WATCH (校准器过拟合, 2026-06-10 诊断确认)
    Rule 3: market_fallback → WATCH (EV循环论证)
    """
    # Rule 1
    if league == 'UEFA Nations League':
        return 'SKIP_LEAGUE'

    # Rule 3: market_fallback 场次即使 margin 高也不推荐（EV是循环论证）
    if model_type == 'market_fallback':
        return 'WATCH'

    # Rule 2: 友谊赛全部降级为 WATCH
    # 2026-06-10 校准曲线诊断: Isotonic校准器在友谊赛上严重过度自信
    # RECOMMEND组校准差 -70.2pp (70%置信度, 0%命中率)
    # 治本方案: 重训sigmoid校准器后可恢复margin门槛
    if '友谊赛' in league or 'Friendly' in league or 'Friendlies' in league:
        return 'WATCH_FRIENDLY'

    return 'RECOMMEND'


def main():
    from team_name_normalizer import normalize_match_pair

    today_str = date.today().isoformat()
    wd = ['一','二','三','四','五','六','日'][date.today().weekday()]
    print(f"{'='*60}")
    print(f"  ⚽ 每日竞彩预测  {today_str} 周{wd}")
    print(f"{'='*60}")

    print("📡 获取今日赛程...")
    matches = get_today_matches()
    _500_odds = scrape_500_odds_today()
    score365_games = load_365scores_today()
    score365_map = build_365_map(score365_games)
    if _500_odds:
        print(f"  📡 500.com: {len(_500_odds)} 场有赔率数据")
    print(f"  📡 365scores: {len(score365_games)} 场增强数据")

    # ── 500.com 分析数据爬取 (FIFA排名/战绩/赢盘率/澳门心水) ──
    _500_analysis = {}
    if _500_odds:
        print(f"\n📡 抓取500.com比赛分析数据...")
        # 从赔率数据构建 match_codes 映射
        match_codes = {}
        for m5 in _500_odds:
            # 尝试从500.com页面提取shuju ID
            match_codes[m5['code']] = {
                'id': m5.get('shuju_id', ''),
                'home': m5['home_cn'],
                'away': m5['away_cn'],
            }
        # 如果没有shuju_id，让 scraper 自动获取
        has_ids = any(v['id'] for v in match_codes.values())
        _500_analysis = scrape_500_analysis(match_codes if has_ids else None)
        if _500_analysis:
            print(f"  📡 500.com分析: {len(_500_analysis)} 场有分析数据")

    use_500_only = False
    if not matches and _500_odds:
        print(f"  📭 football-data.org 无联赛赛事, 使用500.com {len(_500_odds)} 场国际赛")
        use_500_only = True
    elif not matches:
        print("  📭 今日无竞彩赛事\n")
        return 0

    if not use_500_only:
        print(f"  {len(matches)} 场联赛/杯赛 ({len(_500_odds)} 场有500赔率)")

    leagues_needed = set()
    ts = ga = elo_r = None
    if not use_500_only:
        for m in matches:
            leagues_needed.add(m['competition']['code'])

        print("\n📡 拉取历史训练数据...")
        all_hist = []
        for code,lname in JCZQ_LEAGUES:
            if code not in leagues_needed:
                continue
            hist = fetch_league_history(code)
            print(f"  {lname}: {len(hist)} 场历史")
            all_hist.extend(hist)

        if not all_hist:
            print("❌ 无历史数据, 无法预测")
            return 1

        print(f"\n🧠 训练后备模型 (泊松+Elo)...")
        ts, ga, elo_r = train(all_hist)
        print(f"  总训练: {len(all_hist)} 场 | λ={ga:.3f} | 球队: {len(ts)}")

    print(f"\n{'─'*60}")
    print(f"  📊 预测")
    print(f"{'─'*60}")

    hybrid_count = 0
    legacy_count = 0

    _500_map = {}
    for m5 in _500_odds:
        try:
            h_e, a_e = normalize_match_pair(m5['home_cn'], m5['away_cn'])
            _500_map[(h_e, a_e)] = m5
            _500_map[(a_e, h_e)] = m5
        except Exception:
            pass

    bundles = []

    if use_500_only:
        # 打印联赛分类信息
        league_counts = defaultdict(int)
        for m5 in _500_odds:
            league = m5.get('league', '') or '未知'
            league_counts[league] += 1
        league_info = ', '.join(f'{k}{v}场' for k, v in sorted(league_counts.items()))
        print(f"\n  📋 500.com 赛事 ({len(_500_odds)}场) — {league_info}")
        print(f"  {'─'*60}")
        for m5 in _500_odds:
            home_cn, away_cn = m5['home_cn'], m5['away_cn']
            p = predict_match_wrapper(home_cn, away_cn)
            if not p:
                legacy_count += 1
                print(f"  ⚠ {m5['code']} {home_cn} vs {away_cn} — 主模型无数据，回退市场保底")
                p = fallback_market_predict(m5)
            else:
                hybrid_count += 1
            h_norm, a_norm = normalize_match_pair(home_cn, away_cn)
            score_meta = score365_map.get((h_norm, a_norm))

            # ── 动态识别赛事类型 ──
            raw_league = m5.get('league', '') or ''
            # 500.com simpleleague 包含 "世界杯"、"国际友谊赛"、"欧国联" 等
            if '世界杯' in raw_league or 'world cup' in raw_league.lower():
                league_label = '世界杯'
            elif '友谊赛' in raw_league or 'friendly' in raw_league.lower():
                league_label = '友谊赛'
            elif raw_league:
                league_label = raw_league
            else:
                league_label = '友谊赛'  # 无标识时回退原行为

            bundle = build_prediction_bundle(m5['code'], home_cn, away_cn, m5['time'], league_label, p, m5, score_meta)
            enrich_bundle_with_500(bundle, _500_analysis.get(m5['code']))
            apply_euro_fallback(bundle, m5)
            # ── 疲劳度特征计算 ──
            a_data = _500_analysis.get(m5['code'], {})
            if a_data.get('future_fixtures'):
                # 清理队名: "[7]荷兰" → "荷兰", "乌兹别克[58]" → "乌兹别克"
                import re as _re
                clean_home = _re.sub(r'\[\d+\]', '', home_cn).strip()
                clean_away = _re.sub(r'\[\d+\]', '', away_cn).strip()
                fatigue = compute_fatigue_features(
                    clean_home, clean_away, m5.get('time', ''), league_label, a_data['future_fixtures']
                )
                bundle['fatigue'] = fatigue
                # 如果主客队轮换差异显著, 调整概率
                if abs(fatigue.get('rotation_diff', 0)) >= 0.1:
                    orig_h = bundle['pred_h'] / 100
                    orig_d = bundle['pred_d'] / 100
                    orig_a = bundle['pred_a'] / 100
                    adj = fatigue_adjustment(fatigue, {'H': orig_h, 'D': orig_d, 'A': orig_a})
                    bundle['pred_h'] = adj['H'] * 100
                    bundle['pred_d'] = adj['D'] * 100
                    bundle['pred_a'] = adj['A'] * 100
                    bundle['model_note'] = bundle.get('model_note', '') + '+疲劳度调整'
            bundles.append(bundle)
            print_match_bundle(bundle)
    else:
        by_league = defaultdict(list)
        for m in matches:
            by_league[m['competition']['code']].append(m)

        for code,lname in JCZQ_LEAGUES:
            if code not in by_league:
                continue
            ms = by_league[code]
            print(f"\n  📋 {lname} ({len(ms)}场)")
            print(f"  {'─'*60}")
            for m in ms:
                home = m['homeTeam']['shortName']
                away = m['awayTeam']['shortName']
                utc = m['utcDate'][11:16]
                p = predict_match_wrapper(home, away)
                if p:
                    hybrid_count += 1
                else:
                    p = predict_match_legacy(home, away, ts, ga, elo_r)
                    legacy_count += 1

                m5 = _500_map.get((home, away))
                # 使用标准化后的名称查找365scores数据
                h_norm, a_norm = normalize_match_pair(home, away)
                score_meta = score365_map.get((h_norm, a_norm))
                bundle = build_prediction_bundle(
                    m5['code'] if m5 else f"{code}-{home}-{away}",
                    home,
                    away,
                    utc,
                    lname,
                    p,
                    m5,
                    score_meta,
                )
                analysis_key = m5['code'] if m5 else None
                if analysis_key:
                    enrich_bundle_with_500(bundle, _500_analysis.get(analysis_key))
                if m5:
                    apply_euro_fallback(bundle, m5)
                # ── 疲劳度特征计算 (联赛分支) ──
                a_data = _500_analysis.get(analysis_key, {}) if analysis_key else {}
                if a_data.get('future_fixtures'):
                    fatigue = compute_fatigue_features(
                        home, away, utc[:10] if utc else date.today().isoformat(),
                        lname, a_data['future_fixtures']
                    )
                    bundle['fatigue'] = fatigue
                bundles.append(bundle)
                print_match_bundle(bundle)

    for bundle in bundles:
        record_prediction(bundle)

    # ── bet_math: 全局价值投注汇总（过滤 SKIP_LEAGUE/WATCH）──
    all_analyses = []
    for b in bundles:
        ba = b.get('bet_analysis')
        ba_label = b.get('bet_action', '')
        if ba and ba_label not in ('SKIP_LEAGUE', 'WATCH', 'WATCH_FRIENDLY'):
            all_analyses.append(ba)
    n_skipped = sum(1 for b in bundles if b.get('bet_action') in ('SKIP_LEAGUE', 'WATCH', 'WATCH_FRIENDLY'))
    if all_analyses:
        print(bet_math.format_value_summary(all_analyses, min_ev=0.05))
    if n_skipped:
        print(f"  ℹ️ 已过滤 {n_skipped} 场赛事类型不推荐场次 (SKIP_LEAGUE/WATCH)")

    print(f"\n{'='*60}")
    print(f"  💎 购彩建议")
    print(f"{'='*60}")
    print(f"  📆 {today_str} 周{wd}  |  {len(bundles)} 场竞彩赛事")
    print(f"  🧠 国际赛: DC+XGBoost+Form ({hybrid_count}场) | 联赛: 泊松+Elo ({legacy_count}场)")
    print(f"  📡 365scores增强已启用")
    print(f"  💾 已写入: {PREDICTIONS_LOG}")
    print(f"  \n  📌 策略:")
    print(f"  • 输出口径: 90分钟常规时间(含伤停补时)")
    print(f"  • 每场落盘: 胜平负 / 竞彩让球 / 半全场 / 比分 / 总进球")
    print(f"  • 500.com决定可买场次, 365scores提供增强特征")
    print(f"\n  ⚠️ 本预测基于统计数据, 不构成投注建议")
    print(f"  请理性购彩, 切勿沉迷")
    print(f"{'='*60}")
    return 0


if __name__ == '__main__':
    main()
