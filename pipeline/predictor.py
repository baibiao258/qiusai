"""Match outcome predictor — pure prediction logic, zero I/O.

Routing logic
-------------
predict_match_wrapper(home, away) -> dict | None
    1. Club DC+XGB hybrid        (_try_club_predict)
    2. International DC+XGB      (_try_hybrid_predict)
    3. Legacy Poisson+Elo        (predict_match_legacy, called by daily_jczq.py)
    4. Market fallback           (fallback_market_predict, when all models absent)

Returns None only when no model has sufficient data AND caller should
invoke predict_match_legacy / fallback_market_predict directly.

All model artifacts are loaded lazily via pipeline.model_loader.
"""
from __future__ import annotations

import math
import os
from datetime import date
from typing import Optional

import numpy as np
from scipy.stats import poisson as sp_poisson

from config.settings import MAX_GOALS
from pipeline.model_loader import get_intl_models, get_club_models
from pipeline.probability import (
    poisson_pmf,
    elo_expected,
    compute_dynamic_xgb_weight,
    implied_probs_from_odds,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public router
# ─────────────────────────────────────────────────────────────────────────────

def predict_match_wrapper(home: str, away: str) -> Optional[dict]:
    """Primary prediction entry point: club → intl → None.

    Tries club model first, then international.  Returns None when both
    models lack sufficient data; caller must handle legacy fallback.

    Also applies 365scores posterior adjustment when the adjuster module
    is available.
    """
    r = _try_club_predict(home, away)
    source = 'club'

    if r is None:
        r = _try_hybrid_predict(home, away)
        source = 'intl'

    if r is None:
        return None

    r['source'] = source

    # ── 365scores 后验调整 ──
    try:
        from scores365_adjuster import adjust_with_365scores
        model_probs = r.get('probs', {})
        if model_probs:
            adjusted = adjust_with_365scores(home, away, model_probs, date.today().isoformat())
            if adjusted != model_probs:
                r['probs'] = adjusted
                r['scores365_adjusted'] = True
    except Exception:
        pass

    return r


# ─────────────────────────────────────────────────────────────────────────────
# International DC+XGB hybrid
# ─────────────────────────────────────────────────────────────────────────────

def _try_hybrid_predict(home: str, away: str) -> Optional[dict]:
    """DC+XGBoost hybrid for international fixtures.

    Returns None on any data-gap or exception; caller falls through to
    club or legacy model.
    """
    try:
        from team_name_normalizer import normalize_match_pair
        h, a = normalize_match_pair(home, away)

        m = get_intl_models()
        dc      = m['dc']
        xgb     = m['xgb']
        elo_d   = m['elo']
        cals    = m.get('calibrators')
        xgb30   = m.get('xgb30')
        xgb_s   = m.get('xgb_simple')
        cal_s   = m.get('cal_simple')

        lam_h, lam_a = dc.predict_lambda(h, a, neutral=True)
        if lam_h is None or lam_a is None:
            return None

        dc_p   = dc.predict_proba(h, a, neutral=True)
        dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])

        eh = elo_d.get(h, 1500)
        ea = elo_d.get(a, 1500)

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
        gold      = build_gold_features(home, away, match_type='competitive')
        odds_feat = [op_h, op_a, 0.0]
        form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]
        feat      = np.array([b15 + gold + odds_feat + form_feat])

        xgb_p = xgb.predict_proba(feat)[0]

        # ── A/B shadow: 30-dim model ──
        xgb30_p = None
        if xgb30 is not None and op_h > 0:
            try:
                feat_30   = np.array([b15 + gold + odds_feat + form_feat + [1.0 / op_h]])
                raw30     = xgb30.predict_proba(feat_30)[0]
                w30, dw30, _ = compute_dynamic_xgb_weight(raw30)
                hyb30     = dw30 * dc_ado + w30 * raw30
                s30       = hyb30.sum()
                if s30 > 0:
                    hyb30 /= s30
                xgb30_p = hyb30
            except Exception:
                pass

        # ── entropy-based dynamic fusion ──
        xgb_w, dc_w, _ = compute_dynamic_xgb_weight(xgb_p)
        hybrid = dc_w * dc_ado + xgb_w * xgb_p
        s = hybrid.sum()
        if s > 0:
            hybrid /= s

        # ── Isotonic calibration (intl) ──
        if cals:
            calibrated = np.zeros(3)
            for j, key in enumerate(['away', 'draw', 'home']):
                calibrated[j] = cals[key].predict([hybrid[j]])[0] if key in cals else hybrid[j]
            s = calibrated.sum()
            if s > 0:
                calibrated /= s
            hybrid = calibrated

        # ── simple parallel model ──
        simple_pred, simple_conf = _run_simple_model(xgb_s, cal_s, op_h, fh5, fa5)

        hw, dr, aw = float(hybrid[2]), float(hybrid[1]), float(hybrid[0])
        result     = _hda_result(hw, dr, aw)

        from predict_match import _load_form_state
        fs = _load_form_state()
        home_has = home in fs and len(fs[home]) >= 1
        away_has = away in fs and len(fs[away]) >= 1
        form_gap = (not home_has) or (not away_has)

        probs_sorted = sorted([hw, dr, aw], reverse=True)
        margin_pp    = (probs_sorted[0] - probs_sorted[1]) * 100
        best_label   = ['H', 'D', 'A'][[hw, dr, aw].index(probs_sorted[0])]
        bet_action   = 'SKIP_DATA' if form_gap else ('BET' if margin_pp >= 10 else 'SKIP')

        bh, ba = _best_score(lam_h, lam_a)
        return {
            'probs':       {'H': round(hw, 4), 'D': round(dr, 4), 'A': round(aw, 4)},
            'score':       f'{bh}-{ba}',
            'result':      result,
            'min_odds':    {k: round(1.02 / max(v, 1e-6), 2) for k, v in [('H', hw), ('D', dr), ('A', aw)]},
            'matches_data': (0, 0),
            'lambda_ft':   {'home': float(lam_h), 'away': float(lam_a)},
            'model':       'hybrid',
            'form': {
                'home_gf': round(fh5[1], 2), 'home_ga': round(fh5[2], 2),
                'away_gf': round(fa5[1], 2), 'away_ga': round(fa5[2], 2),
            },
            'simple_pred': simple_pred,
            'simple_conf': round(simple_conf, 4) if simple_conf else 0,
            'pred30_h': round(float(xgb30_p[2]), 4) if xgb30_p is not None else None,
            'pred30_d': round(float(xgb30_p[1]), 4) if xgb30_p is not None else None,
            'pred30_a': round(float(xgb30_p[0]), 4) if xgb30_p is not None else None,
            'bet_recommendation': {
                'action':        bet_action,
                'margin_pp':     round(margin_pp, 1),
                'best_pick':     best_label,
                'best_prob_pct': round(probs_sorted[0] * 100, 1),
            },
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Club DC+XGB hybrid
# ─────────────────────────────────────────────────────────────────────────────

def _try_club_predict(home: str, away: str) -> Optional[dict]:
    """DC+XGBoost hybrid for domestic club fixtures.

    Returns None if club models are not loaded or team has insufficient data.
    """
    try:
        from team_name_normalizer import normalize_match_pair
        h, a = normalize_match_pair(home, away)

        cm = get_club_models()
        if cm is None:
            return None

        dc    = cm.get('dc')
        xgb   = cm.get('xgb')
        elo_d = cm.get('elo')
        cals  = cm.get('calibrators')
        form  = cm.get('form')
        xg    = cm.get('xg')

        if dc is None or xgb is None or elo_d is None or form is None:
            return None
        if h not in form or a not in form:
            return None
        if len(form.get(h, [])) < 1 or len(form.get(a, [])) < 1:
            return None

        lam_h, lam_a = dc.predict_lambda(h, a, neutral=True)
        if lam_h is None or lam_a is None:
            return None

        dc_p   = dc.predict_proba(h, a, neutral=True)
        dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])

        eh = elo_d.get(h, 1400)
        ea = elo_d.get(a, 1400)

        fh5  = _recent_form_club(form, h, 5)
        fa5  = _recent_form_club(form, a, 5)
        fh12 = _recent_form_club(form, h, 12)
        fa12 = _recent_form_club(form, a, 12)

        h2h_gd = _load_h2h_gd(h, a)

        op_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        op_a = 1 / (1 + 10 ** ((eh - ea) / 400))

        b15 = [
            (eh - ea) / 400, lam_h, lam_a, lam_h - lam_a,
            math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
            dc_p[0], dc_p[1], dc_p[2],
            fh5[0], fa5[0], fh5[1] - fa5[2], fa5[1] - fh5[2],
            fh5[1] - fa5[1], fh5[0] - fa5[0], 1,
        ]
        gold      = [h2h_gd, 0, 0, fh12[1] - fa12[2], fa12[1] - fh12[0]]
        odds_feat = [op_h, op_a, 0.0]
        form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]
        xg_feat   = _build_xg_feat(xg, h, a)
        feat      = np.array([b15 + gold + odds_feat + form_feat + xg_feat])

        xgb_p = xgb.predict_proba(feat)[0]

        # dynamic weight
        p_clip = np.clip(xgb_p, 1e-10, 1.0)
        p_clip /= p_clip.sum()
        entropy  = -np.sum(p_clip * np.log2(p_clip))
        conf     = 1.0 - entropy / math.log2(3)
        xgb_w    = max(0.10, min(0.90, 0.30 + 0.50 * conf))
        dc_w     = 1.0 - xgb_w

        hybrid = dc_w * dc_ado + xgb_w * xgb_p
        s = hybrid.sum()
        if s > 0:
            hybrid /= s

        # Isotonic calibration (club)
        if cals:
            calibrated = np.zeros(3)
            for j, key in enumerate(['away', 'draw', 'home']):
                calibrated[j] = cals[key].predict([hybrid[j]])[0] if key in cals else hybrid[j]
            s = calibrated.sum()
            if s > 0:
                calibrated /= s
            hybrid = calibrated

        hw, dr, aw = float(hybrid[2]), float(hybrid[1]), float(hybrid[0])
        result     = _hda_result(hw, dr, aw)

        probs_sorted = sorted([hw, dr, aw], reverse=True)
        margin_pp    = (probs_sorted[0] - probs_sorted[1]) * 100

        bh, ba = _best_score(lam_h, lam_a)

        standings = _load_standings(h, a)

        return {
            'probs':      {'H': round(hw, 4), 'D': round(dr, 4), 'A': round(aw, 4)},
            'score':      f'{bh}-{ba}',
            'result':     result,
            'lambda_ft':  {'home': float(lam_h), 'away': float(lam_a)},
            'model':      'club_hybrid',
            'form': {
                'home_gf': round(fh5[1], 2), 'home_ga': round(fh5[2], 2),
                'away_gf': round(fa5[1], 2), 'away_ga': round(fa5[2], 2),
            },
            'margin_pp':  round(margin_pp, 1),
            'standings':  standings,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Poisson+Elo (league fallback)
# ─────────────────────────────────────────────────────────────────────────────

def predict_match_legacy(
    home: str, away: str,
    ts: dict, ga: float, elo_r: dict,
) -> dict:
    """Fallback Poisson+Elo predictor for domestic leagues.

    Parameters
    ----------
    ts  : team-stats dict from pipeline.data_loader.train()
    ga  : global average goals
    elo_r : elo ratings dict
    """
    h_ts = ts.get(home, {'attack': 1.0, 'defense': 1.0})
    a_ts = ts.get(away, {'attack': 1.0, 'defense': 1.0})

    lam_h = max(0.1, min(5.0, ga * h_ts['attack'] * a_ts['defense'] * 1.05))
    lam_a = max(0.1, min(5.0, ga * a_ts['attack'] * h_ts['defense'] * 0.95))

    hw = dr = aw = 0.0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson_pmf(hg, lam_h) * poisson_pmf(ag, lam_a)
            if hg > ag:   hw += p
            elif hg == ag: dr += p
            else:         aw += p

    t = hw + dr + aw
    hw, dr, aw = hw / t, dr / t, aw / t

    eh = elo_r.get(home, 1500)
    ea = elo_r.get(away, 1500)
    ep = elo_expected(eh, ea)
    w  = 0.55

    hw = hw * w + ep * (1 - w)
    aw = aw * w + (1 - ep) * (1 - w)
    dr = dr * w + 0.2 * (1 - w)
    t  = hw + dr + aw
    hw, dr, aw = hw / t, dr / t, aw / t

    bh, ba = _best_score(lam_h, lam_a)
    result = _hda_result(hw, dr, aw)

    return {
        'probs':       {'H': round(hw, 4), 'D': round(dr, 4), 'A': round(aw, 4)},
        'score':       f'{bh}-{ba}',
        'result':      result,
        'min_odds':    {k: round(1.02 / v, 2) for k, v in [('H', hw), ('D', dr), ('A', aw)]},
        'matches_data': (h_ts.get('m', 0), a_ts.get('m', 0)),
        'lambda_ft':   {'home': float(lam_h), 'away': float(lam_a)},
        'model':       'legacy_poisson',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Market fallback (no model data at all)
# ─────────────────────────────────────────────────────────────────────────────

def fallback_market_predict(market_row: dict) -> dict:
    """Derive outcome probabilities directly from 500.com SPF odds.

    Used as last resort when all model routes fail. EV is circular when
    derived from the same odds, so bundle is tagged 'market_fallback'.
    """
    odds_h = market_row.get('odds_h', 0)
    odds_d = market_row.get('odds_d', 0)
    odds_a = market_row.get('odds_a', 0)
    probs  = implied_probs_from_odds(odds_h, odds_d, odds_a)

    # All zero odds → uniform fallback
    if sum(probs.values()) < 0.01:
        probs = {'H': 1/3, 'D': 1/3, 'A': 1/3}

    lam_total = 2.55
    lam_home  = max(0.2, lam_total * (probs['H'] + 0.5 * probs['D']))
    lam_away  = max(0.2, lam_total - lam_home)

    bh, ba = _best_score(lam_home, lam_away)

    ph, pd, pa = probs['H'], probs['D'], probs['A']
    result = _hda_result(ph, pd, pa)

    return {
        'probs':       {'H': round(ph, 4), 'D': round(pd, 4), 'A': round(pa, 4)},
        'score':       f'{bh}-{ba}',
        'result':      result,
        'min_odds':    {k: round(1.02 / max(v, 1e-6), 2) for k, v in [('H', ph), ('D', pd), ('A', pa)]},
        'matches_data': (0, 0),
        'lambda_ft':   {'home': float(lam_home), 'away': float(lam_away)},
        'model':       'market_fallback',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hda_result(hw: float, dr: float, aw: float) -> str:
    if hw >= dr and hw >= aw:
        return 'H'
    if dr >= hw and dr >= aw:
        return 'D'
    return 'A'


def _best_score(lam_h: float, lam_a: float) -> tuple[int, int]:
    """Return the most probable (home_goals, away_goals) score."""
    bp = bh = ba = 0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson_pmf(hg, lam_h) * poisson_pmf(ag, lam_a)
            if p > bp:
                bp, bh, ba = p, hg, ag
    return bh, ba


def _recent_form_club(form: dict, team: str, n: int = 5) -> list:
    """[win_rate, avg_gf, avg_ga, avg_gd] over last n games."""
    games  = form.get(team, [])
    recent = games[-n:] if len(games) >= n else games
    if not recent:
        return [0.5, 0.0, 0.0, 0.0]
    wins = sum(1 for g in recent if g[0] > g[1]) + sum(0.5 for g in recent if g[0] == g[1])
    gf   = sum(g[0] for g in recent) / len(recent)
    ga   = sum(g[1] for g in recent) / len(recent)
    return [wins / len(recent), gf, ga, gf - ga]


def _build_xg_feat(xg: Optional[dict], h: str, a: str) -> list:
    """8-dim xG-proxy feature vector (home 4 dims + away 4 dims)."""
    feat = []
    for team in (h, a):
        s = (xg or {}).get(team, {})
        feat.extend([
            s.get('xg_proxy_5', 0.0),
            s.get('xg_proxy_12', 0.0),
            s.get('xg_streak', 0) / 10.0,
            s.get('xg_volatility', 0.0),
        ])
    return feat


def _load_h2h_gd(h: str, a: str) -> float:
    """Load H2H goal-difference from club cache. Returns 0.0 on miss."""
    try:
        import json as _j
        h2h_path = os.path.join(os.environ.get('DATA_DIR', '/root/data'), 'h2h_cache_club.json')
        if not os.path.exists(h2h_path):
            return 0.0
        with open(h2h_path) as f:
            cache = _j.load(f)
        key    = tuple(sorted([h, a]))
        entry  = cache.get(f'{key[0]}||{key[1]}')
        if entry:
            return entry[1] - entry[2] if h == key[0] else entry[2] - entry[1]
    except Exception:
        pass
    return 0.0


def _load_standings(h: str, a: str) -> Optional[dict]:
    """Load league standings for both teams. Returns None on any failure."""
    try:
        from standings_lookup import load_standings_cache, lookup_both
        cache = load_standings_cache()
        hi, ai, _ = lookup_both(h, a, cache)
        if hi and ai and hi.get('comp_id') == ai.get('comp_id'):
            return {
                'home':      f"#{hi['position']} {hi['points']}pts GD{hi['goal_difference']:+d}",
                'away':      f"#{ai['position']} {ai['points']}pts GD{ai['goal_difference']:+d}",
                'rank_diff': hi['position'] - ai['position'],
                'pt_diff':   hi['points'] - ai['points'],
                'gd_diff':   hi['goal_difference'] - ai['goal_difference'],
                'comp_id':   hi['comp_id'],
            }
    except Exception:
        pass
    return None


def _run_simple_model(
    xgb_s, cal_s,
    op_h: float,
    fh5: list, fa5: list,
) -> tuple[str, float]:
    """Run the simple parallel model; returns (pred_label, confidence)."""
    if xgb_s is None:
        return '', 0.0
    try:
        market_odds_h = 1.0 / max(op_h, 0.01)
        feat = np.array([[
            market_odds_h,
            fh5[0], fh5[1], fh5[2],
            fa5[0], fa5[1], fa5[2],
        ]])
        proba = xgb_s.predict_proba(feat)[0]
        if cal_s is not None:
            cal = np.zeros(3)
            for j, key in enumerate(['home', 'draw', 'away']):
                cal[j] = cal_s[key].predict([proba[j]])[0] if key in cal_s else proba[j]
            s = cal.sum()
            if s > 0:
                cal /= s
            proba = cal
        label = ['H', 'D', 'A'][proba.argmax()]
        return label, float(proba.max())
    except Exception:
        return '', 0.0
