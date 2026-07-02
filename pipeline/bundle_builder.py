"""Prediction bundle assembly, display, and logging.

Replaces five functions previously in daily_jczq.py:
    build_prediction_bundle()      → assemble full output dict
    print_match_bundle()           → terminal display
    record_prediction()            → direct Python call to backtest_jczq
    ensure_log_has_source_fields() → CSV schema migration
    patch_logged_metadata()        → backfill source_tag / model_version

Also hosts pure helpers that belong to this layer:
    compute_bet_action()
    compute_htft_topn()
    pick_best_htft()
    top_market_label()
    estimate_vote_fusion_alpha()
"""
from __future__ import annotations

import csv
import json
import os
from datetime import date
from typing import Optional

import bet_math
from scraper_500_analysis import enrich_bundle_with_500, format_500_analysis_lines
from fatigue_features import format_fatigue_lines
from config.settings import (
    PREDICTIONS_LOG,
    MODEL_VERSION,
    HTFT_SHORT_MAP,
    HTFT_DISPLAY_MAP,
)
from pipeline.probability import (
    compute_rq_probs,
    compute_goals_distribution,
    compute_score_topn,
    compute_htft_topn_math,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def estimate_vote_fusion_alpha(votes: Optional[dict]) -> str:
    """Return string weight for 365scores vote fusion based on sample size."""
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


def top_market_label(odds_map: dict, fallback_label: str) -> str:
    """Return label with the lowest (shortest) market odds, i.e. market favourite."""
    if odds_map:
        available = [(label, odd) for label, odd in odds_map.items() if odd and odd > 0]
        if available:
            return min(available, key=lambda x: x[1])[0]
    return fallback_label


def pick_best_htft(htft_probs: dict, market_htft_odds: Optional[dict] = None) -> str:
    """Select best HTFT label, preferring market-available outcomes."""
    if market_htft_odds:
        available = [
            (label, prob)
            for label, prob in htft_probs.items()
            if market_htft_odds.get(label, 0) > 0
        ]
        if available:
            return max(available, key=lambda x: x[1])[0]
    return max(htft_probs.items(), key=lambda x: x[1])[0]


def compute_htft_topn(
    lambda_home: float,
    lambda_away: float,
    topn: int = 6,
    home: Optional[str] = None,
    away: Optional[str] = None,
) -> tuple[list, dict]:
    """Return top-N HTFT outcomes. Uses XGB model when available, else Poisson math."""
    label_map = {
        '胜胜': '胜胜', '胜平': '胜平', '胜负': '胜负',
        '平胜': '平胜', '平平': '平平', '平负': '平负',
        '负胜': '负胜', '负平': '负平', '负负': '负负',
        'HH': '胜胜', 'HD': '胜平', 'HA': '胜负',
        'DH': '平胜', 'DD': '平平', 'DA': '平负',
        'AH': '负胜', 'AD': '负平', 'AA': '负负',
    }
    try:
        from htft_predictor import predict_htft_probs
        probs = predict_htft_probs(lambda_home, lambda_away, home=home, away=away)
        probs_cn: dict = {}
        for k, v in probs.items():
            cn = label_map.get(k, k)
            probs_cn[cn] = probs_cn.get(cn, 0) + v
    except Exception:
        _, probs_cn = compute_htft_topn_math(lambda_home, lambda_away, topn=9)

    rows = sorted(probs_cn.items(), key=lambda kv: kv[1], reverse=True)
    return rows[:topn], probs_cn


def compute_bet_action(
    league: str,
    model_type: str,
    bet_analysis,
    htft_top6: list,
    handicap: int,
    rq_probs: dict,
) -> str:
    """Classify match for betting action routing.

    Returns
    -------
    'SKIP_LEAGUE'    : systematically unprofitable competition
    'WATCH'          : market-fallback or circular EV
    'WATCH_FRIENDLY' : friendly match — calibrator overfit confirmed
    'RECOMMEND'      : cleared for full analysis
    """
    if league == 'UEFA Nations League':
        return 'SKIP_LEAGUE'
    if model_type == 'market_fallback':
        return 'WATCH'
    if '友谊赛' in league or 'Friendly' in league or 'Friendlies' in league:
        return 'WATCH_FRIENDLY'
    return 'RECOMMEND'


# ─────────────────────────────────────────────────────────────────────────────
# Bundle assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_bundle(
    code: str,
    home: str,
    away: str,
    utc: str,
    league: str,
    p: dict,
    market_row: Optional[dict] = None,
    score_meta: Optional[dict] = None,
) -> dict:
    """Assemble the full per-match output dict from model output + market data."""
    lambda_home = p['lambda_ft']['home']
    lambda_away = p['lambda_ft']['away']
    market_row = market_row or {}
    votes = score_meta.get('votes') if score_meta else None

    pred_h = p['probs']['H']
    pred_d = p['probs']['D']
    pred_a = p['probs']['A']

    odds_h = market_row.get('odds_h', 0)
    odds_d = market_row.get('odds_d', 0)
    odds_a = market_row.get('odds_a', 0)

    if odds_h and odds_d and odds_a and odds_h > 1 and odds_d > 1 and odds_a > 1:
        try:
            from mc_market_weight_helper import market_weight_for_match
            mkt_h, mkt_d, mkt_a = 1 / odds_h, 1 / odds_d, 1 / odds_a
            mkt_total = mkt_h + mkt_d + mkt_a
            mkt_h /= mkt_total; mkt_d /= mkt_total; mkt_a /= mkt_total
            elo_h = p.get('elo_h', 1500)
            elo_a = p.get('elo_a', 1500)
            mkt_w = market_weight_for_match(elo_h, elo_a, neutral=p.get('neutral', True))
            pred_h = (1 - mkt_w) * pred_h + mkt_w * mkt_h
            pred_d = (1 - mkt_w) * pred_d + mkt_w * mkt_d
            pred_a = (1 - mkt_w) * pred_a + mkt_w * mkt_a
            s = pred_h + pred_d + pred_a
            if s > 0:
                pred_h /= s; pred_d /= s; pred_a /= s
        except Exception:
            pass

    pred_h *= 100
    pred_d *= 100
    pred_a *= 100

    spf_pick = max(
        [('主胜', pred_h), ('平', pred_d), ('客胜', pred_a)],
        key=lambda x: x[1],
    )[0]
    market_spf_pick = top_market_label(
        {'主胜': odds_h, '平': odds_d, '客胜': odds_a}, spf_pick,
    )

    handicap = int(market_row.get('handicap', 0) or 0)
    dc_rho = p.get('rho', 0.0)
    rq_probs = compute_rq_probs(lambda_home, lambda_away, handicap, rho=dc_rho)
    rq_pick = max(rq_probs.items(), key=lambda x: x[1])[0]
    market_rq_pick = top_market_label(
        {'让胜': market_row.get('rq_h', 0), '让平': market_row.get('rq_d', 0), '让负': market_row.get('rq_a', 0)},
        rq_pick,
    )

    goals_dist = compute_goals_distribution(lambda_home, lambda_away, rho=dc_rho)
    goals_items = sorted(goals_dist.items(), key=lambda kv: kv[1], reverse=True)
    goals_pick = int(goals_items[0][0]) if goals_items else int(
        sum(int(k) * v for k, v in goals_dist.items())
    )
    goals_top5 = goals_items[:5]
    goals_all  = goals_items

    score_top8 = compute_score_topn(lambda_home, lambda_away, 8, rho=dc_rho)
    pred_top_score = score_top8[0][0] if score_top8 else p['score'].replace('-', ':')
    market_score_pick = top_market_label(market_row.get('bf_odds', {}), pred_top_score)
    score_prob_map = {score: prob for score, prob, _hg, _ag in score_top8}
    score_all = [
        (s, prob) for s, prob, _hg, _ag in
        compute_score_topn(lambda_home, lambda_away, topn=999, rho=dc_rho)
        if prob > 0
    ]

    htft_top6, htft_probs = compute_htft_topn(lambda_home, lambda_away, 6, home=home, away=away)
    pred_top_htft = max(htft_probs.items(), key=lambda x: x[1])[0]
    market_htft_pick = pick_best_htft(htft_probs, market_row.get('htft_odds'))
    htft_all = sorted(htft_probs.items(), key=lambda kv: kv[1], reverse=True)

    def _ev(prob_pct: float, odds: float):
        p_ = prob_pct / 100.0
        return p_ * (odds - 1) - (1 - p_) if odds > 1 and p_ > 0 else ''

    ev_h = _ev(pred_h, odds_h)
    ev_d = _ev(pred_d, odds_d)
    ev_a = _ev(pred_a, odds_a)

    _predictions = {
        'spf':        {'h': pred_h / 100, 'd': pred_d / 100, 'a': pred_a / 100},
        'rq':         {'rq_win': rq_probs['让胜'], 'rq_draw': rq_probs['让平'], 'rq_lose': rq_probs['让负']},
        'score':      [{'score': s, 'prob': pr} for s, pr, _hg, _ag in score_top8[:5]],
        'total_goals': [{'goals': g, 'prob': pr} for g, pr in goals_top5],
        'half_full':  [{'hf': label, 'prob': pr} for label, pr in htft_top6[:4]],
    }
    _odds_d = {
        'spf':   {'h': odds_h, 'd': odds_d, 'a': odds_a} if all([odds_h, odds_d, odds_a]) else {},
        'rq':    {'rq_win': market_row.get('rq_h', 0), 'rq_draw': market_row.get('rq_d', 0), 'rq_lose': market_row.get('rq_a', 0)},
        'score': market_row.get('bf_odds', {}),
        'total_goals': {str(k).replace('球', ''): v for k, v in market_row.get('zjq_odds', {}).items()} if market_row.get('zjq_odds') else {},
        'half_full': market_row.get('htft_odds', {}),
    }
    model_type = p.get('model', '')
    bet_analysis = bet_math.analyze_match(home, away, _predictions, _odds_d, model_type)

    spf_value_tips = sorted(
        [
            {'label': s.pick, 'prob': s.prob * 100, 'odd': s.odds, 'ev': s.ev}
            for s in bet_analysis.scenarios
            if s.play == '胜平负' and s.is_value
        ],
        key=lambda x: x['ev'], reverse=True,
    )

    market_conflicts: list[str] = []
    if market_spf_pick != spf_pick:
        market_conflicts.append(f'SPF市场倾向={market_spf_pick}')
    if market_rq_pick != rq_pick:
        market_conflicts.append(f'RQ市场倾向={market_rq_pick}')
    if market_score_pick != pred_top_score:
        market_conflicts.append(f'比分市场倾向={market_score_pick}')
    if market_htft_pick != pred_top_htft:
        market_conflicts.append(f'半全场市场倾向={HTFT_DISPLAY_MAP.get(market_htft_pick, market_htft_pick)}')
    zjq_odds_map = market_row.get('zjq_odds', {})
    market_zjq_pick = top_market_label(zjq_odds_map, f'{goals_pick}球') if zjq_odds_map else None
    if market_zjq_pick and market_zjq_pick != f'{goals_pick}球':
        market_conflicts.append(f'总进球市场倾向={market_zjq_pick}')

    vote_h     = votes.get('home')  if votes else None
    vote_d     = votes.get('draw')  if votes else None
    vote_a     = votes.get('away')  if votes else None
    vote_count = votes.get('total') if votes else None
    vote_text  = ''
    if votes and vote_h is not None and vote_d is not None and vote_a is not None:
        vote_text = f'公众{vote_h:.1f}/{vote_d:.1f}/{vote_a:.1f}% n={vote_count or 0}'
        if score_meta and score_meta.get('trend_home') and score_meta.get('trend_away'):
            th = score_meta['trend_home']
            ta = score_meta['trend_away']
            vote_text += f' 近况{home}{th[0]}-{th[1]}-{th[2]} {away}{ta[0]}-{ta[1]}-{ta[2]}'

    if handicap > 0:
        rq_text = f'受让{handicap}'
    elif handicap < 0:
        rq_text = f'让{abs(handicap)}'
    else:
        rq_text = '0'

    bet_action = compute_bet_action(league, model_type, bet_analysis, htft_top6, handicap, rq_probs)
    max_hda_prob = max(pred_h, pred_d, pred_a)
    if max_hda_prob < 60 and bet_action.startswith(('RECOMMEND', 'WATCH')):
        bet_action = f'{bet_action} [LOW_CONF]'

    htft_warning = any(
        s.play == '半全场' and s.pick == '胜胜' and s.prob < 0.20 and model_type == 'hybrid'
        for s in bet_analysis.scenarios
    )

    ah_fair_odds: dict = {}
    try:
        from asian_handicap import ah_probs
        base = float(handicap) if handicap != 0 else 0.5
        scan_range = [h_ * 0.25 for h_ in range(int(max(base - 2, 0) * 4), int((base + 2) * 4) + 1)]
        for h_ in scan_range:
            res = ah_probs(lambda_home, lambda_away, h_, max_goals=15)
            ah_fair_odds[h_] = round(res['fair_odds'], 2)
    except Exception:
        pass

    simple_pred = p.get('simple_pred', '')
    simple_conf = p.get('simple_conf', 0)

    return {
        'code': code, 'league': league, 'time': utc,
        'home': home, 'away': away,
        'home_cn': market_row.get('home_cn', home),
        'away_cn': market_row.get('away_cn', away),
        'handicap': handicap, 'rq_text': rq_text,
        'pred_h': pred_h, 'pred_d': pred_d, 'pred_a': pred_a,
        'spf_pick': spf_pick, 'market_spf_pick': market_spf_pick,
        'pred_rq_win':  rq_probs['让胜'] * 100,
        'pred_rq_draw': rq_probs['让平'] * 100,
        'pred_rq_loss': rq_probs['让负'] * 100,
        'rq_pick': rq_pick, 'market_rq_pick': market_rq_pick,
        'pred_top_score': pred_top_score, 'market_score_pick': market_score_pick,
        'score_prob_map': score_prob_map,
        'pred_top_goals': goals_pick, 'pred_top_htft': pred_top_htft,
        'market_htft_pick': market_htft_pick,
        'pred_spf_pick': spf_pick, 'pred_rq_pick': rq_pick,
        'pred_htft_pick': HTFT_DISPLAY_MAP.get(pred_top_htft, pred_top_htft),
        'pred_goals_pick': goals_pick, 'pred_score_pick': pred_top_score,
        'score_top8': score_top8, 'score_all': score_all,
        'goals_top5': goals_top5, 'goals_all': goals_all, 'goals_pick': goals_pick,
        'htft_top6': htft_top6, 'htft_all': htft_all, 'htft_prob_map': htft_probs,
        'market_spf': f'{odds_h:.2f}-{odds_d:.2f}-{odds_a:.2f}' if all([odds_h, odds_d, odds_a]) else '',
        'zjq_odds_str': _fmt_zjq(market_row.get('zjq_odds', {})),
        'spf_value_tips': spf_value_tips, 'bet_analysis': bet_analysis,
        'market_conflicts': market_conflicts, 'votes_text': vote_text,
        'model_note': p.get('model', ''),
        'direction': f"SPF:{spf_pick} | RQ:{rq_pick} | HTFT:{pred_top_htft} | Goals:{goals_pick} | Score:{pred_top_score}",
        'source_tag': '500+365', 'model_version': MODEL_VERSION,
        'simple_pred': simple_pred, 'simple_conf': simple_conf,
        'pred30_h': p.get('pred30_h'), 'pred30_d': p.get('pred30_d'), 'pred30_a': p.get('pred30_a'),
        'odds_h_str': f'{odds_h:.2f}' if odds_h else '',
        'odds_d_str': f'{odds_d:.2f}' if odds_d else '',
        'odds_a_str': f'{odds_a:.2f}' if odds_a else '',
        'ev_h_str': f'{ev_h:.4f}' if ev_h != '' else '',
        'ev_d_str': f'{ev_d:.4f}' if ev_d != '' else '',
        'ev_a_str': f'{ev_a:.4f}' if ev_a != '' else '',
        'vote_h_str':     f'{vote_h:.1f}' if vote_h is not None else '',
        'vote_d_str':     f'{vote_d:.1f}' if vote_d is not None else '',
        'vote_a_str':     f'{vote_a:.1f}' if vote_a is not None else '',
        'vote_count_str': str(vote_count) if vote_count is not None else '',
        'vote_fusion_alpha': estimate_vote_fusion_alpha(votes),
        'pop_rank_home_str': _safe_str(score_meta, 'pop_rank_home'),
        'pop_rank_away_str': _safe_str(score_meta, 'pop_rank_away'),
        'pop_rank_diff_str': _safe_diff_str(score_meta, 'pop_rank_away', 'pop_rank_home'),
        'trend_win_rate_home_str': _safe_fmt(score_meta, 'trend_win_rate_home', '.4f'),
        'trend_win_rate_away_str': _safe_fmt(score_meta, 'trend_win_rate_away', '.4f'),
        'trend_win_rate_diff_str': _safe_diff_fmt(score_meta, 'trend_win_rate_home', 'trend_win_rate_away', '.4f'),
        's365_home_winrate':    score_meta.get('trend_win_rate_home') if score_meta else None,
        's365_away_winrate':    score_meta.get('trend_win_rate_away') if score_meta else None,
        's365_home_fifa':       score_meta.get('fifa_rank_home')      if score_meta else None,
        's365_away_fifa':       score_meta.get('fifa_rank_away')      if score_meta else None,
        's365_rank_diff': (
            score_meta.get('fifa_rank_away', 100) - score_meta.get('fifa_rank_home', 100)
            if score_meta and score_meta.get('fifa_rank_home') is not None
            and score_meta.get('fifa_rank_away') is not None else None
        ),
        's365_popularity_diff': (
            score_meta.get('pop_rank_home', 50000) - score_meta.get('pop_rank_away', 50000)
            if score_meta and score_meta.get('pop_rank_home') is not None
            and score_meta.get('pop_rank_away') is not None else None
        ),
        'ah_fair_odds': ah_fair_odds,
        'bet_action': bet_action,
        'htft_warning': htft_warning,
        'standings': p.get('standings'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print
# ─────────────────────────────────────────────────────────────────────────────

def print_match_bundle(bundle: dict) -> None:
    """Render a single match bundle to stdout."""

    def _fmt_htft(label: str) -> str:
        return HTFT_DISPLAY_MAP.get(label, label)

    def _fmt_prob_list(full_data, top_data, min_prob: float = 0.001) -> str:
        src = full_data or top_data or []
        items = [(label, prob * 100) for label, prob in src if prob > min_prob]
        items.sort(key=lambda x: -x[1])
        return ' '.join(f'{label}({p:.1f}%)' for label, p in items)

    def _fmt_htft_list(full_data, top_data) -> str:
        src = full_data or top_data or []
        items = [(_fmt_htft(label), prob * 100) for label, prob in src if prob > 0.001]
        items.sort(key=lambda x: -x[1])
        return ' '.join(f'{label}({p:.1f}%)' for label, p in items)

    for line in format_500_analysis_lines(bundle):
        print(line)
    for line in format_fatigue_lines(bundle.get('fatigue', {})):
        print(line)

    spf_pick_prob  = {'主胜': bundle['pred_h'], '平': bundle['pred_d'], '客胜': bundle['pred_a']}.get(bundle['spf_pick'])
    rq_pick_prob   = {'让胜': bundle['pred_rq_win'], '让平': bundle['pred_rq_draw'], '让负': bundle['pred_rq_loss']}.get(bundle['rq_pick'])
    score_pick_prob = {k: v * 100 for k, v in bundle.get('score_prob_map', {}).items()}.get(bundle['pred_top_score'])
    htft_pick_prob  = {k: v * 100 for k, v in bundle.get('htft_prob_map', {}).items()}.get(bundle['pred_top_htft'])

    print()
    print('=' * 70)
    print(f"  {bundle.get('code', '')} {bundle['home']} vs {bundle['away']}  ({bundle['time']})  [{bundle.get('league', '')}]")
    print(f"  bet_action: {bundle.get('bet_action', '')}")
    print('=' * 70)

    print(f"  【胜平负】主{bundle['pred_h']:.1f}% / 平{bundle['pred_d']:.1f}% / 客{bundle['pred_a']:.1f}%")
    print(f"  → 推荐: {bundle['spf_pick']}({spf_pick_prob:.1f}%)")
    if bundle.get('odds_h_str') or bundle.get('odds_d_str') or bundle.get('odds_a_str'):
        print(f"  SPF市场赔率: {bundle.get('odds_h_str', '')} / {bundle.get('odds_d_str', '')} / {bundle.get('odds_a_str', '')}")

    rq_text = bundle.get('rq_text', '')
    print(f"  【竞彩让球({rq_text})】让胜{bundle['pred_rq_win']:.1f}% / 让平{bundle['pred_rq_draw']:.1f}% / 让负{bundle['pred_rq_loss']:.1f}%")
    print(f"  → 推荐: {bundle['rq_pick']}({rq_pick_prob:.1f}%)")

    score_str = _fmt_prob_list(bundle.get('score_all'), bundle.get('score_top8'))
    if score_str:
        print(f'  【比分】{score_str}')
        print(f"  → 推荐: {bundle['pred_top_score']}({score_pick_prob:.1f}%)")

    goals_str = _fmt_prob_list(bundle.get('goals_all'), bundle.get('goals_top5'))
    if goals_str:
        print(f'  【总进球】{goals_str}')
        if bundle['goals_top5']:
            print(f"  → 推荐: {bundle['goals_pick']}球({bundle['goals_top5'][0][1] * 100:.1f}%)")

    htft_str = _fmt_htft_list(bundle.get('htft_all'), bundle.get('htft_top6'))
    if htft_str:
        print(f'  【半全场】{htft_str}')
        print(f"  → 推荐: {_fmt_htft(bundle['pred_top_htft'])}({htft_pick_prob:.1f}%)")

    vh, vd, va, vc = (
        bundle.get('vote_h_str'), bundle.get('vote_d_str'),
        bundle.get('vote_a_str'), bundle.get('vote_count_str'),
    )
    if vh or vd or va:
        print(f'  365scores公众投票: 主{vh}% / 平{vd}% / 客{va}% (n={vc})')

    home_wr   = bundle.get('s365_home_winrate')
    away_wr   = bundle.get('s365_away_winrate')
    home_fifa = bundle.get('s365_home_fifa')
    away_fifa = bundle.get('s365_away_fifa')
    if home_wr is not None or home_fifa is not None:
        parts = []
        if home_wr is not None and away_wr is not None:
            parts.append(f'胜率(主{home_wr:.0%} vs 客{away_wr:.0%})')
        if home_fifa is not None and away_fifa is not None:
            parts.append(f'FIFA({home_fifa} vs {away_fifa})')
        rank_diff = bundle.get('s365_rank_diff')
        if rank_diff is not None:
            parts.append(f'差距{rank_diff:+d}')
        if parts:
            print(f"  📊 365基本面: {' | '.join(parts)}")

    si = bundle.get('standings')
    if si:
        rd  = si.get('rank_diff', 0)
        pd_ = si.get('pt_diff', 0)
        gd  = si.get('gd_diff', 0)
        print(f"  🏆 联赛排名: 主{si.get('home', '')} | 客{si.get('away', '')}  (差: {rd:+d}位, {pd_:+d}分, GD{gd:+d})")

    print(f"  模型: {bundle.get('model_version', '')} (生产) / xgb_model_30 (影子后台运行)")

    ba = bundle.get('bet_analysis')
    if ba and ba.scenarios:
        value_bets = sorted(
            [s for s in ba.scenarios if s.ev > 0.02 and bet_math.is_sane_bet(s)],
            key=lambda s: -s.ev,
        )
        if value_bets:
            parts = [
                f'{s.play}{s.pick}(EV={s.ev:+.1%}, Kelly½={s.kelly_half:.1%})'
                for s in value_bets[:3]
            ]
            print(f"  💰 价值投注: {' | '.join(parts)}")

    if bundle.get('market_conflicts'):
        print(f"  市场分歧: {' | '.join(bundle['market_conflicts'])}")


# ─────────────────────────────────────────────────────────────────────────────
# Logging / CSV
# ─────────────────────────────────────────────────────────────────────────────

def ensure_log_has_source_fields() -> None:
    """Back-fill missing columns in predictions_log.csv (schema migration)."""
    if not os.path.exists(PREDICTIONS_LOG):
        return
    with open(PREDICTIONS_LOG, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    if not fieldnames:
        return
    extras = [
        col for col in (
            'source_tag', 'model_version',
            'pred_spf_pick', 'pred_rq_pick', 'pred_htft_pick', 'pred_goals_pick', 'pred_score_pick',
            's365_home_winrate', 's365_away_winrate', 's365_home_fifa', 's365_away_fifa',
            's365_rank_diff', 's365_popularity_diff',
        )
        if col not in fieldnames
    ]
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


def patch_logged_metadata(code: str, source_tag: str, model_version: str) -> None:
    """Backfill source_tag and model_version for today's rows in the CSV."""
    if not os.path.exists(PREDICTIONS_LOG):
        return
    with open(PREDICTIONS_LOG, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    if not rows:
        return
    today = date.today().isoformat()
    changed = False
    for row in rows:
        if row.get('code') == code and row.get('date') == today:
            row['source_tag'] = source_tag
            row['model_version'] = model_version
            changed = True
    if not changed:
        return
    with open(PREDICTIONS_LOG, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def record_prediction(bundle: dict) -> None:
    """Persist a match bundle to predictions_log.csv.

    Calls backtest_jczq.record_match() directly (Python import) instead
    of the previous subprocess.run approach, eliminating 40-line cmd list
    construction and the associated shell-quoting fragility.

    Falls back to a stderr warning on any exception so a single failed
    record never aborts the full prediction run.
    """
    score_full = {s: round(pr, 4) for s, pr, _hg, _ag in bundle.get('score_top8', [])}
    htft_full  = {
        HTFT_SHORT_MAP.get(label, label): round(pr, 4)
        for label, pr in bundle.get('htft_top6', [])
    }
    goals_full = {str(g): round(pr, 4) for g, pr in bundle.get('goals_all', [])}

    record_kwargs = {
        'code':           bundle['code'],
        'home':           bundle['home_cn'],
        'away':           bundle['away_cn'],
        'league':         bundle['league'],
        'time':           bundle['time'],
        'rq':             bundle['handicap'],
        'pred_h':         bundle['pred_h'],
        'pred_d':         bundle['pred_d'],
        'pred_a':         bundle['pred_a'],
        'pred_rq_win':    bundle['pred_rq_win'],
        'pred_rq_draw':   bundle['pred_rq_draw'],
        'pred_rq_loss':   bundle['pred_rq_loss'],
        'pred_score':     bundle['pred_top_score'],
        'pred_goals':     bundle['pred_top_goals'],
        'pred_htft':      HTFT_SHORT_MAP.get(bundle['pred_top_htft'], bundle['pred_top_htft']),
        'pred_spf_pick':  bundle['pred_spf_pick'],
        'pred_rq_pick':   bundle['pred_rq_pick'],
        'pred_htft_pick': bundle['pred_htft_pick'],
        'pred_goals_pick': bundle['pred_goals_pick'],
        'pred_score_pick': bundle['pred_score_pick'],
        'odds_h':         bundle.get('odds_h_str', ''),
        'odds_d':         bundle.get('odds_d_str', ''),
        'odds_a':         bundle.get('odds_a_str', ''),
        'ev_h':           bundle.get('ev_h_str', ''),
        'ev_d':           bundle.get('ev_d_str', ''),
        'ev_a':           bundle.get('ev_a_str', ''),
        'direction':      bundle['direction'],
        'vote_h':         bundle.get('vote_h_str', ''),
        'vote_d':         bundle.get('vote_d_str', ''),
        'vote_a':         bundle.get('vote_a_str', ''),
        'vote_count':     bundle.get('vote_count_str', ''),
        'vote_fusion_alpha':    bundle.get('vote_fusion_alpha', ''),
        'pop_rank_home':        bundle.get('pop_rank_home_str', ''),
        'pop_rank_away':        bundle.get('pop_rank_away_str', ''),
        'pop_rank_diff':        bundle.get('pop_rank_diff_str', ''),
        'trend_win_rate_home':  bundle.get('trend_win_rate_home_str', ''),
        'trend_win_rate_away':  bundle.get('trend_win_rate_away_str', ''),
        'trend_win_rate_diff':  bundle.get('trend_win_rate_diff_str', ''),
        'simple_pred':    str(bundle.get('simple_pred', '')),
        'simple_conf':    bundle.get('simple_conf', 0),
        'bet_action':     bundle.get('bet_action', ''),
        'model_route':    bundle.get('model', ''),
        'match_key':      f"{bundle.get('date', '')}|{bundle.get('league', '')}|{bundle.get('home_cn', '')}|{bundle.get('away_cn', '')}|{bundle.get('time', '')}",
        'pred30_h':       bundle.get('pred30_h'),
        'pred30_d':       bundle.get('pred30_d'),
        'pred30_a':       bundle.get('pred30_a'),
        's365_home_winrate':    bundle.get('s365_home_winrate'),
        's365_away_winrate':    bundle.get('s365_away_winrate'),
        's365_home_fifa':       bundle.get('s365_home_fifa'),
        's365_away_fifa':       bundle.get('s365_away_fifa'),
        's365_rank_diff':       bundle.get('s365_rank_diff'),
        's365_popularity_diff': bundle.get('s365_popularity_diff'),
        'score_full':     json.dumps(score_full, ensure_ascii=False),
        'htft_full':      json.dumps(htft_full,  ensure_ascii=False),
        'goals_full':     json.dumps(goals_full, ensure_ascii=False),
    }

    try:
        import backtest_jczq
        result = backtest_jczq.record_match(**record_kwargs)
        if result:
            print(f'     💾 {result}')
    except Exception as exc:
        print(f'     ⚠ 落盘失败: {bundle["code"]} | {exc}')
        return

    ensure_log_has_source_fields()
    patch_logged_metadata(bundle['code'], bundle['source_tag'], bundle['model_version'])


# ─────────────────────────────────────────────────────────────────────────────
# Private string helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_str(meta: Optional[dict], key: str) -> str:
    return str(meta[key]) if meta and meta.get(key) is not None else ''


def _safe_diff_str(meta: Optional[dict], key_a: str, key_b: str) -> str:
    if meta and meta.get(key_a) is not None and meta.get(key_b) is not None:
        return str(meta[key_a] - meta[key_b])
    return ''


def _safe_fmt(meta: Optional[dict], key: str, fmt: str) -> str:
    return format(meta[key], fmt) if meta and meta.get(key) is not None else ''


def _safe_diff_fmt(meta: Optional[dict], key_a: str, key_b: str, fmt: str) -> str:
    if meta and meta.get(key_a) is not None and meta.get(key_b) is not None:
        return format(meta[key_a] - meta[key_b], fmt)
    return ''


def _fmt_zjq(zjq: dict) -> str:
    if not zjq:
        return ''
    try:
        d = sorted([(k.replace('球', ''), v) for k, v in zjq.items() if '球' in k])
        if len(d) >= 4:
            return f'{int(d[0][0])}球{d[0][1]:.2f}-{int(d[1][0])}球{d[1][1]:.2f}-{int(d[2][0])}球{d[2][1]:.2f}-{int(d[3][0])}球{d[3][1]:.2f}'
    except Exception:
        pass
    return ''
