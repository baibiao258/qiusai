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
from pipeline.predictor import (
    predict_match_wrapper,
    predict_match_legacy,
    fallback_market_predict,
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


def compute_htft_topn(lambda_home, lambda_away, topn=6, home=None, away=None):
    """半全场预测: 优先 XGB 模型, 回退数学推导."""
    try:
        from htft_predictor import predict_htft_probs
        probs = predict_htft_probs(
            lambda_home, lambda_away,
            home=home, away=away,
        )

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
    rq_probs = compute_rq_probs(lambda_home, lambda_away, handicap, rho=dc_rho)
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
