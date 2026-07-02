#!/usr/bin/env python3
"""daily_jczq.py — 每日竞彩足球预测 (v4)  ← 仅剩 main() 骨架

数据流:
  football-data.org → 500.com → 365scores
  → pipeline.predictor (DC+XGB / Poisson+Elo)
  → pipeline.bundle_builder (组包 / 打印 / 落盘)
"""
import os
import re
from collections import defaultdict
from datetime import date

from config.settings import JCZQ_LEAGUES, MODEL_VERSION
from pipeline.data_loader import (
    fetch_league_history,
    get_today_matches,
    load_365scores_today,
    build_365_map,
)
from pipeline.scraper import (
    apply_euro_fallback,
    fetch_live_odds_map,
    scrape_500_odds_today,
)
from pipeline.predictor import (
    predict_match_wrapper,
    predict_match_legacy,
    fallback_market_predict,
)
from pipeline.trainer import train
from pipeline.bundle_builder import (
    build_prediction_bundle,
    print_match_bundle,
    record_prediction,
    compute_bet_action,
)
from scraper_500_analysis import scrape_500_analysis, enrich_bundle_with_500
from fatigue_features import compute_fatigue_features, fatigue_adjustment
import bet_math


def main() -> int:
    from team_name_normalizer import normalize_match_pair

    today_str = date.today().isoformat()
    wd = ['一', '二', '三', '四', '五', '六', '日'][date.today().weekday()]
    print(f"{'=' * 60}")
    print(f"  ⚽ 每日竞彩预测  {today_str} 周{wd}")
    print(f"{'=' * 60}")

    print('📡 获取今日赛程...')
    matches       = get_today_matches()
    _500_odds     = scrape_500_odds_today()
    score365_games = load_365scores_today()
    score365_map  = build_365_map(score365_games)
    if _500_odds:
        print(f'  📡 500.com: {len(_500_odds)} 场有赔率数据')
    print(f'  📡 365scores: {len(score365_games)} 场增强数据')

    # ── 500.com 分析数据 ──
    _500_analysis: dict = {}
    if _500_odds:
        print('\n📡 抓取500.com比赛分析数据...')
        match_codes = {
            m5['code']: {'id': m5.get('shuju_id', ''), 'home': m5['home_cn'], 'away': m5['away_cn']}
            for m5 in _500_odds
        }
        has_ids = any(v['id'] for v in match_codes.values())
        _500_analysis = scrape_500_analysis(match_codes if has_ids else None)
        if _500_analysis:
            print(f'  📡 500.com分析: {len(_500_analysis)} 场有分析数据')

    use_500_only = False
    if not matches and _500_odds:
        print(f'  📭 football-data.org 无联赛赛事, 使用500.com {len(_500_odds)} 场国际赛')
        use_500_only = True
    elif not matches:
        print('  📭 今日无竞彩赛事\n')
        return 0

    if not use_500_only:
        print(f'  {len(matches)} 场联赛/杯赛 ({len(_500_odds)} 场有500赔率)')

    # ── 训练后备模型 ──
    ts = ga = elo_r = None
    if not use_500_only:
        leagues_needed = {m['competition']['code'] for m in matches}
        print('\n📡 拉取历史训练数据...')
        all_hist: list = []
        for code, lname in JCZQ_LEAGUES:
            if code not in leagues_needed:
                continue
            hist = fetch_league_history(code)
            print(f'  {lname}: {len(hist)} 场历史')
            all_hist.extend(hist)
        if not all_hist:
            print('❌ 无历史数据, 无法预测')
            return 1
        print('\n🧠 训练后备模型 (泊松+Elo)...')
        ts, ga, elo_r = train(all_hist)
        print(f'  总训练: {len(all_hist)} 场 | λ={ga:.3f} | 球队: {len(ts)}')

    print(f"\n{'─' * 60}\n  📊 预测\n{'─' * 60}")

    hybrid_count = legacy_count = 0
    _500_map: dict = {}
    for m5 in _500_odds:
        try:
            h_e, a_e = normalize_match_pair(m5['home_cn'], m5['away_cn'])
            _500_map[(h_e, a_e)] = _500_map[(a_e, h_e)] = m5
        except Exception:
            pass

    bundles: list = []

    def _apply_fatigue(bundle, home_label, away_label, time_str, league_label, a_data):
        if not a_data.get('future_fixtures'):
            return
        clean_h = re.sub(r'\[\d+\]', '', home_label).strip()
        clean_a = re.sub(r'\[\d+\]', '', away_label).strip()
        fatigue = compute_fatigue_features(clean_h, clean_a, time_str, league_label, a_data['future_fixtures'])
        bundle['fatigue'] = fatigue
        if abs(fatigue.get('rotation_diff', 0)) >= 0.1:
            adj = fatigue_adjustment(fatigue, {
                'H': bundle['pred_h'] / 100,
                'D': bundle['pred_d'] / 100,
                'A': bundle['pred_a'] / 100,
            })
            bundle['pred_h'] = adj['H'] * 100
            bundle['pred_d'] = adj['D'] * 100
            bundle['pred_a'] = adj['A'] * 100
            bundle['model_note'] = bundle.get('model_note', '') + '+疲劳度调整'

    if use_500_only:
        league_counts: dict = defaultdict(int)
        for m5 in _500_odds:
            league_counts[m5.get('league', '') or '未知'] += 1
        league_info = ', '.join(f'{k}{v}场' for k, v in sorted(league_counts.items()))
        print(f"\n  📋 500.com 赛事 ({len(_500_odds)}场) — {league_info}\n  {'─' * 60}")

        for m5 in _500_odds:
            home_cn, away_cn = m5['home_cn'], m5['away_cn']
            p = predict_match_wrapper(home_cn, away_cn)
            if not p:
                legacy_count += 1
                print(f'  ⚠ {m5["code"]} {home_cn} vs {away_cn} — 主模型无数据，回退市场保底')
                p = fallback_market_predict(m5)
            else:
                hybrid_count += 1
            h_norm, a_norm = normalize_match_pair(home_cn, away_cn)
            score_meta = score365_map.get((h_norm, a_norm))
            raw_league = m5.get('league', '') or ''
            if '世界杯' in raw_league or 'world cup' in raw_league.lower():
                league_label = '世界杯'
            elif '友谊赛' in raw_league or 'friendly' in raw_league.lower():
                league_label = '友谊赛'
            else:
                league_label = raw_league or '友谊赛'

            bundle = build_prediction_bundle(m5['code'], home_cn, away_cn, m5['time'], league_label, p, m5, score_meta)
            enrich_bundle_with_500(bundle, _500_analysis.get(m5['code']))
            apply_euro_fallback(bundle, m5)
            _apply_fatigue(bundle, home_cn, away_cn, m5.get('time', ''), league_label, _500_analysis.get(m5['code'], {}))
            bundles.append(bundle)
            print_match_bundle(bundle)
    else:
        by_league: dict = defaultdict(list)
        for m in matches:
            by_league[m['competition']['code']].append(m)

        for code, lname in JCZQ_LEAGUES:
            if code not in by_league:
                continue
            ms = by_league[code]
            print(f"\n  📋 {lname} ({len(ms)}场)\n  {'─' * 60}")
            for m in ms:
                home = m['homeTeam']['shortName']
                away = m['awayTeam']['shortName']
                utc  = m['utcDate'][11:16]
                p = predict_match_wrapper(home, away)
                if p:
                    hybrid_count += 1
                else:
                    p = predict_match_legacy(home, away, ts, ga, elo_r)
                    legacy_count += 1
                m5 = _500_map.get((home, away))
                h_norm, a_norm = normalize_match_pair(home, away)
                score_meta = score365_map.get((h_norm, a_norm))
                bundle = build_prediction_bundle(
                    m5['code'] if m5 else f'{code}-{home}-{away}',
                    home, away, utc, lname, p, m5, score_meta,
                )
                analysis_key = m5['code'] if m5 else None
                if analysis_key:
                    enrich_bundle_with_500(bundle, _500_analysis.get(analysis_key))
                if m5:
                    apply_euro_fallback(bundle, m5)
                _apply_fatigue(bundle, home, away, utc[:10] if utc else today_str, lname,
                               _500_analysis.get(analysis_key, {}) if analysis_key else {})
                bundles.append(bundle)
                print_match_bundle(bundle)

    for bundle in bundles:
        record_prediction(bundle)

    # ── 全局价值汇总 ──
    all_analyses = [
        b['bet_analysis'] for b in bundles
        if b.get('bet_analysis') and b.get('bet_action') not in ('SKIP_LEAGUE', 'WATCH', 'WATCH_FRIENDLY')
    ]
    n_skipped = sum(1 for b in bundles if b.get('bet_action') in ('SKIP_LEAGUE', 'WATCH', 'WATCH_FRIENDLY'))
    if all_analyses:
        print(bet_math.format_value_summary(all_analyses, min_ev=0.05))
    if n_skipped:
        print(f'  ℹ️ 已过滤 {n_skipped} 场赛事类型不推荐场次 (SKIP_LEAGUE/WATCH)')

    print(f"\n{'=' * 60}\n  💎 购彩建议\n{'=' * 60}")
    print(f'  📆 {today_str} 周{wd}  |  {len(bundles)} 场竞彩赛事')
    print(f'  🧠 国际赛: DC+XGBoost+Form ({hybrid_count}场) | 联赛: 泊松+Elo ({legacy_count}场)')
    print(f'  📡 365scores增强已启用')
    print(f'  💾 已写入: /root/data/predictions_log.csv')
    print(f'  \n  📌 策略:')
    print(f'  • 输出口径: 90分钟常规时间(含伤停补时)')
    print(f'  • 每场落盘: 胜平负 / 竞彩让球 / 半全场 / 比分 / 总进球')
    print(f'  • 500.com决定可买场次, 365scores提供增强特征')
    print(f'\n  ⚠️ 本预测基于统计数据, 不构成投注建议')
    print(f'  请理性购彩, 切勿沉迷')
    print(f"{'=' * 60}")
    return 0


if __name__ == '__main__':
    main()
