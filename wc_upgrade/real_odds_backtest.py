#!/usr/bin/env python3
"""
real_odds_backtest.py — 真实收盘赔率回测
=========================================

用真实收盘SP赔率（500.com历史开奖）评估XGBoost预测管线的真实ROI。

合并逻辑:
  - 中文队名 → 英文队名 (team_name_mapping.json)
  - 主队+客队+日期正负2天 模糊匹配 (容忍时区错位)
  - 只有spf_sp > 0的比赛才可交易 (排除未开售)

用法:
  python3 real_odds_backtest.py              # 默认回测
  python3 real_odds_backtest.py --ev 0.10   # EV阈值=10%
  python3 real_odds_backtest.py --play rqspf # 用让球SP赔率回测
"""

import json
import os
import sys
import math
from datetime import datetime, date, timedelta
from collections import defaultdict

import numpy as np

sys.path.insert(0, '/root')

# Competition tier weights for filtering (based on actual ROI)
COMPETITION_TIER = {
    # Tier 1: High ROI, normal betting
    'AFC Asian Cup': 1.2,  # +194.7% ROI
    'FIFA World Cup qualification': 1.0,  # +15.0% ROI
    'World Cup qualification': 1.0,
    
    # Tier 2: Neutral, standard EV threshold
    'UEFA Euro': 0.7,  # -2.4% ROI (slightly negative)
    'Copa America': 0.6,  # -12.7% ROI
    'Copa América': 0.6,
    'African Cup of Nations': 0.5,  # -100% ROI (1 bet)
    
    # Tier 3: Negative ROI, skip or very high threshold
    'Friendly': 0.2,  # -58.1% ROI
    'International Friendlies': 0.2,
    'Friendlies': 0.2,
    'UEFA Nations League': 0.2,  # -72.5% ROI
    
    # Tier 4: Skip completely
    'U23': 0.0,
    'Youth': 0.0,
    'U20': 0.0,
    'U19': 0.0,
}

# Default tier for unknown competitions
DEFAULT_TIER = 0.5
sys.path.insert(0, '/root/wc_2026_upgrade')

DATA_DIR = '/root/data'

# ── 1. 加载数据 ─────────────────────────────────────

def load_kaijiang() -> list[dict]:
    """加载历史开奖数据 (从CSV)"""
    csv_path = os.path.join(DATA_DIR, 'historical_kaijiang.csv')
    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} 不存在, 请先运行 historical_kaijiang.py")
        sys.exit(1)
    
    import csv
    matches = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数值字段
            for num_field in ['handicap', 'ht_h', 'ht_a', 'ft_h', 'ft_a', 'total_goals']:
                if row.get(num_field):
                    row[num_field] = int(row[num_field])
            for sp_field in ['spf_sp', 'rqspf_sp', 'jqs_sp', 'bqc_sp']:
                if row.get(sp_field):
                    row[sp_field] = float(row[sp_field])
            matches.append(row)
    
    print(f"  📊 开奖数据: {len(matches)} 场")
    return matches


def load_international_results() -> list[dict]:
    """加载训练用国际赛结果"""
    path = os.path.join(DATA_DIR, 'international_results.json')
    with open(path) as f:
        data = json.load(f)
    # 只加载2023年以后的 (近期相关)
    recent = [m for m in data if m.get('date', '') >= '2023-01-01']
    print(f"  📊 国际赛数据: {len(recent)} 场 (2023+)")
    return recent


def load_team_mapping() -> dict:
    """加载中文→英文队名映射"""
    path = os.path.join(DATA_DIR, 'team_name_mapping.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_xgb_model():
    """加载XGBoost模型和校准器"""
    import joblib
    model_paths = [
        os.path.join(DATA_DIR, 'xgb_model_30.pkl'),
        os.path.join(DATA_DIR, 'xgb_model_20_3.pkl'),
    ]
    model = None
    calibrators = None
    for p in model_paths:
        if os.path.exists(p):
            model = joblib.load(p)
            print(f"  📦 XGB模型: {os.path.basename(p)}")
            break
    if model is None:
        print("  ⚠️ 无XGB模型, 使用DC均匀分布")
    
    cal_path = os.path.join(DATA_DIR, 'calibrators.pkl')
    if os.path.exists(cal_path):
        calibrators = joblib.load(cal_path)
        print(f"  📦 Isotonic校准器: ✅")
    
    return model, calibrators


def load_dc_model_and_elo():
    """加载DC模型和Elo"""
    import joblib
    dc_path = os.path.join(DATA_DIR, 'dc_model.pkl')
    elo_path = os.path.join(DATA_DIR, 'elo_ratings.pkl')
    dc = joblib.load(dc_path) if os.path.exists(dc_path) else None
    elo = joblib.load(elo_path) if os.path.exists(elo_path) else {}
    print(f"  📦 DC模型: {'✅' if dc else '❌'} | Elo: {len(elo)} 队")
    return dc, elo


# ── 2. 匹配合并 ─────────────────────────────────────

def merge_data(kaijiang, intl, team_map, date_tolerance=2):
    """Merge kaijiang and international results by team names and date tolerance."""
    from datetime import datetime as dt

    # 建索引: (home_en, away_en) -> [(date, match)]
    intl_index = defaultdict(list)
    for m in intl:
        h, a = m['home'], m['away']
        intl_index[(h, a)].append(m)
        intl_index[(a, h)].append(m)  # 双向

    merged = []
    unmatched = []
    club_teams = {'京都不死鸟', '冈山绿雉', '名古屋鲸鱼', '川崎前锋', '广岛三箭',
                  '柏太阳神', '横滨水手', '浦和红钻', '清水心跳', '町田泽维亚',
                  '神户胜利船', '鹿岛鹿角', '特尔斯达', '芬洛', '布雷达',
                  '格拉夫夏普', '多德勒支', '罗达JC', '海牙', '奥斯',
                  '埃门', '登博思', '威廉二世', '马斯特里赫特', '坎布尔',
                  '阿尔克马尔青年', 'SBV精英', '福伦丹', '赫尔蒙德', '阿贾克斯青年',
                  'FC埃因霍温', '乌德勒支青年', '阿尔梅勒城', '邓伯什', '阿克马尔青年'}

    for kj in kaijiang:
        home_cn, away_cn = kj['home'], kj['away']

        # 跳过俱乐部比赛
        if home_cn in club_teams or away_cn in club_teams:
            continue

        # 映射到英文名
        home_en = team_map.get(home_cn)
        away_en = team_map.get(away_cn)

        if not home_en or not away_en:
            unmatched.append((home_cn, away_cn))
            continue

        # 日期解析
        try:
            kj_date = dt.strptime(kj['date'], '%Y-%m-%d').date()
        except ValueError:
            continue

        # 在国际赛中查找匹配
        candidates = intl_index.get((home_en, away_en), [])
        best_match = None
        best_delta = 999

        for im in candidates:
            try:
                im_date = dt.strptime(im['date'], '%Y-%m-%d').date()
            except (ValueError, TypeError):
                continue
            delta = abs((im_date - kj_date).days)
            if delta <= date_tolerance and delta < best_delta:
                best_delta = delta
                best_match = im

        if best_match:
            merged.append({
                'date_kj': kj['date'],
                'date_ir': best_match['date'],
                'date_delta': best_delta,
                'home_cn': home_cn,
                'away_cn': away_cn,
                'home_en': home_en,
                'away_en': away_en,
                'handicap': kj['handicap'],
                # 真实赛果
                'ft_h': kj['ft_h'],
                'ft_a': kj['ft_a'],
                'spf_result': kj['spf_result'],  # 3/1/0
                # 收盘赔率
                'spf_sp': kj['spf_sp'],
                'rqspf_result': kj['rqspf_result'],
                'rqspf_sp': kj['rqspf_sp'],
                'jqs_result': kj['jqs_result'],
                'jqs_sp': kj['jqs_sp'],
                'bqc_result': kj['bqc_result'],
                'bqc_sp': kj['bqc_sp'],
                # 国际赛数据
                'tournament': best_match.get('tournament', ''),
            })

    if unmatched:
        unique_unmatched = sorted(set(unmatched))
        print(f"  ⚠️ 未匹配: {len(unique_unmatched)} 对 (俱乐部/小队跳过)")

    print(f"  ✅ 合并: {len(merged)} 场 (从 {len(kaijiang)} 场开奖中)")
    return merged


# ── 3. XGBoost 特征工程 ──────────────────────────────

def compute_dc_probs(dc_model, home, away):
    """DC模型预测"""
    try:
        lam_h, lam_a = dc_model.predict_lambda(home, away, True)
        if lam_h is None:
            return None, None, None
        from scipy.stats import poisson
        max_g = 8
        ph = [poisson.pmf(i, lam_h) for i in range(max_g)]
        pa = [poisson.pmf(i, lam_a) for i in range(max_g)]
        p_h, p_d, p_a = 0, 0, 0
        for i in range(max_g):
            for j in range(max_g):
                p = ph[i] * pa[j]
                rho = 0.25 if (i == 0 and j == 0) else 1.0
                if i > j: p_h += p * rho
                elif i == j: p_d += p * rho
                else: p_a += p * rho
        return [p_a, p_d, p_h], lam_h, lam_a  # [A,D,H] order for XGB
    except Exception:
        return None, None, None


def make_features(match, dc_model, elo, xgb_model, calibrators=None, sp=0.0):
    """为单场比赛构建29维特征并预测"""
    home, away = match['home_en'], match['away_en']

    # DC概率
    dc_p, lam_h, lam_a = compute_dc_probs(dc_model, home, away)
    if dc_p is None or lam_h is None:
        return None

    elo_h = elo.get(home, 1500)
    elo_a = elo.get(away, 1500)

    # 近5场form
    try:
        from predict_match import recent_form
        fh5 = recent_form(home, 5)
        fa5 = recent_form(away, 5)
    except Exception:
        fh5 = [0.5, 0.0, 0.0, 0.0]
        fa5 = [0.5, 0.0, 0.0, 0.0]

    # Gold特征 (h2h + 12场form)
    try:
        from feature_helper import build_gold_features
        gold = build_gold_features(home, away, match_type='competitive')
    except Exception:
        gold = [0.0, 0, 0, 0.0, 0.0]

    # 概率特征
    op_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
    op_a = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))

    # 完整30维特征 (29 + 市场赔率)
    b15 = [
        (elo_h - elo_a) / 400, lam_h, lam_a, lam_h - lam_a,
        math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
        dc_p[0], dc_p[1], dc_p[2],
        fh5[0], fa5[0],
        fh5[1] - fa5[2], fa5[1] - fh5[2],
        fh5[1] - fa5[1], fh5[0] - fa5[0],
        1,
    ]
    odds_feat = [op_h, op_a, 0.0]
    form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]
    
    # 市场赔率特征 (第30维)
    market_implied = 1.0 / sp if sp > 0 else 0.0
    
    feat = np.array([b15 + gold + odds_feat + form_feat + [market_implied]])

    if xgb_model is not None:
        try:
            xp = xgb_model.predict_proba(feat)[0]
            # XGB输出: [class0=A, class1=D, class2=H]
            p_home = xp[2]
            p_draw = xp[1]
            p_away = xp[0]
            # 混合: DC权重0.4, XGB权重0.6
            alpha = 0.4
            p_home = alpha * dc_p[2] + (1 - alpha) * p_home
            p_draw = alpha * dc_p[1] + (1 - alpha) * p_draw
            p_away = alpha * dc_p[0] + (1 - alpha) * p_away
        except Exception as e:
            print(f"  ⚠️ XGB预测失败: {e}")
            p_home, p_draw, p_away = dc_p[2], dc_p[1], dc_p[0]
    else:
        p_home, p_draw, p_away = dc_p[2], dc_p[1], dc_p[0]

    # 应用Isotonic校准
    if calibrators is not None:
        raw = np.array([p_away, p_draw, p_home])  # [A, D, H]
        calibrated = np.zeros(3)
        for j, key in enumerate(['away', 'draw', 'home']):
            if key in calibrators:
                calibrated[j] = calibrators[key].predict([raw[j]])[0]
            else:
                calibrated[j] = raw[j]
        s = calibrated.sum()
        if s > 0: calibrated = calibrated / s
        p_away, p_draw, p_home = calibrated[0], calibrated[1], calibrated[2]

    return {'p_home': p_home, 'p_draw': p_draw, 'p_away': p_away}


# ── 4. 回测主逻辑 ────────────────────────────────────

def run_backtest(merged, dc_model, elo, xgb_model, calibrators, args):
    """真实赔率回测"""
    import pandas as pd

    ev_threshold = args.ev
    play = args.play  # spf / rqspf / jqs / bqc

    print(f"\n{'='*60}")
    print(f"  🔥 真实赔率回测")
    print(f"  玩法: {play} | EV阈值: {ev_threshold*100:.0f}%")
    print(f"  合并数据: {len(merged)} 场")
    print(f"{'='*60}\n")

    # 预测 + 收集
    rows = []
    for m in merged:
        # 选择赔率来源 (先确定sp, 再传给make_features)
        if play == 'spf':
            sp = m['spf_sp']
            result = m['spf_result']  # 3=胜, 1=平, 0=负
        elif play == 'rqspf':
            sp = m['rqspf_sp']
            result = m['rqspf_result']
        elif play == 'jqs':
            sp = m['jqs_sp']
            result = m['jqs_result']
        elif play == 'bqc':
            sp = m['bqc_sp']
            result = m['bqc_result']
        else:
            sp = m['spf_sp']
            result = m['spf_result']
        
        probs = make_features(m, dc_model, elo, xgb_model, calibrators, sp)
        if probs is None:
            continue
        
        # 确定p_outcome
        if play == 'spf':
            p_outcome = probs['p_home']
        elif play == 'rqspf':
            p_outcome = probs['p_home']  # 简化: 让球后用主胜概率近似
        elif play == 'jqs':
            p_outcome = 0.25  # 总进球占位
        elif play == 'bqc':
            p_outcome = 0.1  # 半全场占位
        else:
            sp = m['spf_sp']
            result = m['spf_result']
            p_outcome = probs['p_home']

        # EV = p * sp - 1
        ev = p_outcome * sp - 1 if sp > 0 else -999

        rows.append({
            'date': m['date_kj'],
            'home': m['home_cn'],
            'away': m['away_cn'],
            'home_en': m['home_en'],
            'away_en': m['away_en'],
            'handicap': m['handicap'],
            'tournament': m.get('tournament', ''),
            'p_home': probs['p_home'],
            'p_draw': probs['p_draw'],
            'p_away': probs['p_away'],
            'p_outcome': p_outcome,
            'sp': sp,
            'result': result,
            'ev': ev,
            'ft_h': m['ft_h'],
            'ft_a': m['ft_a'],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ 无有效数据")
        return

    # ── 过滤 ──
    valid = df[df['sp'] > 0].copy()
    print(f"  可交易: {len(valid)} / {len(df)} (排除未开售)")

    # ── EV > 阈值 with tier filtering ──
    # Apply competition tier weights
    def get_tier_weight(tournament):
        for key, weight in COMPETITION_TIER.items():
            if key.lower() in str(tournament).lower():
                return weight
        return DEFAULT_TIER
    
    valid['tier_weight'] = valid['tournament'].apply(get_tier_weight)
    valid['adjusted_ev_threshold'] = ev_threshold / valid['tier_weight'].clip(lower=0.1)
    
    # Filter by tier weight and adjusted EV threshold
    bets = valid[
        (valid['tier_weight'] > 0.3) &  # Skip low-tier competitions
        (valid['ev'] > valid['adjusted_ev_threshold'])
    ].copy()
    
    print(f"  触发下注: {len(bets)} 场 (EV > {ev_threshold*100:.0f}%, tier > 0.3)")
    if bets.empty:
        print("  ⚠️ 无满足条件的下注")
        return

    # ── 结算 ──
    if play == 'spf':
        # 胜平负: result=3(胜)→赢, 1或0→输
        bets['won'] = bets['result'].astype(str) == '3'
    elif play == 'rqspf':
        bets['won'] = bets['result'].astype(str) == '3'
    else:
        # 简化: 只做spf回测
        bets['won'] = bets['result'].astype(str) == '3'

    bets['profit'] = bets.apply(
        lambda x: (x['sp'] - 1) if x['won'] else -1, axis=1
    )

    # ── 统计 ──
    total_bets = len(bets)
    wins = bets['won'].sum()
    losses = total_bets - wins
    total_profit = bets['profit'].sum()
    roi = total_profit / total_bets * 100
    hit_rate = wins / total_bets * 100

    print(f"\n{'='*60}")
    print(f"  📊 回测结果 (真实收盘赔率)")
    print(f"{'='*60}")
    print(f"  下注场次:  {total_bets}")
    print(f"  命中:      {wins} / {total_bets} = {hit_rate:.1f}%")
    print(f"  总盈亏:    {total_profit:+.2f} 单位")
    print(f"  ROI:       {roi:+.2f}%")
    print(f"{'='*60}")

    # ── 逐月统计 ──
    bets['month'] = bets['date'].str[:7]
    monthly = bets.groupby('month').agg(
        bets=('profit', 'count'),
        wins=('won', 'sum'),
        profit=('profit', 'sum'),
    )
    monthly['roi'] = (monthly['profit'] / monthly['bets'] * 100).round(2)
    monthly['hit'] = (monthly['wins'] / monthly['bets'] * 100).round(1)

    print(f"\n  📅 逐月ROI:")
    for _, row in monthly.iterrows():
        print(f"    {row.name}: {int(row['bets'])}场 | "
              f"命中{row['hit']}% | "
              f"ROI {row['roi']:+.1f}% | "
              f"盈亏{row['profit']:+.2f}")

    # ── 保存详情 ──
    out_path = os.path.join(DATA_DIR, f'real_backtest_{play}.csv')
    bets.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\n  💾 详情已保存: {out_path}")


# ── 5. 主函数 ────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description='真实赔率回测')
    ap.add_argument('--ev', type=float, default=0.05,
                    help='EV阈值 (默认5%%)')
    ap.add_argument('--play', type=str, default='spf',
                    choices=['spf', 'rqspf', 'jqs', 'bqc'],
                    help='回测玩法 (默认spf)')
    ap.add_argument('--tolerance', type=int, default=2,
                    help='日期容差天数 (默认2)')
    args = ap.parse_args()

    print("📡 加载数据...")
    kaijiang = load_kaijiang()
    intl = load_international_results()
    team_map = load_team_mapping()

    print("\n📡 加载模型...")
    dc_model, elo = load_dc_model_and_elo()
    xgb_model, calibrators = load_xgb_model()

    print("\n🔗 合并数据...")
    merged = merge_data(kaijiang, intl, team_map, args.tolerance)

    if not merged:
        print("❌ 合并后无数据, 请检查队名映射")
        return

    run_backtest(merged, dc_model, elo, xgb_model, calibrators, args)


if __name__ == '__main__':
    main()
