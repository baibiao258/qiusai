#!/usr/bin/env python3
"""
club_data_pipeline.py — 俱乐部数据处理管线
==========================================
功能:
  1. 加载 club_matches.json
  2. 计算俱乐部 Elo 评分 (独立于国家队)
  3. 构建 form_state_club.json (俱乐部近期状态)
  4. 为 XGB 训练准备特征矩阵

关键设计:
  - 俱乐部 Elo 与国家队 Elo 完全隔离
  - 半衰期 150 天 (适应俱乐部高频赛程)
  - 升班马初始 Elo = 联赛降级区均分 (~1400)
"""
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

DATA_DIR = '/root/data'
CLUB_MATCHES_PATH = os.path.join(DATA_DIR, 'club_matches.json')
ELO_CLUB_PATH = os.path.join(DATA_DIR, 'elo_club.pkl')
FORM_CLUB_PATH = os.path.join(DATA_DIR, 'form_club.json')
FORM_12_CLUB_PATH = os.path.join(DATA_DIR, 'form_12_club.json')

# ── 参数 ──
ELO_K = 32
ELO_INIT = 1500
ELO_HALF_LIFE = 150  # 俱乐部: 150 天 (约半赛季)
PROMOTED_ELO = 1400   # 升班马初始 Elo


def load_club_matches():
    """加载俱乐部比赛数据."""
    if not os.path.exists(CLUB_MATCHES_PATH):
        print(f"❌ {CLUB_MATCHES_PATH} 不存在")
        print(f"  请先运行: python3 fetch_league_data.py")
        return []

    with open(CLUB_MATCHES_PATH) as f:
        matches = json.load(f)

    print(f"📊 加载 {len(matches)} 场俱乐部比赛")
    return matches


def compute_elo_ratings(matches, half_life=ELO_HALF_LIFE):
    """
    计算俱乐部 Elo 评分.
    
    使用时间衰减加权, 半衰期 150 天.
    """
    elo = defaultdict(lambda: ELO_INIT)
    
    # 按日期排序
    matches_sorted = sorted(matches, key=lambda m: m['date'])
    
    if not matches_sorted:
        return dict(elo)
    
    cutoff_date = matches_sorted[-1]['date']
    
    for m in matches_sorted:
        home = m['home']
        away = m['away']
        hg = m['h_score']
        ag = m['a_score']
        
        # 时间衰减权重
        try:
            days_diff = (datetime.strptime(cutoff_date, '%Y-%m-%d') - 
                        datetime.strptime(m['date'], '%Y-%m-%d')).days
            weight = 0.5 ** (max(days_diff, 0) / half_life)
        except:
            weight = 1.0
        
        # 实际结果
        if hg > ag:
            sh, sa = 1.0, 0.0
        elif hg == ag:
            sh, sa = 0.5, 0.5
        else:
            sh, sa = 0.0, 1.0
        
        # 期望得分
        e_h = 1.0 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        e_a = 1.0 - e_h
        
        # 更新 (加权 K)
        k_weighted = ELO_K * weight
        elo[home] += k_weighted * (sh - e_h)
        elo[away] += k_weighted * (sa - e_a)
    
    return dict(elo)


def build_form_state(matches, max_games=25):
    """
    构建俱乐部 form_state (最近 N 场比分记录).
    
    格式: {team: [[hg, ag], [hg, ag], ...]}
    """
    form = defaultdict(list)
    
    matches_sorted = sorted(matches, key=lambda m: m['date'])
    
    for m in matches_sorted:
        home = m['home']
        away = m['away']
        hg = m['h_score']
        ag = m['a_score']
        
        # 主队视角
        form[home].append([hg, ag])
        if len(form[home]) > max_games:
            form[home] = form[home][-max_games:]
        
        # 客队视角
        form[away].append([ag, hg])
        if len(form[away]) > max_games:
            form[away] = form[away][-max_games:]
    
    return dict(form)


def recent_form_club(team, form_state, n=5):
    """
    获取俱乐部近 N 场 form.
    
    返回: [win_rate, avg_gf, avg_ga, goal_diff]
    """
    games = form_state.get(team, [])
    if not games:
        return [0.5, 0.0, 0.0, 0.0]
    
    recent = games[-n:] if len(games) >= n else games
    n_games = len(recent)
    
    wins = 0
    total_gf = 0
    total_ga = 0
    for g in recent:
        hg, ag = g[0], g[1]
        total_gf += hg
        total_ga += ag
        if hg > ag:
            wins += 1
        elif hg == ag:
            wins += 0.5
    
    return [
        wins / n_games,           # win_rate
        total_gf / n_games,       # avg_gf
        total_ga / n_games,       # avg_ga
        (total_gf - total_ga) / n_games,  # goal_diff
    ]


def build_gold_features_club(home, away, form_state, h2h_cache=None):
    """
    构建俱乐部 gold 特征 (5 维).
    
    与国际赛 gold 特征格式一致, 但使用俱乐部数据.
    """
    fh12 = recent_form_club(home, form_state, n=12)
    fa12 = recent_form_club(away, form_state, n=12)
    
    # H2H (如果有的话)
    h2h_gd = 0.0
    if h2h_cache:
        key = tuple(sorted([home, away]))
        cache_key = f"{key[0]}||{key[1]}"
        entry = h2h_cache.get(cache_key)
        if entry:
            if home == key[0]:
                h2h_gd = entry[1] - entry[2]  # avg_gf - avg_ga
            else:
                h2h_gd = entry[2] - entry[1]  # flip
    
    # 联赛比赛 tier: [is_friendly, is_major, is_knockout, is_qualifier]
    # 俱乐部比赛: 都不是 friendly, 也不是 national cup (简化处理)
    gold = [
        h2h_gd,                    # H2H goal difference
        0,                         # is_major_cup (俱乐部常规赛=0)
        0,                         # is_friendly
        fh12[1] - fa12[2],         # home 12-game avg_gf - away 12-game avg_ga
        fa12[1] - fh12[0],         # away 12-game avg_gf - home 12-game win_rate
    ]
    
    return gold


def build_features_club(home, away, elo, form_state, dc_model=None,
                        match_type='league', xg_state=None):
    """
    为俱乐部比赛构建完整 37 维特征 (29 基线 + 8 xG-proxy).
    """
    eh = elo.get(home, PROMOTED_ELO)
    ea = elo.get(away, PROMOTED_ELO)
    
    # DC 预测 (如果有)
    lam_h, lam_a = 1.0, 1.0
    dc_p = [1/3, 1/3, 1/3]
    
    if dc_model:
        try:
            dc_p = dc_model.predict_proba(home, away, neutral=True)
            lam_h, lam_a = dc_model.predict_lambda(home, away, neutral=True)
            if lam_h is None:
                lam_h, lam_a = 1.0, 1.0
                dc_p = [1/3, 1/3, 1/3]
        except:
            pass
    
    # 5 场 form
    fh5 = recent_form_club(home, form_state, 5)
    fa5 = recent_form_club(away, form_state, 5)
    
    # b15 (15 维基线特征)
    b15 = [
        (eh - ea) / 400,
        lam_h, lam_a, lam_h - lam_a,
        math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
        dc_p[0], dc_p[1], dc_p[2],
        fh5[0], fa5[0],
        fh5[1] - fa5[2], fa5[1] - fh5[2],
        fh5[1] - fa5[1], fh5[0] - fa5[0],
        1,  # neutral (俱乐部比赛通常非中立)
    ]
    
    # gold (5 维)
    gold = build_gold_features_club(home, away, form_state)
    
    # odds (3 维)
    op_h = 1 / (1 + 10 ** ((ea - eh) / 400))
    op_a = 1 / (1 + 10 ** ((eh - ea) / 400))
    odds_feat = [op_h, op_a, 0.0]
    
    # form (6 维)
    form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]
    
    # xG-proxy (8维: 主客各4)
    xg_feat = []
    for team in [home, away]:
        s = (xg_state or {}).get(team, {})
        xg_feat.extend([
            s.get('xg_proxy_5', 0.0),
            s.get('xg_proxy_12', 0.0),
            s.get('xg_streak', 0) / 10.0,
            s.get('xg_volatility', 0.0),
        ])
    return b15 + gold + odds_feat + form_feat + xg_feat  # 37 维


def train_dc_model(matches):
    """
    训练俱乐部 Dixon-Coles 模型.
    
    使用独立于国家队的参数.
    """
    import sys
    sys.path.insert(0, '/root')
    sys.path.insert(0, '/root/wc_2026_upgrade')
    
    from wc_2026_phase1 import DixonColes
    import pandas as pd
    
    # 转换为 DataFrame
    records = []
    for m in matches:
        records.append({
            'date': m['date'],
            'home': m['home'],
            'away': m['away'],
            'h_score': m['h_score'],
            'a_score': m['a_score'],
            'neutral': m.get('neutral', False),
        })
    
    df = pd.DataFrame(records)
    
    # 训练 DC (俱乐部半衰期更短)
    dc = DixonColes(time_decay_hl=ELO_HALF_LIFE)
    dc.fit(df)
    
    print(f"  DC: ρ={dc.rho_:.4f} γ={dc.gamma_:.4f}")
    return dc


def save_all(elo, form_state, dc_model=None):
    """保存所有缓存."""
    import joblib
    
    # Elo
    joblib.dump(elo, ELO_CLUB_PATH)
    print(f"  ✅ Elo: {len(elo)} 队 → {ELO_CLUB_PATH}")
    
    # Form state
    with open(FORM_CLUB_PATH, 'w') as f:
        json.dump(form_state, f, ensure_ascii=False)
    print(f"  ✅ Form: {len(form_state)} 队 → {FORM_CLUB_PATH}")
    
    # Form 12 (预计算)
    form_12 = {}
    for team, games in form_state.items():
        form_12[team] = recent_form_club(team, form_state, n=12)
    with open(FORM_12_CLUB_PATH, 'w') as f:
        json.dump(form_12, f)
    print(f"  ✅ Form 12: {len(form_12)} 队 → {FORM_12_CLUB_PATH}")
    
    # DC 模型
    if dc_model:
        dc_path = os.path.join(DATA_DIR, 'dc_model_club.pkl')
        joblib.dump(dc_model, dc_path)
        print(f"  ✅ DC 模型 → {dc_path}")
    
    # xG-proxy (依赖 DC 模型)
    if dc_model:
        try:
            sys.path.insert(0, '/root')
            from xg_proxy import compute_luck_factors, build_xg_proxy_state
            luck_data = compute_luck_factors(
                json.load(open(CLUB_MATCHES_PATH)), dc_model
            )
            xg_state = build_xg_proxy_state(luck_data)
            xg_path = os.path.join(DATA_DIR, 'xg_proxy_club.json')
            with open(xg_path, 'w') as f:
                json.dump(xg_state, f, ensure_ascii=False, indent=2)
            print(f"  ✅ xG-proxy: {len(xg_state)} 队 → {xg_path}")
        except Exception as e:
            print(f"  ⚠️ xG-proxy 生成失败: {e}")


def main():
    """主流程: 加载数据 → 计算 Elo → 构建 Form → 训练 DC → 保存."""
    print("="*50)
    print("⚽ 俱乐部数据管线")
    print("="*50)
    
    # 1. 加载数据
    matches = load_club_matches()
    if not matches:
        return
    
    # 2. 计算 Elo
    print("\n📊 计算俱乐部 Elo...")
    elo = compute_elo_ratings(matches)
    
    # 显示 Top 10
    top10 = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10:")
    for team, rating in top10:
        print(f"    {team:30s} {rating:.1f}")
    
    # 3. 构建 Form State
    print("\n📊 构建 form_state...")
    form_state = build_form_state(matches)
    print(f"  {len(form_state)} 队")
    
    # 4. 训练 DC
    print("\n📊 训练 Dixon-Coles...")
    dc_model = train_dc_model(matches)
    
    # 5. 保存
    print("\n💾 保存...")
    save_all(elo, form_state, dc_model)
    
    print("\n✅ 完成!")


if __name__ == '__main__':
    main()
