#!/usr/bin/env python3
"""
feature_helper.py — Gold 特征补全工具
=====================================
修复 predict_match.py 中 gold 特征的 train-serve skew:
  - h2h_gd: 从 international_results.json 预计算 H2H 目标差
  - fh12/fh12: 从 form_state.json 计算 12 场 form
  - tournament_tier: 从赛事名称推断

预计算文件:
  - /root/data/h2h_cache.json: {(team_a, team_b): [win_rate, avg_gf, avg_ga, n]}
  - /root/data/form_12.json: {team: [win_rate, avg_gf, avg_ga, goal_diff]}
"""
import json
import math
import os
from collections import defaultdict
from datetime import datetime

DATA_DIR = '/root/data'
H2H_CACHE = os.path.join(DATA_DIR, 'h2h_cache.json')
FORM_12_CACHE = os.path.join(DATA_DIR, 'form_12.json')


def tournament_tier(tournament):
    """赛事等级: [is_friendly, is_major_cup, is_knockout_capable, is_qualifier]
    
    与 wc_2026_phase1.py 保持一致.
    """
    t = tournament or ''
    friendly = int(t in ('Friendly', 'Friendlies'))
    major = int(any(kw in t for kw in (
        'FIFA World Cup', 'UEFA Euro', 'Copa América',
        'African Cup of Nations', 'AFC Asian Cup',
        'Gold Cup', 'Oceania Nations Cup',
    )))
    final_round = int(any(kw in t for kw in (
        'Final', 'Semi', 'Quarter', 'Round',
        'play-off', 'Play-off', 'knockout',
    )))
    qualifier = int(any(kw in t for kw in ('qualification', 'Qualification')))
    return [friendly, major, final_round, qualifier]


def build_h2h_cache():
    """从 international_results.json 预计算所有 H2H 对."""
    cache_path = os.path.join(DATA_DIR, 'international_results.json')
    if not os.path.exists(cache_path):
        print("  ⚠️ international_results.json 不存在, 跳过 H2H 构建")
        return {}

    with open(cache_path) as f:
        matches = json.load(f)

    # 按 (min(team_a, team_b), max(team_a, team_b)) 分组
    h2h_raw = defaultdict(list)
    for m in matches:
        home = m.get('home', m.get('home_team', ''))
        away = m.get('away', m.get('away_team', ''))
        try:
            hg = int(m.get('h_score', m.get('home_score', -1)))
            ag = int(m.get('a_score', m.get('away_score', -1)))
        except:
            continue
        if hg < 0 or ag < 0 or not home or not away:
            continue

        key = tuple(sorted([home, away]))
        # 记录谁是主队
        h2h_raw[key].append({
            'home': home, 'away': away,
            'hg': hg, 'ag': ag,
            'date': m.get('date', ''),
        })

    # 对每对计算统计 (最近 N 场)
    h2h_cache = {}
    for key, games in h2h_raw.items():
        # 按日期排序
        games.sort(key=lambda x: x['date'], reverse=True)
        # 取最近 3 场
        recent = games[:3]
        if not recent:
            continue

        t1, t2 = key  # sorted
        total_gf = 0  # t1 进球
        total_ga = 0  # t1 失球
        wins = 0
        for g in recent:
            if g['home'] == t1:
                total_gf += g['hg']
                total_ga += g['ag']
                if g['hg'] > g['ag']:
                    wins += 1
                elif g['hg'] == g['ag']:
                    wins += 0.5
            else:
                total_gf += g['ag']
                total_ga += g['hg']
                if g['ag'] > g['hg']:
                    wins += 1
                elif g['ag'] == g['hg']:
                    wins += 0.5

        n = len(recent)
        h2h_cache[f"{t1}||{t2}"] = [
            wins / n,           # win_rate
            total_gf / n,       # avg_gf
            total_ga / n,       # avg_ga
            n,
        ]

    # 保存
    with open(H2H_CACHE, 'w') as f:
        json.dump(h2h_cache, f)

    print(f"  ✅ H2H 缓存: {len(h2h_cache)} 对")
    return h2h_cache


def build_form_12_cache():
    """从 form_state.json 计算 12 场 form (复用已有的 25 场数据)."""
    form_path = os.path.join(DATA_DIR, 'form_state.json')
    if not os.path.exists(form_path):
        print("  ⚠️ form_state.json 不存在, 跳过 form_12 构建")
        return {}

    with open(form_path) as f:
        form_state = json.load(f)

    form_12 = {}
    for team, games in form_state.items():
        if not games or len(games) < 1:
            form_12[team] = [0.5, 0.0, 0.0, 0.0]
            continue

        # 取最近 12 场 (或全部)
        recent = games[-12:] if len(games) >= 12 else games
        n = len(recent)

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

        form_12[team] = [
            wins / n,             # win_rate
            total_gf / n,         # avg_gf
            total_ga / n,         # avg_ga
            (total_gf - total_ga) / n,  # goal_diff
        ]

    with open(FORM_12_CACHE, 'w') as f:
        json.dump(form_12, f)

    print(f"  ✅ Form 12 缓存: {len(form_12)} 队")
    return form_12


def load_h2h_cache():
    """加载 H2H 缓存."""
    if os.path.exists(H2H_CACHE):
        with open(H2H_CACHE) as f:
            return json.load(f)
    return {}


def load_form_12_cache():
    """加载 12 场 form 缓存."""
    if os.path.exists(FORM_12_CACHE):
        with open(FORM_12_CACHE) as f:
            return json.load(f)
    return {}


def get_h2h(home, away, h2h_cache=None):
    """获取 H2H 特征: [win_rate, avg_gf, avg_ga, n] (从 home 视角).
    
    返回格式与 FeatureBuffer.h2h 一致.
    """
    if h2h_cache is None:
        h2h_cache = load_h2h_cache()

    key = tuple(sorted([home, away]))
    cache_key = f"{key[0]}||{key[1]}"
    entry = h2h_cache.get(cache_key)

    if entry is None:
        return [0.5, 0.0, 0.0, 0]

    # entry 是从 sorted 视角 (t1 < t2)
    # 需要调整到 home 视角
    t1, t2 = key
    if home == t1:
        return entry  # home 是 t1, 直接返回
    else:
        # home 是 t2, 需要翻转
        return [entry[0], entry[2], entry[1], entry[3]]  # flip gf/ga


def get_12game_form(team, form_12_cache=None):
    """获取 12 场 form: [win_rate, avg_gf, avg_ga, goal_diff]."""
    if form_12_cache is None:
        form_12_cache = load_form_12_cache()

    return form_12_cache.get(team, [0.5, 0.0, 0.0, 0.0])


def build_gold_features(home, away, match_type='competitive',
                        h2h_cache=None, form_12_cache=None):
    """构建 gold 特征向量 (5 维), 与 wc_2026_final.py 训练时一致.
    
    Returns: [h2h_gd, tier_major, tier_friendly, fh12_gf_minus_fa12_ga, fa12_gf_minus_fh12_wr]
    """
    if h2h_cache is None:
        h2h_cache = load_h2h_cache()
    if form_12_cache is None:
        form_12_cache = load_form_12_cache()

    h2h = get_h2h(home, away, h2h_cache)
    fh12 = get_12game_form(home, form_12_cache)
    fa12 = get_12game_form(away, form_12_cache)

    # tournament_tier flags
    if match_type == 'friendly':
        tier = [1, 0, 0, 0]
    elif match_type == 'competitive':
        tier = [0, 1, 0, 0]
    elif match_type == 'qualifier':
        tier = [0, 0, 0, 1]
    else:
        tier = [0, 0, 0, 0]

    gold = [
        h2h[1] - h2h[2],           # h2h goal difference
        tier[1],                     # is_major_cup
        tier[0],                     # is_friendly
        fh12[1] - fa12[2],          # home 12-game avg_gf - away 12-game avg_ga
        fa12[1] - fh12[0],          # away 12-game avg_gf - home 12-game win_rate
    ]

    return gold


def rebuild_all_caches():
    """重建所有特征缓存."""
    print("🔨 重建特征缓存...")
    build_h2h_cache()
    build_form_12_cache()
    print("✅ 完成")


if __name__ == '__main__':
    rebuild_all_caches()
