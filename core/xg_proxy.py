#!/usr/bin/env python3
"""
xg_proxy.py — 预期进球代理特征 (Performance Residuals)
======================================================
核心逻辑:
  luck_factor = actual_goals - lambda (由 DC 模型输出)
  
  如果一支球队近期实际进球远超 λ → 超常发挥(不可持续)
  如果一支球队近期实际进球远低于 λ → 运气差(可能回归)

输出特征 (每队 4 维):
  - xg_proxy_5:   近 5 场运气因子均值
  - xg_proxy_12:  近 12 场运气因子均值
  - xg_streak:    连续超常/低于预期的场次 (正=超常, 负=低于)
  - xg_volatility:运气因子标准差 (波动越大越不稳定)

集成路径:
  club_data_pipeline.py → 计算 xg_proxy → 保存到 xg_proxy_club.json
  train_xgb_club.py     → 加载 xg_proxy 作为额外特征
  daily_jczq.py         → 实时计算 xg_proxy 特征
"""
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import joblib

DATA_DIR = '/root/data'


def compute_luck_factors(club_matches, dc_model):
    """
    计算每场比赛的运气因子 (actual - lambda).
    
    Returns: list of dict, 每场包含 home_luck, away_luck
    """
    luck_data = []
    
    for m in club_matches:
        home = m['home']
        away = m['away']
        actual_h = m['h_score']
        actual_a = m['a_score']
        
        try:
            lam_h, lam_a = dc_model.predict_lambda(home, away, neutral=True)
            if lam_h is None or lam_a is None:
                continue
        except:
            continue
        
        luck_h = actual_h - lam_h  # 主队运气因子
        luck_a = actual_a - lam_a  # 客队运气因子
        
        luck_data.append({
            'date': m['date'],
            'home': home,
            'away': away,
            'lam_h': round(lam_h, 4),
            'lam_a': round(lam_a, 4),
            'actual_h': actual_h,
            'actual_a': actual_a,
            'luck_h': round(luck_h, 4),
            'luck_a': round(luck_a, 4),
        })
    
    return luck_data


def build_xg_proxy_state(luck_data):
    """
    从运气因子历史数据构建 xG-proxy 状态.
    
    对每支球队追踪:
      - luck_history: [luck_1, luck_2, ...] (按时间排序)
      - recent_5: 近 5 场运气因子
      - recent_12: 近 12 场运气因子
    
    Returns: {team: {recent_5: [], recent_12: [], streak: int, volatility: float}}
    """
    team_luck = defaultdict(list)
    
    # 按日期排序
    luck_data.sort(key=lambda x: x['date'])
    
    for entry in luck_data:
        team_luck[entry['home']].append(entry['luck_h'])
        team_luck[entry['away']].append(entry['luck_a'])
    
    state = {}
    for team, lucks in team_luck.items():
        if len(lucks) < 3:
            continue
        
        recent_5 = lucks[-5:]
        recent_12 = lucks[-12:] if len(lucks) >= 12 else lucks
        
        # 均值
        mean_5 = np.mean(recent_5)
        mean_12 = np.mean(recent_12)
        
        # 连续趋势 (streak)
        streak = 0
        for l in reversed(lucks):
            if l > 0:
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif l < 0:
                if streak <= 0:
                    streak -= 1
                else:
                    break
            else:
                break
        
        # 波动率
        volatility = float(np.std(recent_12))
        
        state[team] = {
            'xg_proxy_5': round(float(mean_5), 4),
            'xg_proxy_12': round(float(mean_12), 4),
            'xg_streak': streak,
            'xg_volatility': round(volatility, 4),
            'n_matches': len(lucks),
        }
    
    return state


def get_xg_proxy_features(team, xg_state, n_5=5, n_12=12):
    """
    获取某球队的 xG-proxy 特征 (4 维).
    
    Returns: [xg_proxy_5, xg_proxy_12, xg_streak, xg_volatility]
    """
    if team not in xg_state:
        return [0.0, 0.0, 0.0, 0.0]
    
    s = xg_state[team]
    return [
        s.get('xg_proxy_5', 0.0),
        s.get('xg_proxy_12', 0.0),
        s.get('xg_streak', 0) / 10.0,  # 归一化到 [-1, 1]
        s.get('xg_volatility', 0.0),
    ]


def build_feat_with_xg_proxy(base_feat, home, away, xg_state):
    """
    在基础 29 维特征上追加 4 维 xG-proxy 特征 → 33 维.
    """
    home_xg = get_xg_proxy_features(home, xg_state)
    away_xg = get_xg_proxy_features(away, xg_state)
    
    # 追加特征 (8 维):
    # - home: xg_proxy_5, xg_proxy_12, xg_streak, xg_volatility
    # - away: xg_proxy_5, xg_proxy_12, xg_streak, xg_volatility
    return base_feat + home_xg + away_xg


def compute_xg_proxy_for_match(home, away, xg_state):
    """为单场比赛计算 xG-proxy 相关信息."""
    home_xg = get_xg_proxy_features(home, xg_state)
    away_xg = get_xg_proxy_features(away, xg_state)
    
    return {
        'home_xg_proxy_5': home_xg[0],
        'home_xg_proxy_12': home_xg[1],
        'home_xg_streak': home_xg[2],
        'away_xg_proxy_5': away_xg[0],
        'away_xg_proxy_12': away_xg[1],
        'away_xg_streak': away_xg[2],
        'xg_advantage': home_xg[0] - away_xg[0],  # 主队运气优势
    }


def main():
    """构建 xG-proxy 状态并保存."""
    print("=" * 50)
    print("🎲 构建 xG-proxy 特征")
    print("=" * 50)
    
    # 加载数据
    with open(os.path.join(DATA_DIR, 'club_matches.json')) as f:
        matches = json.load(f)
    dc = joblib.load(os.path.join(DATA_DIR, 'dc_model_club.pkl'))
    
    print(f"  比赛: {len(matches)} 场")
    print(f"  DC: ρ={dc.rho_:.4f} γ={dc.gamma_:.4f}")
    
    # 计算运气因子
    print("\n📊 计算运气因子...")
    luck_data = compute_luck_factors(matches, dc)
    print(f"  有效场次: {len(luck_data)} 场")
    
    # 显示运气因子分布
    lucks_h = [e['luck_h'] for e in luck_data]
    lucks_a = [e['luck_a'] for e in luck_data]
    all_lucks = lucks_h + lucks_a
    print(f"  运气因子均值: {np.mean(all_lucks):.4f} (理论值=0)")
    print(f"  运气因子标准差: {np.std(all_lucks):.4f}")
    print(f"  运气因子范围: [{np.min(all_lucks):.2f}, {np.max(all_lucks):.2f}]")
    
    # 构建 xg-proxy 状态
    print("\n📊 构建 xG-proxy 状态...")
    xg_state = build_xg_proxy_state(luck_data)
    print(f"  有 xG-proxy 数据的球队: {len(xg_state)} 队")
    
    # 显示示例
    top_teams = sorted(xg_state.items(), key=lambda x: -abs(x[1]['xg_proxy_5']))[:5]
    print("\n  Top 5 xG-proxy (运气因子):")
    for team, s in top_teams:
        direction = "超常" if s['xg_proxy_5'] > 0 else "低于预期"
        print(f"    {team:30s} 5场={s['xg_proxy_5']:+.3f} 12场={s['xg_proxy_12']:+.3f} "
              f"streak={s['xg_streak']:+d} vol={s['xg_volatility']:.3f} ({direction})")
    
    # 保存
    out_path = os.path.join(DATA_DIR, 'xg_proxy_club.json')
    with open(out_path, 'w') as f:
        json.dump(xg_state, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存: {out_path}")
    
    print("\n✅ 完成!")


if __name__ == '__main__':
    main()
