#!/usr/bin/env python3
"""
train_club_dc.py — 训练俱乐部 Dixon-Coles 模型
=============================================
从 500_history_backfill.csv 训练独立的 DC 模型,
覆盖 500+ 俱乐部, 中文队名直接使用 (内部一致即可).

输出: /root/data/dc_club.pkl
"""
import csv, json, math, os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, '/root')
from wc_2026_phase1 import DixonColes

DATA_DIR = '/root/data'
OUTPUT_MODEL = os.path.join(DATA_DIR, 'dc_club.pkl')

# ── 筛选参数 ──
MIN_LEAGUE_MATCHES = 200   # 联赛最少场次
MIN_TEAM_MATCHES = 10      # 球队最少出场
TIME_DECAY_HL = 180        # 俱乐部快节奏, 半年衰减

def load_500_data():
    """加载 500_history_backfill.csv"""
    path = os.path.join(DATA_DIR, '500_history_backfill.csv')
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"📥 500 History: {len(rows)} 场")
    return rows

def parse_scores(rows):
    """解析比分, 筛选有效数据"""
    parsed = []
    for r in rows:
        sf = r.get('score_full', '').strip()
        if not sf or '-' not in sf:
            continue
        try:
            hg = int(sf.split('-')[0].strip())
            ag = int(sf.split('-')[1].strip())
        except (ValueError, IndexError):
            continue
        
        # 有效比分
        parsed.append({
            'date': r['date'],
            'home': r['home'].strip(),
            'away': r['away'].strip(),
            'h_score': hg,
            'a_score': ag,
            'league_id': r.get('league_id', ''),
        })
    return parsed

def filter_data(parsed):
    """按联赛和球队过滤"""
    # 联赛过滤
    from collections import Counter
    league_counts = Counter(p['league_id'] for p in parsed)
    valid_leagues = {lid for lid, c in league_counts.items() if c >= MIN_LEAGUE_MATCHES}
    print(f"  联赛 ≥{MIN_LEAGUE_MATCHES}场: {len(valid_leagues)}/{len(league_counts)}")
    
    filtered = [p for p in parsed if p['league_id'] in valid_leagues]
    print(f"  联赛过滤后: {len(filtered)} 场")
    
    # 球队过滤
    team_counts = Counter()
    for p in filtered:
        team_counts[p['home']] += 1
        team_counts[p['away']] += 1
    valid_teams = {t for t, c in team_counts.items() if c >= MIN_TEAM_MATCHES}
    print(f"  球队 ≥{MIN_TEAM_MATCHES}场: {len(valid_teams)}/{len(team_counts)}")
    
    filtered2 = [p for p in filtered if p['home'] in valid_teams and p['away'] in valid_teams]
    print(f"  球队过滤后: {len(filtered2)} 场")
    
    return filtered2, valid_teams

def build_df(filtered):
    """构建 DC 训练 DataFrame"""
    df = pd.DataFrame(filtered)
    df['neutral'] = False  # 500.com 比赛都有主客场
    return df

def main():
    print("=" * 60)
    print("  🏋️ 俱乐部 Dixon-Coles 训练")
    print("=" * 60)
    
    rows = load_500_data()
    parsed = parse_scores(rows)
    print(f"📊 有效比分: {len(parsed)}/{len(rows)}")
    
    # 日期范围
    dates = sorted(set(p['date'] for p in parsed))
    print(f"  日期范围: {dates[0]} → {dates[-1]}")
    
    # 过滤
    filtered, valid_teams = filter_data(parsed)
    
    if len(filtered) < 1000:
        print(f"❌ 数据不足 ({len(filtered)}), 放宽筛选条件")
        # 回退: 只过滤球队
        team_counts = Counter()
        for p in parsed:
            team_counts[p['home']] += 1
            team_counts[p['away']] += 1
        valid_teams = {t for t, c in team_counts.items() if c >= 5}
        filtered = [p for p in parsed if p['home'] in valid_teams and p['away'] in valid_teams]
        print(f"  回退过滤后: {len(filtered)} 场, {len(valid_teams)} 队")
    
    df = build_df(filtered)
    print(f"\n📊 训练数据:")
    print(f"  比赛: {len(df)}")
    print(f"  球队: {len(valid_teams)}")
    print(f"  场均进球: {df['h_score'].mean() + df['a_score'].mean():.3f}")
    
    # 训练 DC 模型
    print(f"\n🔄 训练 Dixon-Coles (衰减半衰期={TIME_DECAY_HL}天)...")
    model = DixonColes(time_decay_hl=TIME_DECAY_HL)
    model.fit(df)
    
    # 验证: 预测几场
    print(f"\n🔍 验证预测:")
    test_matches = filtered[-10:]
    for m in test_matches:
        try:
            lam_h, lam_a = model.predict_lambda(m['home'], m['away'], neutral=False)
            prob = model.predict_proba(m['home'], m['away'], neutral=False)
            actual = 'H' if m['h_score'] > m['a_score'] else ('D' if m['h_score'] == m['a_score'] else 'A')
            pred = ['A', 'D', 'H'][np.argmax(prob)]
            print(f"  {m['home'][:12]:>12} vs {m['away'][:12]:<12} "
                  f"λ=({lam_h:.2f},{lam_a:.2f}) "
                  f"P=({prob[0]:.2f},{prob[1]:.2f},{prob[2]:.2f}) "
                  f"实={actual} 预={pred} {'✓' if actual==pred else '✗'}")
        except Exception as e:
            print(f"  {m['home']} vs {m['away']}: ERROR {e}")
    
    # 保存模型
    import joblib
    joblib.dump(model, OUTPUT_MODEL)
    print(f"\n✅ 俱乐部 DC 模型保存到: {OUTPUT_MODEL}")
    print(f"  模型参数: {len(model.attack_)} 攻击 + {len(model.defense_)} 防守 + ρ={model.rho_:.4f} + γ={model.gamma_:.4f}")
    print(f"  球队数: {len(model.teams_)}")

if __name__ == '__main__':
    main()
