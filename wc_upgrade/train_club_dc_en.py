#!/usr/bin/env python3
"""
train_club_dc_en.py — 训练英文名俱乐部 DC 模型
=============================================
从 football-data.org CSV 训练, 覆盖英超/西甲/意甲/德甲/法甲等,
队名全是英文, 补充 dc_club 缺的豪门覆盖.
"""
import csv, json, math, os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, '/root')
from wc_2026_phase1 import DixonColes

DATA_DIR = '/root/data'
INPUT_CSV = os.path.join(DATA_DIR, 'football_data_org_clubs.csv')
OUTPUT_MODEL = os.path.join(DATA_DIR, 'dc_club_en.pkl')

def load_data():
    rows = []
    with open(INPUT_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            h = int(r['h_score'])
            a = int(r['a_score'])
            rows.append({
                'date': r['date'],
                'home': r['home'].strip(),
                'away': r['away'].strip(),
                'h_score': h,
                'a_score': a,
                'league': r.get('league', ''),
            })
    return rows

def filter_data(rows):
    # 过滤无效比分
    filtered = [r for r in rows if r['h_score'] >= 0 and r['a_score'] >= 0]
    print(f'  有效比分: {len(filtered)}/{len(rows)}')
    
    # 球队过滤
    from collections import Counter
    team_counts = Counter()
    for r in filtered:
        team_counts[r['home']] += 1
        team_counts[r['away']] += 1
    
    # 只保留 ≥5 场的球队 (小联赛数据有限, 降低阈值)
    valid_teams = {t for t, c in team_counts.items() if c >= 5}
    filtered2 = [r for r in filtered if r['home'] in valid_teams and r['away'] in valid_teams]
    print(f'  球队 ≥5场: {len(valid_teams)}/{len(team_counts)} → {len(filtered2)} 场')
    
    return filtered2

def main():
    print("=" * 60)
    print("  🏋️ 英文俱乐部 DC 训练 (football-data.org)")
    print("=" * 60)
    
    rows = load_data()
    print(f'📥 加载 {len(rows)} 场比赛')
    
    filtered = filter_data(rows)
    
    df = pd.DataFrame(filtered)
    df['neutral'] = False
    
    print(f'\n📊 训练数据:')
    print(f'  比赛: {len(df)}')
    print(f'  联赛: {df["league"].nunique()}')
    print(f'  场均进球: {df["h_score"].mean() + df["a_score"].mean():.3f}')
    
    from collections import Counter
    leagues = Counter(r['league'] for r in filtered)
    print(f'  联赛分布:')
    for l, c in leagues.most_common():
        print(f'    {l}: {c}')
    
    # 训练 DC
    model = DixonColes(time_decay_hl=180)
    model.fit(df)
    
    # 验证
    print(f'\n🔍 验证预测 (最近10场):')
    test = filtered[-10:]
    correct = 0
    for m in test:
        try:
            prob = model.predict_proba(m['home'], m['away'], neutral=False)
            actual = 'H' if m['h_score'] > m['a_score'] else ('D' if m['h_score'] == m['a_score'] else 'A')
            pred = ['A', 'D', 'H'][np.argmax(prob)]
            if actual == pred: correct += 1
            lam_h, lam_a = model.predict_lambda(m['home'], m['away'], neutral=False)
            print(f"  {m['home'][:20]:>20} vs {m['away'][:20]:<20} "
                  f"λ=({lam_h:.2f},{lam_a:.2f}) "
                  f"实={actual} 预={pred} {'✓' if actual==pred else '✗'}")
        except Exception as e:
            print(f"  预测失败: {e}")
    print(f'\n  最近10场准确率: {correct}/10')
    
    # 覆盖检查
    test_teams = ['Arsenal', 'Chelsea', 'Barcelona', 'Real Madrid', 'Bayern Munich', 
                   'AC Milan', 'Juventus', 'Paris Saint-Germain', 'Benfica', 'Ajax']
    print(f'\n📋 豪门覆盖:')
    for t in test_teams:
        if t in model.team_idx_:
            idx = model.team_idx_[t]
            print(f'  ✅ {t:25s} att={model.attack_[idx]:+.3f} def={model.defense_[idx]:+.3f}')
        else:
            print(f'  ❌ {t:25s} 不在模型中')
    
    import joblib
    joblib.dump(model, OUTPUT_MODEL)
    print(f'\n✅ 英文俱乐部 DC 保存到: {OUTPUT_MODEL}')
    print(f'  球队数: {len(model.teams_)}')
    print(f'  ρ={model.rho_:.4f} γ={model.gamma_:.4f}')

if __name__ == '__main__':
    main()
