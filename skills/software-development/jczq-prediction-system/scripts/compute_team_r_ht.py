#!/usr/bin/env python3
"""从 historical_kaijiang.csv 计算 493 队半全场进球比 (r_ht)

用法:
    python3 scripts/compute_team_r_ht.py

输出: /root/data/team_r_ht.json
"""
import csv
import json
import os
from collections import defaultdict

def main():
    csv_path = '/root/data/historical_kaijiang.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f"总场次: {len(rows)}")

    team_goals = defaultdict(lambda: {'half': 0, 'full': 0, 'count': 0})

    for r in rows:
        try:
            home = r.get('home', '') or ''
            away = r.get('away', '') or ''
            hh = float(r.get('half_home', 0) or 0)
            ha = float(r.get('half_away', 0) or 0)
            fh = float(r.get('full_home', 0) or 0)
            fa = float(r.get('full_away', 0) or 0)
        except (ValueError, TypeError):
            continue

        if not home or not away:
            continue

        # 主队半全场进球
        if fh > 0:
            team_goals[home]['half'] += hh
            team_goals[home]['full'] += fh
            team_goals[home]['count'] += 1
        # 客队半全场进球
        if fa > 0:
            team_goals[away]['half'] += ha
            team_goals[away]['full'] += fa
            team_goals[away]['count'] += 1

    # 计算 r_ht = 半场进球 / 全场进球
    team_r = {}
    for team, g in sorted(team_goals.items()):
        if g['full'] > 0 and g['count'] >= 3:  # 至少3场
            team_r[team] = round(g['half'] / g['full'], 4)
        else:
            team_r[team] = None  # 数据不足, 用默认值

    # 全局默认值
    valid_r = [v for v in team_r.values() if v is not None]
    default_r = round(sum(valid_r) / len(valid_r), 4) if valid_r else 0.4423

    print(f"总球队: {len(team_r)}")
    print(f"有数据: {len(valid_r)} 队")
    print(f"全局默认 r_ht: {default_r}")
    print(f"分布范围: {min(valid_r):.4f} ~ {max(valid_r):.4f}")
    print(f"标准差: {__import__('statistics').stdev(valid_r):.4f}")

    # 输出示例
    top10 = sorted(team_r.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)[:10]
    print(f"\n最高 r_ht (半场进球倾向高):")
    for t, r in top10:
        print(f"  {t}: {r} ({team_goals[t]['count']}场)")

    bottom10 = sorted(team_r.items(), key=lambda x: x[1] if x[1] else 0)[:10]
    print(f"\n最低 r_ht (半场进球倾向低):")
    for t, r in bottom10:
        print(f"  {t}: {r} ({team_goals[t]['count']}场)")

    # 保存
    output = {
        '_default': default_r,
        'teams': {t: v if v is not None else default_r for t, v in team_r.items()}
    }
    os.makedirs('/root/data', exist_ok=True)
    with open('/root/data/team_r_ht.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n已保存到 /root/data/team_r_ht.json ({len(team_r)} 队)")

if __name__ == '__main__':
    main()
