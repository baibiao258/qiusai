#!/usr/bin/env python3
"""
compute_team_r_ht.py — 计算每支球队的半全场进球比 (r_ht)
==========================================================
从 historical_kaijiang.csv 统计每支球队的历史半场进球/全场进球比。
r_ht = 半场进球总和 / 全场进球总和

输出: /root/data/team_r_ht.json 
  {team_name: r_ht_value, ...}

供 half_full_model.py / daily_jczq.py 使用:
  from half_full_model import predict_half_full_probs
  # 传入 team_r_ht={team: r_ht, ...}
"""
import csv
import json
import os
from collections import defaultdict

KAJ_PATH = '/root/data/historical_kaijiang.csv'
OUTPUT = '/root/data/team_r_ht.json'


def compute():
    teams = defaultdict(lambda: {'ht_gf': 0, 'ft_gf': 0, 'ht_ga': 0, 'ft_ga': 0, 'n': 0})

    with open(KAJ_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ht_h = int(row['ht_h'])
                ht_a = int(row['ht_a'])
                ft_h = int(row['ft_h'])
                ft_a = int(row['ft_a'])
            except (ValueError, TypeError, KeyError):
                continue

            home = row.get('home', '').strip()
            away = row.get('away', '').strip()

            for team, ht_g, ft_g in [(home, ht_h, ft_h), (away, ht_a, ft_a)]:
                if not team:
                    continue
                teams[team]['ht_gf'] += ht_g
                teams[team]['ft_gf'] += ft_g
                teams[team]['ht_ga'] += ht_a if team == home else ht_h
                teams[team]['ft_ga'] += ft_a if team == home else ft_h
                teams[team]['n'] += 1

    # 计算 r_ht (半场进球/全场进球)
    r_ht_map = {}
    global_summary = {'ht_gf': 0, 'ft_gf': 0, 'teams': 0, 'valid_teams': 0}

    for team, s in sorted(teams.items()):
        global_summary['ht_gf'] += s['ht_gf']
        global_summary['ft_gf'] += s['ft_gf']
        global_summary['teams'] += 1

        if s['ft_gf'] > 0 and s['n'] >= 3:  # 至少3场比赛
            r_ht_val = s['ht_gf'] / s['ft_gf']
            r_ht_val = max(0.1, min(0.9, r_ht_val))  # 钳位
            r_ht_map[team] = round(r_ht_val, 4)
            global_summary['valid_teams'] += 1

    # 全局平均 r_ht (用于无数据球队的默认值)
    if global_summary['ft_gf'] > 0:
        global_avg = global_summary['ht_gf'] / global_summary['ft_gf']
    else:
        global_avg = 0.45

    print(f"📊 球队 r_ht 计算完成")
    print(f"   历史比赛: {sum(s['n'] for s in teams.values())} 场")
    print(f"   球队总数: {global_summary['teams']}")
    print(f"   有效球队: {global_summary['valid_teams']} (n≥3)")
    print(f"   全局平均 r_ht: {global_avg:.4f}")
    print(f"   r_ht 范围: {min(r_ht_map.values()):.4f} ~ {max(r_ht_map.values()):.4f}")

    # 保存时附带全局平均作为默认值
    output_data = {
        '_default': round(global_avg, 4),
        '_updated': __import__('datetime').date.today().isoformat(),
        '_n_matches': sum(s['n'] for s in teams.values()),
        '_teams': r_ht_map,  # {team: r_ht}
    }

    # 也用平铺格式方便直接读取
    flat = r_ht_map.copy()
    flat['_default'] = round(global_avg, 4)

    os.makedirs(os.path.dirname(OUTPUT) or '.', exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    print(f"\n💾 已保存: {OUTPUT}")
    print(f"   ({len(r_ht_map)} 队各有独立 r_ht, 默认 {global_avg:.4f})")

    # 打印分布
    buckets = {'<0.35': 0, '0.35-0.40': 0, '0.40-0.45': 0, '0.45-0.50': 0, '0.50-0.55': 0, '>0.55': 0}
    for v in r_ht_map.values():
        if v < 0.35: buckets['<0.35'] += 1
        elif v < 0.40: buckets['0.35-0.40'] += 1
        elif v < 0.45: buckets['0.40-0.45'] += 1
        elif v < 0.50: buckets['0.45-0.50'] += 1
        elif v < 0.55: buckets['0.50-0.55'] += 1
        else: buckets['>0.55'] += 1
    print(f"\n📈 r_ht 分布:")
    for k, v in buckets.items():
        bar = '█' * v
        print(f"   {k}: {v:>4d} {bar}")


if __name__ == '__main__':
    compute()
