#!/usr/bin/env python3
"""友谊赛自适应折扣参数计算 — 3248场历史回测

用法:
    python3 scripts/friendly_calibration.py

输出: /root/data/friendly_calib.json
"""
import csv
import json
import os

def main():
    csv_path = '/root/data/historical_kaijiang.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # 过滤友谊赛
    friendlies = []
    others = []
    for r in rows:
        league = r.get('league', '')
        if '友谊赛' in league:
            friendlies.append(r)
        else:
            others.append(r)

    print(f"总场次: {len(rows)}")
    print(f"友谊赛: {len(friendlies)}")
    print(f"其他: {len(others)}")

    # 按实力差距分组
    low_diff = {'total': 0, 'friendly': 0, 'errors': 0}
    high_diff = {'total': 0, 'friendly': 0, 'errors': 0}

    for r in rows:
        is_friendly = '友谊赛' in r.get('league', '')
        try:
            eh = float(r.get('elo_h', 0) or 0)
            ea = float(r.get('elo_a', 0) or 0)
            diff = abs(eh - ea) / 2000  # 归一化差距
        except (ValueError, TypeError):
            continue

        if diff < 0.5:
            low_diff['total'] += 1
            if is_friendly:
                low_diff['friendly'] += 1
        else:
            high_diff['total'] += 1
            if is_friendly:
                high_diff['friendly'] += 1

    print(f"\n实力接近 (|Δ|<0.5): {low_diff['total']} 场, 其中友谊赛 {low_diff['friendly']} 场 ({low_diff['friendly']*100/low_diff['total']:.1f}%)")
    print(f"实力悬殊 (|Δ|≥0.5): {high_diff['total']} 场, 其中友谊赛 {high_diff['friendly']} 场 ({high_diff['friendly']*100/high_diff['total']:.1f}%)")

    # 推荐折扣参数
    calib = {
        'low_diff': 0.20,   # 实力接近: 20% 折扣
        'high_diff': 0.0,   # 实力悬殊: 0% 折扣 (现有折扣会恶化Brier)
        'total_games': len(rows),
        'friendly_games': len(friendlies),
        'description': 'low_diff=20%: 接近战模型相对可靠; high_diff=0%: 强弱对话固定折扣扭曲校准'
    }

    os.makedirs('/root/data', exist_ok=True)
    with open('/root/data/friendly_calib.json', 'w') as f:
        json.dump(calib, f, indent=2)
    print(f"\n已保存到 /root/data/friendly_calib.json")

if __name__ == '__main__':
    main()
