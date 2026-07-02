#!/usr/bin/env python3
"""
prepare_training_data.py — 准备带市场赔率特征的训练数据
=======================================================

从 historical_kaijiang.csv 提取收盘SP赔率,
与 international_results.json 合并,
生成带市场赔率特征的 XGBoost 训练集。

输出: /root/data/training_data_with_odds.json
"""

import json
import os
import sys
from datetime import datetime, date, timedelta
from collections import defaultdict

import numpy as np

DATA_DIR = '/root/data'


def load_kaijiang():
    """加载历史开奖数据"""
    import csv
    path = os.path.join(DATA_DIR, 'historical_kaijiang.csv')
    matches = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for num_field in ['handicap', 'ht_h', 'ht_a', 'ft_h', 'ft_a', 'total_goals']:
                if row.get(num_field):
                    row[num_field] = int(row[num_field])
            for sp_field in ['spf_sp', 'rqspf_sp', 'jqs_sp', 'bqc_sp']:
                if row.get(sp_field):
                    row[sp_field] = float(row[sp_field])
            matches.append(row)
    return matches


def load_international():
    """加载国际赛数据"""
    path = os.path.join(DATA_DIR, 'international_results.json')
    with open(path) as f:
        return json.load(f)


def load_team_mapping():
    """加载队名映射"""
    path = os.path.join(DATA_DIR, 'team_name_mapping.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def merge_data(kaijiang, intl, team_map, date_tolerance=2):
    """合并数据, 保留市场赔率和赛事阶段特征"""
    from datetime import datetime as dt

    # 建索引
    intl_index = defaultdict(list)
    for m in intl:
        h, a = m['home'], m['away']
        intl_index[(h, a)].append(m)
        intl_index[(a, h)].append(m)

    merged = []
    club_teams = {'京都不死鸟', '冈山绿雉', '名古屋鲸鱼', '川崎前锋', '广岛三箭',
                  '柏太阳神', '横滨水手', '浦和红钻', '清水心跳', '町田泽维亚',
                  '神户胜利船', '鹿岛鹿角', '特尔斯达', '芬洛', '布雷达',
                  '格拉夫夏普', '多德勒支', '罗达JC', '海牙', '奥斯',
                  '埃门', '登博思', '威廉二世', '马斯特里赫特', '坎布尔',
                  '阿尔克马尔青年', 'SBV精英', '福伦丹', '赫尔蒙德', '阿贾克斯青年',
                  'FC埃因霍温', '乌德勒支青年', '阿尔梅勒城', '邓伯什', '阿克马尔青年'}

    for kj in kaijiang:
        home_cn, away_cn = kj['home'], kj['away']

        if home_cn in club_teams or away_cn in club_teams:
            continue

        home_en = team_map.get(home_cn)
        away_en = team_map.get(away_cn)

        if not home_en or not away_en:
            continue

        try:
            kj_date = dt.strptime(kj['date'], '%Y-%m-%d').date()
        except ValueError:
            continue

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
            # 计算隐含概率 (去除overround)
            sp = kj['spf_sp']
            if sp > 0:
                implied_prob = 1.0 / sp
            else:
                implied_prob = 0.0

            # 赛事阶段特征 (Phase 2: 从tournament推断)
            tournament = best_match.get('tournament', '')
            is_world_cup = 'World Cup' in tournament or '世界杯' in tournament
            is_euro = 'Euro' in tournament or '欧洲杯' in tournament
            is_copa = 'Copa' in tournament or '美洲杯' in tournament
            is_asian_cup = 'Asian Cup' in tournament or '亚洲杯' in tournament
            is_knockout = any(kw in tournament.lower() for kw in ['final', 'semi', 'quarter', 'knockout'])
            
            # 根据tournament类型和日期推断阶段
            # 默认: 小组赛第1轮, 积分=0
            points_diff = 0.0
            rank_diff = 0.333  # 默认排名差
            is_knockout_flag = 1.0 if is_knockout else 0.0
            
            # 根据月份推断轮次 (世界杯通常6-7月)
            month = kj_date.month if kj_date else 6
            if is_world_cup:
                if month == 6:
                    round_num = 1  # 小组赛第1轮
                elif month == 7:
                    round_num = 2  # 小组赛第2-3轮或淘汰赛
                else:
                    round_num = 1
            else:
                round_num = 1  # 非世界杯默认第1轮

            merged.append({
                'date': kj['date'],
                'home_en': home_en,
                'away_en': away_en,
                'tournament': tournament,
                'spf_result': kj['spf_result'],  # 3/1/0
                'spf_sp': kj['spf_sp'],
                'rqspf_sp': kj['rqspf_sp'],
                'handicap': kj['handicap'],
                'ft_h': kj['ft_h'],
                'ft_a': kj['ft_a'],
                # 市场赔率特征
                'market_odds': kj['spf_sp'],
                'market_implied_prob': implied_prob,
                # 赛事阶段特征 (4维)
                'points_diff': points_diff,
                'rank_diff': rank_diff,
                'is_knockout': is_knockout_flag,
                'round_num': round_num / 7.0,  # 归一化
            })

    return merged


def main():
    print("📡 加载数据...")
    kaijiang = load_kaijiang()
    intl = load_international()
    team_map = load_team_mapping()

    print(f"  开奖数据: {len(kaijiang)} 场")
    print(f"  国际赛数据: {len(intl)} 场")

    print("\n🔗 合并数据...")
    merged = merge_data(kaijiang, intl, team_map)

    print(f"  合并: {len(merged)} 场")

    # 保存训练数据
    output_path = os.path.join(DATA_DIR, 'training_data_with_odds.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 保存到: {output_path}")
    print(f"  包含字段: {list(merged[0].keys()) if merged else 'none'}")

    # 统计
    if merged:
        tournaments = defaultdict(int)
        for m in merged:
            tournaments[m['tournament']] += 1
        print(f"\n📊 赛事分布:")
        for t, count in sorted(tournaments.items(), key=lambda x: -x[1])[:10]:
            print(f"  {t}: {count}")


if __name__ == '__main__':
    main()
