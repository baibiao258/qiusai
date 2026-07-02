#!/usr/bin/env python3
"""
build_training_with_365scores.py
================================
将 football_games.csv 的 365scores 投票/FIFA/人气特征
join 到 XGBoost 训练数据中。

10 维 365scores 特征:
  1-3. vote_home, vote_draw, vote_away  (群众投票分布)
  4.    vote_count_log                   (投票量 log, 置信度权重)
  5-6.  pop_rank_diff, pop_log_diff     (人气排名差)
  7.    trend_win_rate_diff              (近5场胜率差)
  8.    trend_goals_diff                 (近5场进球差)
  9-10. fifa_rank_diff, fifa_log_diff   (FIFA 排名差)

用法:
  python3 build_training_with_365scores.py
  python3 build_training_with_365scores.py --min-overlap 50
  python3 build_training_with_365scores.py --stats-only

输出: /root/data/training_with_365scores.json
      (原始训练数据 + s365_* 字段, 仅含能配对的场次)
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict

DATA_DIR = '/root/data'
FOOTBALL_CSV = f'{DATA_DIR}/365scores/football_games.csv'
TRAIN_JSON = f'{DATA_DIR}/training_data_with_odds.json'
OUTPUT_JSON = f'{DATA_DIR}/training_with_365scores.json'
MAPPING_JSON = f'{DATA_DIR}/team_name_mapping.json'


def load_365scores():
    """加载 football_games.csv, 只保留 finished 比赛"""
    if not os.path.exists(FOOTBALL_CSV):
        print(f"  ❌ 找不到 {FOOTBALL_CSV}")
        return [], {}

    rows = []
    index = defaultdict(list)  # {(date, home_lower, away_lower): [row, ...]}
    with open(FOOTBALL_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status', '').lower() != 'finished':
                continue
            if not row.get('score') or ':' not in row.get('score', ''):
                continue
            rows.append(row)
            d = row.get('date', '')
            h = (row.get('home') or '').strip().lower()
            a = (row.get('away') or '').strip().lower()
            index[(d, h, a)].append(row)

    print(f"  ✅ 365scores finished: {len(rows)} 场")
    return rows, index


def load_mapping():
    """加载中英文队名映射: 中文→英文"""
    if not os.path.exists(MAPPING_JSON):
        print(f"  ⚠️ 无队名映射, 直接匹配英文")
        return {}
    with open(MAPPING_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_365_features(row):
    """从 football_games.csv 的一行提取 10 维特征"""
    try:
        vh = float(row.get('vote_home', 0) or 0)
        vd = float(row.get('vote_draw', 0) or 0)
        va = float(row.get('vote_away', 0) or 0)
    except (ValueError, TypeError):
        return None

    try:
        vc = int(row.get('vote_count', 0) or 0)
    except (ValueError, TypeError):
        vc = 0

    try:
        prh = int(row.get('pop_rank_home', 0) or 0)
        pra = int(row.get('pop_rank_away', 0) or 0)
    except (ValueError, TypeError):
        prh, pra = 0, 0

    try:
        thw = float(row.get('trend_home_w', 0) or 0)
        thd = float(row.get('trend_home_d', 0) or 0)
        thl = float(row.get('trend_home_l', 0) or 0)
        taw = float(row.get('trend_away_w', 0) or 0)
        tad = float(row.get('trend_away_d', 0) or 0)
        tal = float(row.get('trend_away_l', 0) or 0)
    except (ValueError, TypeError):
        thw = thd = thl = taw = tad = tal = 0

    th_total = thw + thd + thl
    ta_total = taw + tad + tal
    twr_h = thw / th_total if th_total > 0 else 0.5
    twr_a = taw / ta_total if ta_total > 0 else 0.5

    try:
        frh = int(row.get('fifa_rank_home', 0) or 0)
        fra = int(row.get('fifa_rank_away', 0) or 0)
    except (ValueError, TypeError):
        frh, fra = 0, 0

    rank_diff = pra - prh if prh > 0 and pra > 0 else 0
    fifa_diff = fra - frh if frh > 0 and fra > 0 else 0

    return {
        's365_vote_home': vh,
        's365_vote_draw': vd,
        's365_vote_away': va,
        's365_vote_log': math.log(vc + 1),
        's365_pop_rank_diff': rank_diff,
        's365_pop_log_diff': (math.log(pra + 1) - math.log(prh + 1)) if prh > 0 and pra > 0 else 0,
        's365_trend_winrate_diff': twr_h - twr_a,
        's365_trend_goals_diff': (thw * 3 + thd) / max(th_total, 1) - (taw * 3 + tad) / max(ta_total, 1),
        's365_fifa_rank_diff': fifa_diff,
        's365_fifa_log_diff': (math.log(fra + 1) - math.log(frh + 1)) if frh > 0 and fra > 0 else 0,
    }


def match_team(team_name_in, mapping):
    """尝试匹配队名: 中文→英文, 直接英文, 或去空格后匹配"""
    team_lower = team_name_in.strip().lower()

    # 直接英文匹配
    if team_lower:
        return team_name_in.strip()

    # 中文→英文
    if team_name_in in mapping:
        return mapping[team_name_in]

    # 反转映射找匹配
    rev = {v.lower(): k for k, v in mapping.items()}
    if team_lower in rev:
        return team_lower.title()

    return None


def build(min_overlap=1, stats_only=False):
    """主构建函数"""
    print("📡 加载数据...")

    # 加载 365scores
    s365_rows, s365_index = load_365scores()
    if not s365_rows:
        print("❌ 无有效 365scores 数据")
        return

    # 加载训练数据
    if not os.path.exists(TRAIN_JSON):
        print(f"❌ 找不到 {TRAIN_JSON}")
        return
    with open(TRAIN_JSON, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"  ✅ 训练数据: {len(train_data)} 场")

    # 加载队名映射
    mapping = load_mapping()
    print(f"  ✅ 队名映射: {len(mapping)} 条")

    # 统计: 各来源可配对的场次
    matched = []
    unmatched = []
    match_stats = defaultdict(int)  # {365_daily_csv_file: count}

    for m in train_data:
        date = m.get('date', '')
        home_en = (m.get('home_en') or '').strip()
        away_en = (m.get('away_en') or '').strip()

        if not date or not home_en or not away_en:
            unmatched.append((m, 'missing_date_or_team'))
            continue

        # 尝试直接匹配
        h_key = home_en.lower()
        a_key = away_en.lower()
        candidates = s365_index.get((date, h_key, a_key), [])

        if not candidates:
            # 尝试通过映射 (中文→英文) 匹配
            home_cn = mapping.get(home_en)  # 反转查找
            away_cn = mapping.get(away_en)
            if home_cn and away_cn:
                candidates = s365_index.get((date, home_cn.lower(), away_cn.lower()), [])
                if not candidates:
                    candidates = s365_index.get((date, away_cn.lower(), home_cn.lower()), [])

        if not candidates:
            # 尝试互换主客场
            candidates = s365_index.get((date, a_key, h_key), [])

        if candidates:
            best = candidates[0]
            feat = extract_365_features(best)
            if feat:
                m.update(feat)
                m['s365_matched'] = True
                matched.append(m)
                match_stats[best.get('date', '?')] += 1
            else:
                unmatched.append((m, 'extract_failed'))
        else:
            unmatched.append((m, 'no_s365_match'))

    print(f"\n{'='*50}")
    print(f"📊 配对结果")
    print(f"{'='*50}")
    print(f"  训练数据总数:        {len(train_data)}")
    print(f"  成功配对 365scores: {len(matched)}")
    print(f"  无法配对:            {len(unmatched)}")

    # 按日期分布
    if match_stats:
        print(f"\n  配对按日期分布:")
        for d in sorted(match_stats):
            print(f"    {d}: {match_stats[d]} 场")

    # 未配对原因
    if unmatched:
        reason_counts = defaultdict(int)
        for _, reason in unmatched:
            reason_counts[reason] += 1
        print(f"\n  未配对原因:")
        for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {cnt}")

    if stats_only:
        return

    if len(matched) < min_overlap:
        print(f"\n⚠️ 配对场次 {len(matched)} < {min_overlap}, 暂不输出 (可用 --min-overlap 调整)")
        print(f"  预计 6月底可达 200+ 场")
        return

    # 输出
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(matched, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存: {OUTPUT_JSON} ({len(matched)} 场)")
    print(f"  新增 365scores 特征: s365_vote_home/draw/away/log/pop/fifa/trend")
    print(f"  总特征数: 29 基础 + 10 维 s365 = 39 维")

    # 验证重训脚本
    train_script = '/root/wc_2026_upgrade/retrain_xgb_with_odds.py'
    if os.path.exists(train_script):
        print(f"\n💡 准备重训:")
        print(f"  mv {OUTPUT_JSON} /root/data/training_data_with_odds.json")
        print(f"  python3 {train_script}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="365scores 特征拼接工具")
    parser.add_argument("--min-overlap", type=int, default=1,
                        help="最小配对场次才输出 (默认1, 建议200)")
    parser.add_argument("--stats-only", action="store_true",
                        help="只展示配对统计, 不输出文件")
    args = parser.parse_args()
    build(min_overlap=args.min_overlap, stats_only=args.stats_only)


if __name__ == '__main__':
    main()
