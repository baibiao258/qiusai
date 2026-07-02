#!/usr/bin/env python3
"""
365scores 每日数据收集
=====================
每日收集 365scores 数据，存为 CSV 格式，用于回测验证

用法:
  python3 collect_365scores_daily.py
"""

import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, '/root')

from fetch_365scores import fetch_365scores_data, extract_games

# CSV 字段定义
CSV_FIELDS = [
    'date', 'game_id', 'home', 'away', 'competition', 'time', 'status', 'score',
    'vote_home', 'vote_draw', 'vote_away', 'vote_count',
    'pop_rank_home', 'pop_rank_away', 'pop_rank_diff',
    'fifa_rank_home', 'fifa_rank_away', 'fifa_rank_diff',
    'trend_home_w', 'trend_home_d', 'trend_home_l',
    'trend_away_w', 'trend_away_d', 'trend_away_l',
    'trend_win_rate_home', 'trend_win_rate_away', 'trend_win_rate_diff',
    'venue', 'attendance',
    'goals_home', 'goals_away',
    'yellow_home', 'yellow_away',
    'red_home', 'red_away',
    # 2026-06-07: 365scores 临场元数据
    'has_lineups', 'lineups_status_text',
    'has_doubtful', 'has_missing_players',
    'has_statistics', 'has_news', 'has_buzz',
    'social_comments',
    # 2026-06-14: 半场比分 + 赛果
    'score_ht', 'winner',
]

DATA_DIR = '/root/data/365scores'
MASTER_CSV = f"{DATA_DIR}/football_games.csv"


def game_to_row(date_str, g):
    """将一场比赛转为 CSV 行"""
    votes = g.get('votes') or {}
    trend_h = g.get('trend_home', [])
    trend_a = g.get('trend_away', [])
    
    # 趋势胜率
    th_total = sum(trend_h[:3]) if len(trend_h) >= 3 else 0
    ta_total = sum(trend_a[:3]) if len(trend_a) >= 3 else 0
    wr_h = trend_h[0] / th_total if th_total > 0 else None
    wr_a = trend_a[0] / ta_total if ta_total > 0 else None
    wr_diff = (wr_h - wr_a) if wr_h is not None and wr_a is not None else None
    
    return {
        'date': date_str,
        'game_id': g.get('id', ''),
        'home': g.get('home', ''),
        'away': g.get('away', ''),
        'competition': g.get('competition', ''),
        'time': g.get('time', ''),
        'status': g.get('status', ''),
        'score': g.get('score', ''),
        'score_ht': g.get('score_ht', ''),
        'winner': g.get('winner', ''),
        'vote_home': votes.get('home', ''),
        'vote_draw': votes.get('draw', ''),
        'vote_away': votes.get('away', ''),
        'vote_count': votes.get('total', ''),
        'pop_rank_home': g.get('pop_rank_home', ''),
        'pop_rank_away': g.get('pop_rank_away', ''),
        'pop_rank_diff': (g.get('pop_rank_away', 0) or 0) - (g.get('pop_rank_home', 0) or 0) if g.get('pop_rank_home') and g.get('pop_rank_away') else '',
        'fifa_rank_home': g.get('fifa_rank_home', ''),
        'fifa_rank_away': g.get('fifa_rank_away', ''),
        'fifa_rank_diff': (g.get('fifa_rank_away', 0) or 0) - (g.get('fifa_rank_home', 0) or 0) if g.get('fifa_rank_home') and g.get('fifa_rank_away') else '',
        'trend_home_w': trend_h[0] if len(trend_h) > 0 else '',
        'trend_home_d': trend_h[1] if len(trend_h) > 1 else '',
        'trend_home_l': trend_h[2] if len(trend_h) > 2 else '',
        'trend_away_w': trend_a[0] if len(trend_a) > 0 else '',
        'trend_away_d': trend_a[1] if len(trend_a) > 1 else '',
        'trend_away_l': trend_a[2] if len(trend_a) > 2 else '',
        'trend_win_rate_home': round(wr_h, 4) if wr_h is not None else '',
        'trend_win_rate_away': round(wr_a, 4) if wr_a is not None else '',
        'trend_win_rate_diff': round(wr_diff, 4) if wr_diff is not None else '',
        'venue': g.get('venue', ''),
        'attendance': g.get('attendance', ''),
        'goals_home': g.get('goals_home', ''),
        'goals_away': g.get('goals_away', ''),
        'yellow_home': g.get('yellow_cards_home', ''),
        'yellow_away': g.get('yellow_cards_away', ''),
        'red_home': g.get('red_cards_home', ''),
        'red_away': g.get('red_cards_away', ''),
        # 2026-06-07: 365scores 临场元数据
        'has_lineups': g.get('has_lineups', ''),
        'lineups_status_text': g.get('lineups_status_text', ''),
        'has_doubtful': g.get('has_doubtful', ''),
        'has_missing_players': g.get('has_missing_players', ''),
        'has_statistics': g.get('has_statistics', ''),
        'has_news': g.get('has_news', ''),
        'has_buzz': g.get('has_buzz', ''),
        'social_comments': g.get('social_comments', ''),
    }


def append_to_master_csv(rows):
    """追加到主 CSV 文件（去重）"""
    existing_ids = set()
    if os.path.exists(MASTER_CSV):
        with open(MASTER_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_ids.add(f"{r['date']}_{r['game_id']}")
    
    new_rows = [r for r in rows if f"{r['date']}_{r['game_id']}" not in existing_ids]
    
    write_header = not os.path.exists(MASTER_CSV) or os.path.getsize(MASTER_CSV) == 0
    with open(MASTER_CSV, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    
    return len(new_rows)


def main():
    """主函数"""
    print("=== 365scores 每日数据收集 (CSV) ===")
    print()
    
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"日期: {today}")
    print()
    
    # 获取数据
    print("📡 获取 365scores 数据...")
    scores365_raw = fetch_365scores_data()
    if not scores365_raw:
        print("❌ 获取数据失败")
        return
    
    scores365_games = extract_games(scores365_raw, filter_sid=1)
    print(f"✓ 获取到 {len(scores365_games)} 场足球比赛")
    
    friendly = [g for g in scores365_games if 'friendly' in g['competition'].lower()]
    print(f"  其中友谊赛: {len(friendly)} 场")
    print()
    
    # 转为 CSV 行 (已通过 filter_sid=1 过滤为仅足球)
    os.makedirs(DATA_DIR, exist_ok=True)
    all_rows = [game_to_row(today, g) for g in scores365_games]
    print(f"  (足球: {len(all_rows)} 场)")
    
    # 保存当日 CSV (足球, 含原始 API 生数据以备后用)
    daily_csv = f"{DATA_DIR}/{today}.csv"
    with open(daily_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows([game_to_row(today, g) for g in scores365_games])
    print(f"✓ 当日全量: {daily_csv} ({len(scores365_games)} 行)")

    # 追加到足球主 CSV (仅足球)
    new_count = append_to_master_csv(all_rows)
    print(f"✓ 足球主文件追加: {MASTER_CSV} (+{new_count} 新行)")
    print()
    
    # 显示友谊赛摘要
    print("=== 友谊赛摘要 ===")
    print()
    friendly_rows = [r for r in all_rows if 'friendly' in r['competition'].lower()]
    for r in friendly_rows:
        vote_str = ""
        if r['vote_count']:
            vote_str = f" | 投票: {r['vote_home']}%/{r['vote_draw']}%/{r['vote_away']}%"
        print(f"  {r['home']} vs {r['away']} ({r['status']}){vote_str}")
    
    print()
    print(f"=== 收集完成 | 累计主文件行数 ===")
    with open(MASTER_CSV, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f) - 1  # 减去 header
    print(f"  {MASTER_CSV}: {total} 行")


if __name__ == "__main__":
    main()
