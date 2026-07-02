#!/usr/bin/env python3
"""
fetch_league_data.py — 从 football-data.org 批量拉取联赛历史数据
=================================================================
功能:
  1. 拉取 9 联赛近 N 个赛季的比赛数据 (含半场比分)
  2. 限流保护: 每次请求间隔 6.5 秒 (API 限制 10 次/分钟)
  3. 输出统一格式 JSON: /root/data/club_matches.json

用法:
  python3 fetch_league_data.py              # 拉取所有联赛
  python3 fetch_league_data.py --league PL  # 仅拉取英超
  python3 fetch_league_data.py --seasons 3  # 近 3 个赛季

数据格式:
  {
    "date": "2025-08-17",
    "home": "Arsenal",
    "away": "Wolverhampton Wanderers",
    "h_score": 2, "a_score": 0,
    "ht_h": 1, "ht_a": 0,
    "tournament": "Premier League",
    "season": "2025",
    "matchday": 1,
    "neutral": false
  }
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

DATA_DIR = '/root/data'
OUTPUT_PATH = os.path.join(DATA_DIR, 'club_matches.json')

API_KEY = os.environ.get('FOOTBALL_API_KEY', '5d07c80baa2645d0809b6ec96d6b49c6')
API_HDR = {'X-Auth-Token': API_KEY, 'Accept': 'application/json'}
API_BASE = 'https://api.football-data.org/v4'

# 请求间隔 (秒) — API 限制 10 次/分钟
REQUEST_INTERVAL = 6.5

# 联赛配置
LEAGUES = {
    'PL':  {'name': 'Premier League',      'country': 'England'},
    'BL1': {'name': 'Bundesliga',          'country': 'Germany'},
    'PD':  {'name': 'Primera Division',    'country': 'Spain'},
    'SA':  {'name': 'Serie A',             'country': 'Italy'},
    'FL1': {'name': 'Ligue 1',             'country': 'France'},
    'DED': {'name': 'Eredivisie',          'country': 'Netherlands'},
    'PPL': {'name': 'Liga Portugal',       'country': 'Portugal'},
    'ELC': {'name': 'Championship',        'country': 'England'},
    'BSA': {'name': 'Série A',             'country': 'Brazil'},
}

# 默认拉取近 5 个赛季
DEFAULT_SEASONS = 5


def api_get(path, retries=3):
    """带重试的 API 请求."""
    url = f"{API_BASE}{path}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=API_HDR)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 60 * (attempt + 1)
                print(f"    ⚠️ 限流, 等待 {wait}s...")
                time.sleep(wait)
            elif e.code == 403:
                print(f"    ❌ 403 Forbidden — 需要更高权限或赛季不存在")
                return None
            else:
                print(f"    ⚠️ HTTP {e.code}, 重试 {attempt+1}/{retries}")
                time.sleep(REQUEST_INTERVAL)
        except Exception as e:
            print(f"    ⚠️ 请求失败: {e}, 重试 {attempt+1}/{retries}")
            time.sleep(REQUEST_INTERVAL)
    print(f"    ❌ 请求失败, 跳过")
    return None


def fetch_league_matches(league_code, years_back=5):
    """
    拉取联赛近 N 年的比赛数据.
    
    不依赖 seasons 端点, 直接尝试不同年份.
    """
    all_matches = []
    current_year = datetime.now().year
    
    for year in range(current_year, current_year - years_back - 1, -1):
        print(f"  📅 {year}...", end=' ', flush=True)
        
        # football-data.org 赛季格式: 2024 赛季 = 2024-08-01 ~ 2025-05-31
        data = api_get(f"/competitions/{league_code}/matches?season={year}")
        
        if data:
            matches = data.get('matches', [])
            finished = [m for m in matches if m.get('status') == 'FINISHED']
            print(f"{len(finished)} 场 (已完成)")
            
            for m in finished:
                home = m.get('homeTeam', {}).get('name', '')
                away = m.get('awayTeam', {}).get('name', '')
                if not home or not away:
                    continue
                
                score = m.get('score', {})
                ft = score.get('fullTime', {})
                ht = score.get('halfTime', {})
                
                hg = ft.get('home')
                ag = ft.get('away')
                if hg is None or ag is None:
                    continue
                
                ht_h = ht.get('home', 0) if ht else 0
                ht_a = ht.get('away', 0) if ht else 0
                
                all_matches.append({
                    'date': m.get('utcDate', '')[:10],
                    'home': home,
                    'away': away,
                    'h_score': hg,
                    'a_score': ag,
                    'ht_h': ht_h,
                    'ht_a': ht_a,
                    'tournament': LEAGUES.get(league_code, {}).get('name', league_code),
                    'season': str(year),
                    'matchday': m.get('matchday'),
                    'neutral': False,
                    'league': league_code,
                })
        else:
            print("无数据")
        
        time.sleep(REQUEST_INTERVAL)
    
    return all_matches


def main():
    parser = argparse.ArgumentParser(description='拉取联赛历史数据')
    parser.add_argument('--league', type=str, help='指定联赛代码 (如 PL)')
    parser.add_argument('--seasons', type=int, default=DEFAULT_SEASONS, help='近 N 个赛季')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH, help='输出路径')
    args = parser.parse_args()

    leagues_to_fetch = [args.league] if args.league else list(LEAGUES.keys())

    print(f"📡 拉取联赛数据...")
    print(f"  联赛: {', '.join(leagues_to_fetch)}")
    print(f"  赛季: 近 {args.seasons} 个赛季")
    print(f"  限流: 每次请求间隔 {REQUEST_INTERVAL}s")
    print()

    # 增量模式: 加载已有数据
    all_matches = []
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                all_matches = json.load(f)
            print(f"  📂 已有 {len(all_matches)} 场, 增量追加")
        except:
            all_matches = []
    seen = set((m['date'], m['home'], m['away']) for m in all_matches)

    total_api_calls = 0

    for league_code in leagues_to_fetch:
        info = LEAGUES.get(league_code, {})
        print(f"🏟️  {info.get('name', league_code)} ({info.get('country', '?')})")

        # 拉取比赛数据
        matches = fetch_league_matches(league_code, years_back=args.seasons)
        
        # 去重并追加
        new_count = 0
        for m in matches:
            key = (m['date'], m['home'], m['away'])
            if key not in seen:
                seen.add(key)
                all_matches.append(m)
                new_count += 1
        
        # 每个联赛完成后立即保存
        if new_count > 0:
            with open(args.output, 'w') as f:
                json.dump(all_matches, f, ensure_ascii=False, indent=2)
            print(f"  💾 保存 +{new_count} 场, 总计 {len(all_matches)} 场")

    # 统计各联赛
    from collections import Counter
    league_counts = Counter(m['league'] for m in all_matches)
    print(f"\n{'='*50}")
    print(f"✅ 完成! 总计 {len(all_matches)} 场")
    print(f"  各联赛场次:")
    for code, count in sorted(league_counts.items(), key=lambda x: -x[1]):
        name = LEAGUES.get(code, {}).get('name', code)
        print(f"    {name:25s} {count:5d} 场")


if __name__ == '__main__':
    main()
