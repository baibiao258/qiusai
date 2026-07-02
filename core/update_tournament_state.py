#!/usr/bin/env python3
"""
update_tournament_state.py — 动态更新杯赛状态 (使用 standings API)
================================================================

从 football-data.org API 的 /standings 端点获取世界杯积分榜,
保存到 /root/data/tournament_state.json 供 daily_jczq.py 读取。

用法:
  python3 update_tournament_state.py              # 使用缓存或更新
  python3 update_tournament_state.py --force       # 强制刷新缓存
  python3 update_tournament_state.py --dry-run     # 仅打印不保存
"""

import argparse
import json
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import requests

DATA_DIR = Path('/root/data')
OUTPUT_FILE = DATA_DIR / 'tournament_state.json'
CACHE_FILE = DATA_DIR / 'tournament_api_cache.json'
CACHE_TTL_HOURS = 24

API_BASE = 'https://api.football-data.org/v4'
API_KEY = os.environ.get('FOOTBALL_API_KEY', '5d07c80baa2645d0809b6ec96d6b49c6')

TEAM_NAME_MAP = {
    'Mexico': '墨西哥', 'South Africa': '南非',
    'South Korea': '韩国', 'Czechia': '捷克',
    'Canada': '加拿大', 'Bosnia-Herzegovina': '波黑',
    'United States': '美国', 'Paraguay': '巴拉圭',
    'Qatar': '卡塔尔', 'Switzerland': '瑞士',
    'Brazil': '巴西', 'Morocco': '摩洛哥',
    'Haiti': '海地', 'Scotland': '苏格兰',
    'Australia': '澳大利亚', 'Turkey': '土耳其',
    'Germany': '德国', 'Cura\u00e7ao': '库拉索',
    'Netherlands': '荷兰', 'Japan': '日本',
    'Ivory Coast': '科特迪瓦', 'Ecuador': '厄瓜多尔',
    'Sweden': '瑞典', 'Tunisia': '突尼斯',
    'Spain': '西班牙', 'Cape Verde Islands': '佛得角',
    'Belgium': '比利时', 'Egypt': '埃及',
    'Saudi Arabia': '沙特阿拉伯', 'Uruguay': '乌拉圭',
    'Iran': '伊朗', 'New Zealand': '新西兰',
    'France': '法国', 'Senegal': '塞内加尔',
    'Iraq': '伊拉克', 'Norway': '挪威',
    'Argentina': '阿根廷', 'Algeria': '阿尔及利亚',
    'Austria': '奥地利', 'Jordan': '约旦',
    'Portugal': '葡萄牙', 'Congo DR': '刚果(金)',
    'England': '英格兰', 'Croatia': '克罗地亚',
    'Ghana': '加纳', 'Panama': '巴拿马',
    "Uzbekistan": '乌兹别克', 'Colombia': '哥伦比亚',
}


def load_cache():
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        cached_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
        if datetime.now() - cached_time > timedelta(hours=CACHE_TTL_HOURS):
            print(f'  \u23f0 缓存已过期 ({CACHE_TTL_HOURS}h)')
            return None
        print(f'  \U0001f4e6 使用本地缓存 (更新于 {cached_time.strftime("%Y-%m-%d %H:%M")})')
        return cache.get('standings', [])
    except Exception as e:
        print(f'  \u26a0\ufe0f 加载缓存失败: {e}')
        return None


def save_cache(standings):
    cache = {'timestamp': datetime.now().isoformat(), 'standings': standings}
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    print(f'  \U0001f4be 缓存已保存 (有效期 {CACHE_TTL_HOURS}h)')


def fetch_standings_from_api():
    url = f'{API_BASE}/competitions/WC/standings'
    headers = {'X-Auth-Token': API_KEY}
    print(f'  \U0001f4e1 调用 API: {url}')
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 429:
        print(f'  \u274c API 限流 (429), 使用缓存')
        return None
    resp.raise_for_status()
    return resp.json().get('standings', [])


def build_tournament_state(standings_data):
    state = {}
    for table in standings_data:
        group = table.get('group', '')
        if 'Group' not in str(group):
            continue
        current_round = 1
        for row in table.get('table', []):
            played = row.get('playedGames', 0)
            if played > 0:
                current_round = max(current_round, played)
        for row in table.get('table', []):
            team_name = row.get('team', {}).get('name', '')
            team_cn = TEAM_NAME_MAP.get(team_name, team_name)
            state[team_cn] = {
                'home_group_points': row.get('points', 0),
                'away_group_points': 0,
                'home_group_rank': row.get('position', 0),
                'away_group_rank': row.get('position', 0),
                'is_knockout': False,
                'round_num': current_round,
            }
    return state


def main():
    parser = argparse.ArgumentParser(description='\u66f4\u65b0\u676f\u8d5b\u72b6\u6001')
    parser.add_argument('--force', action='store_true', help='\u5f3a\u5236\u5237\u65b0\u7f13\u5b58')
    parser.add_argument('--dry-run', action='store_true', help='\u4ec5\u6253\u5370\u4e0d\u4fdd\u5b58')
    args = parser.parse_args()

    print('=' * 60)
    print('\U0001f504 \u66f4\u65b0\u676f\u8d5b\u72b6\u6001 (\u4f7f\u7528 standings API)')
    print('=' * 60)

    standings_data = None
    if not args.force:
        standings_data = load_cache()

    if standings_data is None:
        print('\n\U0001f4e1 \u4ece API \u83b7\u53d6\u6700\u65b0\u79ef\u5206\u699c...')
        standings_data = fetch_standings_from_api()
        if standings_data:
            save_cache(standings_data)
        else:
            print('\n\u26a0\ufe0f API \u5931\u8d25, \u5c1d\u8bd5\u4f7f\u7528\u65e7\u7f13\u5b58...')
            if CACHE_FILE.exists():
                with open(CACHE_FILE) as f:
                    cache = json.load(f)
                standings_data = cache.get('standings', [])
                if standings_data:
                    print(f'  \U0001f4e6 \u4f7f\u7528\u65e7\u7f13\u5b58 ({len(standings_data)} \u4e2a\u79ef\u5206\u699c)')

    if not standings_data:
        print('\n\u274c \u65e0\u6cd5\u83b7\u53d6\u79ef\u5206\u699c\u6570\u636e')
        return

    print(f'\n\u2705 \u83b7\u53d6\u5230 {len(standings_data)} \u4e2a\u79ef\u5206\u699c')
    state = build_tournament_state(standings_data)
    print(f'\u2705 \u8ba1\u7b97 {len(state)} \u652f\u7403\u961f\u7684\u72b6\u6001')

    print(f'\n\U0001f4ca \u72b6\u6001\u6458\u8981:')
    for team, info in list(state.items())[:8]:
        print(f'  {team}: {info["home_group_points"]}\u5206, \u6392\u540d{info["home_group_rank"]}, \u7b2c{info["round_num"]}\u8f6e')

    if args.dry_run:
        print(f'\n📋 Dry run 模式, 不保存文件')
    else:
        DATA_DIR.mkdir(exist_ok=True)
        # ── 主文件 ──
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f'\n✅ 保存到: {OUTPUT_FILE}')
        print(f'   文件大小: {os.path.getsize(OUTPUT_FILE)} bytes')

        # ── 日期快照 (供 _load_tournament_state 回退链使用) ──
        today_str = date.today().isoformat()
        dated_file = DATA_DIR / f'tournament_state.{today_str}.json'
        with open(dated_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f'   📆 日期快照: {dated_file}')

        # ── 清理 7 天前的过期快照 ──
        import glob
        import re as _re
        ts_re = _re.compile(r'tournament_state\.(\d{4}-\d{2}-\d{2})\.json$')
        seen_dated = 0
        cleaned = 0
        for fpath in sorted(glob.glob(str(DATA_DIR / 'tournament_state.*.json'))):
            m = ts_re.search(fpath)
            if m:
                seen_dated += 1
                fdate = m.group(1)
                if fdate < (date.today() - timedelta(days=7)).isoformat():
                    os.remove(fpath)
                    cleaned += 1
        if cleaned:
            print(f'   🧹 清理过期快照: {cleaned} 个 (保留最近 7 天, 共 {seen_dated} 个)')


if __name__ == '__main__':
    main()
