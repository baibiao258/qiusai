#!/usr/bin/env python3
"""
update_form_state.py — 每日更新 form_state.json
================================================
从 football-data.org 拉取昨日比赛结果, 追加到 form_state.json.

用法:
  python3 update_form_state.py              # 更新昨日数据
  python3 update_form_state.py --date 2026-06-07  # 指定日期
  python3 update_form_state.py --rebuild    # 从头重建

Cron 建议: 每日 06:00 运行 (比赛结束后)
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, date, timedelta

DATA_DIR = '/root/data'
FORM_STATE_PATH = os.path.join(DATA_DIR, 'form_state.json')
INTERNATIONAL_RESULTS = os.path.join(DATA_DIR, 'international_results.json')

API_KEY = os.environ.get('FOOTBALL_API_KEY', '5d07c80baa2645d0809b6ec96d6b49c6')
API_HDR = {'X-Auth-Token': API_KEY, 'Accept': 'application/json'}

# 竞彩覆盖联赛
LEAGUES = ['PL', 'BL1', 'PD', 'SA', 'FL1', 'DED', 'PPL', 'ELC']


def api_get(path):
    """通用 API 请求"""
    url = f"https://api.football-data.org/v4{path}"
    req = urllib.request.Request(url, headers=API_HDR)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode('utf-8'))


def load_form_state():
    """加载 form_state.json."""
    if os.path.exists(FORM_STATE_PATH):
        with open(FORM_STATE_PATH) as f:
            return json.load(f)
    return {}


def save_form_state(state):
    """保存 form_state.json."""
    with open(FORM_STATE_PATH, 'w') as f:
        json.dump(state, f, ensure_ascii=False)


def fetch_yesterday_results():
    """从 football-data.org 拉取昨日比赛结果."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    results = []

    for code in LEAGUES:
        try:
            data = api_get(f"/competitions/{code}/matches?dateFrom={yesterday}&dateTo={yesterday}")
            for m in data.get('matches', []):
                if m['status'] == 'FINISHED':
                    home = m['homeTeam'].get('name', m['homeTeam'].get('shortName', ''))
                    away = m['awayTeam'].get('name', m['awayTeam'].get('shortName', ''))
                    hg = m.get('score', {}).get('fullTime', {}).get('home')
                    ag = m.get('score', {}).get('fullTime', {}).get('away')
                    if home and away and hg is not None and ag is not None:
                        results.append({
                            'date': yesterday,
                            'home': home,
                            'away': away,
                            'h_score': hg,
                            'a_score': ag,
                            'tournament': m.get('competition', {}).get('name', ''),
                        })
        except Exception as e:
            print(f"  ⚠️ {code}: {e}")

    return results


def fetch_international_results昨日():
    """从 international_results.json 提取昨日比赛."""
    if not os.path.exists(INTERNATIONAL_RESULTS):
        return []

    yesterday = (date.today() - timedelta(days=1)).isoformat()

    with open(INTERNATIONAL_RESULTS) as f:
        all_matches = json.load(f)

    results = []
    for m in all_matches:
        if m.get('date') == yesterday:
            try:
                hg = int(m.get('h_score', m.get('home_score', -1)))
                ag = int(m.get('a_score', m.get('away_score', -1)))
                if hg >= 0 and ag >= 0:
                    results.append({
                        'date': yesterday,
                        'home': m.get('home', m.get('home_team', '')),
                        'away': m.get('away', m.get('away_team', '')),
                        'h_score': hg,
                        'a_score': ag,
                    })
            except:
                continue

    return results


def update_form_state(matches, state):
    """将比赛结果追加到 form_state."""
    updated = 0
    for m in matches:
        home = m['home']
        away = m['away']
        hg = m['h_score']
        ag = m['a_score']

        # 追加到主队记录 (主队视角: gf=hg, ga=ag)
        if home not in state:
            state[home] = []
        state[home].append([hg, ag])

        # 追加到客队记录 (客队视角: gf=ag, ga=hg)
        if away not in state:
            state[away] = []
        state[away].append([ag, hg])

        updated += 1

    return state, updated


def rebuild_from_international():
    """从 international_results.json 完全重建 form_state."""
    if not os.path.exists(INTERNATIONAL_RESULTS):
        print("❌ international_results.json 不存在")
        return {}

    with open(INTERNATIONAL_RESULTS) as f:
        all_matches = json.load(f)

    # 按日期排序
    all_matches.sort(key=lambda x: x.get('date', ''))

    state = {}
    for m in all_matches:
        try:
            home = m.get('home', m.get('home_team', ''))
            away = m.get('away', m.get('away_team', ''))
            hg = int(m.get('h_score', m.get('home_score', -1)))
            ag = int(m.get('a_score', m.get('away_score', -1)))
            if hg < 0 or ag < 0 or not home or not away:
                continue
        except:
            continue

        # 每队保留最近 25 场
        if home not in state:
            state[home] = []
        state[home].append([hg, ag])
        if len(state[home]) > 25:
            state[home] = state[home][-25:]

        if away not in state:
            state[away] = []
        state[away].append([ag, hg])
        if len(state[away]) > 25:
            state[away] = state[away][-25:]

    return state


def main():
    parser = argparse.ArgumentParser(description='更新 form_state.json')
    parser.add_argument('--date', type=str, help='指定日期 (YYYY-MM-DD)')
    parser.add_argument('--rebuild', action='store_true', help='从头重建')
    args = parser.parse_args()

    if args.rebuild:
        print("🔨 从头重建 form_state.json...")
        state = rebuild_from_international()
        save_form_state(state)
        print(f"✅ 重建完成: {len(state)} 队")
        return

    print(f"📡 更新 form_state.json...")
    state = load_form_state()

    # 收集昨日比赛
    matches = []

    # 1. 从 football-data.org (联赛)
    fd_matches = fetch_yesterday_results()
    matches.extend(fd_matches)
    print(f"  football-data.org: {len(fd_matches)} 场")

    # 2. 从 international_results.json (国际赛)
    intl_matches = fetch_international_results昨日()
    matches.extend(intl_matches)
    print(f"  international: {len(intl_matches)} 场")

    if not matches:
        print("  ⚠️ 昨日无比赛数据")
        return

    # 更新
    state, updated = update_form_state(matches, state)
    save_form_state(state)
    print(f"✅ 更新完成: {updated} 场比赛, {len(state)} 队")

    # 同时重建 H2H 和 form_12 缓存
    try:
        from feature_helper import rebuild_all_caches
        rebuild_all_caches()
    except Exception as e:
        print(f"  ⚠️ 缓存重建失败: {e}")


if __name__ == '__main__':
    main()
