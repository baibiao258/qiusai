"""
fetch_team_squads.py — 从 TheStatsAPI 拉取48支世界杯球队阵容
===========================================================
输出:
  /root/data/star_players.json — 球队阵容+核心球员标记

核心球员识别策略（不用写死名单）:
  1. 位置权重: F > M > D > G (前锋最重要)
  2. 年龄权重: 24-30岁是巅峰期
  3. 首发率: 如果该球员在历史lineup中常出现 → 核心
  4. 综合得分 = 位置系数 × 年龄系数
"""

import requests, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

THE_KEY = os.environ.get('THE_KEY', '') or os.environ.get('THE_STATS_KEY', 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww')

HEADERS = {"Authorization": f"Bearer {THE_KEY}"}
BASE = "https://api.thestatsapi.com/api/football"
DATA_DIR = "/root/data"
OUTPUT = f"{DATA_DIR}/star_players.json"

# === 获取48支世界杯球队 ===
def get_wc_teams():
    url = f"{BASE}/competitions/comp_6107/seasons/sn_118868/standings"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"Standings error: {r.status_code}")
        return []
    teams = {}
    for row in r.json().get('data', []):
        team = row.get('team', {})
        tid = team.get('id')
        tname = team.get('name')
        group = row.get('group_label', '')
        if tid and tname and tid not in teams:
            teams[tid] = {'name': tname, 'group': group}
    print(f"  WC teams: {len(teams)}")
    return teams

# === 拉取每队的完整阵容 ===
def fetch_players(team_id, team_name):
    """Return list of player dicts"""
    url = f"{BASE}/teams/{team_id}/players"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json()
            players = data.get('data', data)
            if isinstance(players, list):
                return players
    except Exception as e:
        print(f"  Error {team_name}: {e}")
    return []

# === 核心球员评分 ===
# 位置权重: F(前锋)=1.0, M(中场)=0.7, D(后卫)=0.4, G(门将)=0.1
POS_WEIGHT = {'F': 1.0, 'M': 0.7, 'D': 0.4, 'G': 0.1, None: 0.3}

def calc_star_score(player):
    """计算球员核心度分数 (0~1)"""
    pos = player.get('position', '')
    age = player.get('age', 25)
    
    pos_w = POS_WEIGHT.get(pos, 0.3)
    
    # 年龄权重: 24-30岁巅峰期
    if 24 <= age <= 30:
        age_w = 1.0
    elif 20 <= age <= 23 or 31 <= age <= 33:
        age_w = 0.7
    elif 17 <= age <= 19 or 34 <= age <= 36:
        age_w = 0.4
    else:
        age_w = 0.2
    
    # 综合得分
    score = pos_w * age_w * 0.7 + pos_w * 0.3
    
    return round(score, 3)

def is_starter_caliber(player):
    """判断是否为大概率首发球员"""
    score = calc_star_score(player)
    pos = player.get('position', '')
    # 前锋+中场得分>0.5是首发达人, 后卫>0.3, 门将不参与此判断
    if pos == 'G':
        return False  # 门将单独处理
    thresholds = {'F': 0.5, 'M': 0.4, 'D': 0.25}
    return score >= thresholds.get(pos, 0.3)

def is_star(player):
    """判断是否为绝对核心 (缺阵影响大)"""
    score = calc_star_score(player)
    pos = player.get('position', '')
    if pos == 'F' and score >= 0.7:
        return True
    if pos == 'M' and score >= 0.6:
        return True
    if pos == 'D' and score >= 0.5:
        return True
    return False

# === 主流程 ===
def main():
    print("=" * 60)
    print("  Fetching WC 2026 Team Squads")
    print("=" * 60)
    
    teams = get_wc_teams()
    if not teams:
        print("No teams found!")
        return
    
    # 并发拉取所有球队的阵容
    all_squads = {}
    lock = Lock()
    
    def fetch_one(tid):
        info = teams[tid]
        players = fetch_players(tid, info['name'])
        with lock:
            all_squads[tid] = {'info': info, 'players': players}
        return len(players)
    
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(fetch_one, list(teams.keys())))
    
    total_players = sum(results)
    print(f"  Total players fetched: {total_players}")
    
    # 构建 star_players.json
    star_data = {}
    for tid in teams:
        entry = all_squads[tid]
        info = entry['info']
        players = entry['players']
        
        # 标记每个球员
        enriched = []
        for p in players:
            player_rec = {
                'id': p.get('id', ''),
                'name': p.get('name', ''),
                'position': p.get('position', ''),
                'age': p.get('age', 0),
                'nationality': p.get('nationality', ''),
                'club': p.get('current_team', {}).get('name', '') if isinstance(p.get('current_team'), dict) else '',
                'star_score': calc_star_score(p),
                'is_starter': is_starter_caliber(p),
                'is_star': is_star(p),
            }
            enriched.append(player_rec)
        
        # 按核心度排序
        enriched.sort(key=lambda x: x['star_score'], reverse=True)
        
        # 提取 star_ids (绝对核心)
        star_ids = [p['id'] for p in enriched if p['is_star']]
        
        # 提取 starter_ids (大概率首发)
        starter_ids = [p['id'] for p in enriched if p['is_starter']]
        
        # 按位置分组
        by_pos = {'F': [], 'M': [], 'D': [], 'G': []}
        for p in enriched:
            by_pos.get(p['position'], []).append(p['name'])
        
        star_data[tid] = {
            'name': info['name'],
            'group': info['group'],
            'squad_size': len(enriched),
            'star_ids': star_ids,
            'starter_ids': starter_ids,
            'players': enriched,
            'by_position': {k: v for k, v in by_pos.items() if v},
        }
        
        # 统计信息
        n_stars = len(star_ids)
        n_starters = len(starter_ids)
        print(f"  {info['name']:20s} ({info['group']}): {len(players):2d} players | "
              f"{n_stars} stars, {n_starters} starters")
    
    # 保存
    with open(OUTPUT, 'w') as f:
        json.dump(star_data, f, indent=2)
    print(f"\n  Saved: {OUTPUT} ({os.path.getsize(OUTPUT)/1024:.0f} KB)")
    print(f"  Total teams: {len(star_data)}")
    print(f"  Total players: {total_players}")
    
    # 统计
    star_total = sum(len(v['star_ids']) for v in star_data.values())
    starter_total = sum(len(v['starter_ids']) for v in star_data.values())
    print(f"  Stars: {star_total} | Starters: {starter_total}")

if __name__ == '__main__':
    main()
