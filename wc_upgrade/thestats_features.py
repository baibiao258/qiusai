"""
thestats_features.py — TheStatsAPI 特征模块 v2 (球员数据库驱动)
===============================================================
功能:
  1. 加载 star_players.json (全48队阵容+核心标记)
  2. detect_rotation() — 基于首发vs核心名单检测轮换
  3. adjust_with_lineups() — 后处理调幅 (最大20%概率偏移)
  4. 回退机制: lineup无队名时通过球员名反向推断

数据源:
  /root/data/star_players.json   — 48队阵容+核心评分 (fetch_team_squads.py生成)
  /root/data/thestats_lineups.json — 当日赛前首发 (cron抓取)
"""

import json, os, math, re, unicodedata
import numpy as np

DATA_DIR = "/root/data"
STAR_PATH = f"{DATA_DIR}/star_players.json"
LINEUPS_PATH = f"{DATA_DIR}/thestats_lineups.json"
TEAM_STATS_PATH = f"{DATA_DIR}/thestats_team_stats.json"

# ── 名字归一化 (去重音/空格/大小写) ──
_NORM_CACHE = {}

def _normalize_name(name):
    """标准化球员名用于匹配: 去重音→小写→去空格"""
    if name in _NORM_CACHE:
        return _NORM_CACHE[name]
    # 去掉重音符号
    nfkd = unicodedata.normalize('NFKD', str(name))
    ascii_str = nfkd.encode('ascii', 'ignore').decode('ascii')
    # 小写 + 去多余空格
    norm = re.sub(r'\s+', '', ascii_str.lower())
    _NORM_CACHE[name] = norm
    return norm


# ═══════════════════════════════════════
# 1. 加载星球员数据库
# ═══════════════════════════════════════
_STAR_DATA_CACHE = None

def _load_star_data(force_reload=False):
    """加载并缓存 star_players.json"""
    global _STAR_DATA_CACHE
    if _STAR_DATA_CACHE is not None and not force_reload:
        return _STAR_DATA_CACHE
    if not os.path.exists(STAR_PATH):
        return None
    with open(STAR_PATH) as f:
        _STAR_DATA_CACHE = json.load(f)
    return _STAR_DATA_CACHE


def build_team_name_index():
    """
    构建 team_name → team_id 索引
    同时构建 team_name → {star_name_set, starter_name_set}
    """
    star_data = _load_star_data()
    if not star_data:
        return {}, {}
    
    name2id = {}
    name2stars = {}
    name2starters = {}
    
    for tid, v in star_data.items():
        name = v['name']
        name2id[name] = tid
        
        # 核心球员名集合 (用于轮换检测)
        star_names = set()
        starter_names = set()
        for p in v['players']:
            if p['is_star']:
                star_names.add(_normalize_name(p['name']))
            if p['is_starter']:
                starter_names.add(_normalize_name(p['name']))
        name2stars[name] = star_names
        name2starters[name] = starter_names
    
    return name2id, name2stars, name2starters


# ═══════════════════════════════════════
# 2. 球场匹配 (名字→队)
# ═══════════════════════════════════════
def _infer_team_from_names(players_names, name2stars, name2id):
    """
    通过首发名单反向推断是哪支队
    返回: team_name 或 None
    
    方法: 统计每个队的星球员名覆盖最多的匹配
    """
    norm_names = {_normalize_name(n) for n in players_names}
    best_team = None
    best_score = 0
    
    for tname, star_set in name2stars.items():
        overlap = len(norm_names & star_set)
        if overlap > best_score:
            best_score = overlap
            best_team = tname
    
    # 至少命中3个星球员才算可靠匹配
    if best_score >= 3:
        return best_team
    return None


# ═══════════════════════════════════════
# 3. 轮换检测核心逻辑
# ═══════════════════════════════════════
def compute_rotation_penalty(team_name, starter_names, star_data):
    """
    计算轮换惩罚系数
    
    输入:
        team_name: 球队名 (中文/英文均可, 会用索引匹配)
        starter_names: list[str] 本场首发球员名
        star_data: star_players.json 的解析结果 (从加载函数获取)
    
    返回:
        penalty: float (0.0 = 无轮换, ~0.20 = 最大20%)
        detail: dict 诊断信息
    """
    if not star_data or not starter_names:
        return 0.0, {'reason': 'no_data'}
    
    # 找队名
    _, name2stars, name2starters = build_team_name_index()
    name2id, _, _ = name2stars, name2starters, _  # unpack properly
    
    # 重建索引避免重复调用
    name2id, name2star_set, name2starter_set = build_team_name_index()
    
    # 找匹配球队 (直接匹配或模糊匹配)
    tid = name2id.get(team_name)
    if not tid:
        # 尝试部分匹配
        for name in name2id:
            if team_name.lower() in name.lower() or name.lower() in team_name.lower():
                tid = name2id[name]
                team_name = name  # 标准化
                break
    
    if not tid:
        return 0.0, {'reason': f'team_not_found: {team_name}'}
    
    star_set = name2star_set.get(team_name, set())
    starter_set = name2starter_set.get(team_name, set())
    
    if not star_set and not starter_set:
        return 0.0, {'reason': 'no_star_data'}
    
    # 归一化首发名单
    norm_starters = {_normalize_name(n) for n in starter_names}
    
    # 核心球员缺失检测
    # star = 绝对核心 (F/M高分)
    missing_stars = star_set - norm_starters
    n_missing = len(missing_stars)
    n_stars = len(star_set)
    
    # 总体首发预期球员缺失检测
    # starter = 大概率首发 (含star+非star常用球员)
    missing_starters = starter_set - norm_starters
    n_starters = len(starter_set)
    
    # --- 惩罚计算 ---
    # 宽容度: 允许2名星球员轮换 (正常杯赛轮换)
    # 从第3个缺失开始每多1人+3.5% penalty
    excess_missing = max(0, n_missing - 2)
    
    # 线性渐近: 3人→3.5%, 5人→10.5%, 7人→17.5%, 8人+→20%
    penalty = min(max(0, excess_missing * 0.035), 0.20)
    
    # 明确哪些星球员缺阵
    missing_star_names = [n for n in star_set if n not in norm_starters]
    # 反查原始名字
    star_players = []
    for tid_inner, v in _load_star_data().items():
        if v['name'] == team_name:
            for p in v['players']:
                if p['is_star'] and _normalize_name(p['name']) in missing_star_names:
                    star_players.append(p['name'])
            break
    
    detail = {
        'team': team_name,
        'stars_total': n_stars,
        'stars_missing': n_missing,
        'stars_missing_names': star_players[:5],
        'excess_missing': excess_missing,
        'penalty': round(penalty, 4),
    }
    
    return penalty, detail


# ═══════════════════════════════════════
# 4. 后处理调幅主入口
# ═══════════════════════════════════════
def detect_rotation(lineup_entry, star_data, team_name_side):
    """
    检测轮换 (外部接口)
    
    输入:
        lineup_entry: dict 单场lineup数据
        star_data: star_players.json loaded
        team_name_side: str 队名 + "||home" 或 "||away" 标记
                        格式: "TeamName||side"
    
    返回: penalty, detail 或 None/Nothing
    """
    if not lineup_entry or not lineup_entry.get('confirmed'):
        return None, None
    
    # 解析队名和侧边
    if '||' in team_name_side:
        team_name, side = team_name_side.split('||', 1)
    else:
        team_name = team_name_side
        side = 'home'
    
    starter_key = f'{side}_starters'
    starters = lineup_entry.get(starter_key, [])
    
    if not starters or len(starters) < 5:
        return None, None
    
    return compute_rotation_penalty(team_name, starters, star_data)


def load_lineups():
    """加载lineup缓存"""
    if os.path.exists(LINEUPS_PATH):
        with open(LINEUPS_PATH) as f:
            return json.load(f)
    return {}


def adjust_with_lineups(probs, home_team, away_team):
    """
    基于首发阵容的完整后处理调幅
    
    输入:
        probs: dict {'H':, 'D':, 'A':} 概率 (0~1)
        home_team, away_team: str 主/客队名
    
    返回:
        (adjusted_probs, adjustments_list)
    """
    star_data = _load_star_data()
    if not star_data:
        return probs, []
    
    lineups = load_lineups()
    if not lineups:
        return probs, []
    
    adjustments = []
    h_penalty = 0.0
    a_penalty = 0.0
    h_detail = {}
    a_detail = {}
    
    # 构建队名→星球员查询索引
    name2id, name2stars, name2starters = build_team_name_index()
    
    for mid, lu in lineups.items():
        # 优先用lineup里存的队名
        lu_home = lu.get('home_team', '')
        lu_away = lu.get('away_team', '')
        
        # 判断哪个对应主客队
        home_matched = False
        away_matched = False
        
        # 方法1: 直接队名匹配
        if lu_home and away_team and lu_home.lower() == home_team.lower():
            home_matched = True
        if lu_home and away_team and lu_home.lower() == away_team.lower():
            away_matched = True
        if lu_away and home_team and lu_away.lower() == home_team.lower():
            home_matched = True
        if lu_away and away_team and lu_away.lower() == away_team.lower():
            away_matched = True
        
        # 方法2: 双向检查 (阵容名单推断)
        if not home_matched:
            # 检查home_starters是否匹配主队
            h_starters = lu.get('home_starters', [])
            inferred_home = _infer_team_from_names(h_starters, name2stars, name2id)
            if inferred_home and (inferred_home.lower() == home_team.lower() or 
                                  home_team.lower() in inferred_home.lower()):
                home_matched = True
        
        if not away_matched:
            a_starters = lu.get('away_starters', [])
            inferred_away = _infer_team_from_names(a_starters, name2stars, name2id)
            if inferred_away and (inferred_away.lower() == away_team.lower() or
                                  away_team.lower() in inferred_away.lower()):
                away_matched = True
        
        if not home_matched and not away_matched:
            continue
        
        # 计算轮换惩罚
        if home_matched:
            h_starters = lu.get('home_starters', [])
            h_penalty, h_detail = compute_rotation_penalty(home_team, h_starters, star_data)
        
        if away_matched:
            a_starters = lu.get('away_starters', [])
            a_penalty, a_detail = compute_rotation_penalty(away_team, a_starters, star_data)
        
        if h_detail:
            adjustments.append({
                'source': 'thestats_lineup',
                'home_rotation': h_detail.get('penalty', 0),
                'stars_missing_home': h_detail.get('stars_missing_names', []),
                'away_rotation': a_detail.get('penalty', 0) if a_detail else 0,
                'stars_missing_away': a_detail.get('stars_missing_names', []) if a_detail else [],
            })
    
    # 应用调整
    if abs(h_penalty) > 0.001 or abs(a_penalty) > 0.001:
        new_probs = dict(probs)
        
        # 主队轮换 → 主胜下降, 平/客升
        if h_penalty > 0:
            shift = h_penalty * new_probs['H']
            new_probs['H'] = max(0.05, new_probs['H'] - shift)
            # redistribution
            remaining = new_probs['D'] + new_probs['A']
            if remaining > 0:
                ratio_d = new_probs['D'] / remaining
                ratio_a = new_probs['A'] / remaining
                new_probs['D'] += shift * ratio_d
                new_probs['A'] += shift * ratio_a
            else:
                new_probs['D'] += shift * 0.5
                new_probs['A'] += shift * 0.5
        
        # 客队轮换 → 客胜下降, 主/平升
        if a_penalty > 0:
            shift = a_penalty * new_probs['A']
            new_probs['A'] = max(0.05, new_probs['A'] - shift)
            remaining = new_probs['H'] + new_probs['D']
            if remaining > 0:
                ratio_h = new_probs['H'] / remaining
                ratio_d = new_probs['D'] / remaining
                new_probs['H'] += shift * ratio_h
                new_probs['D'] += shift * ratio_d
            else:
                new_probs['H'] += shift * 0.5
                new_probs['D'] += shift * 0.5
        
        # 重新归一化确保加起来=1.0
        total = new_probs['H'] + new_probs['D'] + new_probs['A']
        if total > 0:
            for k in ['H', 'D', 'A']:
                new_probs[k] /= total
        
        if not adjustments:
            adjustments.append({
                'source': 'thestats_lineup',
                'home_rotation': h_penalty,
                'stars_missing_home': h_detail.get('stars_missing_names', []) if h_detail else [],
                'away_rotation': a_penalty,
                'stars_missing_away': a_detail.get('stars_missing_names', []) if a_detail else [],
            })
        
        return new_probs, adjustments
    
    return probs, adjustments


def load_team_stats():
    """加载缓存的球队统计"""
    if os.path.exists(TEAM_STATS_PATH):
        with open(TEAM_STATS_PATH) as f:
            return json.load(f)
    return {}


def get_form_string(team_name):
    """获取球队近期 form 字符串"""
    stats = load_team_stats()
    for tid, v in stats.items():
        if v.get('name', '') == team_name:
            return v.get('form', '')
    return ''


def compute_form_features(team_name):
    """从 TheStatsAPI 统计计算 form 特征向量"""
    stats = load_team_stats()
    for tid, v in stats.items():
        if v.get('name', '') == team_name:
            mp = v.get('mp', 0)
            if mp < 2:
                return None
            win_rate = v.get('w', 0) / max(mp, 1)
            avg_gf = v.get('gf', 0) / max(mp, 1)
            avg_ga = v.get('ga', 0) / max(mp, 1)
            form_str = v.get('form', '')
            form_score = 0.5
            if form_str and len(form_str) >= 3:
                scores = []
                for ch in form_str:
                    if ch == 'W': scores.append(1)
                    elif ch == 'D': scores.append(0.5)
                    elif ch == 'L': scores.append(0)
                if scores:
                    form_score = sum(scores) / len(scores)
            return [win_rate, avg_gf, avg_ga, form_score]
    return None


# ═══════════════════════════════════════
# 5. 自测
# ═══════════════════════════════════════
if __name__ == '__main__':
    print("=== TheStatsAPI Features v2 (Star DB) ===\n")
    
    star_data = _load_star_data()
    if star_data:
        print(f"Star DB: {len(star_data)} teams, {sum(len(v['players']) for v in star_data.values())} players")
    else:
        print("Star DB: NOT FOUND — run fetch_team_squads.py first")
    
    lineups = load_lineups()
    if lineups:
        print(f"Lineups: {len(lineups)} matches cached")
        for mid, lu in lineups.items():
            ht = lu.get('home_team', '?')
            at = lu.get('away_team', '?')
            hf = lu.get('home_formation', '?')
            af = lu.get('away_formation', '?')
            print(f"  {ht} vs {at} | {hf} vs {af} | confirmed={lu.get('confirmed')}")
            
            # Test rotation detection
            h_starters = lu.get('home_starters', [])
            if h_starters and star_data:
                penalty, detail = compute_rotation_penalty(ht, h_starters, star_data)
                print(f"    Home rotation: penalty={penalty:.3f} missing={detail.get('stars_missing_names', [])}")
                a_starters = lu.get('away_starters', [])
                penalty2, detail2 = compute_rotation_penalty(at, a_starters, star_data)
                print(f"    Away rotation: penalty={penalty2:.3f} missing={detail2.get('stars_missing_names', [])}")
    else:
        print("Lineups: none cached (wait for cron or fetch manually)")
    
    # Test adjust_with_lineups
    print("\n--- adjust_with_lineups test ---")
    test_probs = {'H': 0.45, 'D': 0.28, 'A': 0.27}
    result, adj = adjust_with_lineups(test_probs, "Sweden", "Tunisia")
    if adj:
        print(f"  Before: {test_probs}")
        print(f"  After:  {result}")
        for a in adj:
            print(f"  Adjustment: {a}")
    else:
        print("  No adjustment (no lineup match or no rotation)")
