#!/usr/bin/env python3
"""
thestats_advanced_features.py — TheStatsAPI 高阶特征工程模块
==========================================================
每天预加载三个维度的高阶特征并缓存为 JSON, 在 _try_hybrid_predict 中
拼接到 Numpy 数组末尾, 让 XGBoost 模型看见"暗线"。

数据源:
  - /football/matches/{id}/odds        → Pinnacle/Bet365 隐含概率 (3维)
  - /football/matches/{id}/stats       → 近5场球队压制力聚合 (5维)
  - /football/matches/{id}             → 裁判+巡场数据 (5维)
  Total: 13维

使用方式:
  from thestats_advanced_features import get_all_advanced_features
  feat = get_all_advanced_features(match_id, target_date)
  # 返回 [float]*13 (无数据时全0)
"""

import requests, json, os, math
from datetime import datetime, date as dt_date
from collections import defaultdict

API_KEY = "fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww"
BASE = "https://api.thestatsapi.com/api"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
DATA_DIR = "/root/data"

# ─── 缓存路径 ───
CACHE_FILE = os.path.join(DATA_DIR, "thestats_adv_cache.json")
REFEREE_DB = os.path.join(DATA_DIR, "referee_strictness.json")
TEAM_STATS_CACHE = os.path.join(DATA_DIR, "team_recent_stats.json")

# ─── 内部缓存 (本次进程) ───
_adv_cache = {}       # match_id -> 13-dim list
_referee_db = {}      # referee_name -> {avg_yellow, avg_red, match_count}
_team_stats_cache = {} # team_name -> recent 5 match stats aggregate

def _api_get(path, timeout=10):
    """安全 API GET"""
    url = f"{BASE}{path}" if not path.startswith('http') else path
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("data")
        return None
    except:
        return None

# ═══════════════════════════════════════════════
# 维度 1: 国际市场隐含概率 (3维)
# ═══════════════════════════════════════════════

def _get_odds(match_id):
    """获取 Pinnacle 开盘赔率 → [prob_h, prob_d, prob_a]"""
    data = _api_get(f"/football/matches/{match_id}/odds")
    if not data:
        return [0.0, 0.0, 0.0]
    
    bookmakers = data.get("bookmakers", [])
    # 优先 Pinnacle, 其次 Bet365, 再次 Betfair
    preferred = ["Pinnacle", "Bet365", "Betfair Exchange"]
    for pname in preferred:
        for b in bookmakers:
            if b.get("bookmaker") == pname:
                mo = b.get("markets", {}).get("match_odds", {})
                home_open = mo.get("home", {}).get("opening") or mo.get("home", {}).get("last_seen")
                draw_open = mo.get("draw", {}).get("opening") or mo.get("draw", {}).get("last_seen")
                away_open = mo.get("away", {}).get("opening") or mo.get("away", {}).get("last_seen")
                if home_open and draw_open and away_open:
                    h_odds = float(home_open)
                    d_odds = float(draw_open)
                    a_odds = float(away_open)
                    # 去抽水 (remove margin)
                    implied = 1.0/h_odds + 1.0/d_odds + 1.0/a_odds
                    if implied > 0:
                        return [
                            (1.0/h_odds) / implied,
                            (1.0/d_odds) / implied,
                            (1.0/a_odds) / implied
                        ]
    return [0.0, 0.0, 0.0]


# ═══════════════════════════════════════════════
# 维度 2: 球队近期压制力聚合 (5维)
# ═══════════════════════════════════════════════
# 提取主客队近5场比赛的技术统计均值:
#   [SoT比率差, xG均值差, 控球率差, 进攻压制系数, 防守脆弱系数]

def _get_team_recent_stats(team_name, n=5):
    """获取球队近n场比赛的聚合统计 (从训练数据或API)"""
    cache_key = f"{team_name}_{n}"
    if cache_key in _team_stats_cache:
        return _team_stats_cache[cache_key]
    
    # 先从训练数据中查找 (更快)
    try:
        with open(f"{DATA_DIR}/thestats_training_data.json") as f:
            all_matches = json.load(f)
    except:
        all_matches = []
    
    # 过滤该球队最近的比赛 (作为主队或客队)
    team_matches = []
    for m in all_matches:
        if m.get("home") == team_name or m.get("away") == team_name:
            team_matches.append(m)
    
    # 按日期排序, 取最近n场
    team_matches.sort(key=lambda x: x.get("date", ""), reverse=True)
    recent = team_matches[:n]
    
    if len(recent) < 1:
        _team_stats_cache[cache_key] = None
        return None
    
    # 对于这n场比赛, 尝试获取 stats
    total_sot, total_xg, total_poss, total_shots = 0.0, 0.0, 0.0, 0.0
    total_dangerous = 0.0
    counted = 0
    
    for m in recent:
        mid = m.get("match_id")
        if not mid:
            continue
        stats = _api_get(f"/football/matches/{mid}/stats")
        if not stats:
            continue
        
        # 提取该球队的数据
        is_home = m.get("home") == team_name
        side = "home" if is_home else "away"
        opp_side = "away" if is_home else "home"
        
        overview = stats.get("overview") or {}
        shots = stats.get("shots") or {}
        attack = stats.get("attack") or {}
        np_xg = stats.get("np_expected_goals") or {}
        
        def _safe_get(d, *keys, default=0):
            """Safe nested dict access, handling None at any level."""
            for k in keys:
                if not isinstance(d, dict):
                    return default
                d = d.get(k)
                if d is None:
                    return default
            return d if d is not None else default
        
        # 射正数
        sot = _safe_get(overview, "shots_on_target", "all", side)
        opp_sot = _safe_get(overview, "shots_on_target", "all", opp_side)
        
        # xG
        xg = _safe_get(overview, "expected_goals", "all", side)
        
        # 控球率
        poss = _safe_get(overview, "ball_possession", "all", side)
        
        # 总射门
        ts = _safe_get(overview, "total_shots", "all", side)
        
        # 危险进攻: 用 shots_inside_box + big_chances 代理
        shots_ibox = _safe_get(shots, "shots_inside_box", "all", side)
        big_chances_val = _safe_get(overview, "big_chances", "all", side)
        
        total_sot += sot
        total_xg += xg
        total_poss += poss
        total_shots += ts
        total_dangerous += (shots_ibox or 0) + (big_chances_val or 0)
        counted += 1
    
    if counted == 0:
        _team_stats_cache[cache_key] = None
        return None
    
    result = {
        "avg_sot": total_sot / counted,
        "avg_xg": total_xg / counted,
        "avg_poss": total_poss / counted,
        "avg_shots": total_shots / counted,
        "avg_dangerous": total_dangerous / counted,
        "count": counted,
    }
    _team_stats_cache[cache_key] = result
    return result


def _build_pressure_features(home, away):
    """构造压制力特征 (5维)"""
    hs = _get_team_recent_stats(home)
    aws = _get_team_recent_stats(away)
    
    if hs is None or aws is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # 1. 射正比率差: 主队射正比率 vs 客队被射正比率
    sot_diff = (hs["avg_sot"] - aws["avg_sot"]) / max(hs["avg_sot"] + aws["avg_sot"], 0.1)
    
    # 2. xG均值差
    xg_diff = hs["avg_xg"] - aws["avg_xg"]
    
    # 3. 控球率差
    poss_diff = (hs["avg_poss"] - aws["avg_poss"]) / max(hs["avg_poss"] + aws["avg_poss"], 1)
    
    # 4. 进攻压制系数: 主队危险进攻 / 客队危险进攻
    danger_ratio = hs["avg_dangerous"] / max(aws["avg_dangerous"], 0.5)
    danger_ratio = min(max(danger_ratio, 0.1), 10.0)  # 截断
    
    # 5. 防守脆弱系数: 客队场均被射正 vs 联赛均值 (用绝对值)
    defend_ratio = aws["avg_sot"] / max(hs["avg_sot"], 0.5)
    defend_ratio = min(max(defend_ratio, 0.1), 10.0)
    
    return [round(sot_diff, 4), round(xg_diff, 4), round(poss_diff, 4),
            round(danger_ratio, 4), round(defend_ratio, 4)]


# ═══════════════════════════════════════════════
# 维度 3: 裁判执法尺度 + 得牌预期 (5维)
# ═══════════════════════════════════════════════

def _load_referee_db():
    """加载裁判数据库"""
    global _referee_db
    if _referee_db:
        return _referee_db
    if os.path.exists(REFEREE_DB):
        try:
            with open(REFEREE_DB) as f:
                _referee_db = json.load(f)
                return _referee_db
        except:
            pass
    _referee_db = {"_default": {"avg_yellow": 3.5, "avg_red": 0.15, "match_count": 0}}
    return _referee_db


def get_referee_data(match_id):
    """获取本场裁判历史出牌数据"""
    detail = _api_get(f"/football/matches/{match_id}")
    if not detail:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    ref_info = detail.get("referee")
    if not ref_info:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    ref_name = ref_info.get("name", "")
    if not ref_name:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    db = _load_referee_db()
    ref_data = db.get(ref_name, db.get("_default", {"avg_yellow": 3.5, "avg_red": 0.15, "match_count": 10}))
    
    # 裁判严厉指数: 归一化到0-1 (5张黄牌=极端严厉)
    ref_strictness = min(ref_data.get("avg_yellow", 3.5) / 5.0, 1.0)
    
    return [
        round(ref_strictness, 4),              # referee_strictness (0-1)
        round(ref_data.get("avg_yellow", 2.5), 2),  # 场均黄牌
        round(ref_data.get("avg_red", 0.15), 3),    # 场均红牌
        0.5,  # home_card_tendency (暂缺)
        0.5,  # away_card_tendency
    ]


# ═══════════════════════════════════════════════
# 主入口: 获取全部高阶特征 (13维)
# ═══════════════════════════════════════════════

_ADV_CACHE = {}

def get_all_advanced_features(match_id, target_date=None, home_team=None, away_team=None):
    """
    返回13维高阶特征向量.
    
    Args:
        match_id: TheStatsAPI match ID
        target_date: 比赛日期 (字符串 "YYYY-MM-DD" 或 date 对象)
        home_team: 主队名 (可选, 用于压制力计算)
        away_team: 客队名 (可选, 用于压制力计算)
    
    Returns: [13 floats] — 无数据时返回全0向量
    """
    if match_id in _ADV_CACHE:
        return _ADV_CACHE[match_id]
    
    # 维度1: 市场隐含概率 (3维)
    odds_feat = _get_odds(match_id)
    
    # 维度2: 球队压制力 (5维)
    pressure_feat = _build_pressure_features(home_team, away_team) if home_team and away_team else [0.0]*5
    
    # 维度3: 裁判特征 (5维)
    referee_feat = get_referee_data(match_id)
    
    # 合体: 3 + 5 + 5 = 13
    result = odds_feat + pressure_feat + referee_feat
    _ADV_CACHE[match_id] = result
    return result


# ═══════════════════════════════════════════════
# 批量缓存: 每天预加载当天比赛的13维特征
# ═══════════════════════════════════════════════

def preload_today_matches():
    """预加载今天所有比赛的高阶特征"""
    today = dt_date.today().isoformat()
    print(f"📡 预加载 {today} 高阶特征...")
    
    # 获取今天的比赛
    r = requests.get(f"{BASE}/football/matches?date_from={today}&date_to={today}&per_page=100",
                     headers=HEADERS, timeout=30)
    matches = r.json().get("data", [])
    print(f"   今天共 {len(matches)} 场比赛")
    
    cache = {}
    for m in matches:
        mid = m["id"]
        ht = m.get("home_team", {}).get("name", "")
        at = m.get("away_team", {}).get("name", "")
        feat = get_all_advanced_features(mid, today, ht, at)
        cache[mid] = {
            "match_id": mid,
            "home": ht,
            "away": at,
            "features": feat,
            "cached_at": datetime.now().isoformat(),
        }
        print(f"   {ht} vs {at}: {feat}")
    
    # 写入缓存
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"✅ 缓存 {len(cache)} 场比赛到 {CACHE_FILE}")
    return cache


def load_adv_cache():
    """加载已缓存的比赛特征"""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except:
        return {}


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ("preload", "build"):
        preload_today_matches()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        mid = sys.argv[2] if len(sys.argv) > 2 else "mt_209798753"
        ht = sys.argv[3] if len(sys.argv) > 3 else "Sweden"
        at = sys.argv[4] if len(sys.argv) > 4 else "Tunisia"
        feat = get_all_advanced_features(mid, "2026-06-15", ht, at)
        print(f"🔬 {ht} vs {at} (id={mid}):")
        print(f"   市场隐含:   H={feat[0]:.1%} D={feat[1]:.1%} A={feat[2]:.1%}")
        print(f"   压制力:     SoT_diff={feat[3]:.3f} xG_diff={feat[4]:.3f} "
              f"Poss_diff={feat[5]:.3f} Dang_ratio={feat[6]:.3f} Def_ratio={feat[7]:.3f}")
        print(f"   裁判尺度:   strict={feat[8]:.3f} avg_yc={feat[9]:.2f} avg_rc={feat[10]:.3f}")
        print(f"   得牌倾向:   home={feat[11]:.2f} away={feat[12]:.2f}")
    else:
        print("用法: python3 thestats_advanced_features.py [preload|test <match_id> <home> <away>]")
