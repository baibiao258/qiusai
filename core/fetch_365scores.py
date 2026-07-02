#!/usr/bin/env python3
"""
365scores 数据抓取器
- 抓取投票数据 (WhoWillWinReults) - 市场情绪
- 抓取趋势数据 (Trend) - 球队近期状态
- 输出与 500.com 赛程匹配的结果
"""

import requests
import json
import sys
from datetime import datetime

API_URL = "https://webws.365scores.com/data/games/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}

def fetch_365scores_data(sport_type=1):
    """抓取 365scores 数据"""
    params = {
        "lang": 1,
        "app-type": 1,
        "cid": 2,
        "sport-type": sport_type
    }
    
    try:
        resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ 抓取失败: {e}", file=sys.stderr)
        return None

def parse_trend(trend_list):
    """解析趋势数据: [胜, 平, 负, ?, ?] → 近期状态描述"""
    if not trend_list or len(trend_list) < 3:
        return "无数据"
    
    wins = trend_list[0]
    draws = trend_list[1]
    losses = trend_list[2]
    total = wins + draws + losses
    
    if total == 0:
        return "无比赛"
    
    # 胜率
    win_rate = wins / total
    
    if win_rate >= 0.8:
        status = "🔥 状态火热"
    elif win_rate >= 0.6:
        status = "✅ 状态良好"
    elif win_rate >= 0.4:
        status = "➡️ 状态一般"
    elif win_rate >= 0.2:
        status = "⚠️ 状态低迷"
    else:
        status = "🔴 状态糟糕"
    
    return f"{status} ({wins}胜{draws}平{losses}负)"

def extract_games(data, target_date=None, filter_sid=None):
    """提取比赛数据（含 Events/Venue/Lineups 等）

    Args:
        data: API 响应数据
        target_date: 过滤日期
        filter_sid: 只保留指定 SID（体育项目 ID, 1=足球）
    """
    if not data:
        return []

    # 联赛映射
    comps_map = {}
    for c in data.get("Competitions", []):
        comps_map[c["ID"]] = c["Name"]

    games = data.get("Games", [])
    results = []

    for g in games:
        # 体育项目过滤 (SID: 1=足球, 2=篮球, 3=网球, 7=棒球, 8=排球)
        if filter_sid is not None and g.get("SID") != filter_sid:
            continue
        # 提取队伍名
        comps = g.get("Comps", [])
        if len(comps) < 2:
            continue
        
        home = comps[0].get("Name", "Unknown")
        away = comps[1].get("Name", "Unknown")
        
        # 联赛名
        comp_name = comps_map.get(g.get("Comp"), "Unknown")
        
        # 时间
        stime = g.get("STime", "")
        
        # 日期过滤
        if target_date and target_date not in stime:
            continue
        
        # 投票数据
        votes = g.get("WhoWillWinReults", {})
        vote1 = votes.get("Vote1", 0)
        voteX = votes.get("VoteX", 0)
        vote2 = votes.get("Vote2", 0)
        total_votes = vote1 + voteX + vote2
        
        if total_votes > 0:
            vote_pct = {
                "home": round(vote1 / total_votes * 100, 1),
                "draw": round(voteX / total_votes * 100, 1),
                "away": round(vote2 / total_votes * 100, 1),
                "total": total_votes
            }
        else:
            vote_pct = None
        
        # 趋势数据
        trend_home = comps[0].get("Trend", [])
        trend_away = comps[1].get("Trend", [])
        
        # 人气排名
        pop_rank_home = comps[0].get("PopularityRank")
        pop_rank_away = comps[1].get("PopularityRank")
        
        # FIFA 排名 (从 Rankings 数组提取)
        def _get_fifa_rank(comp):
            rankings = comp.get("Rankings", [])
            for r in rankings:
                if r.get("Name") == "FIFA":
                    return r.get("Position")
            return None
        
        fifa_rank_home = _get_fifa_rank(comps[0])
        fifa_rank_away = _get_fifa_rank(comps[1])
        
        # 比赛状态
        is_finished = g.get("IsFinished", False)
        is_active = g.get("Active", False)
        
        if is_finished:
            status = "finished"
        elif is_active:
            status = "live"
        else:
            status = "upcoming"
        
        # 比分
        scrs = g.get("Scrs", [])
        score_home = int(scrs[0]) if len(scrs) > 0 and scrs[0] >= 0 else None
        score_away = int(scrs[1]) if len(scrs) > 1 and scrs[1] >= 0 else None
        
        # 半场比分 (Scrs[2:4] = HT score for SID=1)
        ht_home = int(scrs[2]) if len(scrs) > 2 and scrs[2] >= 0 else None
        ht_away = int(scrs[3]) if len(scrs) > 3 and scrs[3] >= 0 else None
        ht_score = f"{ht_home}:{ht_away}" if ht_home is not None else None
        
        # Winner (-1=平, 1=主, 2=客)
        winner = g.get("Winner", -2)
        winner_str = {1: "home", -1: "draw", 2: "away"}.get(winner, "unknown")
        
        game_id = g.get("ID")
        
        # Events (比赛事件)
        events = g.get("Events", [])
        goals_home = 0
        goals_away = 0
        yellow_cards_home = 0
        yellow_cards_away = 0
        red_cards_home = 0
        red_cards_away = 0
        
        for e in events:
            etype = e.get("Type")
            stype = e.get("SType")
            comp_num = e.get("Comp")
            
            if etype == 0:  # 进球
                if comp_num == 1:
                    goals_home += 1
                else:
                    goals_away += 1
            elif etype == 1:  # 红黄牌
                if stype == -1:  # 黄牌
                    if comp_num == 1:
                        yellow_cards_home += 1
                    else:
                        yellow_cards_away += 1
                else:  # 红牌
                    if comp_num == 1:
                        red_cards_home += 1
                    else:
                        red_cards_away += 1
        
        # Venue (场地)
        venue = g.get("Venue", {})
        venue_name = venue.get("Name") if venue else None
        
        # Attendance (出席人数)
        attendance = g.get("Attendance")
        
        # RedCardsCount (红牌数)
        red_cards_count = g.get("RedCardsCount", [])
        
        # Lineups (阵容)
        lineups = g.get("Lineups", [])
        avg_age_home = None
        avg_age_away = None
        
        if len(lineups) >= 2:
            # 主队
            players_home = lineups[0].get("Players", [])
            ages_home = [p.get("Age") for p in players_home if p.get("Age")]
            if ages_home:
                avg_age_home = round(sum(ages_home) / len(ages_home), 1)
            
            # 客队
            players_away = lineups[1].get("Players", [])
            ages_away = [p.get("Age") for p in players_away if p.get("Age")]
            if ages_away:
                avg_age_away = round(sum(ages_away) / len(ages_away), 1)
        
        # ── 2026-06-07 新增：临场元数据 (API 已返回, 之前未读) ──
        has_lineups = bool(g.get("HasLineups", False))
        lineups_status_text = g.get("LineupsStatusText") or None
        has_doubtful = bool(g.get("HasDoubtful", False))
        has_missing_players = bool(g.get("HasMissingPlayers", False))
        has_statistics = bool(g.get("HasStatistics", False))
        has_news = bool(g.get("HasNews", False))
        has_buzz = bool(g.get("HasBuzz", False))
        social_stats = g.get("SocialStats") or {}
        social_comments = social_stats.get("Comments", 0) if isinstance(social_stats, dict) else 0

        results.append({
            "id": game_id,
            "home": home,
            "away": away,
            "competition": comp_name,
            "time": stime,
            "status": status,
            "score": f"{score_home}:{score_away}" if score_home is not None else None,
            "votes": vote_pct,
            "trend_home": trend_home,
            "trend_away": trend_away,
            "trend_home_desc": parse_trend(trend_home),
            "trend_away_desc": parse_trend(trend_away),
            "pop_rank_home": pop_rank_home,
            "pop_rank_away": pop_rank_away,
            "fifa_rank_home": fifa_rank_home,
            "fifa_rank_away": fifa_rank_away,
            "venue": venue_name,
            "attendance": attendance,
            "score_ht": ht_score,
            "winner": winner_str,
            "goals_home": goals_home,
            "goals_away": goals_away,
            "yellow_cards_home": yellow_cards_home,
            "yellow_cards_away": yellow_cards_away,
            "red_cards_home": red_cards_home,
            "red_cards_away": red_cards_away,
            "avg_age_home": avg_age_home,
            "avg_age_away": avg_age_away,
            # 新增临场元数据
            "has_lineups": has_lineups,
            "lineups_status_text": lineups_status_text,
            "has_doubtful": has_doubtful,
            "has_missing_players": has_missing_players,
            "has_statistics": has_statistics,
            "has_news": has_news,
            "has_buzz": has_buzz,
            "social_comments": social_comments,
        })
    
    return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="365scores 数据抓取器")
    parser.add_argument("--date", help="过滤日期 (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--competition", help="过滤联赛关键词")
    args = parser.parse_args()
    
    # 抓取数据
    data = fetch_365scores_data()
    if not data:
        sys.exit(1)
    
    # 提取比赛
    target_date = args.date or datetime.now().strftime("%d-%m-%Y")
    games = extract_games(data, target_date)
    
    # 联赛过滤
    if args.competition:
        games = [g for g in games if args.competition.lower() in g["competition"].lower()]
    
    if args.json:
        print(json.dumps(games, ensure_ascii=False, indent=2))
    else:
        print(f"=== 365scores 数据 ({target_date}) ===")
        print(f"共 {len(games)} 场比赛\n")
        
        for g in games:
            print(f"⚽ {g['home']} vs {g['away']}")
            print(f"   联赛: {g['competition']} | 时间: {g['time']} | 状态: {g['status']}")
            
            if g["votes"]:
                v = g["votes"]
                print(f"   投票: 主{v['home']}% / 平{v['draw']}% / 客{v['away']}% ({v['total']}人)")
            
            print(f"   主队趋势: {g['trend_home_desc']}")
            print(f"   客队趋势: {g['trend_away_desc']}")
            print()

if __name__ == "__main__":
    main()
