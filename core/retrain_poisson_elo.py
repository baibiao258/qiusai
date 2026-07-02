#!/usr/bin/env python3
"""
retrain_poisson_elo.py — 5年全史 Elo + 时间衰减 Poisson λ 训练
===============================================================

数据范围: 2021-01-01 至今天
Elo: 从 1500 起步，严格按时间序滚动更新
Poisson λ: 指数时间衰减 (半衰期 1.5 年) + 最近 30 场硬截断
输出: /root/data/poisson_elo_prior.json
"""

import requests
import json
import math
from datetime import date, datetime, timedelta
from collections import defaultdict
import os

KEY = "fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww"
HDR = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api"

# 目标赛事: World Cup (含预选/正赛/友谊赛) + 主流联赛
TARGET_COMPETITIONS = {
    "comp_6107": "FIFA World Cup",
    "comp_29967": "International Friendly",
    "comp_3039": "Premier League",
    "comp_4643": "Bundesliga",
    "comp_8814": "LaLiga",
    "comp_0256": "Ligue 1",
    "comp_8385": "Liga Portugal Betclic",
    "comp_3809": "Eredivisie",
    "comp_8321": "Championship",
    "comp_2949": "EURO",
    "comp_3759": "EURO Qualification",
    "comp_5749": "Copa America",
    "comp_9799": "MLS",
    "comp_6240": "J1 League",
    "comp_1646": "K League 1",
    "comp_4795": "Brasileirão Série A",
    "comp_4893": "Austrian Bundesliga",
    "comp_84287": "Egyptian Premier League",
    "comp_9711": "Ekstraklasa",
    "comp_1992": "Eliteserien",
    "comp_1941": "HNL (Croatia)",
    "comp_19603": "Indian Super League",
}

# 历史拉取窗口
START_DATE = "2021-01-01"
END_DATE = date.today().isoformat()

# Poisson λ 时间衰减参数
HALF_LIFE_DAYS = 1.5 * 365.25  # 1.5 年半衰期
MAX_RECENT_MATCHES = 30         # 硬截断：最多用最近 30 场

# Elo 参数
ELO_INIT = 1500
ELO_K = 20


def pull_matches_for_comp(comp_id: str, start: str, end: str) -> list:
    """分页拉取某赛事的完场比赛"""
    all_matches = []
    page = 1
    per_page = 100
    
    while True:
        url = f"{BASE}/football/matches?competition_id={comp_id}&status=finished&date_from={start}&date_to={end}&per_page={per_page}&page={page}"
        r = requests.get(url, headers=HDR, timeout=30)
        if r.status_code != 200:
            print(f"  ❌ {comp_id} page {page}: HTTP {r.status_code}")
            break
        data = r.json().get("data", [])
        if not data:
            break
        all_matches.extend(data)
        if len(data) < per_page:
            break
        page += 1
    return all_matches


def extract_match_info(m: dict) -> tuple | None:
    """从 match dict 提取 (date, home, away, hg, ag)"""
    try:
        utc = m.get("utc_date", "")
        if not utc:
            return None
        d = utc[:10]  # YYYY-MM-DD
        
        h = m["home_team"]["name"]
        a = m["away_team"]["name"]
        hg = int(m["score"]["home"])
        ag = int(m["score"]["away"])
        
        return (d, h, a, hg, ag)
    except Exception:
        return None


def time_weight(match_date: str, ref_date: str) -> float:
    """指数时间衰减权重: exp(-ln2 * days_diff / half_life)"""
    try:
        md = datetime.fromisoformat(match_date)
        rd = datetime.fromisoformat(ref_date)
        days = (rd - md).days
        if days < 0:
            return 1.0  # 未来比赛不衰减（理论不应出现）
        return math.exp(-math.log(2) * days / HALF_LIFE_DAYS)
    except Exception:
        return 1.0


def main():
    print(f"🚀 开始 5 年全史训练 ({START_DATE} → {END_DATE})")
    print(f"   目标赛事: {len(TARGET_COMPETITIONS)} 个")
    print(f"   Poisson λ: 指数衰减(半衰期=1.5年) + 近期30场截断")
    print(f"   Elo: {ELO_INIT} 起步, K={ELO_K}, 严格时间序滚动")
    
    # 1) 拉取所有历史完场比赛
    all_matches = []
    for comp_id, comp_name in TARGET_COMPETITIONS.items():
        print(f"\n📡 拉取 {comp_name} ({comp_id})...")
        matches = pull_matches_for_comp(comp_id, START_DATE, END_DATE)
        print(f"   完场场次: {len(matches)}")
        for m in matches:
            info = extract_match_info(m)
            if info:
                d, h, a, hg, ag = info
                all_matches.append({
                    "date": d,
                    "home": h,
                    "away": a,
                    "hg": hg,
                    "ag": ag,
                    "comp": comp_name,
                })
    
    if not all_matches:
        print("❌ 无数据")
        return
    
    # 2) 按日期严格排序 (Elo 必须按时间序)
    all_matches.sort(key=lambda x: x["date"])
    print(f"\n✅ 总计 {len(all_matches)} 场完赛数据")
    print(f"   时间跨度: {all_matches[0]['date']} → {all_matches[-1]['date']}")
    
    # 3) 计算 Poisson λ (带时间衰减 + 近期截断)
    # 先按队伍分组收集所有比赛
    team_matches = defaultdict(list)  # team -> list of (date, gf, ga)
    for m in all_matches:
        team_matches[m["home"]].append((m["date"], m["hg"], m["ag"]))
        team_matches[m["away"]].append((m["date"], m["ag"], m["hg"]))
    
    print(f"\n🧮 计算 Poisson λ 先验...")
    lambda_prior = {}
    ref_date = END_DATE
    
    for team, matches in team_matches.items():
        if len(matches) < 5:
            continue
        
        # 按日期倒序，取最近 MAX_RECENT_MATCHES 场
        matches.sort(key=lambda x: x[0], reverse=True)
        recent = matches[:MAX_RECENT_MATCHES]
        
        # 加权平均
        w_gf = w_ga = w_sum = 0.0
        for d, gf, ga in recent:
            w = time_weight(d, ref_date)
            w_gf += w * gf
            w_ga += w * ga
            w_sum += w
        
        if w_sum > 0:
            lambda_prior[team] = {
                "lambda_home": round(w_gf / w_sum, 4),
                "lambda_away": round(w_ga / w_sum, 4),
                "n_matches": len(recent),
                "total_n": len(matches),
            }
    
    print(f"   输出球队数: {len(lambda_prior)}")
    
    # 4) Elo 严格时间序滚动
    print(f"\n⚽ Elo 滚动更新...")
    elo = defaultdict(lambda: ELO_INIT)
    
    for i, m in enumerate(all_matches):
        h, a = m["home"], m["away"]
        hg, ag = m["hg"], m["ag"]
        
        eh = elo[h]
        ea = elo[a]
        
        # 主场优势 +100 Elo (约等于 0.64 胜率)
        ph = 1 / (1 + 10 ** ((ea - eh - 100) / 400))
        
        if hg > ag:
            res = 1.0
        elif hg == ag:
            res = 0.5
        else:
            res = 0.0
        
        elo[h] += ELO_K * (res - ph)
        elo[a] += ELO_K * ((1 - res) - (1 - ph))
        
        if i % 500 == 0:
            print(f"   进度: {i+1}/{len(all_matches)} ({m['date']} {h} vs {a})")
    
    # 转换为普通 dict (Elo 保留 1 位小数)
    elo_dict = {t: round(v, 1) for t, v in elo.items()}
    print(f"   最终 Elo 球队数: {len(elo_dict)}")
    
    # 5) 保存主场优势系数 (联赛级)
    # 统计各联赛主场胜率
    home_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    for m in all_matches:
        comp = m["comp"]
        hg, ag = m["hg"], m["ag"]
        home_stats[comp]["total"] += 1
        if hg > ag:
            home_stats[comp]["wins"] += 1
    
    home_advantage = {}
    for comp, s in home_stats.items():
        if s["total"] >= 20:
            home_advantage[comp] = round(s["wins"] / s["total"], 4)
    
    # 6) 落盘
    output = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "total_matches": len(all_matches),
            "teams_with_elo": len(elo_dict),
            "teams_with_lambda": len(lambda_prior),
            "half_life_days": HALF_LIFE_DAYS,
            "max_recent_matches": MAX_RECENT_MATCHES,
            "elo_init": ELO_INIT,
            "elo_k": ELO_K,
        },
        "elo": elo_dict,
        "lambda_prior": lambda_prior,
        "home_advantage": home_advantage,
    }
    
    out_path = "/root/data/poisson_elo_prior.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完成! 已保存至 {out_path}")
    print(f"   文件大小: {os.path.getsize(out_path) / 1024:.1f} KB")
    
    # 示例输出
    print(f"\n📊 示例 Elo Top 10:")
    top = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for t, v in top:
        print(f"   {t}: {v}")
    
    return output


def incremental_update(new_matches: list) -> None:
    """
    增量更新: 传入新比赛列表 [{date, home, away, hg, ag, comp}, ...]
    仅更新 Elo 和 Poisson λ，不重算全量
    """
    if not new_matches:
        return
    
    # 读取当前 prior
    prior_path = "/root/data/poisson_elo_prior.json"
    if not os.path.exists(prior_path):
        print("⚠️ prior 不存在，需先全量训练")
        return
    
    with open(prior_path) as f:
        prior = json.load(f)
    
    elo = defaultdict(lambda: ELO_INIT, prior["elo"])
    lambda_data = prior["lambda_prior"]
    home_adv = prior["home_advantage"]
    
    # 按日期排序
    new_matches.sort(key=lambda x: x["date"])
    
    for m in new_matches:
        h, a = m["home"], m["away"]
        hg, ag = m["hg"], m["ag"]
        d = m["date"]
        comp = m.get("comp", "")
        
        # Elo 更新
        eh = elo[h]
        ea = elo[a]
        ph = 1 / (1 + 10 ** ((ea - eh - 100) / 400))
        
        if hg > ag:
            res = 1.0
        elif hg == ag:
            res = 0.5
        else:
            res = 0.0
        
        elo[h] += ELO_K * (res - ph)
        elo[a] += ELO_K * ((1 - res) - (1 - ph))
        
        # Poisson λ 增量更新 (指数移动平均，alpha 基于半衰期)
        # 简化: 重新计算该队最近 30 场 (需维护队伍比赛历史队列)
        # 这里只做 Elo 增量，λ 建议每周全量重算
        
        # 主场优势增量
        if comp and comp in home_adv:
            s = home_adv[comp]
            # 简化的在线更新
            home_adv[comp] = round(0.99 * s + 0.01 * (1 if hg > ag else 0), 4)
    
    # 保存
    prior["elo"] = {t: round(v, 1) for t, v in elo.items()}
    prior["home_advantage"] = home_adv
    prior["meta"]["updated_at"] = datetime.now().isoformat()
    prior["meta"]["total_matches"] += len(new_matches)
    
    with open(prior_path, "w") as f:
        json.dump(prior, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 增量更新完成: +{len(new_matches)} 场")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "incremental":
        # 供 cron 调用: 拉取昨天的比赛并增量更新
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        print(f"🔄 增量更新: {yesterday}")
        for comp_id in TARGET_COMPETITIONS:
            matches = pull_matches_for_comp(comp_id, yesterday, yesterday)
            if matches:
                new_m = []
                for m in matches:
                    info = extract_match_info(m)
                    if info:
                        d, h, a, hg, ag = info
                        new_m.append({"date": d, "home": h, "away": a, "hg": hg, "ag": ag, "comp": TARGET_COMPETITIONS[comp_id]})
                if new_m:
                    incremental_update(new_m)
    else:
        main()