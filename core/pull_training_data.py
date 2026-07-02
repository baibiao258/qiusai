#!/usr/bin/env python3
"""
pull_training_data.py — 全量特征训练数据拉取 (TheStatsAPI)
=============================================================
从 TheStatsAPI 拉取 2021-01-01 → 今日所有完场数据,
追加 Elo / Poisson λ 先验 + 高阶统计特征,
保存为 /root/data/thestats_training_data.json.

特点:
  - 断点续传 (checkpoint)
  - tqdm 进度条
  - dry-run 模式仅拉 50 场预览

用法:
  python3 pull_training_data.py              # 全量拉取
  python3 pull_training_data.py --dry-run    # 仅拉 50 场预览
  python3 pull_training_data.py --resume     # 断点续传
"""

import os, sys, json, math, time
from datetime import datetime, date
from collections import defaultdict
import requests

KEY = "fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww"
HDR = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api"

START_DATE = "2021-01-01"
END_DATE = date.today().isoformat()

OUTPUT = "/root/data/thestats_training_data.json"
CHECKPOINT = "/root/data/thestats_training_checkpoint.json"
PRIOR_PATH = "/root/data/poisson_elo_prior.json"
THE_STATS_ADV = "/root/thestats_advanced_features.py"

# 兼容原有 API 的赛事 ID
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


def load_prior():
    """加载 Elo + Poisson λ 先验"""
    if not os.path.exists(PRIOR_PATH):
        print(f"  ⚠️ {PRIOR_PATH} 不存在, 无法追加特征")
        return {}, {}, {}
    with open(PRIOR_PATH) as f:
        prior = json.load(f)
    elo = prior.get("elo", {})
    lam = prior.get("lambda_prior", {})
    meta = prior.get("meta", {})
    print(f"  ✅ 加载先验: {len(elo)} Elo, {len(lam)} λ")
    return elo, lam, meta


def load_checkpoint():
    """读取断点续传状态"""
    if not os.path.exists(CHECKPOINT):
        return {"completed_comps": [], "total_saved": 0}
    with open(CHECKPOINT) as f:
        return json.load(f)


def save_checkpoint(cp):
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f)


def pull_matches(comp_id, start, end, page=1):
    """拉取一页完赛数据"""
    url = f"{BASE}/football/matches?competition_id={comp_id}&status=finished&date_from={start}&date_to={end}&per_page=100&page={page}"
    r = requests.get(url, headers=HDR, timeout=30)
    if r.status_code != 200:
        print(f"    ❌ HTTP {r.status_code}")
        return [], False
    data = r.json().get("data", [])
    meta = r.json().get("meta", {})
    total_pages = meta.get("total_pages", 1)
    return data, page >= total_pages


def extract_match(m, comp_name, elo_dict, lam_dict):
    """从 API 响应提取一条特征记录"""
    try:
        utc = m.get("utc_date", "")
        if not utc:
            return None
        d = utc[:10]
        ht = m.get("home_team", {}) or {}
        at = m.get("away_team", {}) or {}
        home = ht.get("name", "")
        away = at.get("name", "")
        score = m.get("score", {}) or {}
        hg = score.get("home")
        ag = score.get("away")
        if hg is None or ag is None:
            return None

        # Elo
        eh = elo_dict.get(home, None)
        ea = elo_dict.get(away, None)
        have_elo = eh is not None and ea is not None

        # Poisson λ 先验
        lh = lam_dict.get(home, {})
        la = lam_dict.get(away, {})
        lh_h = lh.get("lambda_home")
        la_a = la.get("lambda_away")
        have_lam = lh_h is not None and la_a is not None

        rec = {
            "match_id": m.get("id", ""),
            "date": d,
            "comp_id": m.get("competition_id", ""),
            "comp_name": comp_name,
            "home": home,
            "away": away,
            "h_score": int(hg),
            "a_score": int(ag),
            "neutral": False,  # TheStatsAPI: home/away always set
            # Elo 特征
            "elo_h": round(eh, 1) if have_elo else 1500.0,
            "elo_a": round(ea, 1) if have_elo else 1500.0,
            "have_elo": have_elo,
            # Poisson λ 先验特征
            "lambda_h": round(lh_h, 4) if have_lam else None,
            "lambda_a": round(la_a, 4) if have_lam else None,
            "have_lambda": have_lam,
        }

        # 半场比分
        ht_score = score.get("half_time_home"), score.get("half_time_away")
        if ht_score[0] is not None and ht_score[1] is not None:
            rec["ht_h"] = int(ht_score[0])
            rec["ht_a"] = int(ht_score[1])

        return rec
    except Exception as e:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="全量特征训练数据拉取")
    parser.add_argument("--dry-run", action="store_true", help="仅拉前 50 场预览格式")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    args = parser.parse_args()

    DRY_RUN = args.dry_run
    RESUME = args.resume

    print(f"{'='*60}")
    print(f"  📡 全量训练数据拉取 (TheStatsAPI)")
    print(f"  窗口: {START_DATE} → {END_DATE}")
    print(f"  赛事: {len(TARGET_COMPETITIONS)} 个")
    print(f"  模式: {'DRY-RUN (50场)' if DRY_RUN else '全量'}")
    if RESUME:
        print(f"  模式: 断点续传")
    print(f"{'='*60}\n")

    # 加载先验
    elo_dict, lam_dict, prior_meta = load_prior()

    # 初始化
    all_records = []
    if RESUME:
        cp = load_checkpoint()
        completed = set(cp.get("completed_comps", []))
        total_before = cp.get("total_saved", 0)
        if total_before > 0:
            print(f"  📌 断点续传: 已保存 {total_before} 条, 已完成 {len(completed)} 个赛事")
        # 加载已保存的记录
        if os.path.exists(OUTPUT):
            with open(OUTPUT) as f:
                all_records = json.load(f)
                print(f"  📌 加载已有输出: {len(all_records)} 条")
    else:
        cp = {"completed_comps": [], "total_saved": 0}
        completed = set()

    total_pulled = 0
    dry_run_limit = 50

    # 遍历所有赛事
    comps = sorted(TARGET_COMPETITIONS.items())
    for comp_id, comp_name in comps:
        if RESUME and comp_id in completed:
            print(f"  ⏭️ 跳过 {comp_name} (已完成)")
            continue

        print(f"\n📡 拉取 {comp_name} ({comp_id})...")
        page = 1
        done = False
        comp_records = []

        while not done:
            matches, done = pull_matches(comp_id, START_DATE, END_DATE, page)
            if not matches:
                break

            for m in matches:
                rec = extract_match(m, comp_name, elo_dict, lam_dict)
                if rec:
                    all_records.append(rec)
                    comp_records.append(rec)
                    total_pulled += 1

                if DRY_RUN and total_pulled >= dry_run_limit:
                    done = True
                    break

            # 进度提示
            sys.stdout.write(f"\r    拉取中... 当前页 {page}, 累计 {len(comp_records)} 场")
            sys.stdout.flush()
            page += 1

            if DRY_RUN and total_pulled >= dry_run_limit:
                break

            time.sleep(0.5)  # 防 throttling

        print(f"\n    ✅ {comp_name}: {len(comp_records)} 场")

        # 更新断点
        if total_pulled > 0 and not DRY_RUN:
            cp["completed_comps"].append(comp_id)
            cp["total_saved"] = len(all_records)
            save_checkpoint(cp)

            # 阶段性保存 (每完成一个赛事写一次)
            with open(OUTPUT, "w") as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)

        if DRY_RUN and total_pulled >= dry_run_limit:
            print(f"\n  ⏹️ Dry-run 达到 {dry_run_limit} 场, 停止")
            break

    if DRY_RUN:
        # ── 展示前 2 条特征宽表 ──
        print(f"\n{'='*80}")
        print(f"  📋 特征宽表预览 (前 {min(2, len(all_records))} 条)")
        print(f"{'='*80}")
        for i, rec in enumerate(all_records[:2]):
            print(f"\n  --- 记录 {i+1}: {rec['match_id']} ---")
            print(f"  date={rec['date']}  comp={rec['comp_name']}")
            print(f"  {rec['home']} {rec['h_score']}-{rec['a_score']} {rec['away']}")
            print(f"  elo_h={rec['elo_h']}  elo_a={rec['elo_a']}  (have_elo={rec['have_elo']})")
            print(f"  lambda_h={rec.get('lambda_h')}  lambda_a={rec.get('lambda_a')}  (have_lambda={rec.get('have_lambda')})")
            if 'ht_h' in rec:
                print(f"  HT: {rec['ht_h']}-{rec['ht_a']}")
            # 特征名称列表
            keys = list(rec.keys())
            print(f"  feature_cols ({len(keys)}): {keys}")

        print(f"\n  📊 总计: {len(all_records)} 条")
        return all_records

    # ── 全量统计数据 ──
    print(f"\n{'='*60}")
    print(f"  ✅ 全量拉取完成")
    print(f"{'='*60}")
    print(f"  总记录: {len(all_records)}")
    print(f"  Elo 覆盖率: {sum(1 for r in all_records if r['have_elo'])}/{len(all_records)} ({sum(1 for r in all_records if r['have_elo'])/len(all_records)*100:.1f}%)")
    print(f"  Poisson λ 覆盖率: {sum(1 for r in all_records if r.get('have_lambda'))}/{len(all_records)} ({sum(1 for r in all_records if r.get('have_lambda'))/len(all_records)*100:.1f}%)")
    print(f"  半场比分覆盖率: {sum(1 for r in all_records if 'ht_h' in r)}/{len(all_records)}")

    # 统计 Elo 先验的真实性
    have_elo_both = sum(1 for r in all_records if r['have_elo'])
    print(f"\n  📊 基于 {len(elo_dict)} 支 Elo 队伍, {have_elo_both}/{len(all_records)} 场比赛双方均有 Elo 分数")

    # 保存
    with open(OUTPUT, "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    file_sz = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"\n  💾 已保存至 {OUTPUT} ({file_sz:.1f} MB)")

    # 清理 checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
        print(f"  🧹 清理断点文件")

    return all_records


if __name__ == "__main__":
    main()
