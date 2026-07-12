#!/usr/bin/env python3
"""
竞彩足球回测系统 (Backtest Pipeline)
=====================================
Phase 1: 记录预测 → predictions_log.csv
Phase 2: 拉取赛果 → 更新实际结果
Phase 3: 对比分析 → 输出准确率报告

用法:
  python3 backtest_jczq.py record --code 周三201 --home 丹麦 --away 刚果(金) \\
      --pred-h 49.4 --pred-d 33.8 --pred-a 16.7 --rq -1 \\
      --pred-rq-win 83.5 --pred-rq-draw 12.5 --pred-rq-loss 4.0 \\
      --pred-score "1:0" --pred-goals 1 \\
      --pred-htft "DH" --odds-h 2.18 --odds-d 3.34 --odds-a 2.70

  python3 backtest_jczq.py fetch 2026-06-02      # 拉某天赛果
  python3 backtest_jczq.py report                  # 生成回测报告
  python3 backtest_jczq.py report --full           # 详细报告含每场明细
"""

import csv
import json
import re
import sys
import os
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict

BASE_DIR = "/root/data"
LOG_FILE = f"{BASE_DIR}/predictions_log.csv"
RESULTS_DIR = f"{BASE_DIR}/results"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============ CSV 字段 ============
FIELDS = [
    "code", "date", "match_date", "time",          # 场次信息
    "home_cn", "away_cn",            # 中文队名
    "league", "rq",                  # 联赛类型、让球
    "pred_h", "pred_d", "pred_a",    # 模型胜平负概率(%)
    "pred_rq_win", "pred_rq_draw", "pred_rq_loss",  # 让球概率
    "pred_top_score",                # 最可能比分
    "pred_top_goals",                # 最可能总进球
    "pred_top_htft",                 # 最可能半全场
    "pred_spf_pick",                 # 胜平负推荐
    "pred_rq_pick",                  # 让球推荐
    "pred_htft_pick",                # 半全场推荐
    "pred_goals_pick",               # 总进球推荐
    "pred_score_pick",               # 比分推荐
    "odds_h", "odds_d", "odds_a",    # 市场赔率
    "ev_h", "ev_d", "ev_a",          # EV值
    "direction",                     # 方向判断
    # 365scores 数据
    "vote_h", "vote_d", "vote_a",    # 投票数据(%)
    "vote_count",                    # 投票人数
    "vote_fusion_alpha",             # 投票融合权重
    "pop_rank_home", "pop_rank_away",  # 人气排名
    "pop_rank_diff",                 # 人气排名差异
    "trend_win_rate_home", "trend_win_rate_away",  # 趋势胜率
    "trend_win_rate_diff",           # 趋势胜率差异
    # 365基本面特征 (预埋)
    "s365_home_winrate", "s365_away_winrate",  # 近5场胜率
    "s365_home_fifa", "s365_away_fifa",        # FIFA排名
    "s365_rank_diff",                         # FIFA排名差
    "s365_popularity_diff",                   # 人气指数差
    "source_tag",                    # 数据源标签
    "model_version",                 # 预测版本
    "score_full",                    # 比分完整概率分布 (JSON)
    "htft_full",                     # 半全场完整概率分布 (JSON)
    "goals_full",                    # 总进球完整概率分布 (JSON)
    "simple_pred",                   # 并行模型预测
    "simple_conf",                   # 并行模型置信度
    "bet_action",                    # 赛事过滤标签 (RECOMMEND/WATCH/SKIP_LEAGUE)
    "model_route",                   # 模型路由 (hybrid/market_fallback/club)
    "match_key",                     # 稳定主键: date+league+home+away+time
    "pred30_h",                      # A/B: 30维模型主胜概率(%)
    "pred30_d",                      # A/B: 30维模型平局概率(%)
    "pred30_a",                      # A/B: 30维模型客胜概率(%)
    "kelly_pct",                     # Quarter-Kelly 建议仓位 (小数)
    # 实际赛果（回填）
    "actual_score", "actual_ht",     # 全场/半场比分
    "actual_hda",                    # 胜平负彩果
    "actual_rq_result",              # 让球彩果
    "actual_goals",                  # 总进球数
    "actual_htft",                   # 半全场彩果
    "brier_spf",                     # 单场Brier Score (胜平负)
    "brier_rq",                      # 让球Brier Score
    "acc_score_top1",                # 比分Top-1准确率
    "acc_goals_top1",                # 总进球Top-1准确率
    "goals_mae",                     # 总进球平均绝对误差
    "acc_htft_top1",                 # 半全场Top-1准确率
    "result_status",                 # missing/filled/conflict/postponed
    "settled_at",                    # 回填完成时间 (ISO)
    "backfill_source",               # 回填数据来源
    "checked",                       # 是否已回测
]

# ============ 辅助函数 ============

def load_log():
    """加载预测日志"""
    rows = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows

def save_log(rows):
    """保存预测日志"""
    with open(LOG_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def get_today():
    """获取今天日期(北京时间)"""
    bj = datetime.utcnow() + timedelta(hours=8)
    return bj.strftime("%Y-%m-%d")

# ============ Phase 1: 记录预测 ============

# ── CLI arg name → CSV column mapping ──
CLI_TO_COLUMN = {
    'code': 'code', 'home': 'home_cn', 'away': 'away_cn',
    'pred-h': 'pred_h', 'pred-d': 'pred_d', 'pred-a': 'pred_a',
    'rq': 'rq',
    'pred-rq-win': 'pred_rq_win', 'pred-rq-draw': 'pred_rq_draw', 'pred-rq-loss': 'pred_rq_loss',
    'pred-score': 'pred_top_score', 'pred-goals': 'pred_top_goals', 'pred-htft': 'pred_top_htft',
    'pred-spf-pick': 'pred_spf_pick', 'pred-rq-pick': 'pred_rq_pick',
    'pred-htft-pick': 'pred_htft_pick', 'pred-goals-pick': 'pred_goals_pick', 'pred-score-pick': 'pred_score_pick',
    'odds-h': 'odds_h', 'odds-d': 'odds_d', 'odds-a': 'odds_a',
    'ev-h': 'ev_h', 'ev-d': 'ev_d', 'ev-a': 'ev_a',
    'dir': 'direction', 'league': 'league', 'time': 'time',
    'vote-h': 'vote_h', 'vote-d': 'vote_d', 'vote-a': 'vote_a',
    'vote-count': 'vote_count', 'vote-fusion-alpha': 'vote_fusion_alpha',
    'pop-rank-home': 'pop_rank_home', 'pop-rank-away': 'pop_rank_away', 'pop-rank-diff': 'pop_rank_diff',
    'trend-win-rate-home': 'trend_win_rate_home', 'trend-win-rate-away': 'trend_win_rate_away',
    'trend-win-rate-diff': 'trend_win_rate_diff',
    's365-home-winrate': 's365_home_winrate', 's365-away-winrate': 's365_away_winrate',
    's365-home-fifa': 's365_home_fifa', 's365-away-fifa': 's365_away_fifa',
    's365-rank-diff': 's365_rank_diff', 's365-popularity-diff': 's365_popularity_diff',
    'score-full': 'score_full', 'htft-full': 'htft_full', 'goals-full': 'goals_full',
    'simple-pred': 'simple_pred', 'simple-conf': 'simple_conf',
    'bet-action': 'bet_action', 'kelly-pct': 'kelly_pct', 'model-route': 'model_route',
    'match-key': 'match_key', 'match-date': 'match_date',
    'pred30-h': 'pred30_h', 'pred30-d': 'pred30_d', 'pred30-a': 'pred30_a',
}


def record_match(**kwargs) -> str:
    """Record a prediction row to CSV. Returns a status message.

    Accepts the same keyword args as the CLI args (without ``--`` prefix).
    Callers pass native Python types; they are str()-ified internally.

    Example
    -------
    >>> record_match(code='周三201', home='丹麦', away='刚果(金)',
    ...              pred_h=49.4, pred_d=33.8, pred_a=16.7)
    '✅ 已记录: 周三201 丹麦 vs 刚果(金)'
    """
    row: dict = {f: '' for f in FIELDS}
    row['date'] = get_today()
    row['checked'] = '0'
    row['result_status'] = 'missing'

    for key, value in kwargs.items():
        col = CLI_TO_COLUMN.get(key)
        if col is None:
            continue
        row[col] = str(value) if value is not None else ''

    rows = load_log()
    # 同一天同场次：更新现有记录，保留已回填字段
    for idx, r in enumerate(rows):
        if r['code'] == row['code'] and r['date'] == row['date']:
            for k, v in row.items():
                if k in ('actual_score', 'actual_ht', 'actual_hda', 'actual_rq_result',
                         'actual_goals', 'actual_htft', 'brier_spf', 'result_status',
                         'settled_at', 'backfill_source', 'pred30_h', 'pred30_d', 'pred30_a'):
                    continue
                if v != '':
                    r[k] = v
            rows[idx] = r
            save_log(rows)
            msg = f"♻️ 已更新: {row['code']} {row['home_cn']} vs {row['away_cn']}"
            print(msg)
            return msg
    rows.append(row)
    save_log(rows)
    msg = f"✅ 已记录: {row['code']} {row['home_cn']} vs {row['away_cn']}"
    print(msg)
    return msg


def cmd_record(args):
    """记录一条预测到日志（CLI 入口，委托给 record_match）"""
    kwargs = {}
    pairs = iter(args)
    for k, v in zip(pairs, pairs):
        kwargs[k.lstrip('--')] = v
    return record_match(**kwargs)

# ============ Phase 2: 从500.com拉取赛果 ============

def fetch_results(date_str):
    """从500.com拉取指定日期的赛果"""
    url = f"https://zx.500.com/jczq/kaijiang.php?d={date_str}"
    result = subprocess.run(
        ["curl", "-sL", url, "-H", "User-Agent: Mozilla/5.0"],
        capture_output=True, timeout=60
    )
    raw = result.stdout
    # GBK -> UTF-8
    html = raw.decode("gbk", errors="ignore")

    # 解析表格行
    pattern = r'<tr>.*?<td>(\w+)</td>.*?<td.*?>(.*?)</td>.*?<td class="eng">(.*?)</td>'
    # 更好的方法: 按<tr>拆分
    rows_raw = re.findall(r'<tr>.*?</tr>', html, re.DOTALL)

    matches = []
    for row_html in rows_raw:
        # 跳过表头
        if '<th' in row_html:
            continue

        cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html)
        if len(cells) < 12:
            continue

        code = re.sub(r'<[^>]+>', '', cells[0]).strip()
        if not re.match(r'周[一二三四五六日]\d{3}', code):
            continue

        try:
            league_raw = re.search(r'style="background-color:#([^"]*)">([^<]*)', cells[1])
            league = league_raw.group(2) if league_raw else re.sub(r'<[^>]+>', '', cells[1]).strip()

            time_raw = re.sub(r'<[^>]+>', '', cells[2]).strip()
            home = re.search(r'>([^<]+)</a>', cells[3])
            home = home.group(1) if home else re.sub(r'<[^>]+>', '', cells[3]).strip()
            rq_raw = re.sub(r'<[^>]+>', '', cells[4]).strip()
            away = re.search(r'>([^<]+)</a>', cells[5])
            away = away.group(1) if away else re.sub(r'<[^>]+>', '', cells[5]).strip()

            score_raw = re.sub(r'<[^>]+>', '', cells[6]).strip()
            # 比分格式: (半场比分) 全场比分 或 直接比分
            score_full = score_raw
            score_ht = ""

            ht_match = re.search(r'\(([^)]*)\)', score_raw)
            ft_match = re.search(r'\)\s*([\d:]+)', score_raw)
            if ht_match and ft_match:
                score_ht = ht_match.group(1)
                score_full = ft_match.group(1)
            elif re.match(r'^\d+:\d+$', score_raw):
                score_full = score_raw

            # 让球彩果 (cells[8])
            rq_result = re.sub(r'<[^>]+>', '', cells[8]).strip()

            # 胜平负彩果 (cells[11])
            hda_result = re.sub(r'<[^>]+>', '', cells[11]).strip()

            # 总进球 (cells[14])
            goals = re.sub(r'<[^>]+>', '', cells[14]).strip()

            # 半全场 (cells[17])
            htft = re.sub(r'<[^>]+>', '', cells[17]).strip()

            # 比分分解
            home_goals = ""
            away_goals = ""
            if ':' in score_full:
                parts = score_full.split(':')
                home_goals = parts[0]
                away_goals = parts[1]

            match = {
                "code": code,
                "league": league,
                "time": time_raw,
                "home": home,
                "away": away,
                "rq": rq_raw,
                "score_ht": score_ht,
                "score_full": score_full,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "rq_result": rq_result,
                "hda_result": hda_result,
                "goals": goals,
                "htft": htft,
            }
            matches.append(match)
        except Exception as e:
            print(f"  解析行失败: {e}")

    # 文件缓存
    out_path = f"{RESULTS_DIR}/{date_str}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    print(f"✅ {date_str}: 获取 {len(matches)} 场比赛结果")
    for m in matches:
        print(f"  {m['code']} {m['home']} vs {m['away']}: {m['score_full']} -> {m['hda_result']}")

    return matches

def cmd_fetch(args):
    """拉取某天赛果并回填"""
    date_str = args[0] if args else get_today()
    matches = fetch_results(date_str)

    # 回填到预测日志 — 用 code 匹配而非日期
    rows = load_log()
    updated = 0
    for r in rows:
        if r["checked"] == "1":
            continue
        for m in matches:
            if m["code"] == r["code"]:
                    r["actual_score"] = m["score_full"]
                    r["actual_ht"] = m["score_ht"]
                    r["actual_hda"] = m["hda_result"]
                    r["actual_rq_result"] = m["rq_result"]
                    r["actual_goals"] = m["goals"]
                    r["actual_htft"] = m["htft"]
                    r["checked"] = "1"
                    updated += 1
                    print(f"  回填: {r['code']} {r['home_cn']} vs {r['away_cn']} -> {m['score_full']}")
                    break

    if updated > 0:
        save_log(rows)
        print(f"✅ 已回填 {updated} 场比赛")
    else:
        print("⚠️ 没有需要回填的预测记录")
        print(f"   日志中共 {len(rows)} 条记录，日期 {date_str} 的待回测记录: {sum(1 for r in rows if r['date']==date_str and r['checked']!='1')}")

# ============ Phase 3: 回测报告 ============

def cmd_report(args):
    """生成回测报告"""
    full = "--full" in args

    rows = load_log()
    checked = [r for r in rows if r["checked"] == "1"]

    if not checked:
        print("❌ 没有已回测的记录。请先运行: python3 backtest_jczq.py fetch YYYY-MM-DD")
        return

    def best_hda_pick(r):
        if r.get("pred_spf_pick"):
            return r.get("pred_spf_pick")
        pred_vals = {"主胜": float(r.get("pred_h", 0) or 0),
                     "平": float(r.get("pred_d", 0) or 0),
                     "客胜": float(r.get("pred_a", 0) or 0)}
        return max(pred_vals, key=pred_vals.get)

    def best_rq_pick(r):
        if r.get("pred_rq_pick"):
            return r.get("pred_rq_pick")
        rq_vals = {
            "让胜": float(r.get("pred_rq_win", 0) or 0),
            "让平": float(r.get("pred_rq_draw", 0) or 0),
            "让负": float(r.get("pred_rq_loss", 0) or 0),
        }
        return max(rq_vals, key=rq_vals.get) if max(rq_vals.values()) > 0 else ""

    def norm_hda(x):
        return {"主胜": "胜", "客胜": "负", "胜": "胜", "平": "平", "负": "负",
                "H": "胜", "D": "平", "A": "负"}.get(x, x)

    def norm_htft(x):
        mapping = {
            "HH": "胜胜", "HD": "胜平", "HA": "胜负",
            "DH": "平胜", "DD": "平平", "DA": "平负",
            "AH": "负胜", "AD": "负平", "AA": "负负",
            "胜胜": "胜胜", "胜平": "胜平", "胜负": "胜负",
            "平胜": "平胜", "平平": "平平", "平负": "平负",
            "负胜": "负胜", "负平": "负平", "负负": "负负",
        }
        clean = (x or "").replace("/", "").replace("-", "")
        return mapping.get(clean, clean)

    # 按日期分组
    by_date = defaultdict(list)
    for r in checked:
        by_date[r["date"]].append(r)

    total = len(checked)
    hda_ok = 0
    score_ok = 0
    goals_ok = 0
    htft_ok = 0
    rq_ok = 0

    detail_lines = []

    for date in sorted(by_date.keys()):
        day_matches = by_date[date]
        if full:
            detail_lines.append(f"\n--- {date} ---")

        for r in day_matches:
            pred_vals = {"主胜": float(r.get("pred_h", 0) or 0),
                         "平": float(r.get("pred_d", 0) or 0),
                         "客胜": float(r.get("pred_a", 0) or 0)}
            best_dir = max(pred_vals, key=pred_vals.get)
            actual_hda = r.get("actual_hda", "")

            spread = max(pred_vals.values()) - min(pred_vals.values())
            is_close = spread < 5

            hda_correct = False
            if norm_hda(best_dir) == norm_hda(actual_hda):
                hda_correct = True
                hda_ok += 1
            elif is_close:
                pass
            else:
                pass

            pred_score = r.get("pred_top_score", "").strip()
            act_score = r.get("actual_score", "").strip()
            if pred_score and act_score and pred_score == act_score:
                score_ok += 1

            pred_g = r.get("pred_top_goals", "").strip()
            act_g = r.get("actual_goals", "").strip()
            if pred_g and act_g and pred_g == act_g:
                goals_ok += 1

            pred_ht = r.get("pred_top_htft", "").strip()
            act_ht = r.get("actual_htft", "").strip()
            if pred_ht and act_ht:
                if norm_htft(pred_ht) == norm_htft(act_ht):
                    htft_ok += 1

            act_rq = r.get("actual_rq_result", "")
            pred_rq_dir = ""
            rq_vals = {
                "让胜": float(r.get("pred_rq_win", 0) or 0),
                "让平": float(r.get("pred_rq_draw", 0) or 0),
                "让负": float(r.get("pred_rq_loss", 0) or 0),
            }
            if max(rq_vals.values()) > 0:
                pred_rq_dir = max(rq_vals, key=rq_vals.get)
                act_rq_norm = act_rq if act_rq.startswith("让") else (f"让{act_rq}" if act_rq in ("胜", "平", "负") else act_rq)
                if pred_rq_dir == act_rq_norm:
                    rq_ok += 1

            if full:
                hda_mark = "✅HDA" if hda_correct else ("⚠均势" if is_close else "❌HDA")
                score_mark = "✅比分" if (pred_score and act_score and pred_score == act_score) else ""
                goals_mark = "✅进球" if (pred_g and act_g and pred_g == act_g) else ""
                rq_mark = "✅让球" if (act_rq and pred_rq_dir and pred_rq_dir == (act_rq if act_rq.startswith('让') else f'让{act_rq}')) else ""
                htft_mark = "✅半全" if (pred_ht and act_ht and norm_htft(pred_ht) == norm_htft(act_ht)) else ""
                hits = [m for m in [hda_mark, score_mark, goals_mark, rq_mark, htft_mark] if m]
                detail_lines.append(
                    f"  {r['code']} {r['home_cn']:{6}}vs {r['away_cn']:{6}}  "
                    f"预测{best_dir}({pred_vals[best_dir]:.0f}%)→实际{actual_hda}  "
                    f"比分{r.get('actual_score','?'):6} 进球{act_g}球 半全场{act_ht} 让球{act_rq}  "
                    f"→ {' '.join(hits)}"
                )

    print("=" * 60)
    print("  📊 竞彩足球回测报告")
    print(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print()
    print(f"  累计预测: {len(rows)} 场")
    print(f"  已回测:   {total} 场")
    print(f"  未回测:   {sum(1 for r in rows if r['checked']!='1')} 场")
    print()

    if total == 0:
        return

    hda_real_ok = sum(1 for r in checked if norm_hda(best_hda_pick(r)) == norm_hda(r.get("actual_hda", "")))

    print(f"  📈 核心指标")
    print(f"  ─────────────────────────────")
    print(f"  HDA方向准确率:     {hda_real_ok}/{total} ({hda_real_ok/total*100:.1f}%)")
    print(f"  精确比分准确率:   {score_ok}/{total} ({score_ok/total*100:.1f}%)")
    print(f"  总进球准确率:     {goals_ok}/{total} ({goals_ok/total*100:.1f}%)")
    print(f"  半全场准确率:     {htft_ok}/{total} ({htft_ok/total*100:.1f}%)")
    print(f"  让球方向准确率:   {rq_ok}/{total} ({rq_ok/total*100:.1f}%)")
    print()

    print(f"  🧩 分玩法报表")
    print(f"  ─────────────────────────────")
    spf_total = sum(1 for r in checked if best_hda_pick(r))
    spf_ok = sum(1 for r in checked if norm_hda(best_hda_pick(r)) == norm_hda(r.get('actual_hda', '')))
    rq_rows = [r for r in checked if best_rq_pick(r)]
    rq_total = len(rq_rows)
    rq_real_ok = sum(1 for r in rq_rows if best_rq_pick(r) == (r.get('actual_rq_result','') if r.get('actual_rq_result','').startswith('让') else (f"让{r.get('actual_rq_result','')}" if r.get('actual_rq_result','') in ('胜','平','负') else r.get('actual_rq_result',''))))
    htft_rows = [r for r in checked if (r.get('pred_htft_pick') or r.get('pred_top_htft')) and r.get('actual_htft')]
    htft_total = len(htft_rows)
    htft_real_ok = sum(1 for r in htft_rows if norm_htft(r.get('pred_htft_pick') or r.get('pred_top_htft')) == norm_htft(r.get('actual_htft')))
    goals_rows = [r for r in checked if (r.get('pred_goals_pick') or r.get('pred_top_goals')) and r.get('actual_goals')]
    goals_total = len(goals_rows)
    goals_real_ok = sum(1 for r in goals_rows if str(r.get('pred_goals_pick') or r.get('pred_top_goals')) == str(r.get('actual_goals')))
    score_rows = [r for r in checked if (r.get('pred_score_pick') or r.get('pred_top_score')) and r.get('actual_score')]
    score_total = len(score_rows)
    score_real_ok = sum(1 for r in score_rows if str(r.get('pred_score_pick') or r.get('pred_top_score')) == str(r.get('actual_score')))
    print(f"  SPF:     {spf_ok}/{spf_total} ({(spf_ok/spf_total*100 if spf_total else 0):.1f}%)")
    print(f"  让球:    {rq_real_ok}/{rq_total} ({(rq_real_ok/rq_total*100 if rq_total else 0):.1f}%)")
    print(f"  半全场:  {htft_real_ok}/{htft_total} ({(htft_real_ok/htft_total*100 if htft_total else 0):.1f}%)")
    print(f"  总进球:  {goals_real_ok}/{goals_total} ({(goals_real_ok/goals_total*100 if goals_total else 0):.1f}%)")
    print(f"  比分:    {score_real_ok}/{score_total} ({(score_real_ok/score_total*100 if score_total else 0):.1f}%)")
    print()

    draw_cal_rows = [r for r in checked if r.get('league') == '友谊赛']
    if draw_cal_rows:
        on_rows = [r for r in draw_cal_rows if r.get('friendly_draw_calibrated') == '1']
        off_rows = [r for r in draw_cal_rows if r.get('friendly_draw_calibrated') != '1']
        print(f"  🎯 友谊赛SPF平局校准追踪")
        print(f"  ─────────────────────────────")
        for label, rows_part in [('ON', on_rows), ('OFF', off_rows)]:
            n = len(rows_part)
            if n == 0:
                print(f"  {label}: 0场")
                continue
            ok = sum(1 for r in rows_part if norm_hda(best_hda_pick(r)) == norm_hda(r.get('actual_hda', '')))
            factors = sorted({r.get('friendly_draw_factor', '') for r in rows_part if r.get('friendly_draw_factor', '')})
            factor_text = ','.join(factors) if factors else '-'
            print(f"  {label}: {ok}/{n} ({ok/n*100:.1f}%) | factor={factor_text}")
        print()

    review = {"SPF": [], "让球": [], "半全场": [], "总进球": [], "比分": []}
    for r in checked:
        spf_pred = best_hda_pick(r)
        spf_act = r.get('actual_hda', '')
        spf_hit = norm_hda(spf_pred) == norm_hda(spf_act)
        review['SPF'].append((spf_hit, f"{r['code']} {r['home_cn']} vs {r['away_cn']} | 预测={spf_pred} | 实际={spf_act}"))

        rq_pred = best_rq_pick(r)
        rq_act_raw = r.get('actual_rq_result', '')
        rq_act = rq_act_raw if rq_act_raw.startswith('让') else (f"让{rq_act_raw}" if rq_act_raw in ('胜','平','负') else rq_act_raw)
        if rq_pred:
            review['让球'].append((rq_pred == rq_act, f"{r['code']} {r['home_cn']} vs {r['away_cn']} | 预测={rq_pred} | 实际={rq_act_raw}"))

        htft_pred = r.get('pred_htft_pick') or r.get('pred_top_htft')
        htft_act = r.get('actual_htft', '')
        if htft_pred and htft_act:
            review['半全场'].append((norm_htft(htft_pred) == norm_htft(htft_act), f"{r['code']} {r['home_cn']} vs {r['away_cn']} | 预测={htft_pred} | 实际={htft_act}"))

        goals_pred = str(r.get('pred_goals_pick') or r.get('pred_top_goals') or '')
        goals_act = str(r.get('actual_goals', ''))
        if goals_pred and goals_act:
            review['总进球'].append((goals_pred == goals_act, f"{r['code']} {r['home_cn']} vs {r['away_cn']} | 预测={goals_pred} | 实际={goals_act}"))

        score_pred = str(r.get('pred_score_pick') or r.get('pred_top_score') or '')
        score_act = str(r.get('actual_score', ''))
        if score_pred and score_act:
            review['比分'].append((score_pred == score_act, f"{r['code']} {r['home_cn']} vs {r['away_cn']} | 预测={score_pred} | 实际={score_act}"))

    print(f"  🧾 玩法复盘清单")
    print(f"  ─────────────────────────────")
    for play in ['SPF', '让球', '半全场', '总进球', '比分']:
        items = review[play]
        misses = [text for ok, text in items if not ok]
        hits = [text for ok, text in items if ok]
        print(f"  {play}:")
        if misses:
            for line in misses[:10]:
                print(f"    ❌ {line}")
        elif hits:
            print(f"    ✅ 本玩法当前无错单")
        else:
            print(f"    - 无可复盘样本")
        if full and hits:
            for line in hits[:10]:
                print(f"    ✅ {line}")
    print()

    print(f"  📅 按日期分布")
    print(f"  ─────────────────────────────")
    for date in sorted(by_date.keys()):
        day_rows = by_date[date]
        day_n = len(day_rows)
        day_hda = sum(1 for r in day_rows if norm_hda(best_hda_pick(r)) == norm_hda(r.get("actual_hda", "")))
        day_goals = sum(1 for r in day_rows if str(r.get("pred_goals_pick") or r.get("pred_top_goals", "")) == str(r.get("actual_goals", "")))
        day_rq_total = sum(1 for r in day_rows if best_rq_pick(r))
        day_rq_ok = sum(1 for r in day_rows if best_rq_pick(r) and best_rq_pick(r) == (r.get('actual_rq_result','') if r.get('actual_rq_result','').startswith('让') else (f"让{r.get('actual_rq_result','')}" if r.get('actual_rq_result','') in ('胜','平','负') else r.get('actual_rq_result',''))))
        print(f"  {date}: {day_n}场 SPF{day_hda}/{day_n} 让球{day_rq_ok}/{day_rq_total} 进球{day_goals}/{day_n}")

    print()

    if full:
        print(f"  📋 场次明细")
        print(f"  ─────────────────────────────")
        for line in detail_lines:
            print(line)
        print()

    if goals_ok / total > 0.7:
        print("  💡 模型强项: 总进球预测")
    if rq_total and rq_real_ok / rq_total > 0.7:
        print("  💡 模型强项: 让球方向判断")
    if hda_real_ok / total < 0.5 and total >= 10:
        print("  ⚠️ 需要关注: HDA准确率偏低，可能需调参")
    print()
    print("=" * 60)

# ============ 自动补录预测 ============

def cmd_autolog(args):
    """将 recent_predictions.txt 中的预测批量导入"""
    path = args[0] if args else "/tmp/recent_predictions.txt"
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return
    print(f"📖 从 {path} 导入...")
    # 这是一个placeholder，实际使用时可以从标准格式文本批量导入
    print("用法: 将预测数据按以下格式写入文件，然后导入")
    print("  周三201 丹麦 刚果(金) 49.4 33.8 16.7 -1 83.5 12.5 4.0 1:0 1 DH")

# ============ 主入口 ============

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "record":
        cmd_record(args)
    elif cmd == "fetch":
        cmd_fetch(args)
    elif cmd == "report":
        cmd_report(args)
    elif cmd == "autolog":
        cmd_autolog(args)
    else:
        print(f"❌ 未知命令: {cmd}")
        print(__doc__)
