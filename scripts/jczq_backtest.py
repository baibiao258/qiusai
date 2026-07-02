#!/usr/bin/env python3
"""
竞彩足球 5 维回测脚本 (基于 500.com kaijiang.php)

用法:
  python3 jczq_backtest.py <YYYY-MM-DD> --predictions preds.json [--odds odds.json] [--report report.html]
  python3 jczq_backtest.py 2026-06-06 --predictions preds_0606.json

输入预测 JSON 格式 (示例):
{
  "周六201": {
    "spf": "胜",          # 胜/平/负 (可选: null = 未推)
    "spf_probs": {"胜": 0.21, "平": 0.24, "负": 0.55},  # 全概率分布 (可选, 用于 Brier/校准)
    "rq":  "让胜",         # 让胜/让平/让负 (可选: null = 未推)
    "rq_probs": {"让胜": 0.34, "让平": 0.25, "让负": 0.41},
    "scores": ["0:1","0:2","1:1"],   # Top N 比分 (可选: [])
    "score_probs": {"0:1": 0.13, "0:2": 0.13, "1:1": 0.10, ...},  # 全比分分布 (可选)
    "size": "大2.5",       # 大2.5/小2.5 (可选: null)
    "size_probs": {"大2.5": 0.50, "小2.5": 0.50},
    "htft": "客胜",        # 9 选 1: 胜胜/胜平/胜负/平胜/平平/平负/负胜/负平/负负 (可选: null)
    "htft_probs": {"胜胜": 0.35, "平胜": 0.29, ...}
  },
  ...
}

校准分析 (--calibration) 需要 *_probs 全概率分布, 否则只算命中率.

输出赔率 JSON (可选, 用于 ROI 模拟):
{
  "周六201": {"spf": {"胜": 1.95, "平": 3.20, "负": 3.80}, "rq": {"让胜": 1.95, ...}, ...}
}
"""
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# ---------- 抓取 500.com 赛果 ----------
def fetch_jczq_kaijiang(date_str: str, playid: int = None) -> list[dict]:
    """抓取 500.com kaijiang.php 指定日期的赛果"""
    url = "https://zx.500.com/jczq/kaijiang.php"
    params = {"d": date_str}
    if playid:
        params["playid"] = playid
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://zx.500.com/jczq/",
    }
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.encoding = "gb2312"  # 500.com 用 GB2312
    soup = BeautifulSoup(r.text, "html.parser")
    return _parse_ld_table(soup, date_str)


def _parse_ld_table(soup, date_str: str) -> list[dict]:
    table = soup.find("table", class_="ld_table")
    if not table:
        return []
    rows = table.find_all("tr")[1:]
    results = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 19:
            continue
        match_id = cells[0].get_text(strip=True)
        league = cells[1].get_text(strip=True)
        match_time = cells[2].get_text(strip=True)
        home = cells[3].get_text(strip=True)
        handicap_span = cells[4].find("span")
        handicap = handicap_span.get_text(strip=True) if handicap_span else cells[4].get_text(strip=True)
        away = cells[5].get_text(strip=True)
        score_raw = cells[6].get_text(strip=True)
        score_match = re.match(r'\((\d+):(\d+)\)\s*(\d+):(\d+)', score_raw)
        if score_match:
            ht_h, ht_a, ft_h, ft_a = [int(g) for g in score_match.groups()]
        else:
            ht_h, ht_a, ft_h, ft_a = None, None, None, None
        # 表格列结构 (含 &nbsp; 分隔): 0编号 1联赛 2时间 3主 4让 5客 6比分 7空 8让球 9让球奖 10空 11SPF 12SPF奖 13空 14总进球 15总进球奖 16空 17半全场 18半全场奖
        rq_result = cells[8].get_text(strip=True)
        spf_result = cells[11].get_text(strip=True)
        goals_result = cells[14].get_text(strip=True)
        htft_result = cells[17].get_text(strip=True)
        results.append({
            "match_id": match_id, "league": league, "time": match_time,
            "home": home, "away": away, "handicap": handicap,
            "ht_home": ht_h, "ht_away": ht_a, "ft_home": ft_h, "ft_away": ft_a,
            "rq_result": rq_result, "spf_result": spf_result,
            "goals_result": goals_result, "htft_result": htft_result,
        })
    return results


# ---------- 5 维回测 ----------
def normalize_htft(p: str) -> str:
    """'客-客' -> '负负'"""
    return p.replace("主", "胜").replace("客", "负").replace("-", "")


def normalize_rq(p: str) -> str:
    """'让胜' -> '胜'"""
    return p.replace("让", "")


def backtest_one(pred: dict, actual: dict) -> dict:
    """单场回测, 返回每维命中"""
    out = {"spf": None, "rq": None, "score": None, "size": None, "htft": None}

    # 1) SPF
    if pred.get("spf") in ("胜", "平", "负"):
        out["spf"] = (pred["spf"] == actual["spf_result"])

    # 2) 让球
    if pred.get("rq") in ("让胜", "让平", "让负"):
        out["rq"] = (normalize_rq(pred["rq"]) == actual["rq_result"])

    # 3) 精准比分
    if pred.get("scores"):
        actual_score = f"{actual['ft_home']}:{actual['ft_away']}"
        out["score"] = (actual_score in pred["scores"])

    # 4) 总进球大小
    if pred.get("size") in ("大2.5", "小2.5"):
        total = int(actual["goals_result"]) if actual["goals_result"] != "7+" else 7
        size_actual = "大2.5" if total > 2.5 else "小2.5"
        out["size"] = (pred["size"] == size_actual)

    # 5) 半全场
    if pred.get("htft"):
        out["htft"] = (normalize_htft(pred["htft"]) == actual["htft_result"])

    return out


def build_report(actuals: list[dict], predictions: dict, odds: dict = None) -> pd.DataFrame:
    """构建回测明细 DataFrame"""
    rows = []
    for a in actuals:
        mid = a["match_id"]
        p = predictions.get(mid, {})
        b = backtest_one(p, a)
        row = {
            "match_id": mid,
            "match": f"{a['home']} vs {a['away']}",
            "league": a["league"],
            "rq": a["handicap"],
            "ht": f"{a['ht_home']}:{a['ht_away']}",
            "ft": f"{a['ft_home']}:{a['ft_away']}",
            "actual_spf": a["spf_result"],
            "actual_rq": a["rq_result"],
            "actual_goals": a["goals_result"],
            "actual_htft": a["htft_result"],
            "pred_spf": p.get("spf", "—"),
            "pred_rq": p.get("rq", "—"),
            "pred_size": p.get("size", "—"),
            "pred_htft": p.get("htft", "—"),
            "hit_spf": "✅" if b["spf"] else ("—" if b["spf"] is None else "❌"),
            "hit_rq": "✅" if b["rq"] else ("—" if b["rq"] is None else "❌"),
            "hit_score": "✅" if b["score"] else ("—" if b["score"] is None else "❌"),
            "hit_size": "✅" if b["size"] else ("—" if b["size"] is None else "❌"),
            "hit_htft": "✅" if b["htft"] else ("—" if b["htft"] is None else "❌"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summary_stats(df: pd.DataFrame) -> dict:
    """5 维命中率统计 (仅统计有预测的场次)"""
    stats = {}
    for dim in ["spf", "rq", "score", "size", "htft"]:
        col = f"hit_{dim}"
        valid = df[df[col].isin(["✅", "❌"])]
        n = len(valid)
        hits = (valid[col] == "✅").sum()
        rate = hits / n if n > 0 else None
        stats[dim] = {"hits": hits, "total": n, "rate": rate}
    return stats


def roi_simulation(df: pd.DataFrame, actuals: list[dict], predictions: dict, odds: dict) -> pd.DataFrame:
    """按 5 维计算单注 ROI, 假设每注 1 元本金"""
    rows = []
    for a in actuals:
        mid = a["match_id"]
        p = predictions.get(mid, {})
        o = odds.get(mid, {}) if odds else {}
        for dim in ["spf", "rq", "size", "htft"]:
            pred_key = {"spf": p.get("spf"), "rq": normalize_rq(p.get("rq", "")), "size": p.get("size"), "htft": normalize_htft(p.get("htft", ""))}[dim]
            if not pred_key or dim not in o or pred_key not in o[dim]:
                continue
            odds_v = o[dim][pred_key]
            # 判断命中
            actual_map = {"spf": a["spf_result"], "rq": a["rq_result"],
                          "size": "大2.5" if int(a["goals_result"] if a["goals_result"] != "7+" else 7) > 2.5 else "小2.5",
                          "htft": a["htft_result"]}
            win = (actual_map[dim] == pred_key)
            pnl = (odds_v - 1) if win else -1
            rows.append({"match_id": mid, "dim": dim, "pred": pred_key, "actual": actual_map[dim],
                         "odds": odds_v, "stake": 1, "pnl": pnl, "win": win})
    return pd.DataFrame(rows)


# ---------- 校准分析 (sklearn-style) ----------
def brier_multiclass(y_true_onehot: np.ndarray, y_prob: np.ndarray) -> float:
    """多类 Brier: mean over samples of sum_k (p_k - y_k)^2
    y_true_onehot: (n, k), y_prob: (n, k), 各列对应同一类"""
    return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / n) * |acc_b - conf_b|"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece, n = 0.0, len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """可靠性表: 每个 bin 的样本数 / 平均预测概率 / 实际频率 / 偏差"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"[{bins[b]:.2f}, {bins[b+1]:.2f})",
            "n": int(mask.sum()),
            "mean_pred": float(y_prob[mask].mean()),
            "actual_rate": float(y_true[mask].mean()),
            "gap": float(y_true[mask].mean() - y_prob[mask].mean()),
        })
    return pd.DataFrame(rows)


def plot_reliability(cal_df: pd.DataFrame, dim_name: str, save_path: str = None):
    """可靠性图 (matplotlib)"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib 未安装, 跳过可靠性图")
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="完美校准")
    ax.plot(cal_df["mean_pred"], cal_df["actual_rate"], "o-", lw=2,
            markersize=10, color="#1f77b4", label=dim_name)
    # 画 gap 柱
    for _, r in cal_df.iterrows():
        ax.plot([r["mean_pred"], r["mean_pred"]], [r["mean_pred"], r["actual_rate"]],
                color="red", alpha=0.4, lw=1.5)
    ax.set_xlabel("预测概率 (Mean Predicted)", fontsize=11)
    ax.set_ylabel("实际命中率 (Actual Rate)", fontsize=11)
    ax.set_title(f"Reliability Plot: {dim_name}", fontsize=13)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
        print(f"   📊 可靠性图: {save_path}")
    plt.close(fig)
    return fig


def compute_calibration(actuals: list[dict], predictions: dict, n_bins: int = 5) -> pd.DataFrame:
    """对 5 维计算 Brier / LogLoss / ECE / 可靠性表"""
    # 5 维定义: (key, classes, actual_field, normalize_fn)
    dims = [
        ("spf", ["胜", "平", "负"], "spf_result", None),
        ("rq",  ["让胜", "让平", "让负"], "rq_result", lambda x: "让" + x),
        ("size", ["大2.5", "小2.5"], None, None),  # 特殊: 从 goals_result 推
        ("htft", ["胜胜", "胜平", "胜负", "平胜", "平平", "平负", "负胜", "负平", "负负"],
         "htft_result", None),
    ]
    results = []
    for dim_key, classes, actual_field, norm_fn in dims:
        y_true_idx, y_prob_list, valid_count = [], [], 0
        for a in actuals:
            p = predictions.get(a["match_id"], {})
            probs_key = f"{dim_key}_probs"
            if probs_key not in p:
                continue
            probs = p[probs_key]
            # 获取实际标签
            if dim_key == "size":
                total = int(a["goals_result"]) if a["goals_result"] != "7+" else 7
                actual = "大2.5" if total > 2.5 else "小2.5"
            elif dim_key == "rq":
                # 500.com rq_result 是 "胜/平/负" 无 "让" 前缀, classes 带前缀
                actual = "让" + a["rq_result"]
            else:
                actual = a[actual_field]
            if actual not in classes:
                continue
            valid_count += 1
            y_true_idx.append(classes.index(actual))
            # 对齐 prob 顺序
            probs_aligned = [probs.get(c, 0.0) for c in classes]
            y_prob_list.append(probs_aligned)
        if valid_count < 3:
            continue
        y_true = np.array(y_true_idx)
        y_prob = np.array(y_prob_list)
        # one-hot
        y_onehot = np.zeros_like(y_prob)
        y_onehot[np.arange(len(y_true)), y_true] = 1.0
        # 二分类化: 取"真类"概率 vs 1-真类概率, 用于 ECE 二元分析
        # 多类场景下 ECE 简化: 用"最高预测概率"作为置信度, 实际=是否命中
        top_pred_idx = y_prob.argmax(axis=1)
        top_conf = y_prob.max(axis=1)
        hit = (top_pred_idx == y_true).astype(float)
        brier = brier_multiclass(y_onehot, y_prob)
        # log loss (clip 防 log(0))
        y_prob_safe = np.clip(y_prob, 1e-15, 1 - 1e-15)
        ll = float(-np.mean(np.log(y_prob_safe[np.arange(len(y_true)), y_true])))
        ece = expected_calibration_error(hit, top_conf, n_bins=n_bins)
        results.append({
            "dim": dim_key,
            "n": valid_count,
            "brier": round(brier, 4),
            "log_loss": round(ll, 4),
            "ece": round(ece, 4),
            "top1_acc": round(hit.mean(), 4),
            "avg_conf": round(float(top_conf.mean()), 4),
        })
    return pd.DataFrame(results)


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="竞彩足球 5 维回测 (500.com kaijiang.php)")
    ap.add_argument("date", help="日期 YYYY-MM-DD")
    ap.add_argument("--predictions", required=True, help="预测 JSON 文件")
    ap.add_argument("--odds", help="赔率 JSON 文件 (可选, 用于 ROI 模拟)")
    ap.add_argument("--calibration", action="store_true", help="启用 Brier/LogLoss/ECE 校准分析 (需要 *_probs)")
    ap.add_argument("--report", help="HTML 报告输出路径 (可选)")
    ap.add_argument("--cal-plot-dir", help="可靠性图输出目录 (可选, 配合 --calibration)")
    args = ap.parse_args()

    # 1) 抓取赛果
    print(f"📥 抓取 500.com 赛果: {args.date} ...")
    actuals = fetch_jczq_kaijiang(args.date)
    print(f"   共 {len(actuals)} 场")
    if not actuals:
        print("❌ 未抓到赛果, 请检查日期或网络")
        return 1

    # 2) 加载预测
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    print(f"📋 加载预测: {len(predictions)} 场")

    # 3) 构建报告
    df = build_report(actuals, predictions)
    stats = summary_stats(df)

    # 4) 控制台输出
    print("\n" + "=" * 100)
    print(f"📊 5 维回测报告: {args.date}")
    print("=" * 100)
    print(df[["match_id", "match", "ft", "pred_spf", "actual_spf", "hit_spf",
              "pred_rq", "actual_rq", "hit_rq", "pred_size", "pred_htft",
              "hit_score", "hit_size", "hit_htft"]].to_string(index=False))

    print("\n📈 5 维命中率 (仅统计有预测的场次):")
    for dim, s in stats.items():
        rate_str = f"{s['rate']*100:.1f}%" if s['rate'] is not None else "N/A"
        print(f"  {dim:<6}: {s['hits']:>3}/{s['total']:>3} = {rate_str}")

    # 5) ROI 模拟
    if args.odds:
        with open(args.odds, "r", encoding="utf-8") as f:
            odds = json.load(f)
        roi_df = roi_simulation(df, actuals, predictions, odds)
        if len(roi_df) > 0:
            print("\n💰 ROI 模拟 (1 元/注):")
            for dim in ["spf", "rq", "size", "htft"]:
                sub = roi_df[roi_df["dim"] == dim]
                if len(sub) == 0:
                    continue
                total_stake = sub["stake"].sum()
                total_pnl = sub["pnl"].sum()
                roi = total_pnl / total_stake * 100
                wins = sub["win"].sum()
                print(f"  {dim:<6}: {wins}/{len(sub)} 命中, ROI = {roi:+.1f}%, 总盈亏 = {total_pnl:+.2f}/{total_stake:.0f}元")

    # 6) 校准分析 (sklearn-style)
    if args.calibration:
        print("\n🎯 校准分析 (Brier / LogLoss / ECE):")
        cal_df = compute_calibration(actuals, predictions)
        if len(cal_df) == 0:
            print("   ⚠️  预测 JSON 中缺少 *_probs 字段, 跳过校准")
        else:
            print(cal_df.to_string(index=False))
            print("\n   指标解读:")
            print("   • Brier ↓     : 多类 Brier 分数, 越低越好, 随机三分类 ≈ 0.667, 完美 = 0")
            print("   • LogLoss ↓   : 对数损失, 越低越好, 反映概率分布的尖锐度")
            print("   • ECE ↓       : 期望校准误差, 0 = 完美校准, >0.1 = 严重失校")
            print("   • top1_acc ↑  : Top-1 命中率 (与上节命中率相同, 此处用 N 样本)")
            print("   • avg_conf    : 平均最高预测概率, 显著高于 top1_acc → 过度自信")
            # 打印每个维度的可靠性表
            print("\n📋 各维度可靠性表 (bin 实际命中率 vs 预测置信度):")
            for dim_key in ["spf", "rq", "size", "htft"]:
                classes_map = {
                    "spf": ["胜", "平", "负"], "rq": ["让胜", "让平", "让负"],
                    "size": ["大2.5", "小2.5"],
                    "htft": ["胜胜","胜平","胜负","平胜","平平","平负","负胜","负平","负负"],
                }
                classes = classes_map[dim_key]
                top_conf, hit = [], []
                for a in actuals:
                    p = predictions.get(a["match_id"], {})
                    probs = p.get(f"{dim_key}_probs")
                    if not probs:
                        continue
                    if dim_key == "size":
                        total = int(a["goals_result"]) if a["goals_result"] != "7+" else 7
                        actual = "大2.5" if total > 2.5 else "小2.5"
                    elif dim_key == "rq":
                        actual = "让" + a["rq_result"]
                    else:
                        actual = a.get(f"{dim_key}_result")
                    if actual not in classes:
                        continue
                    idx = classes.index(actual)
                    probs_aligned = [probs.get(c, 0.0) for c in classes]
                    pred_idx = int(np.argmax(probs_aligned))
                    top_conf.append(probs_aligned[pred_idx])
                    hit.append(1.0 if pred_idx == idx else 0.0)
                if not top_conf:
                    continue
                rt = reliability_table(np.array(hit), np.array(top_conf), n_bins=5)
                print(f"\n   ── {dim_key.upper()} (top1_acc={np.mean(hit)*100:.1f}%, avg_conf={np.mean(top_conf)*100:.1f}%, ECE={expected_calibration_error(np.array(hit), np.array(top_conf), 10):.4f}) ──")
                if len(rt) > 0:
                    print(rt.to_string(index=False))
            # 画可靠性图
            if args.cal_plot_dir:
                from pathlib import Path
                Path(args.cal_plot_dir).mkdir(parents=True, exist_ok=True)
                print(f"\n   📊 生成可靠性图 → {args.cal_plot_dir}/")
                # 对每个维度, 用 top_conf vs 命中画二元校准
                for dim_key in ["spf", "rq", "size", "htft"]:
                    classes_map = {
                        "spf": ["胜", "平", "负"], "rq": ["让胜", "让平", "让负"],
                        "size": ["大2.5", "小2.5"],
                        "htft": ["胜胜","胜平","胜负","平胜","平平","平负","负胜","负平","负负"],
                    }
                    classes = classes_map[dim_key]
                    top_conf, hit = [], []
                    for a in actuals:
                        p = predictions.get(a["match_id"], {})
                        probs = p.get(f"{dim_key}_probs")
                        if not probs:
                            continue
                        if dim_key == "size":
                            total = int(a["goals_result"]) if a["goals_result"] != "7+" else 7
                            actual = "大2.5" if total > 2.5 else "小2.5"
                        elif dim_key == "rq":
                            actual = "让" + a["rq_result"]
                        else:
                            actual = a.get(f"{dim_key}_result")
                        if actual not in classes:
                            continue
                        idx = classes.index(actual)
                        probs_aligned = [probs.get(c, 0.0) for c in classes]
                        pred_idx = int(np.argmax(probs_aligned))
                        top_conf.append(probs_aligned[pred_idx])
                        hit.append(1.0 if pred_idx == idx else 0.0)
                    if not top_conf:
                        continue
                    rt = reliability_table(np.array(hit), np.array(top_conf), n_bins=5)
                    if len(rt) > 0:
                        plot_reliability(rt, dim_key.upper(),
                                         save_path=f"{args.cal_plot_dir}/reliability_{dim_key}.png")

    # 7) HTML 报告
    if args.report:
        html = df.to_html(index=False, classes="table table-striped")
        full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>回测 {args.date}</title>
<style>body{{font-family:sans-serif;padding:20px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px}}
th{{background:#f0f0f0}}.hit{{color:green;font-weight:bold}}.miss{{color:red}}</style>
</head><body>
<h1>竞彩足球回测: {args.date}</h1>
<p>共 {len(df)} 场, 命中 5 维统计：</p>
<ul>{''.join(f'<li>{d}: {s["hits"]}/{s["total"]} = {s["rate"]*100:.1f}%</li>' for d,s in stats.items() if s['rate'] is not None)}</ul>
{html}
</body></html>"""
        Path(args.report).write_text(full_html, encoding="utf-8")
        print(f"\n📄 HTML 报告: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
