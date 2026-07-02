#!/usr/bin/env python3
"""
evaluate_brier.py — Brier Score A/B 评估 + 数据清洗
======================================================
自动清洗：
  1. 过滤 0% 假预测 (pred_h/pred_d/pred_a = 0) → market_fallback 无意义输出
  2. 按 (home_cn, away_cn, match_date) 去重，保留最新预测
     (match_date 为空时兜底 (home_cn, away_cn))

用法:
  python3 evaluate_brier.py              # 全量评估（raw + clean 对比）
  python3 evaluate_brier.py --ab         # A/B 对比 (补丁前 vs 补丁后)
  python3 evaluate_brier.py --new-only   # 仅新数据
  python3 evaluate_brier.py --raw        # 跳过清洗，用原始逻辑
"""

import csv, sys, os
from collections import defaultdict
from datetime import datetime

LOG_PATH = '/root/data/predictions_log.csv'
pred_map = {'主胜': 'H', '平': 'D', '客胜': 'A'}


def load():
    with open(LOG_PATH, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def clean(rows):
    """清洗：去伪(过滤0%预测) + 去重(按主客队+比赛日保留最新)"""
    # 取有 brier_spf 的行
    raw = [r for r in rows if r.get('brier_spf', '').strip()]
    if not raw:
        return [], {'zero_removed': 0, 'dup_removed': 0, 'rows_in': 0, 'rows_out': 0}

    # 1) 去伪: 过滤 pred_h/pred_d/pred_a 全部为 0
    cleaned = []
    zero_count = 0
    for r in raw:
        pred_h = float(r.get('pred_h', 0) or 0)
        pred_d = float(r.get('pred_d', 0) or 0)
        pred_a = float(r.get('pred_a', 0) or 0)
        if pred_h == 0.0 and pred_d == 0.0 and pred_a == 0.0:
            zero_count += 1
            continue
        cleaned.append(r)

    # 2) 去重
    seen = {}
    dup_count = 0
    for r in cleaned:
        key = (r.get('home_cn', '').strip(), r.get('away_cn', '').strip())
        # 如果 match_date 非空，加入去重键
        md = r.get('match_date', '').strip()
        if md:
            key = (key[0], key[1], md)
        date = r.get('date', '')
        if key in seen:
            # 保留 date 较新的
            if date >= seen[key][0]:
                dup_count += 1
                seen[key] = (date, r)
            else:
                dup_count += 1
        else:
            seen[key] = (date, r)

    deduped = [v[1] for v in seen.values()]
    removed = len(cleaned) - len(deduped)

    metrics = {'zero_removed': zero_count, 'dup_removed': removed,
               'rows_in': len(raw), 'rows_out': len(deduped)}
    return deduped, metrics


def brier_report(label, rows, detail=False, show_clean_meta=False):
    """打印一组记录的 Brier 报告"""
    brier_rows = [r for r in rows if r.get('brier_spf', '').strip()]
    if not brier_rows:
        print(f"\n{label}: 无 Brier 数据（比赛尚未回填）")
        return

    briers = [float(r['brier_spf']) for r in brier_rows]
    correct = sum(1 for r in brier_rows
                  if r.get('actual_hda', '').strip() == pred_map.get(r.get('pred_spf_pick', ''), ''))
    n = len(brier_rows)
    avg_brier = sum(briers) / n

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    if show_clean_meta:
        print(f"  (清洗: ✂ {show_clean_meta['zero_removed']}行0%预测, ✂ {show_clean_meta['dup_removed']}行重复)")
    print(f"  Brier Score:  {avg_brier:.4f}")
    print(f"  HDA 准确率:   {correct}/{n} = {correct / n * 100:.1f}%")
    print(f"  样本量:       n={n}")

    # 校准曲线
    print(f"\n  校准曲线:")
    print(f"  {'置信区间':<15} {'预测均值':<9} {'实际胜率':<9} {'样本':<5} {'偏差'}")
    print(f"  {'-' * 50}")
    for lo, hi in [(0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]:
        subset = []
        for r in brier_rows:
            pick = r['pred_spf_pick'].strip()
            k = {'主胜': 'pred_h', '平': 'pred_d', '客胜': 'pred_a'}.get(pick, 'pred_h')
            p = float(r.get(k, 0))
            if lo <= p < hi:
                subset.append((r, p))
        if subset:
            act = sum(1 for r, _ in subset if r['actual_hda'].strip() == pred_map.get(r['pred_spf_pick'].strip(), ''))
            avg_p = sum(p for _, p in subset) / len(subset)
            act_pct = act / len(subset) * 100
            bias = act_pct - avg_p
            mark = " ← 严重过自信" if bias < -20 else (" ← 严重保守" if bias > 20 else "")
            print(f"  [{lo:>3}-{hi:<3}%):   {avg_p:<7.1f}%   {act_pct:<7.1f}%  {len(subset):<4} {bias:+.1f}%{mark}")

    # 平局预测
    draw_pred = sum(1 for r in brier_rows if r.get('pred_spf_pick', '').strip() == '平')
    draw_actual = sum(1 for r in brier_rows if r.get('actual_hda', '').strip() == 'D')
    draw_correct = sum(1 for r in brier_rows
                       if r.get('pred_spf_pick', '').strip() == '平' and r.get('actual_hda', '').strip() == 'D')
    print(f"\n  平局预测: {draw_pred} 场 (正确 {draw_correct}) | 实际平局: {draw_actual} 场")

    if detail:
        print(f"\n  明细:")
        for r in sorted(brier_rows, key=lambda x: float(x['brier_spf'])):
            pick = r['pred_spf_pick']
            k = {'主胜': 'pred_h', '平': 'pred_d', '客胜': 'pred_a'}.get(pick, 'pred_h')
            print(f"    {r['brier_spf']:>6s} | {r['home_cn']:<12} vs {r['away_cn']:<12} "
                  f"→ 预测={pick}({r.get(k, '?'):>4s}%) 实际={r['actual_hda']} route={r.get('model_route', '?')}")


def five_market_summary(rows, label=""):
    """打印 5 玩法独立校准概览"""
    prefix = f"  [{label}] " if label else "  "
    markets = [
        ('brier_rq', '让球 RQ Brier'),
        ('acc_score_top1', '比分 Score Acc'),
        ('acc_goals_top1', '总进球 Goals Acc'),
        ('goals_mae', '总进球 Goals MAE'),
        ('acc_htft_top1', '半全场 HTFT Acc'),
    ]
    for key, mlabel in markets:
        vals = []
        n_ok = 0
        n_total = 0
        for r in rows:
            v = r.get(key, '').strip()
            if v:
                n_total += 1
                if key in ('brier_rq', 'goals_mae'):
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
                else:
                    if v == '1':
                        n_ok += 1
        if key in ('brier_rq', 'goals_mae') and vals:
            if key == 'goals_mae':
                print(f"{prefix}{mlabel}: {sum(vals) / len(vals):.1f}  (n={len(vals)})")
            else:
                print(f"{prefix}{mlabel}: {sum(vals) / len(vals):.4f}  (n={len(vals)})")
        elif n_total > 0:
            print(f"{prefix}{mlabel}: {n_ok}/{n_total} = {n_ok / n_total * 100:.1f}%")


def main():
    rows = load()
    brier_all = [r for r in rows if r.get('brier_spf', '').strip()]

    # 旧/新分组
    old = [r for r in brier_all if not r.get('model_route', '').strip() or r['model_route'] == 'unknown']
    new = [r for r in brier_all if r.get('model_route', '').strip() and r['model_route'] != 'unknown']

    # 清洗: 去伪 + 去重
    clean_new, metrics = clean(new)
    clean_old, _ = clean(old)

    if '--raw' in sys.argv:
        # 跳过清洗
        brier_report("📊 全量 Brier 评估 (raw)", brier_all)
        brier_report("🧓 补丁前 (raw)", old)
        brier_report("🆕 补丁后 (raw)", new, detail=True)
        return

    # ── 对比报告: RAW vs CLEAN ──
    print("=" * 60)
    print("  📊 Brier 评估报告 — RAW vs CLEAN (去伪+去重)")
    print("=" * 60)

    if '--ab' in sys.argv or '--all' in sys.argv:
        # Raw 全量
        brier_report("📊 RAW 全量 Brier", brier_all)
        brier_report("🧓 RAW 补丁前", old)
        brier_report("🆕 RAW 补丁后", new, detail=True)

        print(f"\n{'─' * 60}")
        print("  ▼ 清洗后 (去伪: 过滤0%预测, 去重: 保留最新)")
        print(f"{'─' * 60}")

        # Clean 全量
        clean_all, m_all = clean(brier_all)
        show_clean = m_all if m_all else None
        brier_report("📊 CLEAN 全量 Brier", clean_all, show_clean_meta=show_clean)
        brier_report("🧓 CLEAN 补丁前", clean_old)
        brier_report("🆕 CLEAN 补丁后", clean_new, detail=True, show_clean_meta=metrics)

    elif '--new-only' in sys.argv:
        brier_report("🆕 RAW 补丁后", new, detail=True)
        print(f"\n{'─' * 60}")
        print("  ▼ 清洗后")
        print(f"{'─' * 60}")
        brier_report("🆕 CLEAN 补丁后", clean_new, detail=True, show_clean_meta=metrics)

    elif '--old-only' in sys.argv:
        brier_report("🧓 CLEAN 补丁前", clean_old, detail=True)

    else:
        # 默认: RAW 全量 + CLEAN 全量 + 对比
        clean_all, m_all = clean(brier_all)
        brier_report("📊 RAW 全量 Brier", brier_all)
        brier_report("📊 CLEAN 全量 Brier", clean_all, show_clean_meta=m_all)

        if clean_old:
            brier_report("🧓 CLEAN 补丁前", clean_old)
        if clean_new:
            brier_report("🆕 CLEAN 补丁后", clean_new, detail=True, show_clean_meta=metrics)

    # ── 对比摘要 ──
    print(f"\n{'=' * 60}")
    print("  A/B 对比摘要")
    print(f"{'=' * 60}")

    for group_name, group in [("补丁前", old), ("补丁后", new)]:
        if not group:
            continue
        raw_b = [float(r['brier_spf']) for r in group]
        raw_acc = sum(1 for r in group if r['actual_hda'].strip() == pred_map.get(r['pred_spf_pick'].strip(), ''))
        print(f"  {group_name} RAW:   Brier={sum(raw_b) / len(raw_b):.4f}  Acc={raw_acc}/{len(group)}={raw_acc / len(group) * 100:.1f}%  n={len(group)}")

        cg = clean(group)[0]
        if cg:
            cl_b = [float(r['brier_spf']) for r in cg]
            cl_acc = sum(1 for r in cg if r['actual_hda'].strip() == pred_map.get(r['pred_spf_pick'].strip(), ''))
            print(f"  {group_name} CLEAN: Brier={sum(cl_b) / len(cl_b):.4f}  Acc={cl_acc}/{len(cg)}={cl_acc / len(cg) * 100:.1f}%  n={len(cg)}")

    # ── 5 玩法 ──
    clean_all, _ = clean(brier_all)
    print(f"\n  ── 5 玩法校准 (CLEAN) ──")
    five_market_summary(clean_all)
    # Route breakdown
    if clean_new:
        print()
        for route in sorted(set(r.get('model_route', '?') for r in clean_new)):
            grp = [r for r in clean_new if r.get('model_route', '?') == route]
            if grp:
                b = [float(r['brier_spf']) for r in grp]
                a = sum(1 for r in grp if r['actual_hda'].strip() == pred_map.get(r['pred_spf_pick'].strip(), ''))
                print(f"  {route:30s} Brier={sum(b)/len(b):.4f}  Acc={a}/{len(grp)}={a/len(grp)*100:.1f}%  n={len(grp)}")


if __name__ == '__main__':
    main()
