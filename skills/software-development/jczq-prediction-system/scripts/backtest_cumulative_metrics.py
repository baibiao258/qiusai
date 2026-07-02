#!/usr/bin/env python3
"""
backtest_cumulative_metrics.py — 从 predictions_log.csv 计算累计核验指标

用途: 当 `backtest_pipeline.py --verify` 因输出截断/无新增而无法查看汇总时,
      用此脚本直接读取 CSV 计算所有已核验 (`checked=1`) 比赛的累计指标。

输出:
  - 总行数、已核验/未核验统计
  - Brier Score / RPS / Log Loss / 准确率
  - 按结果类型 (H/D/A) 拆分
  - 与 backtest_results.json 对比

用法:
  python3 [skill-path]/scripts/backtest_cumulative_metrics.py
  python3 [skill-path]/scripts/backtest_cumulative_metrics.py --json-only  # JSON 输出
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

CSV_PATH = '/root/data/predictions_log.csv'
BACKTEST_PATH = '/root/data/backtest_results.json'
TODAY = os.environ.get('BACKTEST_TODAY') or __import__('datetime').date.today().isoformat()  # 可被环境变量覆盖, 默认取实际今天


def score_to_hda(hg, ag):
    if hg > ag:
        return 'H'
    elif hg == ag:
        return 'D'
    else:
        return 'A'


def brier_score(y_idx, probs_3):
    y = np.zeros(3)
    y[y_idx] = 1.0
    p = np.asarray(probs_3, dtype=float)
    return float(np.sum((y - p) ** 2))


def rps_single(y_idx, probs_3):
    cdf_true = np.cumsum(np.eye(3)[y_idx])
    cdf_pred = np.cumsum(np.asarray(probs_3, dtype=float))
    return float(np.sum((cdf_true - cdf_pred) ** 2) / 2.0)


def log_loss_single(y_idx, probs_3):
    p = np.clip(np.asarray(probs_3, dtype=float), 1e-10, 1.0)
    p = p / p.sum()
    return -math.log(p[y_idx])


def main():
    parser = argparse.ArgumentParser(description='Cumulative backtest metrics from predictions_log.csv')
    parser.add_argument('--json-only', action='store_true', help='Output only JSON summary')
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        print(f'❌ {CSV_PATH} 不存在')
        sys.exit(1)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    checked = sum(1 for r in rows if r.get('checked') == '1')
    unchecked = total - checked
    have_score = sum(1 for r in rows if r.get('actual_score', '').strip())
    no_score = total - have_score

    # === 诊断: 未核验比赛的状态 ===
    unchecked_rows = [r for r in rows if r.get('checked') != '1']
    past_due = sum(1 for r in unchecked_rows if r.get('date', '') < TODAY)
    today_count = sum(1 for r in unchecked_rows if r.get('date', '') == TODAY)
    future_count = sum(1 for r in unchecked_rows if r.get('date', '') > TODAY)

    # === 计算已核验比赛的指标 ===
    verified = [r for r in rows if r.get('checked') == '1' and r.get('actual_score', '').strip()]
    metrics = []

    for r in verified:
        s = r.get('actual_score', '').strip()
        try:
            parts = s.split(':')
            hg, ag = int(parts[0].strip()), int(parts[1].strip())
        except (ValueError, IndexError):
            continue
        hda = score_to_hda(hg, ag)
        idx_map = {'A': 0, 'D': 1, 'H': 2}
        y_idx = idx_map[hda]

        try:
            pred_h = float(r.get('pred_h', 0)) / 100.0
            pred_d = float(r.get('pred_d', 0)) / 100.0
            pred_a = float(r.get('pred_a', 0)) / 100.0
        except (ValueError, TypeError):
            continue

        # 跳过全零预测（market_fallback 无赔率时的假预测）
        if pred_h == 0 and pred_d == 0 and pred_a == 0:
            continue

        probs = [pred_a, pred_d, pred_h]  # [A, D, H]
        bs = brier_score(y_idx, probs)
        rps = rps_single(y_idx, probs)
        ll = log_loss_single(y_idx, probs)
        acc = 1.0 if np.argmax(probs) == y_idx else 0.0

        metrics.append({
            'code': r['code'],
            'home': r.get('home_cn', '?'),
            'away': r.get('away_cn', '?'),
            'date': r.get('date', ''),
            'actual_hda': hda,
            'pred_hda': ['A', 'D', 'H'][np.argmax(probs)],
            'score': s,
            'brier': bs,
            'rps': rps,
            'log_loss': ll,
            'acc': acc,
            'correct': acc == 1.0,
            'model_route': r.get('model_route', ''),
            'bet_action': r.get('bet_action', ''),
        })

    n = len(metrics)
    if n == 0:
        if args.json_only:
            print(json.dumps({'error': 'no computable metrics', 'total_rows': total, 'checked': checked}))
        else:
            print(f'总行数: {total}')
            print(f'已核验: {checked}, 未核验: {unchecked}')
            print(f'有比分: {have_score}, 缺比分: {no_score}')
            print('❌ 无有效指标可计算（所有已核验行均缺实际比分或全零预测）')
        return

    brier_avg = np.mean([m['brier'] for m in metrics])
    rps_avg = np.mean([m['rps'] for m in metrics])
    ll_avg = np.mean([m['log_loss'] for m in metrics])
    acc_avg = np.mean([m['acc'] for m in metrics])
    correct = sum(1 for m in metrics if m['correct'])

    # 按结果类型拆分
    by_hda = {}
    for hda in ['H', 'D', 'A']:
        subset = [m for m in metrics if m['actual_hda'] == hda]
        if subset:
            by_hda[hda] = {
                'n': len(subset),
                'acc': np.mean([m['acc'] for m in subset]),
                'brier': np.mean([m['brier'] for m in subset]),
            }
        else:
            by_hda[hda] = {'n': 0, 'acc': 0.0, 'brier': 0.0}

    # 读取 backtest_results.json 对比
    last_daily = None
    if os.path.exists(BACKTEST_PATH):
        try:
            bt = json.load(open(BACKTEST_PATH))
            daily_records = [r for r in bt if r.get('type') == 'daily_verify']
            if daily_records:
                last_daily = daily_records[-1]
        except Exception:
            pass

    result = {
        'total_rows': total,
        'checked': checked,
        'unchecked': unchecked,
        'unchecked_past_due': past_due,
        'unchecked_today': today_count,
        'unchecked_future': future_count,
        'computable_metrics': n,
        'metrics': {
            'brier': round(brier_avg, 4),
            'rps': round(rps_avg, 4),
            'log_loss': round(ll_avg, 4),
            'accuracy': round(acc_avg, 4),
            'correct': correct,
        },
        'by_outcome': {k: {
            'n': v['n'],
            'accuracy': round(v['acc'], 4),
            'brier': round(v['brier'], 4),
        } for k, v in by_hda.items()},
        'details': metrics,
    }

    if args.json_only:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # === 人类可读输出 ===
    print(f'总行数: {total}')
    print(f'已核验 (checked=1): {checked}')
    print(f'未核验: {unchecked} (已过期 {past_due}, 今日 {today_count}, 未来 {future_count})')
    print()

    print(f'=== 累计指标 ({n} 场有效) ===')
    print(f'Brier Score:  {brier_avg:.4f}  (随机=0.667, 越低越好)')
    print(f'RPS:          {rps_avg:.4f}  (随机=0.333, 越低越好)')
    print(f'Log Loss:     {ll_avg:.4f}  (随机=1.099, 越低越好)')
    print(f'准确率:        {acc_avg*100:.1f}%  (随机=33.3%)')
    print(f'正确:         {correct}/{n}')

    print()
    print('=== 按结果类型拆分 ===')
    for hda in ['H', 'D', 'A']:
        info = by_hda[hda]
        if info['n'] > 0:
            print(f'  {hda}: {info["n"]} 场, 准确率={info["acc"]*100:.1f}%, Brier={info["brier"]:.4f}')

    print()
    print('=== 与 backtest_results.json 对比 ===')
    if last_daily:
        print(f'上次 daily_verify: {last_daily["timestamp"][:19]}')
        print(f'  场次: {last_daily["n_matches"]} → 当前累计: {n}')
        print(f'  Brier: {last_daily["brier"]:.4f} → {brier_avg:.4f}')
        print(f'  准确率: {last_daily["accuracy"]*100:.1f}% → {acc_avg*100:.1f}%')
    else:
        print('  backtest_results.json 无 daily_verify 记录')

    # 模型路由分布
    route_dist = defaultdict(int)
    for m in verified:
        route = m.get('model_route', 'unknown') or 'unknown'
        route_dist[route] += 1
    print()
    print('=== model_route 分布 ===')
    for route, count in sorted(route_dist.items(), key=lambda x: -x[1]):
        subset = [m for m in metrics if m['model_route'] == route]
        if subset:
            acc_r = np.mean([m['acc'] for m in subset])
            print(f'  {route}: {count} 场, 准确率={acc_r*100:.1f}%')
        else:
            print(f'  {route}: {count} 场')

    # 样本错误预测
    wrong = [m for m in metrics if not m['correct']]
    if wrong:
        print()
        print(f'=== 错误预测样本 (显示前 10 条) ===')
        for w in wrong[:10]:
            print(f'  {w["code"]} {w["home"]} vs {w["away"]}: 预测={w["pred_hda"]}, 实际={w["actual_hda"]} ({w["score"]})')


if __name__ == '__main__':
    main()
