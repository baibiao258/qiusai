#!/usr/bin/env python3
"""
verify_daily.py — 截断安全的每日核验替代脚本

在 backtest_pipeline.py --verify 因异常行警告过多被截断时使用。
直接读取 predictions_log.csv, 无 subprocess 调用, 输出精简。

用法:
  python3 verify_daily.py                         # 默认模式, 只输出汇总
  python3 verify_daily.py --verbose                # 输出逐场明细
  python3 verify_daily.py --dry-run                # 预览, 不写回CSV
  python3 verify_daily.py --silent                 # 只输出 JSON (cron 友好)
"""
import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, date

import numpy as np

DATA_DIR = '/root/data'
PREDICTIONS_LOG = os.path.join(DATA_DIR, 'predictions_log.csv')
BACKTEST_RESULTS = os.path.join(DATA_DIR, 'backtest_results.json')


def score_to_hda(hg, ag):
    if hg > ag:
        return 'H'
    elif hg == ag:
        return 'D'
    return 'A'


def outcome_to_idx(hda_str):
    return {'A': 0, 'D': 1, 'H': 2}.get(hda_str)


def brier_score(y_true_idx, probs_3):
    y = np.zeros(3)
    y[y_true_idx] = 1.0
    return float(np.sum((y - np.asarray(probs_3)) ** 2))


def rps_single(y_true_idx, probs_3):
    cdf_true = np.cumsum(np.eye(3)[y_true_idx])
    cdf_pred = np.cumsum(np.asarray(probs_3, dtype=float))
    return float(np.sum((cdf_true - cdf_pred) ** 2) / 2.0)


def log_loss_single(y_true_idx, probs_3):
    p = np.clip(np.asarray(probs_3, dtype=float), 1e-10, 1.0)
    p = p / p.sum()
    return -math.log(p[y_true_idx])


def accuracy_from_probs(y_true_idx, probs_3):
    return 1.0 if np.argmax(probs_3) == y_true_idx else 0.0


def main():
    parser = argparse.ArgumentParser(description='每日核验 (截断安全版)')
    parser.add_argument('--verbose', action='store_true', help='输出逐场明细')
    parser.add_argument('--dry-run', action='store_true', help='预览模式, 不写回')
    parser.add_argument('--silent', action='store_true', help='静默模式, 只输出 JSON')
    args = parser.parse_args()

    if not os.path.exists(PREDICTIONS_LOG):
        if args.silent:
            print(json.dumps({'error': 'predictions_log.csv not found'}))
        else:
            print('❌ predictions_log.csv 不存在')
        return

    with open(PREDICTIONS_LOG, 'r') as f:
        rows = list(csv.DictReader(f))

    today = date.today().isoformat()
    verified = []
    skipped_unchecked = 0
    skipped_past_no_score = 0
    skipped_today = 0
    skipped_corrupted = 0
    skipped_checked = 0

    for i, r in enumerate(rows):
        # 已核验跳过
        if r.get('checked') == '1':
            skipped_checked += 1
            continue

        # 未来比赛跳过
        match_date = r.get('date', '')
        if match_date >= today:
            skipped_today += 1
            continue

        # 无实际比分跳过
        actual_score = r.get('actual_score', '').strip()
        if not actual_score:
            skipped_past_no_score += 1
            continue

        # 解析比分
        try:
            parts = actual_score.split(':')
            hg, ag = int(parts[0].strip()), int(parts[1].strip())
        except (ValueError, IndexError):
            skipped_corrupted += 1
            continue

        # 解析预测概率
        try:
            pred_h = float(r.get('pred_h', 0)) / 100.0
            pred_d = float(r.get('pred_d', 0)) / 100.0
            pred_a = float(r.get('pred_a', 0)) / 100.0
        except (ValueError, TypeError):
            skipped_corrupted += 1
            continue

        actual_hda = score_to_hda(hg, ag)
        y_idx = outcome_to_idx(actual_hda)
        probs = [pred_a, pred_d, pred_h]  # [A, D, H]

        bs = brier_score(y_idx, probs)
        rps = rps_single(y_idx, probs)
        ll = log_loss_single(y_idx, probs)
        acc = accuracy_from_probs(y_idx, probs)

        verified.append({
            'idx': i,
            'code': r.get('code', ''),
            'home': r.get('home_cn', ''),
            'away': r.get('away_cn', ''),
            'league': r.get('league', ''),
            'date': r.get('date', ''),
            'actual_score': actual_score,
            'actual_hda': actual_hda,
            'pred_h': pred_h,
            'pred_d': pred_d,
            'pred_a': pred_a,
            'brier': round(bs, 4),
            'rps': round(rps, 4),
            'log_loss': round(ll, 4),
            'acc': acc,
        })

        # 在不干模式中不写回
        if not args.dry_run:
            rows[i]['actual_hda'] = actual_hda
            rows[i]['actual_goals'] = actual_score
            rows[i]['checked'] = '1'

    # 写回 CSV
    if verified and not args.dry_run:
        with open(PREDICTIONS_LOG, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # 汇总
    if args.silent:
        result = {
            'total_rows': len(rows),
            'skipped_checked': skipped_checked,
            'skipped_past_no_score': skipped_past_no_score,
            'skipped_today': skipped_today,
            'skipped_corrupted': skipped_corrupted,
            'verified_count': len(verified),
            'verified': [],
        }
        if verified:
            brier_avg = float(np.mean([m['brier'] for m in verified]))
            rps_avg = float(np.mean([m['rps'] for m in verified]))
            acc_avg = float(np.mean([m['acc'] for m in verified]))
            result['metrics'] = {
                'brier': round(brier_avg, 4),
                'rps': round(rps_avg, 4),
                'acc': round(acc_avg, 4),
                'correct': sum(1 for m in verified if m['acc'] == 1.0),
                'wrong': sum(1 for m in verified if m['acc'] == 0.0),
            }
            if args.verbose:
                result['verified'] = [
                    {k: v for k, v in m.items() if k in ('code', 'home', 'away', 'actual_score', 'brier', 'acc')}
                    for m in verified
                ]
        else:
            result['metrics'] = None
        print(json.dumps(result, ensure_ascii=False))
        return

    # 人类可读输出
    print(f'predictions_log.csv: {len(rows)} 行')
    print(f'  已核验: {skipped_checked}')
    print(f'  今日/未来比赛: {skipped_today}')
    print(f'  过去无比分: {skipped_past_no_score}')
    print(f'  数据损坏: {skipped_corrupted}')
    print(f'  本次核验: {len(verified)}')

    if not verified:
        print('\n📭 今日无新增可核验比赛')
        return

    brier_avg = np.mean([m['brier'] for m in verified])
    rps_avg = np.mean([m['rps'] for m in verified])
    ll_avg = np.mean([m['log_loss'] for m in verified])
    acc_avg = np.mean([m['acc'] for m in verified])
    correct = sum(1 for m in verified if m['acc'] == 1.0)
    wrong = len(verified) - correct

    print(f'\n{"=" * 60}')
    print(f'📊 每日核验结果 ({len(verified)} 场)')
    print(f'{"=" * 60}')
    print(f'  ✅ 正确: {correct}  ({acc_avg * 100:.1f}%)')
    print(f'  ❌ 错误: {wrong}  ({(1 - acc_avg) * 100:.1f}%)')
    print(f'  Brier Score: {brier_avg:.4f}  (随机=0.667)')
    print(f'  RPS:         {rps_avg:.4f}  (随机=0.333)')
    print(f'  Log Loss:    {ll_avg:.4f}  (随机=1.099)')
    print(f'  准确率:       {acc_avg * 100:.1f}%  (随机=33.3%)')

    if args.verbose:
        for m in verified:
            correct_str = '✅' if m['acc'] == 1.0 else '❌'
            print(f'  {correct_str} {m["code"]} {m["home"]} vs {m["away"]}: '
                  f'实际={m["actual_score"]} '
                  f'预测 H={m["pred_h"] * 100:.1f}% D={m["pred_d"] * 100:.1f}% A={m["pred_a"] * 100:.1f}% '
                  f'Brier={m["brier"]:.4f} RPS={m["rps"]:.4f}')

    # 追加到 backtest_results.json
    record = {
        'timestamp': datetime.now().isoformat(),
        'type': 'daily_verify',
        'n_matches': len(verified),
        'brier': round(brier_avg, 4),
        'rps': round(rps_avg, 4),
        'log_loss': round(ll_avg, 4),
        'accuracy': round(acc_avg, 4),
        'correct_count': correct,
        'wrong_count': wrong,
        'details': [{
            'code': m['code'],
            'home': m['home'],
            'away': m['away'],
            'score': m['actual_score'],
            'actual_hda': m['actual_hda'],
            'brier': m['brier'],
            'rps': m['rps'],
            'log_loss': m['log_loss'],
            'acc': m['acc'],
        } for m in verified],
    }

    if not args.dry_run:
        existing = []
        if os.path.exists(BACKTEST_RESULTS):
            try:
                with open(BACKTEST_RESULTS) as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except Exception:
                existing = []

        existing.append(record)
        with open(BACKTEST_RESULTS, 'w') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f'\n  💾 已保存到 {BACKTEST_RESULTS}')
    else:
        print(f'\n  (dry-run 模式, 未保存)')


if __name__ == '__main__':
    main()
