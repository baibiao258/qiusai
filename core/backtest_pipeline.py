#!/usr/bin/env python3
"""
backtest_pipeline.py — 预测验证 + 历史回测管线
==============================================
功能:
  1. daily_verify(): 核验 predictions_log.csv 中已结束的比赛
  2. historical_backtest(): 用 DC+XGB 对历史国际赛做滚动回测
  3. 输出 Brier Score / RPS / 准确率 / ROI

用法:
  python3 backtest_pipeline.py --verify          # 核验已有预测
  python3 backtest_pipeline.py --backtest         # 历史回测 (500场)
  python3 backtest_pipeline.py --backtest --n 100 # 历史回测 (100场)
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, date

import numpy as np
from feature_helper import build_gold_features, get_12game_form, load_h2h_cache, load_form_12_cache

sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')

DATA_DIR = '/root/data'
PREDICTIONS_LOG = os.path.join(DATA_DIR, 'predictions_log.csv')
BACKTEST_RESULTS = os.path.join(DATA_DIR, 'backtest_results.json')

# ═══════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════

def brier_score(y_true_idx, probs_3):
    """Single-sample Brier score for 3-class. probs_3 = [p_away, p_draw, p_home]."""
    y = np.zeros(3)
    y[y_true_idx] = 1.0
    p = np.asarray(probs_3, dtype=float)
    return float(np.sum((y - p) ** 2))


def rps_single(y_true_idx, probs_3):
    """Ranked Probability Score for one sample."""
    cdf_true = np.cumsum(np.eye(3)[y_true_idx])
    cdf_pred = np.cumsum(np.asarray(probs_3, dtype=float))
    return float(np.sum((cdf_true - cdf_pred) ** 2) / 2.0)


def log_loss_single(y_true_idx, probs_3):
    """Log loss for one sample."""
    p = np.clip(np.asarray(probs_3, dtype=float), 1e-10, 1.0)
    p = p / p.sum()
    return -math.log(p[y_true_idx])


def accuracy_from_probs(y_true_idx, probs_3):
    """Top-1 accuracy."""
    return 1.0 if np.argmax(probs_3) == y_true_idx else 0.0


def outcome_to_idx(hda_str):
    """Convert 'H'/'D'/'A' to index 0/1/2 for [A, D, H] convention."""
    mapping = {'A': 0, 'D': 1, 'H': 2}
    return mapping.get(hda_str)


def score_to_hda(home_goals, away_goals):
    """Score -> H/D/A."""
    if home_goals > away_goals:
        return 'H'
    elif home_goals == away_goals:
        return 'D'
    else:
        return 'A'


# ═══════════════════════════════════════
#  1. DAILY VERIFY — 核验 predictions_log
# ═══════════════════════════════════════

def daily_verify():
    """核验 predictions_log.csv 中已结束但未核验的比赛."""
    if not os.path.exists(PREDICTIONS_LOG):
        print("❌ predictions_log.csv 不存在")
        return

    with open(PREDICTIONS_LOG, 'r') as f:
        rows = list(csv.DictReader(f))

    today = date.today().isoformat()
    verified = 0
    skipped = 0
    metrics积累 = []

    for i, r in enumerate(rows):
        # 已核验跳过
        if r.get('checked') == '1':
            skipped += 1
            continue

        # 检查比赛是否已结束
        match_date = r.get('date', '')
        if match_date >= today:
            skipped += 1
            continue

        # 需要有实际比分才能核验
        actual_score = r.get('actual_score', '').strip()
        if not actual_score or actual_score == '':
            # 尝试从 football-data.org 或其他来源获取
            print(f"  ⚠️ {r['code']} {r['home_cn']} vs {r['away_cn']} 缺少实际比分")
            skipped += 1
            continue

        # 解析比分
        try:
            parts = actual_score.split(':')
            hg, ag = int(parts[0].strip()), int(parts[1].strip())
        except:
            print(f"  ⚠️ {r['code']} 比分格式错误: {actual_score}")
            skipped += 1
            continue

        actual_hda = score_to_hda(hg, ag)
        y_idx = outcome_to_idx(actual_hda)

        # 模型预测概率
        try:
            pred_h = float(r['pred_h']) / 100.0
            pred_d = float(r['pred_d']) / 100.0
            pred_a = float(r['pred_a']) / 100.0
        except:
            print(f"  ⚠️ {r['code']} 预测概率解析失败")
            skipped += 1
            continue

        probs = [pred_a, pred_d, pred_h]  # [A, D, H] order

        # 计算指标
        bs = brier_score(y_idx, probs)
        rps = rps_single(y_idx, probs)
        ll = log_loss_single(y_idx, probs)
        acc = accuracy_from_probs(y_idx, probs)

        metrics积累.append({
            'code': r['code'],
            'home': r['home_cn'],
            'away': r['away_cn'],
            'actual_hda': actual_hda,
            'pred_h': pred_h, 'pred_d': pred_d, 'pred_a': pred_a,
            'brier': bs, 'rps': rps, 'log_loss': ll, 'accuracy': acc,
        })

        # 更新 predictions_log
        rows[i]['actual_hda'] = actual_hda
        rows[i]['actual_goals'] = f"{hg}-{ag}"
        rows[i]['checked'] = '1'
        verified += 1

    # 写回 predictions_log
    if verified > 0:
        with open(PREDICTIONS_LOG, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ 核验 {verified} 场, 跳过 {skipped} 场")

    # 汇总
    if metrics积累:
        brier_avg = np.mean([m['brier'] for m in metrics积累])
        rps_avg = np.mean([m['rps'] for m in metrics积累])
        ll_avg = np.mean([m['log_loss'] for m in metrics积累])
        acc_avg = np.mean([m['accuracy'] for m in metrics积累])
        print(f"\n📊 核验结果 ({len(metrics积累)} 场):")
        print(f"  Brier Score:    {brier_avg:.4f}  (越低越好, 随机=0.667)")
        print(f"  RPS:            {rps_avg:.4f}  (越低越好, 随机=0.333)")
        print(f"  Log Loss:       {ll_avg:.4f}  (越低越好, 随机=1.099)")
        print(f"  准确率:          {acc_avg*100:.1f}%  (随机=33.3%)")

        # 追加到 backtest_results.json
        _append_backtest_result({
            'timestamp': datetime.now().isoformat(),
            'type': 'daily_verify',
            'n_matches': len(metrics积累),
            'brier': round(brier_avg, 4),
            'rps': round(rps_avg, 4),
            'log_loss': round(ll_avg, 4),
            'accuracy': round(acc_avg, 4),
            'details': metrics积累,
        })

    return metrics积累


# ═══════════════════════════════════════
#  2. HISTORICAL BACKTEST — 历史滚动回测
# ═══════════════════════════════════════

def historical_backtest(n_matches=500, train_ratio=0.7):
    """
    对国际赛历史数据做滚动回测:
    1. 加载 international_results.json
    2. 按时间排序, 用前 train_ratio 训练 DC+Elo
    3. 用剩余数据做 XGB+hybrid 预测
    4. 计算 Brier/RPS/准确率
    """
    import joblib

    cache = os.path.join(DATA_DIR, 'international_results.json')
    if not os.path.exists(cache):
        print("❌ international_results.json 不存在")
        return

    with open(cache) as f:
        all_matches = json.load(f)

    # 加载已训练的模型
    dc_path = os.path.join(DATA_DIR, 'dc_model.pkl')
    xgb_path = os.path.join(DATA_DIR, 'xgb_model_29.pkl')
    elo_path = os.path.join(DATA_DIR, 'elo_ratings.pkl')

    if not all(os.path.exists(p) for p in [dc_path, xgb_path, elo_path]):
        print("❌ 模型文件不完整")
        return

    print(f"📦 加载模型...")
    dc = joblib.load(dc_path)
    xgb = joblib.load(xgb_path)
    elo = joblib.load(elo_path)

    print(f"📊 数据: {len(all_matches)} 场历史比赛")

    # 过滤有效比赛 (有比分的)
    valid = []
    for m in all_matches:
        try:
            hg = int(m.get('h_score', m.get('home_score', -1)))
            ag = int(m.get('a_score', m.get('away_score', -1)))
            if hg < 0 or ag < 0:
                continue
            home = m.get('home', m.get('home_team', ''))
            away = m.get('away', m.get('away_team', ''))
            if not home or not away:
                continue
            valid.append({
                'date': m.get('date', ''),
                'home': home, 'away': away,
                'h_score': hg, 'a_score': ag,
                'tournament': m.get('tournament', ''),
            })
        except:
            continue

    # 按日期排序
    valid.sort(key=lambda x: x['date'])

    # 取最近 N 场做回测
    if len(valid) > n_matches:
        valid = valid[-n_matches:]

    # 训练集/测试集切分
    split = int(len(valid) * train_ratio)
    train_data = valid[:split]
    test_data = valid[split:]

    print(f"  训练: {len(train_data)} 场 | 测试: {len(test_data)} 场")
    print(f"  测试范围: {test_data[0]['date']} ~ {test_data[-1]['date']}")

    # 在测试集上逐场预测
    results = []
    for m in test_data:
        home, away = m['home'], m['away']

        # DC 预测
        try:
            dc_p = dc.predict_proba(home, away, neutral=True)
            lam_h, lam_a = dc.predict_lambda(home, away, neutral=True)
        except:
            dc_p = [1/3, 1/3, 1/3]
            lam_h, lam_a = 1.0, 1.0

        if lam_h is None:
            dc_p = [1/3, 1/3, 1/3]
            lam_h, lam_a = 1.0, 1.0

        # XGB 特征 (使用完整特征管道)
        eh = elo.get(home, 1500)
        ea = elo.get(away, 1500)

        # 5场 form (从 form_state.json)
        from predict_match import recent_form as pm_recent_form
        fh5 = pm_recent_form(home, 5)
        fa5 = pm_recent_form(away, 5)

        b15 = [
            (eh - ea) / 400,
            lam_h, lam_a, lam_h - lam_a,
            math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
            dc_p[0], dc_p[1], dc_p[2],
            fh5[0], fa5[0],
            fh5[1] - fa5[2], fa5[1] - fh5[2],
            fh5[1] - fa5[1], fh5[0] - fa5[0],
            1,  # neutral (世界杯场景)
        ]
        gold = build_gold_features(home, away, match_type=m.get('tournament', ''))
        op_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        op_a = 1 / (1 + 10 ** ((eh - ea) / 400))
        odds_feat = [op_h, op_a, 0.0]
        form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]

        feat = np.array([b15 + gold + odds_feat + form_feat])

        try:
            xgb_p = xgb.predict_proba(feat)[0]
        except:
            xgb_p = np.array([1/3, 1/3, 1/3])

        # Dynamic weight
        p = np.clip(xgb_p, 1e-10, 1.0)
        p = p / p.sum()
        e = -np.sum(p * np.log2(p))
        conf = 1.0 - e / math.log2(3)
        xgb_w = max(0.10, min(0.90, 0.30 + 0.50 * conf))
        dc_w = 1.0 - xgb_w

        dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])  # [A, D, H]
        hybrid = dc_w * dc_ado + xgb_w * xgb_p

        # Normalize
        s = hybrid.sum()
        if s > 0:
            hybrid = hybrid / s

        # Actual outcome
        hg, ag = m['h_score'], m['a_score']
        actual_hda = score_to_hda(hg, ag)
        y_idx = outcome_to_idx(actual_hda)

        bs = brier_score(y_idx, hybrid)
        rps = rps_single(y_idx, hybrid)
        ll = log_loss_single(y_idx, hybrid)
        acc = accuracy_from_probs(y_idx, hybrid)

        results.append({
            'date': m['date'],
            'home': home, 'away': away,
            'score': f"{hg}-{ag}",
            'actual_hda': actual_hda,
            'pred_h': round(float(hybrid[2]) * 100, 1),
            'pred_d': round(float(hybrid[1]) * 100, 1),
            'pred_a': round(float(hybrid[0]) * 100, 1),
            'brier': round(bs, 4),
            'rps': round(rps, 4),
            'log_loss': round(ll, 4),
            'accuracy': acc,
            'xgb_w': round(xgb_w, 3),
            'confidence': round(conf, 3),
        })

    # 汇总
    if results:
        brier_avg = np.mean([r['brier'] for r in results])
        rps_avg = np.mean([r['rps'] for r in results])
        ll_avg = np.mean([r['log_loss'] for r in results])
        acc_avg = np.mean([r['accuracy'] for r in results])

        print(f"\n{'='*60}")
        print(f"📊 历史回测结果 ({len(results)} 场)")
        print(f"{'='*60}")
        print(f"  Brier Score:    {brier_avg:.4f}  (随机=0.667, 完美=0.0)")
        print(f"  RPS:            {rps_avg:.4f}  (随机=0.333, 完美=0.0)")
        print(f"  Log Loss:       {ll_avg:.4f}  (随机=1.099, 完美=0.0)")
        print(f"  准确率:          {acc_avg*100:.1f}%  (随机=33.3%)")

        # 按结果类型分组
        for hda in ['H', 'D', 'A']:
            subset = [r for r in results if r['actual_hda'] == hda]
            if subset:
                acc_hda = np.mean([r['accuracy'] for r in subset])
                brier_hda = np.mean([r['brier'] for r in subset])
                print(f"\n  {hda} 场 ({len(subset)} 场): 准确率={acc_hda*100:.1f}%, Brier={brier_hda:.4f}")

        # 校准度检查: 预测概率 vs 实际频率
        print(f"\n  校准度检查:")
        for bucket_name, lo, hi in [('10-20%', 0.10, 0.20), ('20-30%', 0.20, 0.30),
                                     ('30-40%', 0.30, 0.40), ('40-50%', 0.40, 0.50),
                                     ('50%+', 0.50, 1.01)]:
            bucket = [r for r in results if lo <= max(r['pred_h'], r['pred_d'], r['pred_a']) < hi]
            if bucket:
                avg_pred = np.mean([max(r['pred_h'], r['pred_d'], r['pred_a']) for r in bucket]) / 100
                avg_actual = np.mean([r['accuracy'] for r in bucket])
                n = len(bucket)
                print(f"    预测{bucket_name}: n={n:3d}, 平均预测={avg_pred*100:.1f}%, 实际命中={avg_actual*100:.1f}%, 偏差={(avg_actual-avg_pred)*100:+.1f}pp")

        # 保存结果
        _append_backtest_result({
            'timestamp': datetime.now().isoformat(),
            'type': 'historical_backtest',
            'n_matches': len(results),
            'train_size': len(train_data),
            'test_size': len(test_data),
            'date_range': f"{test_data[0]['date']} ~ {test_data[-1]['date']}",
            'brier': round(brier_avg, 4),
            'rps': round(rps_avg, 4),
            'log_loss': round(ll_avg, 4),
            'accuracy': round(acc_avg, 4),
        })

        # 保存详细结果
        detail_path = os.path.join(DATA_DIR, 'backtest_details.json')
        with open(detail_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  详细结果: {detail_path}")

        return results


def _append_backtest_result(record):
    """追加一条回测结果到 backtest_results.json."""
    existing = []
    if os.path.exists(BACKTEST_RESULTS):
        try:
            with open(BACKTEST_RESULTS) as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
        except:
            existing = []

    existing.append(record)
    with open(BACKTEST_RESULTS, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  💾 已保存到 {BACKTEST_RESULTS}")


# ═══════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预测回测管线')
    parser.add_argument('--verify', action='store_true', help='核验已有预测')
    parser.add_argument('--backtest', action='store_true', help='历史回测')
    parser.add_argument('--n', type=int, default=500, help='历史回测场数')
    args = parser.parse_args()

    if args.verify:
        daily_verify()
    elif args.backtest:
        historical_backtest(n_matches=args.n)
    else:
        print("用法:")
        print("  python3 backtest_pipeline.py --verify       # 核验已有预测")
        print("  python3 backtest_pipeline.py --backtest      # 历史回测 (500场)")
        print("  python3 backtest_pipeline.py --backtest --n 100")
