#!/usr/bin/env python3
"""
train_sigmoid_calibrator.py — 用 Platt Scaling (Sigmoid) 替换 Isotonic 校准器
==============================================================================
"""
import csv, json, os, sys
import numpy as np
from datetime import datetime

DATA_DIR = '/root/data'
TRAINING_DATA = f'{DATA_DIR}/training_data_with_odds.json'
PREDICTIONS_LOG = f'{DATA_DIR}/predictions_log.csv'
OUTPUT_CALIBRATORS = f'{DATA_DIR}/calibrators_sigmoid.pkl'
OUTPUT_REPORT = f'{DATA_DIR}/calibration_report.json'


def load_data():
    """加载所有可用数据"""
    X_list, y_list, meta = [], [], []

    # 1. 2024 训练数据 (market_implied 作为概率输入)
    if os.path.exists(TRAINING_DATA):
        data = json.load(open(TRAINING_DATA, encoding='utf-8'))
        result_map = {'3': 'H', '1': 'D', '0': 'A'}
        for m in data:
            spf_sp = m.get('spf_sp', 0)
            hda = result_map.get(str(m.get('spf_result', '')), '')
            if not hda or spf_sp <= 0:
                continue
            # market implied 作为 XGB 输出的近似
            X_list.append([1.0/spf_sp, 0.33, 0.33])
            y_list.append(hda)
            meta.append({'date': m.get('date',''), 'source': 'training_2024'})
        print(f"  训练数据: {sum(1 for m in meta if '2024' in m['source'])} 场")

    # 2. 2026 回填数据 (真实 XGB 概率)
    if os.path.exists(PREDICTIONS_LOG):
        for r in csv.DictReader(open(PREDICTIONS_LOG, encoding='utf-8')):
            hda = r.get('actual_hda', '').strip()
            if hda not in ('H', 'D', 'A'):
                continue
            try:
                pH = float(r.get('pred_h', 0)) / 100
                pD = float(r.get('pred_d', 0)) / 100
                pA = float(r.get('pred_a', 0)) / 100
            except:
                continue
            if pH + pD + pA < 0.01:
                continue
            X_list.append([pH, pD, pA])
            y_list.append(hda)
            meta.append({'date': r.get('date',''), 'source': 'backfilled_2026'})
        print(f"  回填数据: {sum(1 for m in meta if '2026' in m['source'])} 场")

    # 归一化概率
    X = np.array(X_list)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = X / row_sums

    return X, np.array(y_list), meta


def diagnose(X, y, label=""):
    """诊断校准状态"""
    classes = ['H', 'D', 'A']
    results = {}
    for i, cls in enumerate(classes):
        y_bin = (y == cls).astype(int)
        p = X[:, i]
        avg_pred = p.mean()
        actual_rate = y_bin.mean()
        gap = actual_rate - avg_pred
        brier = np.mean((y_bin - p)**2)
        results[cls] = {'avg_pred': float(avg_pred), 'actual_rate': float(actual_rate),
                        'gap_pp': float(gap*100), 'brier': float(brier)}
        flag = 'OVERCONF' if gap < -0.10 else 'CONSERV' if gap > 0.10 else 'OK'
        print(f"    {cls}: pred={avg_pred*100:.1f}% actual={actual_rate*100:.1f}% "
              f"gap={gap*100:+.1f}pp brier={brier:.4f} [{flag}]")
    overall = sum(np.mean((y == cls).astype(int) - X[:, i])**2 for i, cls in enumerate(classes)) / 3
    # 正确的 overall Brier
    ob = 0
    for i, cls in enumerate(classes):
        ob += np.mean(((y == cls).astype(int) - X[:, i])**2)
    ob /= 3
    results['overall_brier'] = float(ob)
    print(f"    Overall Brier: {ob:.4f}")
    return results


def train_sigmoid(X, y, cv=5):
    """用 LogisticRegression 做 Platt Scaling"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    # 用原始概率作为特征, 训练 LR 校准器
    # 这等价于 Platt Scaling: 对每个类别的 logits 做 sigmoid 变换
    lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')

    # 概率 → logits (避免 log(0))
    eps = 1e-6
    X_clipped = np.clip(X, eps, 1 - eps)
    logits = np.log(X_clipped / (1 - X_clipped + eps))

    # 交叉验证预测
    lr.fit(logits, y)

    # 交叉验证校准后的概率
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    X_cal = np.zeros_like(X)
    for train_idx, val_idx in skf.split(logits, y):
        lr_fold = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        lr_fold.fit(logits[train_idx], y[train_idx])
        X_cal[val_idx] = lr_fold.predict_proba(logits[val_idx])

    return lr, X_cal


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("  校准器重训 (Platt Scaling / Sigmoid)")
    print("=" * 60)

    X, y, meta = load_data()
    print(f"\n  合并数据: {len(X)} 场")

    print(f"\n  ── 当前校准状态 (原始概率) ──")
    raw_diag = diagnose(X, y, "raw")

    if args.dry_run:
        print("\n  (dry-run)")
        return

    # 时间序列划分
    dates = [m.get('date', '') for m in meta]
    sorted_idx = np.argsort(dates)
    X_sorted = X[sorted_idx]
    y_sorted = y[sorted_idx]
    split = int(len(X_sorted) * 0.8)
    X_train, X_val = X_sorted[:split], X_sorted[split:]
    y_train, y_val = y_sorted[:split], y_sorted[split:]
    print(f"\n  训练: {len(X_train)}, 验证: {len(X_val)}")

    # 训练 sigmoid 校准器
    print(f"\n  ── 训练 Sigmoid 校准器 (LogisticRegression Platt Scaling) ──")
    lr_model, X_cal_cv = train_sigmoid(X_sorted, y_sorted, cv=5)

    print(f"\n  ── 校准后 (5-fold CV) ──")
    cal_diag = diagnose(X_cal_cv, y_sorted, "calibrated")

    # 验证集效果
    eps = 1e-6
    X_val_clip = np.clip(X_val, eps, 1-eps)
    logits_val = np.log(X_val_clip / (1 - X_val_clip + eps))
    X_val_cal = lr_model.predict_proba(logits_val)

    print(f"\n  ── 校准后 (验证集) ──")
    val_diag = diagnose(X_val_cal, y_val, "val_calibrated")

    # 保存
    import joblib
    joblib.dump(lr_model, OUTPUT_CALIBRATORS)
    print(f"\n  校准器已保存: {OUTPUT_CALIBRATORS}")

    # 报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'method': 'sigmoid_platt_scaling',
        'base_estimator': 'LogisticRegression(C=1.0, multinomial)',
        'n_total': len(X),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'raw_calibration': raw_diag,
        'calibrated_cv': cal_diag,
        'calibrated_val': val_diag,
        'model_params': {
            'coef_shape': lr_model.coef_.tolist(),
            'intercept': lr_model.intercept_.tolist(),
        }
    }
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  诊断报告: {OUTPUT_REPORT}")

    # 对比
    if args.compare:
        print(f"\n  ── 对比总结 ──")
        print(f"  {'指标':<20} {'原始':>10} {'Sigmoid':>10} {'改善':>10}")
        print(f"  {'-'*50}")
        for cls in ['H', 'D', 'A']:
            raw_brier = raw_diag[cls]['brier']
            cal_brier = cal_diag[cls]['brier']
            raw_gap = abs(raw_diag[cls]['gap_pp'])
            cal_gap = abs(cal_diag[cls]['gap_pp'])
            print(f"  Brier({cls}){'':<14} {raw_brier:>10.4f} {cal_brier:>10.4f} {(raw_brier-cal_brier):>+10.4f}")
            print(f"  |gap|pp({cls}){'':<11} {raw_gap:>10.1f} {cal_gap:>10.1f} {(raw_gap-cal_gap):>+10.1f}")
        raw_ob = raw_diag['overall_brier']
        cal_ob = cal_diag['overall_brier']
        print(f"  {'Overall Brier':<20} {raw_ob:>10.4f} {cal_ob:>10.4f} {(raw_ob-cal_ob):>+10.4f}")

    print(f"\n  下一步:")
    print(f"  1. daily_jczq.py: _load_shared_models() 的 cal_path 改为 calibrators_sigmoid.pkl")
    print(f"  2. 用 backtest_pipeline.py 重跑回测验证 Brier 下降")
    print(f"  3. 确认后恢复友谊赛的 margin 门槛逻辑")


if __name__ == '__main__':
    main()
