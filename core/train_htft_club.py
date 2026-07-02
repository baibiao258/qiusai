#!/usr/bin/env python3
"""
train_htft_club.py — 俱乐部半全场独立模型训练
=============================================
利用 club_matches.json 中的 HT/FT 比分数据，
训练 9 分类 XGB 模型替代 r_ht=0.45 的纯数学推导。

类别: HH/HD/HA/DH/DD/DA/AH/AD/AA (9 类)

输出:
  - /root/data/xgb_htft_club.pkl   (XGB 9 分类器)
  - /root/data/htft_calibrators.pkl (Isotonic 校准器)
"""
import json
import math
import os
import sys

import numpy as np
import joblib
from collections import defaultdict

sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')

DATA_DIR = '/root/data'

# HT/FT 类别映射 (固定顺序)
HTFT_LABELS = ['HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA']
HTFT_TO_IDX = {label: i for i, label in enumerate(HTFT_LABELS)}

def htft_class(ht_h, ht_a, ft_h, ft_a):
    """半场+全场比分 → 9 类标签."""
    if ht_h > ht_a: ht = 'H'
    elif ht_h == ht_a: ht = 'D'
    else: ht = 'A'
    if ft_h > ft_a: ft = 'H'
    elif ft_h == ft_a: ft = 'D'
    else: ft = 'A'
    return ht + ft


def load_htft_data():
    """加载带 HT/FT 的比赛数据."""
    path = os.path.join(DATA_DIR, 'club_matches.json')
    with open(path) as f:
        matches = json.load(f)

    valid = []
    for m in matches:
        ht_h = m.get('ht_h')
        ht_a = m.get('ht_a')
        ft_h = m.get('h_score')
        ft_a = m.get('a_score')

        if ht_h is None or ht_a is None or ft_h is None or ft_a is None:
            continue
        if not (0 <= ht_h <= 10 and 0 <= ht_a <= 10 and 0 <= ft_h <= 15 and 0 <= ft_a <= 15):
            continue
        if ft_h < ht_h or ft_a < ht_a:  # 全场不能小于半场
            continue

        label = htft_class(ht_h, ht_a, ft_h, ft_a)
        valid.append({
            **m,
            'ht_h': ht_h, 'ht_a': ht_a,
            'ft_h': ft_h, 'ft_a': ft_a,
            'htft_label': label,
            'htft_idx': HTFT_TO_IDX[label],
        })

    return valid


def build_htft_features(m, elo, dc, form_state):
    """
    构建 HT/FT 预测特征 (12 维):
      - lambda_ft_home, lambda_ft_away (DC 预测)
      - elo_diff, elo_expected
      - form_home_wr, form_away_wr (近5场胜率)
      - form_home_gf, form_away_gf (近5场进球)
      - form_home_ga, form_away_ga (近5场失球)
      - h2h_diff (H2H 目标差, 占位)
      - neutral_flag
    """
    h, a = m['home'], m['away']
    eh = elo.get(h, 1400)
    ea = elo.get(a, 1400)

    try:
        lam_h, lam_a = dc.predict_lambda(h, a, neutral=True)
        if lam_h is None: raise ValueError
    except:
        lam_h, lam_a = 1.0, 1.0

    # Form
    def _form(team, n=5):
        games = form_state.get(team, [])
        recent = games[-n:] if len(games) >= n else games
        if not recent:
            return [0.5, 0.0, 0.0]
        wins = sum(1 for g in recent if g[0] > g[1]) + \
               sum(0.5 for g in recent if g[0] == g[1])
        gf = sum(g[0] for g in recent) / len(recent)
        ga = sum(g[1] for g in recent) / len(recent)
        return [wins / len(recent), gf, ga]

    fh = _form(h, 5)
    fa = _form(a, 5)

    elo_diff = (eh - ea) / 400
    elo_expected = 1.0 / (1 + 10 ** ((ea - eh) / 400))

    feat = [
        lam_h, lam_a,
        elo_diff, elo_expected,
        fh[0], fa[0],
        fh[1], fa[1],
        fh[2], fa[2],
        0.0,  # h2h_diff placeholder
        1.0,  # neutral
    ]
    return feat


def main():
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.isotonic import IsotonicRegression

    print("=" * 55)
    print("🏟️  训练俱乐部半全场 (HT/FT) 独立模型")
    print("=" * 55)

    # ── 加载数据 ──
    elo = joblib.load(os.path.join(DATA_DIR, 'elo_club.pkl'))
    dc = joblib.load(os.path.join(DATA_DIR, 'dc_model_club.pkl'))
    with open(os.path.join(DATA_DIR, 'form_club.json')) as f:
        form_state = json.load(f)

    matches = load_htft_data()
    matches.sort(key=lambda m: m['date'])

    print(f"  有效 HT/FT 数据: {len(matches)} 场")

    # 类别分布
    from collections import Counter
    dist = Counter(m['htft_label'] for m in matches)
    print(f"\n  类别分布:")
    for label in HTFT_LABELS:
        n = dist.get(label, 0)
        pct = n / len(matches) * 100
        bar = '█' * int(pct / 2)
        print(f"    {label}: {n:5d} ({pct:5.1f}%) {bar}")

    # ── 时序切分: 60/20/20 ──
    n = len(matches)
    train_end = int(n * 0.6)
    cal_end = int(n * 0.8)
    ms_train = matches[:train_end]
    ms_cal = matches[train_end:cal_end]
    ms_val = matches[cal_end:]

    print(f"\n  训练: {len(ms_train)} 场 ({ms_train[0]['date']} ~ {ms_train[-1]['date']})")
    print(f"  校准: {len(ms_cal)} 场 ({ms_cal[0]['date']} ~ {ms_cal[-1]['date']})")
    print(f"  验证: {len(ms_val)} 场 ({ms_val[0]['date']} ~ {ms_val[-1]['date']})")

    # ── 构建特征 ──
    X_train = np.array([build_htft_features(m, elo, dc, form_state) for m in ms_train])
    y_train = np.array([m['htft_idx'] for m in ms_train])
    X_cal = np.array([build_htft_features(m, elo, dc, form_state) for m in ms_cal])
    y_cal = np.array([m['htft_idx'] for m in ms_cal])
    X_val = np.array([build_htft_features(m, elo, dc, form_state) for m in ms_val])
    y_val = np.array([m['htft_idx'] for m in ms_val])

    print(f"\n  特征维度: {X_train.shape[1]}")
    print(f"  Train: {X_train.shape} | Cal: {X_cal.shape} | Val: {X_val.shape}")

    # ── 训练 XGB (9 分类) ──
    print("\n  🌲 训练 XGBoost (9 分类)...")
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    sw = np.array([cw[list(classes).index(c)] for c in y_train])

    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=9,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=400,
        reg_alpha=2.0,
        reg_lambda=2.0,
        colsample_bytree=0.6,
        subsample=0.7,
        min_child_weight=10,
        n_jobs=-1,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, sample_weight=sw,
            eval_set=[(X_cal, y_cal)], verbose=False)

    # ── 评估 ──
    y_proba_raw = xgb.predict_proba(X_val)
    y_pred_raw = np.argmax(y_proba_raw, axis=1)

    acc_raw = accuracy_score(y_val, y_pred_raw)
    brier_raw = np.mean(np.sum((y_proba_raw - np.eye(9)[y_val]) ** 2, axis=1))

    # Top-3 accuracy
    top3_correct = 0
    for i in range(len(y_val)):
        top3 = np.argsort(y_proba_raw[i])[-3:]
        if y_val[i] in top3:
            top3_correct += 1
    acc_top3 = top3_correct / len(y_val)

    print(f"  ✅ Val (raw): acc={acc_raw:.4f} top3_acc={acc_top3:.4f} brier={brier_raw:.4f}")

    # ── Isotonic 校准 (per-class, 在 cal 集训练) ──
    print("\n  🌡 Isotonic 校准...")
    y_proba_cal = xgb.predict_proba(X_cal)

    calibrators = {}
    for c in range(9):
        y_bin = (y_cal == c).astype(int)
        if y_bin.sum() < 20:
            continue
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
        ir.fit(y_proba_cal[:, c], y_bin)
        calibrators[c] = ir

    # 校准后评估
    y_proba_cal_val = np.zeros_like(y_proba_raw)
    for c in range(9):
        if c in calibrators:
            y_proba_cal_val[:, c] = calibrators[c].predict(y_proba_raw[:, c])
        else:
            y_proba_cal_val[:, c] = y_proba_raw[:, c]

    # 归一化
    row_sums = y_proba_cal_val.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_proba_cal_val = y_proba_cal_val / row_sums

    y_pred_cal = np.argmax(y_proba_cal_val, axis=1)
    acc_cal = accuracy_score(y_val, y_pred_cal)
    brier_cal = np.mean(np.sum((y_proba_cal_val - np.eye(9)[y_val]) ** 2, axis=1))

    top3_cal = 0
    for i in range(len(y_val)):
        top3 = np.argsort(y_proba_cal_val[i])[-3:]
        if y_val[i] in top3:
            top3_cal += 1
    acc_top3_cal = top3_cal / len(y_val)

    print(f"  ✅ Val (cal): acc={acc_cal:.4f} top3_acc={acc_top3_cal:.4f} brier={brier_cal:.4f}")

    # ── 各类别准确率 ──
    print(f"\n  各类别表现 (校准后):")
    for c in range(9):
        mask = y_val == c
        if mask.sum() == 0:
            continue
        acc_c = accuracy_score(y_val[mask], y_pred_cal[mask])
        avg_prob_c = y_proba_cal_val[mask, c].mean()
        print(f"    {HTFT_LABELS[c]}: n={mask.sum():4d} acc={acc_c:.3f} avg_prob={avg_prob_c:.3f}")

    # ── 与纯数学推导对比 (r_ht=0.45) ──
    print(f"\n  📊 对比基线 (r_ht=0.45 纯数学推导):")
    # 用 r_ht=0.45 生成基线预测
    baseline_probs = []
    for m in ms_val:
        lam_h = 1.0
        lam_a = 1.0
        try:
            lh, la = dc.predict_lambda(m['home'], m['away'], neutral=True)
            if lh is not None:
                lam_h, lam_a = lh, la
        except:
            pass

        r_ht = 0.45
        lam_ht_h = lam_h * r_ht
        lam_ht_a = lam_a * r_ht
        lam_ft_h = lam_h * (1 - r_ht)
        lam_ft_a = lam_a * (1 - r_ht)

        probs = np.zeros(9)
        for h1 in range(6):
            for a1 in range(6):
                p_ht = (lam_ht_h**h1 * math.exp(-lam_ht_h) / math.factorial(h1)) * \
                       (lam_ht_a**a1 * math.exp(-lam_ht_a) / math.factorial(a1))
                for h2 in range(h1, 8):
                    for a2 in range(a1, 8):
                        sh, sa = h2 - h1, a2 - a1
                        if sh < 0 or sa < 0:
                            continue
                        p_sec = (lam_ft_h**sh * math.exp(-lam_ft_h) / math.factorial(sh)) * \
                                (lam_ft_a**sa * math.exp(-lam_ft_a) / math.factorial(sa))
                        p = p_ht * p_sec
                        def rs(hg, ag): return 0 if hg>ag else (1 if hg==ag else 2)
                        idx = rs(h1, a1) * 3 + rs(h2, a2)
                        probs[idx] += p
        s = probs.sum()
        if s > 0: probs /= s
        baseline_probs.append(probs)

    baseline_probs = np.array(baseline_probs)
    baseline_pred = np.argmax(baseline_probs, axis=1)
    acc_baseline = accuracy_score(y_val, baseline_pred)
    brier_baseline = np.mean(np.sum((baseline_probs - np.eye(9)[y_val]) ** 2, axis=1))

    top3_baseline = 0
    for i in range(len(y_val)):
        top3 = np.argsort(baseline_probs[i])[-3:]
        if y_val[i] in top3:
            top3_baseline += 1
    acc_top3_baseline = top3_baseline / len(y_val)

    print(f"    基线 (r_ht=0.45): acc={acc_baseline:.4f} top3={acc_top3_baseline:.4f} brier={brier_baseline:.4f}")
    print(f"    XGB (raw):        acc={acc_raw:.4f} top3={acc_top3:.4f} brier={brier_raw:.4f}")
    print(f"    XGB (cal):        acc={acc_cal:.4f} top3={acc_top3_cal:.4f} brier={brier_cal:.4f}")

    brier_improvement = (brier_baseline - brier_cal) / brier_baseline * 100
    acc_improvement = (acc_cal - acc_baseline) / acc_baseline * 100
    print(f"\n    🎯 Brier 改善: {brier_improvement:+.1f}%")
    print(f"    🎯 Acc 改善:   {acc_improvement:+.1f}%")

    # ── 保存 ──
    print("\n💾 保存模型...")
    joblib.dump(xgb, os.path.join(DATA_DIR, 'xgb_htft_club.pkl'))
    joblib.dump(calibrators, os.path.join(DATA_DIR, 'htft_calibrators.pkl'))
    print(f"  ✅ xgb_htft_club.pkl ({xgb.n_features_in_} 维, {xgb.n_estimators} 棵, 9 类)")
    print(f"  ✅ htft_calibrators.pkl ({len(calibrators)} 类)")

    print("\n✅ 完成!")


if __name__ == '__main__':
    main()
