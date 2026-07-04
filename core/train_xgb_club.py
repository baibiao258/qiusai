#!/usr/bin/env python3
"""
train_xgb_club.py — 俱乐部 XGBoost 训练
=========================================
基于 club_matches.json + elo_club.pkl + dc_model_club.pkl + xg_proxy 训练 XGB 分类器。

特征: 37 维 (29 基线 + 8 xG-proxy)
训练/校准/验证切分: 60/20/20 (时序严格分离)
输出: /root/data/xgb_model_club.pkl + calibrated_xgb_club.pkl
"""
import json
import math
import os
import sys

import numpy as np
import joblib
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')

DATA_DIR = '/root/data'

# ── 加载数据 ──
def load_all():
    with open(os.path.join(DATA_DIR, 'club_matches.json')) as f:
        matches = json.load(f)
    elo = joblib.load(os.path.join(DATA_DIR, 'elo_club.pkl'))
    dc = joblib.load(os.path.join(DATA_DIR, 'dc_model_club.pkl'))
    with open(os.path.join(DATA_DIR, 'form_club.json')) as f:
        form_state = json.load(f)
    xg_proxy_path = os.path.join(DATA_DIR, 'xg_proxy_club.json')
    xg_proxy_state = {}
    if os.path.exists(xg_proxy_path):
        with open(xg_proxy_path) as f:
            xg_proxy_state = json.load(f)
    xg_real_path = os.path.join(DATA_DIR, 'xg_real_club.json')
    xg_real_state = None
    if os.path.exists(xg_real_path):
        with open(xg_real_path) as f:
            xg_real_state = json.load(f)
    return matches, elo, dc, form_state, xg_proxy_state, xg_real_state


# ── Feature Buffer (简化版) ──
class ClubFeatureBuffer:
    def __init__(self, elo, form_state):
        self.elo = elo
        self.form_state = form_state
        self.team_games = defaultdict(list)
        self.h2h_cache = defaultdict(lambda: defaultdict(list))

    def add_match(self, m):
        h, a = m['home'], m['away']
        self.team_games[h].append(m)
        self.team_games[a].append(m)
        key = (min(h, a), max(h, a))
        self.h2h_cache[key[0]][key[1]].append(m)

    def recent_form(self, team, n=5):
        games = self.form_state.get(team, [])
        recent = games[-n:] if len(games) >= n else games
        if not recent:
            return [0.5, 0.0, 0.0, 0.0]
        wins = sum(1 for g in recent if g[0] > g[1]) + \
               sum(0.5 for g in recent if g[0] == g[1])
        gf = sum(g[0] for g in recent) / len(recent)
        ga = sum(g[1] for g in recent) / len(recent)
        return [wins / len(recent), gf, ga, gf - ga]

    def h2h(self, home, away, n=3):
        key = (min(home, away), max(home, away))
        raw = self.h2h_cache[key[0]][key[1]][-n:]
        if not raw:
            return [0.5, 0.0, 0.0]
        wins = 0; gf = 0; ga = 0
        for m in raw:
            if m['home'] == home:
                gf += m['h_score']; ga += m['a_score']
                wins += 1 if m['h_score'] > m['a_score'] else (0.5 if m['h_score'] == m['a_score'] else 0)
            else:
                gf += m['a_score']; ga += m['h_score']
                wins += 1 if m['a_score'] > m['h_score'] else (0.5 if m['a_score'] == m['h_score'] else 0)
        return [wins / len(raw), gf / len(raw), ga / len(raw)]


def build_feat(fb, dc, h, a, xg_state=None, xg_real_state=None):
    """构建 41 维特征 (29 基线 + 8 xG-proxy + 4 xG-real).

    特征顺序（必须与推理侧 _build_xg_feat 的输出顺序完全一致）:
      b15(15) + gold(5) + odds(3) + form(6) + xg_proxy(8) + xg_real(4) = 41
    """
    eh = fb.elo.get(h, 1400)
    ea = fb.elo.get(a, 1400)

    # DC 预测
    try:
        dc_p = dc.predict_proba(h, a, neutral=True)
        lam_h, lam_a = dc.predict_lambda(h, a, neutral=True)
        if lam_h is None: raise ValueError
    except:
        dc_p = [1/3, 1/3, 1/3]
        lam_h, lam_a = 1.0, 1.0

    fh5 = fb.recent_form(h, 5)
    fa5 = fb.recent_form(a, 5)
    fh12 = fb.recent_form(h, 12)
    fa12 = fb.recent_form(a, 12)
    h2h = fb.h2h(h, a, 3)

    op_h = 1 / (1 + 10 ** ((ea - eh) / 400))
    op_a = 1 / (1 + 10 ** ((eh - ea) / 400))

    b15 = [
        (eh - ea) / 400,
        lam_h, lam_a, lam_h - lam_a,
        math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
        dc_p[0], dc_p[1], dc_p[2],
        fh5[0], fa5[0],
        fh5[1] - fa5[2], fa5[1] - fh5[2],
        fh5[1] - fa5[1], fh5[0] - fa5[0],
        1,  # neutral
    ]
    gold = [
        h2h[0] - h2h[2],  # h2h goal diff
        0, 0,  # tier flags
        fh12[1] - fa12[2],
        fa12[1] - fh12[0],
    ]
    odds_feat = [op_h, op_a, 0.0]
    form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]

    # ── xG-proxy 特征 (8维: 主客各4) ──
    if xg_state is not None:
        xg_proxy_feat = []
        for team in [h, a]:
            s = xg_state.get(team, {})
            xg_proxy_feat.extend([
                s.get('xg_proxy_5', 0.0),
                s.get('xg_proxy_12', 0.0),
                s.get('xg_streak', 0) / 10.0,
                s.get('xg_volatility', 0.0),
            ])
    else:
        xg_proxy_feat = [0.0] * 8

    # ── xG-real 特征 (4维: 主客各2) ──
    # 使用 NaN 表示缺失（XGBoost 原生支持缺失值分裂）
    rh = (xg_real_state or {}).get(h)
    ra = (xg_real_state or {}).get(a)
    xg_real_feat = [
        rh.get('xg_recent_avg') if rh else float('nan'),
        rh.get('xg_diff_avg')   if rh else float('nan'),
        ra.get('xg_recent_avg') if ra else float('nan'),
        ra.get('xg_diff_avg')   if ra else float('nan'),
    ]

    return b15 + gold + odds_feat + form_feat + xg_proxy_feat + xg_real_feat


def main():
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.isotonic import IsotonicRegression

    print("=" * 50)
    print("🌲 训练俱乐部 XGBoost (含 xG-proxy)")
    print("=" * 50)

    matches, elo, dc, form_state, xg_state, xg_real_state = load_all()
    matches.sort(key=lambda m: m['date'])

    # 去重
    seen = set()
    unique = []
    for m in matches:
        key = (m['date'], m['home'], m['away'])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    n_total = len(unique)
    train_end = int(n_total * 0.6)
    cal_end = int(n_total * 0.8)

    ms_train = unique[:train_end]
    ms_cal = unique[train_end:cal_end]
    ms_val = unique[cal_end:]

    print(f"  xG-proxy: {len(xg_state)} 队")
    print(f"  总计: {n_total} 场")
    print(f"  训练: {len(ms_train)} 场 ({ms_train[0]['date']} ~ {ms_train[-1]['date']})")
    print(f"  校准: {len(ms_cal)} 场 ({ms_cal[0]['date']} ~ {ms_cal[-1]['date']})")
    print(f"  验证: {len(ms_val)} 场 ({ms_val[0]['date']} ~ {ms_val[-1]['date']})")

    # 构建特征 (只用 train 的 buffer)
    fb = ClubFeatureBuffer(elo, form_state)
    X_train, y_train = [], []
    for m in ms_train:
        feat = build_feat(fb, dc, m['home'], m['away'], xg_state, xg_real_state)
        X_train.append(feat)
        if m['h_score'] > m['a_score']: y_train.append(2)
        elif m['h_score'] == m['a_score']: y_train.append(1)
        else: y_train.append(0)
        fb.add_match(m)

    # Cal/Val 不加入 buffer (防止泄漏)
    X_cal, y_cal = [], []
    for m in ms_cal:
        feat = build_feat(fb, dc, m['home'], m['away'], xg_state, xg_real_state)
        X_cal.append(feat)
        if m['h_score'] > m['a_score']: y_cal.append(2)
        elif m['h_score'] == m['a_score']: y_cal.append(1)
        else: y_cal.append(0)

    X_val, y_val = [], []
    for m in ms_val:
        feat = build_feat(fb, dc, m['home'], m['away'], xg_state, xg_real_state)
        X_val.append(feat)
        if m['h_score'] > m['a_score']: y_val.append(2)
        elif m['h_score'] == m['a_score']: y_val.append(1)
        else: y_val.append(0)

    X_train = np.array(X_train)
    X_cal = np.array(X_cal)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_cal = np.array(y_cal)
    y_val = np.array(y_val)

    print(f"\n  特征: train={X_train.shape} cal={X_cal.shape} val={X_val.shape}")

    # ── 训练 XGB ──
    print("\n  🌲 训练 XGBoost (Optuna 参数)...")
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    sw = np.array([cw[list(classes).index(c)] for c in y_train])

    xgb = XGBClassifier(
        max_depth=4,
        learning_rate=0.03,
        n_estimators=300,
        reg_alpha=3.0,
        reg_lambda=2.7,
        colsample_bytree=0.45,
        subsample=0.65,
        min_child_weight=8,
        n_jobs=-1,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, sample_weight=sw,
            eval_set=[(X_cal, y_cal)], verbose=False)

    # 评估
    y_pred = xgb.predict(X_val)
    y_proba = xgb.predict_proba(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    val_nll = log_loss(y_val, y_proba)
    val_brier = np.mean((y_proba - np.eye(3)[y_val]) ** 2)

    print(f"  ✅ Val: acc={val_acc:.4f} nll={val_nll:.4f} brier={val_brier:.4f}")
    print(f"  n_estimators={xgb.n_estimators} n_features={xgb.n_features_in_}")

    # ── Isotonic 校准 (在 cal 集训练, val 集评估) ──
    print("\n  🌡 Isotonic 校准...")
    dc_ado_all = []
    for m in ms_cal:
        try:
            dp = dc.predict_proba(m['home'], m['away'], neutral=True)
            dc_ado_all.append([dp[2], dp[1], dp[0]])
        except:
            dc_ado_all.append([1/3, 1/3, 1/3])

    xgb_cal_probs = xgb.predict_proba(X_cal)

    # Dynamic weight for each sample
    hybrid_cal = []
    for i in range(len(X_cal)):
        p = np.clip(xgb_cal_probs[i], 1e-10, 1.0)
        p = p / p.sum()
        e = -np.sum(p * np.log2(p))
        conf = 1.0 - e / math.log2(3)
        xgb_w = max(0.10, min(0.90, 0.30 + 0.50 * conf))
        dc_w = 1.0 - xgb_w
        h = dc_w * np.array(dc_ado_all[i]) + xgb_w * xgb_cal_probs[i]
        s = h.sum()
        if s > 0: h = h / s
        hybrid_cal.append(h)

    hybrid_cal = np.array(hybrid_cal)

    # Per-class Isotonic
    calibrators = {}
    for c, name in enumerate(['away', 'draw', 'home']):
        y_bin = (y_cal == c).astype(int)
        if y_bin.sum() < 30:
            continue
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
        ir.fit(hybrid_cal[:, c], y_bin)
        calibrators[name] = ir

    # 验证校准效果
    if calibrators:
        hybrid_val = []
        for i in range(len(X_val)):
            p = np.clip(y_proba[i], 1e-10, 1.0)
            p = p / p.sum()
            e = -np.sum(p * np.log2(p))
            conf = 1.0 - e / math.log2(3)
            xgb_w = max(0.10, min(0.90, 0.30 + 0.50 * conf))
            dc_w = 1.0 - xgb_w
            try:
                dp = dc.predict_proba(ms_val[i]['home'], ms_val[i]['away'], neutral=True)
                dc_ado = [dp[2], dp[1], dp[0]]
            except:
                dc_ado = [1/3, 1/3, 1/3]
            h = dc_w * np.array(dc_ado) + xgb_w * y_proba[i]
            s = h.sum()
            if s > 0: h = h / s

            # Apply isotonic
            calibrated = np.zeros(3)
            for j, key in enumerate(['away', 'draw', 'home']):
                if key in calibrators:
                    calibrated[j] = calibrators[key].predict([h[j]])[0]
                else:
                    calibrated[j] = h[j]
            s = calibrated.sum()
            if s > 0: calibrated = calibrated / s
            hybrid_val.append(calibrated)

        hybrid_val = np.array(hybrid_val)
        val_brier_cal = np.mean((hybrid_val - np.eye(3)[y_val]) ** 2)
        val_acc_cal = accuracy_score(y_val, np.argmax(hybrid_val, axis=1))
        print(f"  ✅ 校准后 Val: acc={val_acc_cal:.4f} brier={val_brier_cal:.4f}")
    else:
        print(f"  ⚠️ 样本不足, 跳过 Isotonic 校准")

    # ── 保存 ──
    print("\n💾 保存模型...")
    joblib.dump(xgb, os.path.join(DATA_DIR, 'xgb_model_club.pkl'))
    joblib.dump(calibrators, os.path.join(DATA_DIR, 'calibrators_club.pkl'))
    print(f"  ✅ xgb_model_club.pkl ({xgb.n_features_in_} 维, {xgb.n_estimators} 棵)")
    print(f"  ✅ calibrators_club.pkl ({len(calibrators)} 类)")

    # ── 交叉验证: 俱乐部 vs 国际赛 ──
    print(f"\n📊 与国际赛模型对比:")
    intl_brier = 0.4613  # from backtest
    print(f"  国际赛 Brier: {intl_brier:.4f}")
    print(f"  俱乐部 Brier: {val_brier:.4f} (raw) / {val_brier_cal:.4f} (cal)" if calibrators else f"  俱乐部 Brier: {val_brier:.4f}")
    print(f"  国际赛 Acc: 64.5%")
    print(f"  俱乐部 Acc: {val_acc*100:.1f}% (raw) / {val_acc_cal*100:.1f}% (cal)" if calibrators else f"  俱乐部 Acc: {val_acc*100:.1f}%")

    print("\n✅ 完成!")


if __name__ == '__main__':
    main()
