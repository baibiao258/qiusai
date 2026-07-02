#!/usr/bin/env python3
"""
htft_predictor.py — 半全场独立预测器
====================================
替代 r_ht=0.45 的纯数学推导，使用训练好的 XGB 9 分类模型。

用法:
  from htft_predictor import predict_htft_probs
  probs = predict_htft_probs(lam_h, lam_a, home, away)
  # 返回 dict: {'HH': 0.31, 'HD': 0.05, ...}
"""
import json
import math
import os
import sys

import numpy as np
import joblib

sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')

DATA_DIR = '/root/data'

# 模型 (lazy load)
_xgb_htft = None
_calibrators_htft = None
_elo_club_htft = None
_dc_club_htft = None
_form_club_htft = None

HTFT_LABELS = ['HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA']


def _load_htft_model():
    global _xgb_htft, _calibrators_htft, _elo_club_htft, _dc_club_htft, _form_club_htft
    if _xgb_htft is not None:
        return

    xgb_path = os.path.join(DATA_DIR, 'xgb_htft_club.pkl')
    cal_path = os.path.join(DATA_DIR, 'htft_calibrators.pkl')

    if not os.path.exists(xgb_path):
        return

    _xgb_htft = joblib.load(xgb_path)
    if os.path.exists(cal_path):
        _calibrators_htft = joblib.load(cal_path)

    try:
        _elo_club_htft = joblib.load(os.path.join(DATA_DIR, 'elo_club.pkl'))
        _dc_club_htft = joblib.load(os.path.join(DATA_DIR, 'dc_model_club.pkl'))
        with open(os.path.join(DATA_DIR, 'form_club.json')) as f:
            _form_club_htft = json.load(f)
    except:
        pass


def predict_htft_probs(lam_h, lam_a, home=None, away=None, elo_h=None, elo_a=None,
                       form_home=None, form_away=None):
    """
    预测半全场 9 类概率.

    优先使用 XGB 模型 (如果可用)，回退到纯数学推导。

    Args:
        lam_h, lam_a: 全场期望进球 (来自 DC/Poisson)
        home, away: 球队名 (可选, 用于查 form/elo)
        elo_h, elo_a: Elo 评分 (可选)
        form_home, form_away: [win_rate, avg_gf, avg_ga] (可选)

    Returns:
        dict: {'HH': p, 'HD': p, ..., 'AA': p} (9 类概率)
    """
    _load_htft_model()

    # 尝试 XGB 模型
    if _xgb_htft is not None and home and away:
        try:
            feat = _build_htft_feat(lam_h, lam_a, home, away, elo_h, elo_a,
                                    form_home, form_away)
            if feat is not None:
                feat_arr = np.array([feat])
                probs_raw = _xgb_htft.predict_proba(feat_arr)[0]

                # Isotonic 校准
                if _calibrators_htft:
                    calibrated = np.zeros(9)
                    for c in range(9):
                        if c in _calibrators_htft:
                            calibrated[c] = _calibrators_htft[c].predict([probs_raw[c]])[0]
                        else:
                            calibrated[c] = probs_raw[c]
                    s = calibrated.sum()
                    if s > 0:
                        calibrated /= s
                    probs_raw = calibrated

                # 归一化
                s = probs_raw.sum()
                if s > 0:
                    probs_raw /= s

                return {label: float(probs_raw[i]) for i, label in enumerate(HTFT_LABELS)}
        except Exception:
            pass

    # 回退: 纯数学推导 (r_ht=0.45)
    return _math_htft_probs(lam_h, lam_a)


def _build_htft_feat(lam_h, lam_a, home, away, elo_h, elo_a,
                     form_home, form_away):
    """构建 HT/FT 特征 (12 维)."""
    eh = elo_h
    ea = elo_a

    if eh is None and _elo_club_htft:
        eh = _elo_club_htft.get(home, _elo_club_htft.get(away, 1400))
    if ea is None and _elo_club_htft:
        ea = _elo_club_htft.get(away, _elo_club_htft.get(home, 1400))
    if eh is None: eh = 1400
    if ea is None: ea = 1400

    # Form
    if form_home is None and _form_club_htft:
        games = _form_club_htft.get(home, [])
        recent = games[-5:] if len(games) >= 5 else games
        if recent:
            wins = sum(1 for g in recent if g[0] > g[1]) + \
                   sum(0.5 for g in recent if g[0] == g[1])
            form_home = [wins / len(recent),
                        sum(g[0] for g in recent) / len(recent),
                        sum(g[1] for g in recent) / len(recent)]
        else:
            form_home = [0.5, 0.0, 0.0]

    if form_away is None and _form_club_htft:
        games = _form_club_htft.get(away, [])
        recent = games[-5:] if len(games) >= 5 else games
        if recent:
            wins = sum(1 for g in recent if g[0] > g[1]) + \
                   sum(0.5 for g in recent if g[0] == g[1])
            form_away = [wins / len(recent),
                        sum(g[0] for g in recent) / len(recent),
                        sum(g[1] for g in recent) / len(recent)]
        else:
            form_away = [0.5, 0.0, 0.0]

    if form_home is None: form_home = [0.5, 0.0, 0.0]
    if form_away is None: form_away = [0.5, 0.0, 0.0]

    elo_diff = (eh - ea) / 400
    elo_expected = 1.0 / (1 + 10 ** ((ea - eh) / 400))

    feat = [
        lam_h, lam_a,
        elo_diff, elo_expected,
        form_home[0], form_away[0],
        form_home[1], form_away[1],
        form_home[2], form_away[2],
        0.0,  # h2h_diff placeholder
        1.0,  # neutral
    ]
    return feat


def _math_htft_probs(lam_h, lam_a, r_ht=0.45):
    """纯数学推导 (r_ht=0.45) — 回退方案."""
    lam_ht_h = lam_h * r_ht
    lam_ht_a = lam_a * r_ht
    lam_ft_h = lam_h * (1 - r_ht)
    lam_ft_a = lam_a * (1 - r_ht)

    probs = {}
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

                    def rs(hg, ag):
                        return '胜' if hg > ag else ('平' if hg == ag else '负')
                    label = rs(h1, a1) + rs(h2, a2)
                    probs[label] = probs.get(label, 0.0) + p

    s = sum(probs.values()) or 1.0
    return {k: v / s for k, v in probs.items()}


if __name__ == '__main__':
    # 测试
    print("=== XGB 模型测试 ===")
    probs = predict_htft_probs(1.5, 0.8, home='Arsenal FC', away='Manchester City FC')
    for label, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {label}: {p*100:.1f}%")

    print("\n=== 数学推导回退 ===")
    probs2 = _math_htft_probs(1.5, 0.8)
    for label, p in sorted(probs2.items(), key=lambda x: -x[1]):
        print(f"  {label}: {p*100:.1f}%")
