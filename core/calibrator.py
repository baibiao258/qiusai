#!/usr/bin/env python3
"""Probability calibrator for predict_match.py outputs (v2).

Uses sklearn's CalibratedClassifierCV (Platt/Isotonic) which handles
multi-class properly (avoids renormalization drift from per-class Isotonic).

Pipeline:
  1. Build X_train, y_train from 2020-2022 competitive matches (~1059 games)
  2. Wrap XGBoost in CalibratedClassifierCV (cv=5, method='isotonic')
  3. Save wrapped model → /root/data/calibrated_xgb.pkl
  4. predict_match.py auto-detects and uses calibrated model

Why CalibratedClassifierCV > per-class Isotonic:
  - Handles multi-class natively (no renorm drift)
  - Cross-validated isotonic (avoids overfitting on small samples)
  - Single .pkl file (no separate {away, draw, home} dict)

Usage:
  python3 calibrator.py             # train + save (default: isotonic, cv=5)
  python3 calibrator.py --sigmoid   # use Platt scaling instead
  python3 calibrator.py --eval      # evaluate trained model on 2022 WC
"""
import sys, os, json
sys.path.insert(0, '/root')

import numpy as np
import joblib
from collections import defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score

CACHE = '/root/data/international_results.json'
DATA_DIR = '/root/data'

# 29-dim feature builder (mirrors predict_match.py)
def build_features(home, away, dc, elo, neutral=True, form_state=None):
    """Build 29-dim feature vector for a match.
    Returns feature array (29,) or None if DC not converged.
    """
    from team_name_normalizer import normalize_match_pair
    home, away = normalize_match_pair(home, away)
    if home not in dc.team_idx_ or away not in dc.team_idx_:
        return None

    dc_p = dc.predict_proba(home, away, neutral)
    lam_h, lam_a = dc.predict_lambda(home, away, neutral)
    if lam_h is None:
        return None

    eh = elo.get(home, 1500)
    ea = elo.get(away, 1500)
    dh = ea - eh; da = eh - ea
    op = [1 / (10 ** (-dh / 400) + 1), 1 / (10 ** (-da / 400) + 1), 0.0]

    # Form features (simplified — use defaults for historical data)
    fh5 = [0.5, 1.0, 1.0, 0.0]  # win_rate, gf, ga, gd (defaults)
    fa5 = [0.5, 1.0, 1.0, 0.0]

    import math
    b15 = [
        (eh - ea) / 400,
        lam_h, lam_a, lam_h - lam_a,
        math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
        dc_p[0], dc_p[1], dc_p[2],
        fh5[0], fa5[0],
        fh5[1] - fa5[2], fa5[1] - fh5[2],
        fh5[1] - fa5[1], fh5[0] - fa5[0],
        0 if not neutral else 1,
    ]
    gold = [0.0, 1, 0, 0.0, 0.0]
    odds_feat = [op[0], op[1], op[2] if op[2] else 0.0]
    form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]

    return np.array(b15 + gold + odds_feat + form_feat)


def get_calibration_set(matches, since='2020-06-01', until='2022-12-31'):
    """Competitive matches 2020-2022 (excludes friendlies)."""
    tournaments = {
        'FIFA World Cup', 'FIFA World Cup qualification',
        'UEFA Euro', 'UEFA Euro qualification',
        'Copa America', 'CONCACAF Gold Cup',
        'African Cup of Nations', 'African Cup of Nations qualification',
    }
    subset = [m for m in matches
              if since <= m.get('date', '') <= until
              and m.get('tournament', '') in tournaments]
    return sorted(subset, key=lambda m: m['date'])


def get_wc_2022(matches):
    """2022 FIFA World Cup tournament only (64 games)."""
    wc = [m for m in matches
          if m.get('tournament', '') == 'FIFA World Cup'
          and m.get('date', '').startswith('2022')]
    return sorted(wc, key=lambda m: m['date'])


def outcome_to_label(m):
    """Map match result to class label: 0=away, 1=draw, 2=home."""
    if m['h_score'] > m['a_score']:
        return 2
    elif m['h_score'] == m['a_score']:
        return 1
    return 0


def train_calibrator(method='isotonic'):
    """Train CalibratedClassifierCV on 2020-2022 competitive matches."""
    from predict_match import _xgb_model as xgb_raw, _dc, _elo

    # Load all matches
    with open(CACHE) as f:
        matches = json.load(f)

    cal_set = get_calibration_set(matches)
    print(f"📊 校准集: 2020-2022 正式国际赛, {len(cal_set)} 场")

    X, y = [], []
    failed = 0
    for m in cal_set:
        is_qualifier = 'qualification' in m.get('tournament', '').lower()
        feat = build_features(m['home'], m['away'], _dc, _elo, neutral=not is_qualifier)
        if feat is None:
            failed += 1
            continue
        X.append(feat)
        y.append(outcome_to_label(m))

    X = np.array(X)
    y = np.array(y)
    print(f"  ✅ 训练样本: {len(X)} (失败{failed})")

    # Use a CLEAN XGBoost copy (without early_stopping_rounds) for calibration
    # The trained xgb_model_29 has early_stopping_rounds=20 which conflicts
    # with CalibratedClassifierCV's internal cv (no eval_set passed to fit)
    template_path = os.path.join(DATA_DIR, 'xgb_template.pkl')
    if os.path.exists(template_path):
        xgb_for_cal = joblib.load(template_path)
    else:
        # Build on-the-fly
        params = {k: v for k, v in xgb_raw.get_params().items()
                  if k not in ('early_stopping_rounds', 'callbacks', 'eval_metric')}
        params['early_stopping_rounds'] = None
        params['random_state'] = 42
        xgb_for_cal = XGBClassifier(**params)

    # Wrap in CalibratedClassifierCV
    print(f"\n  🔧 训练 CalibratedClassifierCV (method={method}, cv=5)...")
    cal_model = CalibratedClassifierCV(xgb_for_cal, method=method, cv=5, n_jobs=-1)
    cal_model.fit(X, y)

    # Save
    out_path = os.path.join(DATA_DIR, 'calibrated_xgb.pkl')
    joblib.dump(cal_model, out_path)
    print(f"  💾 已保存: {out_path}")
    return cal_model


def evaluate(cal_model):
    """Show before/after on 2022 WC."""
    from predict_match import _xgb_model as xgb_raw, _dc, _elo

    with open(CACHE) as f:
        matches = json.load(f)
    wc = get_wc_2022(matches)

    n = 0
    correct_before = 0
    correct_after = 0
    brier_before = 0.0
    brier_after = 0.0

    for m in wc:
        feat = build_features(m['home'], m['away'], _dc, _elo, neutral=True)
        if feat is None:
            continue
        X = np.array([feat])

        probs_before = xgb_raw.predict_proba(X)[0]
        probs_after = cal_model.predict_proba(X)[0]

        actual = outcome_to_label(m)

        if np.argmax(probs_before) == actual:
            correct_before += 1
        if np.argmax(probs_after) == actual:
            correct_after += 1

        actual_onehot = np.zeros(3)
        actual_onehot[actual] = 1
        brier_before += float(np.sum((probs_before - actual_onehot) ** 2))
        brier_after += float(np.sum((probs_after - actual_onehot) ** 2))
        n += 1

    if n == 0:
        return
    print(f"\n  📊 2022 WC 校准效果 (n={n}):")
    print(f"    HDA准确率: {correct_before}/{n} ({correct_before/n*100:.1f}%) → {correct_after}/{n} ({correct_after/n*100:.1f}%)")
    print(f"    Brier Score: {brier_before/(3*n):.4f} → {brier_after/(3*n):.4f}")


if __name__ == '__main__':
    if '--eval' in sys.argv:
        cals_path = os.path.join(DATA_DIR, 'calibrated_xgb.pkl')
        if not os.path.exists(cals_path):
            print(f"❌ 未找到 {cals_path}, 请先训练: python3 calibrator.py")
            sys.exit(1)
        cals = joblib.load(cals_path)
        evaluate(cals)
    else:
        method = 'sigmoid' if '--sigmoid' in sys.argv else 'isotonic'
        cal = train_calibrator(method=method)
        print("\n" + "="*60)
        evaluate(cal)
