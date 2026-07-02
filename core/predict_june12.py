#!/usr/bin/env python3
"""Predict June 12, 2026 World Cup matches"""
import sys, os, json, math
sys.path.insert(0, '/root')
import numpy as np
import pandas as pd
from scipy.stats import poisson
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from wc_2026_phase1 import *

# ── Load ──
cache = os.path.join(DATA_DIR, 'international_results.json')
all_m = load_data(cache)
matches = filter_matches(all_m)
elo = compute_elo(all_m)
df = pd.DataFrame(matches)

# ── DC ──
dc = DixonColes(time_decay_hl=540)
dc.fit(df)

# ── Odds function ──
def elo_to_odds_prob(elo_h, elo_a):
    e_h = 1.0 / (1 + 10**((elo_a - elo_h) / 400))
    e_draw = 0.26 * math.exp(-((elo_h-elo_a)/200)**2)
    e_home = e_h * (1 - e_draw)
    e_away = (1 - e_h) * (1 - e_draw)
    total = e_home + e_draw + e_away
    margin = 0.06
    odds_h = 1.0 / ((e_home/total) * (1-margin))
    odds_d = 1.0 / ((e_draw/total) * (1-margin))
    odds_a = 1.0 / ((e_away/total) * (1-margin))
    margin_check = 1/odds_h + 1/odds_d + 1/odds_a
    return np.array([(1/odds_h)/margin_check, (1/odds_d)/margin_check, (1/odds_a)/margin_check])

# ── Build features ──
X_list, y_list = [], []
ms = sorted(matches, key=lambda m: m['date'])
for i, m in enumerate(ms):
    h, a = m['home'], m['away']
    elo_h, elo_a = elo.get(h, 1500), elo.get(a, 1500)
    lam_h, lam_a = dc.predict_lambda(h, a, neutral=m.get('neutral', False))
    if lam_h is None: continue
    dc_probs = dc.predict_proba(h, a, neutral=m.get('neutral', False))
    odds_probs = elo_to_odds_prob(elo_h, elo_a)
    fh = compute_recent_form(ms[:i], h, m['date'])
    fa = compute_recent_form(ms[:i], a, m['date'])
    feat = [(elo_h-elo_a)/400, lam_h, lam_a, lam_h-lam_a,
            math.log(max(lam_h,0.01)/max(lam_a,0.01)),
            dc_probs[0], dc_probs[1], dc_probs[2],
            odds_probs[0], odds_probs[1], odds_probs[2],
            fh[0], fa[0], fh[1]-fa[2], fa[1]-fh[2],
            fh[1]-fa[1], fh[0]-fa[0], int(m.get('neutral',False))]
    X_list.append(feat)
    if m['h_score'] > m['a_score']: y_list.append(2)
    elif m['h_score'] == m['a_score']: y_list.append(1)
    else: y_list.append(0)

X = np.array(X_list); y = np.array(y_list)
split = int(len(X)*0.8)
X_train, y_train = X[:split], y[:split]
classes = np.unique(y_train)
cw = compute_class_weight('balanced', classes=classes, y=y_train)
sw = np.array([cw[list(classes).index(c)] for c in y_train])

xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                    eval_metric='mlogloss', verbosity=0)
xgb.fit(X_train, y_train, sample_weight=sw, verbose=False)

# ── Predict ──
def predict_match(dc, xgb, elo, home, away, neutral=True):
    dc_p = dc.predict_proba(home, away, neutral)
    lam_h, lam_a = dc.predict_lambda(home, away, neutral)
    elo_h, elo_a = elo.get(home,1500), elo.get(away,1500)
    odds_probs = elo_to_odds_prob(elo_h, elo_a)
    
    if lam_h is None:
        feat = np.array([[(elo_h-elo_a)/400,1.0,1.0,0.0,0.0,
                         1/3,1/3,1/3,odds_probs[0],odds_probs[1],odds_probs[2],
                         0.5,0.5,0.0,0.0,0.0,0.0,1]])
    else:
        feat = np.array([[(elo_h-elo_a)/400, lam_h, lam_a, lam_h-lam_a,
                         math.log(max(lam_h,0.01)/max(lam_a,0.01)),
                         dc_p[0], dc_p[1], dc_p[2],
                         odds_probs[0], odds_probs[1], odds_probs[2],
                         0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1]])
    
    xgb_p = xgb.predict_proba(feat)[0]  # [A,D,H]
    dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])
    hybrid = 0.6 * dc_ado + 0.4 * xgb_p  # [A,D,H]
    
    result = ['Away', 'Draw', 'Home'][np.argmax(hybrid)]
    
    lam_h = max(0.1, min(5.0, lam_h if lam_h else 1.0))
    lam_a = max(0.1, min(5.0, lam_a if lam_a else 1.0))
    
    best_score, best_prob = (0,0), 0
    for i in range(7):
        for j in range(7):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            if p > best_prob:
                best_prob = p
                best_score = (i, j)
    
    return (home, away, dc_p, hybrid, result, best_score, best_prob, lam_h, lam_a)

# Match 1
print("=" * 60)
print("  2026 World Cup - June 12 Predictions")
print("  (DC + XGBoost + Market Calibrated Odds)")
print("=" * 60)

home1, away1 = "Canada", "Bosnia and Herzegovina"
print(f"\nMatch 1: {home1} vs {away1}")
print("  BMO Field, Toronto | 15:00 ET")
r1 = predict_match(dc, xgb, elo, home1, away1, neutral=False)
dc1 = r1[2]; hyb1 = r1[3]; res1 = r1[4]; sc1 = r1[5]; pr1 = r1[6]; lam1 = r1[7:9]

elo_c = elo.get(home1,1500); elo_b = elo.get(away1,1500)
print(f"  Elo: {home1}={elo_c:.0f} {away1}={elo_b:.0f} (diff={elo_c-elo_b:+.0f})")
print(f"  Expected goals: {home1}={lam1[0]:.2f} {away1}={lam1[1]:.2f}")
print(f"  DC model (H/D/A): {dc1[0]*100:.1f}% / {dc1[1]*100:.1f}% / {dc1[2]*100:.1f}%")
print(f"  Hybrid model (H/D/A): {hyb1[2]*100:.1f}% / {hyb1[1]*100:.1f}% / {hyb1[0]*100:.1f}%")
print(f"  Predicted result: {res1}")
print(f"  Most likely score: {sc1[0]}-{sc1[1]} ({pr1*100:.1f}%)")

# Match 2
home2, away2 = "United States", "Paraguay"
print(f"\nMatch 2: {home2} vs {away2}")
print("  SoFi Stadium, Inglewood | 18:00 PT")
r2 = predict_match(dc, xgb, elo, home2, away2, neutral=False)
dc2 = r2[2]; hyb2 = r2[3]; res2 = r2[4]; sc2 = r2[5]; pr2 = r2[6]; lam2 = r2[7:9]

elo_u = elo.get(home2,1500); elo_p = elo.get(away2,1500)
print(f"  Elo: {home2}={elo_u:.0f} {away2}={elo_p:.0f} (diff={elo_u-elo_p:+.0f})")
print(f"  Expected goals: {home2}={lam2[0]:.2f} {away2}={lam2[1]:.2f}")
print(f"  DC model (H/D/A): {dc2[0]*100:.1f}% / {dc2[1]*100:.1f}% / {dc2[2]*100:.1f}%")
print(f"  Hybrid model (H/D/A): {hyb2[2]*100:.1f}% / {hyb2[1]*100:.1f}% / {hyb2[0]*100:.1f}%")
print(f"  Predicted result: {res2}")
print(f"  Most likely score: {sc2[0]}-{sc2[1]} ({pr2*100:.1f}%)")

print("\n" + "=" * 60)
print("  Final Predictions:")
print(f"  {home1} {sc1[0]}-{sc1[1]} {away1} ({res1})")
print(f"  {home2} {sc2[0]}-{sc2[1]} {away2} ({res2})")
print("=" * 60)
