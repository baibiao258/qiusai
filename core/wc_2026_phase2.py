#!/usr/bin/env python3
"""
wc_2026_phase2.py — Phase 2: Stacking Ensemble + Weight Tuning
===============================================================
架构:
  Level 0: DC + XGBoost + RandomForest + HistGradientBoosting
  Level 1: LogisticRegression (5-fold CV)
  叠加权重优化 (grid search DC vs XGB ratio)
"""

import sys, os, json, math, random
from datetime import datetime
from collections import defaultdict, Counter
sys.path.insert(0, '/root')

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from wc_2026_phase1 import *

def log(s=""):
    print(s, flush=True)

log("="*65)
log("  PHASE 2: Weight Tuning + Stacking Ensemble")
log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("="*65)

# ── 1. Load Data ──
cache = os.path.join(DATA_DIR, 'international_results.json')
all_m = load_data(cache)
matches = filter_matches(all_m)
elo = compute_elo(all_m)

# ── 2. DC + XGBoost (same as Phase 1) ──
df = pd.DataFrame(matches)
dc = DixonColes(time_decay_hl=540)
dc.fit(df)
log(f"  DC done: rho={dc.rho_:.4f} gamma={dc.gamma_:.4f}")

X, y, match_keys = build_features(matches, dc, elo)
log(f"  Features: {X.shape}")

# Time-series split (80/20)
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
log(f"  Train: {len(X_train)} Val: {len(X_val)}")

# ── 3. Weight Optimization ──
log("\n  ─── Weight Optimization ───")
xgb_model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                          eval_metric='mlogloss', verbosity=0)
classes = np.unique(y_train)
cw = compute_class_weight('balanced', classes=classes, y=y_train)
sw = np.array([cw[list(classes).index(c)] for c in y_train])
xgb_model.fit(X_train, y_train, sample_weight=sw, verbose=False)

# DC probabilities on validation set
dc_val_probs = np.array([dc.predict_proba(m[1], m[2], True) 
                         for m in match_keys[split:]])
# XGB probabilities
xgb_val_probs = xgb_model.predict_proba(X_val)
# XGB output: [away, draw, home] → already aligned with y_val encoding (0=away,1=draw,2=home)
# BUT dc_val_probs is [home, draw, away]. Convert DC to [away, draw, home]:
dc_val_ado = dc_val_probs[:, [2, 1, 0]]  # [H,D,A] → [A,D,H]

best_w, best_acc = 0, 0
log(f"  {'DC权重':>8s} {'XGB权重':>8s} {'准确率':>8s}")
for w in np.linspace(0.1, 0.9, 9):
    hybrid = w * dc_val_ado + (1-w) * xgb_val_probs  # both in [A,D,H]
    acc = accuracy_score(y_val, np.argmax(hybrid, axis=1))
    log(f"  {w:>7.1f}  {1-w:>7.1f}  {acc*100:>6.1f}%")
    if acc > best_acc:
        best_acc, best_w = acc, w

log(f"\n  \u2705 最优权重: DC={best_w:.1f} + XGB={1-best_w:.1f}, Acc={best_acc*100:.1f}%")

# ── 4. Build Stacking Ensemble ──
log("\n  ─── Stacking Ensemble ───")

# Get DC+XGB probabilities for BOTH train and val splits
train_keys = match_keys[:split]
val_keys = match_keys[split:]

dc_train_probs = np.array([dc.predict_proba(m[1], m[2], True) for m in train_keys])
dc_train_ado = dc_train_probs[:, [2, 1, 0]]  # [H,D,A] → [A,D,H]
dc_val_ado2 = dc_val_ado  # already computed above

xgb_train_probs = xgb_model.predict_proba(X_train)  # [A,D,H] already

# Stacking features: DC probs [A,D,H] + XGB probs [A,D,H] + top 5 raw features
X_stack_train = np.column_stack([
    dc_train_ado,         # 3 cols
    xgb_train_probs,      # 3 cols
    X_train[:, :5],       # elo_diff, lam_h, lam_a, lam_diff, lam_ratio
])

X_stack_val = np.column_stack([
    dc_val_ado2,          # 3 cols
    xgb_val_probs,        # 3 cols
    X_val[:, :5],
])

log(f"  Stacking input shape: {X_stack_train.shape}")

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=8,
                                  class_weight='balanced', random_state=42,
                                  n_jobs=-1)),
    ('histgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                              random_state=42)),
    ('xgb', XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbosity=0)),
]

# Meta model
meta = LogisticRegression(C=1.0, max_iter=1000,
                          class_weight='balanced', random_state=42)

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=meta,
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1,
)

log("  Training base models + stacking...")
stack.fit(X_stack_train, y_train)

# Evaluate stacking
stack_val_probs = stack.predict_proba(X_stack_val)  # [A,D,H]
stack_acc = accuracy_score(y_val, np.argmax(stack_val_probs, axis=1))
stack_nll = log_loss(y_val, stack_val_probs)
y_oh = np.zeros((len(y_val), 3))
y_oh[np.arange(len(y_val)), y_val] = 1
stack_brier = np.mean(np.sum((stack_val_probs - y_oh)**2, axis=1))

log(f"\n  Validation Results:")
log(f"  {'Model':<25s} {'Accuracy':>10s} {'LogLoss':>10s} {'Brier':>10s}")
log(f"  {'─'*55}")
log(f"  {'DC alone':<25s} {'':>10s} {'':>10s} {'':>10s}")
log(f"  {'XGBoost alone':<25s} {best_acc*100:>9.1f}%")
log(f"  {'Best Hybrid (DC+XGB)':<25s} {best_acc*100:>9.1f}%")
log(f"  {'Stacking Ensemble':<25s} {stack_acc*100:>9.1f}% {stack_nll:>10.4f} {stack_brier:>10.4f}")

# ── 5. 2022 World Cup Backtest ──
log("\n  ─── 2022 WC Backtest (Stacking) ───")
wc = [m for m in all_m if m['tournament'] == 'FIFA World Cup'
      and '2022-11-20' <= m['date'] <= '2022-12-18']
log(f"  Matches: {len(wc)}")

elo_wc = defaultdict(lambda: 1500.0)
pre = [m for m in all_m if m['date'] < '2022-11-20']
for m in pre:
    h,a = m['home'],m['away']
    e_h = 1/(1+10**((elo_wc[a]-elo_wc[h])/400))
    sh = 1 if m['h_score']>m['a_score'] else (0.5 if m['h_score']==m['a_score'] else 0)
    elo_wc[h] += 32*(sh-e_h); elo_wc[a] += 32*((1-sh)-(1-e_h))

correct_dc = correct_xgb = correct_hybrid = correct_stack = 0
brier_dc = brier_xgb = brier_hybrid = brier_stack = 0

# Y encoding: 0=away, 1=draw, 2=home (same as training)
actual_map = {'A':0, 'D':1, 'H':2}

for m in wc:
    h, a = m['home'], m['away']
    dc_p_hda = dc.predict_proba(h, a, True)     # [H,D,A]
    lam_h, lam_a = dc.predict_lambda(h, a, True)
    
    if lam_h is None:
        feat = np.array([[(elo_wc.get(h,1500)-elo_wc.get(a,1500))/400,
                          1.0, 1.0, 0.0, 0.0, 1/3, 1/3, 1/3,
                          0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1]])
    else:
        feat = np.array([[(elo_wc.get(h,1500)-elo_wc.get(a,1500))/400,
                          lam_h, lam_a, lam_h-lam_a,
                          math.log(max(lam_h,0.01)/max(lam_a,0.01)),
                          dc_p_hda[0], dc_p_hda[1], dc_p_hda[2],
                          0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1]])
    
    xgb_p_ado = xgb_model.predict_proba(feat)[0]   # [A,D,H]
    # Hybrid: convert DC to [A,D,H] to match XGB
    dc_p_ado = np.array([dc_p_hda[2], dc_p_hda[1], dc_p_hda[0]])  # [H,D,A] → [A,D,H]
    hybrid_ado = best_w * dc_p_ado + (1-best_w) * xgb_p_ado
    
    # Stacking: features in [A,D,H] order
    stack_in = np.array([[dc_p_ado[0], dc_p_ado[1], dc_p_ado[2],
                          xgb_p_ado[0], xgb_p_ado[1], xgb_p_ado[2],
                          feat[0,0], feat[0,1], feat[0,2],
                          feat[0,3], feat[0,4]]])
    stack_ado = stack.predict_proba(stack_in)[0]    # [A,D,H] already
    
    # Actual result in [0=A, 1=D, 2=H] encoding
    if m['h_score'] > m['a_score']: actual = 2  # H
    elif m['h_score'] == m['a_score']: actual = 1  # D
    else: actual = 0  # A
    
    if np.argmax(dc_p_ado) == actual: correct_dc += 1
    if np.argmax(xgb_p_ado) == actual: correct_xgb += 1
    if np.argmax(hybrid_ado) == actual: correct_hybrid += 1
    if np.argmax(stack_ado) == actual: correct_stack += 1
    
    yoh = np.array([1 if actual==0 else 0, 1 if actual==1 else 0, 1 if actual==2 else 0])
    brier_stack += np.sum((stack_ado - yoh)**2)

n_wc = len(wc)
log(f"\n  2022 WC Results ({n_wc} matches):")
log(f"  {'Model':<25s} {'Accuracy':>10s}")
log(f"  {'─'*35}")
log(f"  {'DC alone':<25s} {correct_dc/n_wc*100:>9.1f}%")
log(f"  {'XGBoost alone':<25s} {correct_xgb/n_wc*100:>9.1f}%")
log(f"  {'Best Hybrid':<25s} {correct_hybrid/n_wc*100:>9.1f}%")
log(f"  {'Stacking Ensemble':<25s} {correct_stack/n_wc*100:>9.1f}%")
log(f"\n  Stacking Brier: {brier_stack/(3*n_wc):.4f}")

# ── 6. Monte Carlo with Stacking ──
# Precompute all model outputs for 48×47 matchups
log("\n  ─── Precomputing Stacking Matchups ───")
mc_cache = {}
count = 0
for h in TEAMS_2026:
    for a in TEAMS_2026:
        if h == a: continue
        dc_p = dc.predict_proba(h, a, True)         # [H,D,A]
        lam_h, lam_a = dc.predict_lambda(h, a, True)
        if lam_h is None:
            feat = np.array([[(elo.get(h,1500)-elo.get(a,1500))/400,
                              1.0, 1.0, 0.0, 0.0, 1/3, 1/3, 1/3,
                              0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1]])
        else:
            feat = np.array([[(elo.get(h,1500)-elo.get(a,1500))/400,
                              lam_h, lam_a, lam_h-lam_a,
                              math.log(max(lam_h,0.01)/max(lam_a,0.01)),
                              dc_p[0], dc_p[1], dc_p[2],
                              0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1]])
        xgb_p = xgb_model.predict_proba(feat)[0]       # [A,D,H]
        dc_p_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])  # [H,D,A] → [A,D,H]
        hybrid_ado = best_w * dc_p_ado + (1-best_w) * xgb_p
        
        # Stacking: features in [A,D,H] order
        stack_in = np.array([[dc_p_ado[0], dc_p_ado[1], dc_p_ado[2],
                              xgb_p[0], xgb_p[1], xgb_p[2],
                              feat[0,0], feat[0,1], feat[0,2],
                              feat[0,3], feat[0,4]]])
        stack_ado = stack.predict_proba(stack_in)[0]    # [A,D,H] already
        
        # CDF from DC lambdas (same as Phase 1)
        def make_cdf(lam):
            cut = 0.0; cdf = []
            for k in range(MAX_GOALS+1):
                cut += poisson.pmf(k, lam)
                cdf.append(cut)
            return cdf
        
        lam_h = max(0.1, min(5.0, lam_h if lam_h else 1.0))
        lam_a = max(0.1, min(5.0, lam_a if lam_a else 1.0))
        mc_cache[(h,a)] = (stack_ado[0], stack_ado[1], stack_ado[2],  # A,D,H → stores as-is for sim
                           lam_h, lam_a, make_cdf(lam_h), make_cdf(lam_a))
        count += 1
log(f"  \u2705 {count} matchups cached")

def sim_from_cache(mc, elo, h, a):
    """Fast simulation from precomputed cache (uses [H,D,A] for result gen)"""
    entry = mc.get((h,a), mc.get((a,h), (1/3, 1/3, 1/3, 1.0, 1.0, list(range(7)), list(range(7)))))
    pa, pd, ph, lam_h, lam_a, cdf_h, cdf_a = entry  # A,D,H probs
    def sample(cdf):
        r = random.random()
        for k, cp in enumerate(cdf):
            if r <= cp: return k
        return MAX_GOALS
    hg, ag = sample(cdf_h), sample(cdf_a)
    r2 = random.random()
    if r2 < ph:  # H
        if hg <= ag: hg = ag + max(1, random.randint(1, 3))
    elif r2 < ph + pd:  # D
        if hg != ag: sg = max(hg, ag); hg, ag = sg, sg
    else:  # A
        if ag <= hg: ag = hg + max(1, random.randint(1, 3))
    return hg, ag

N = 50000
log(f"\n  MC {N:,} (Stacking)...")
champ = defaultdict(int)
BATCH = 10000
for batch in range(N // BATCH):
    for _ in range(BATCH):
        st = sorted(TEAMS_2026, key=lambda t: elo.get(t,1500), reverse=True)
        pots = [st[i:i+16] for i in range(0, 48, 16)]
        groups = {}
        for pi, pot in enumerate(pots):
            sh = list(pot); random.shuffle(sh)
            for gi, tm in enumerate(sh):
                gn = chr(ord('A')+gi)
                if gn not in groups: groups[gn] = []
                groups[gn].append(tm)
        q = []
        for gn in sorted(groups):
            gt = groups[gn]
            if len(gt) != 3: continue
            pts = {t:0 for t in gt}; gd = {t:0 for t in gt}; gf = {t:0 for t in gt}
            for t1,t2 in [(gt[0],gt[1]),(gt[0],gt[2]),(gt[1],gt[2])]:
                hg,ag = sim_from_cache(mc_cache, elo, t1, t2)
                gf[t1]+=hg; gf[t2]+=ag; gd[t1]+=hg-ag; gd[t2]+=ag-hg
                if hg>ag: pts[t1]+=3
                elif hg==ag: pts[t1]+=1; pts[t2]+=1
                else: pts[t2]+=3
            rk = sorted(gt, key=lambda t:(pts[t],gd[t],gf[t]), reverse=True)
            q.append(rk[:2])
        if len(q)!=16: continue
        r32 = []
        for i in range(0,16,2):
            r32.append((q[i][0], q[i+1][1]))
            r32.append((q[i+1][0], q[i][1]))
        cur = r32
        for _ in range(5):
            if len(cur)<=1: break
            nxt = []
            for i in range(0,len(cur),2):
                t1,t2=cur[i][0],cur[i+1][0]
                hg,ag = sim_from_cache(mc_cache, elo, t1, t2)
                if hg==ag:
                    hg2,ag2 = sim_from_cache(mc_cache, elo, t1, t2)
                    hg+=hg2; ag+=ag2
                    if hg==ag:
                        e1,e2=elo.get(t1,1500),elo.get(t2,1500)
                        pp=0.5+(1/(1+10**((e2-e1)/400))-0.5)*0.3
                        winner=t1 if random.random()<pp else t2
                        nxt.append((winner,None))
                        continue
                winner=t1 if hg>ag else t2
                nxt.append((winner,None))
            cur=nxt
        if cur: champ[cur[0][0]]+=1
    log(f"    batch {batch+1}/{N//BATCH} done")

total = sum(champ.values())
log(f"\n  \u2705 {total:,} simulations")
log(f"\n{'='*65}")
log(f"  \U0001f3c6 2026 Champion Probabilities (Stacking Ensemble)")
log(f"{'='*65}")
champs = sorted(champ.items(), key=lambda x:-x[1])
best_pct = champs[0][1]/total*100 if champs else 0
for i,(t,c) in enumerate(champs[:20], 1):
    pct = c/total*100
    bar = chr(9608)*int(pct/best_pct*20) + chr(9617)*(20-int(pct/best_pct*20))
    log(f"  {i:>3d}. {t:<25s} {c:>6,d} {pct:>6.2f}% {bar}")
if len(champs)>20:
    oc = sum(c for _,c in champs[20:])
    log(f"  {' '*4} Others ({len(champs)-20}) {oc:>6,d} {oc/total*100:>6.2f}%")

# Save
result = {'model':'Stacking Ensemble (RF+HistGB+XGB+DC)','ts':datetime.now().isoformat(),
          'sims':total,'dc_weight':best_w,'stacking_cv':'5-fold',
          'backtest_wc2022':{
              'dc_acc':correct_dc/n_wc,'hybrid_acc':correct_hybrid/n_wc,
              'stacking_acc':correct_stack/n_wc,'n':n_wc},
          'champs':[(t,c,c/total*100) for t,c in champs[:30]]}
with open(os.path.join(DATA_DIR,'phase2_results.json'),'w') as f:
    json.dump(result, f, indent=2, default=str)
log(f"\n  \U0001f4be Saved to {DATA_DIR}/phase2_results.json")
log(f"\n{'='*65}")
log("  PHASE 2 COMPLETE")
log(f"{'='*65}")
