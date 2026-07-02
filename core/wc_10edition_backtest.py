#!/usr/bin/env python3
"""
wc_10edition_backtest.py — 29维特征 + Stacking + 让球-1 (不限时)
=========================================================================
数据: openfootball/worldcup.json (1986-2022, 10届, 604场)
优化: 
  1. 29维特征 (基线14 + h2h_n + form12 + odds3 + form6)
  2. Stacking: LR meta-learner 融合 Elo/DC/XGB
  3. 让球-1: 用 Elo 反解的市场赔率 (代理) + DC 反解
"""
import sys, os, json, math, time
from datetime import datetime
from collections import defaultdict
sys.path.insert(0, '/root')

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from wc_2026_phase1 import DixonColes, compute_elo, filter_matches, A_MATCH_TOURNAMENTS

DATA_DIR = '/root/data'
WC_HIST_PATH = os.path.join(DATA_DIR, 'wc_historical_matches.json')
INTL_PATH = os.path.join(DATA_DIR, 'international_results.json')
WC_YEARS = [1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]

def log(s=""): print(s, flush=True)

# ========== 让球评估 ==========
def handicap_probs(lh, la, handicap, max_g=6):
    p = np.zeros(3)
    for i in range(max_g+1):
        for j in range(max_g+1):
            pp = poisson.pmf(i, lh) * poisson.pmf(j, la)
            adj = i - j + handicap
            if adj > 0: p[0] += pp
            elif adj == 0: p[1] += pp
            else: p[2] += pp
    p /= p.sum()
    return p

# ========== 29维特征 (向量化加速版) ==========
def build_features_29(matches_sorted, elo, dc, all_known_teams, tier_map):
    """返回 X[N,29]"""
    # 预索引
    team_games = defaultdict(list)
    h2h_pairs = defaultdict(list)
    for m in matches_sorted:
        team_games[m['home']].append(m)
        team_games[m['away']].append(m)
        k = (m['home'], m['away']) if m['home'] < m['away'] else (m['away'], m['home'])
        h2h_pairs[k].append(m)
    for t in team_games: team_games[t].sort(key=lambda x: x['date'])
    for k in h2h_pairs: h2h_pairs[k].sort(key=lambda x: x['date'])
    
    def elo_odds(eh, ea):
        e_h = 1.0/(1+10**((ea-eh)/400))
        e_d = 0.26 * math.exp(-((eh-ea)/200)**2)
        return [e_h*(1-e_d), e_d, (1-e_h)*(1-e_d)]
    
    def recent(team, date, n):
        gs = team_games.get(team, [])
        res = []
        for mm in reversed(gs):
            if mm['date'] < date:
                if mm['home'] == team:
                    res.append((mm['h_score'], mm['a_score']))
                else:
                    res.append((mm['a_score'], mm['h_score']))
                if len(res) >= n: break
        if not res:
            return [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # w, gf, ga, gf5, gf12, ga12
        gf = sum(g for g,_ in res) / len(res)
        ga = sum(g for _,g in res) / len(res)
        w = sum(1 for g,g2 in res if g>g2) + sum(0.5 for g,g2 in res if g==g2)
        return [w/len(res), gf, ga, gf, gf, ga]  # 简: 5和12用相同
    
    def h2h_full(h, a, date, n):
        k = (h, a) if h < a else (a, h)
        ms = [mm for mm in h2h_pairs.get(k, []) if mm['date'] < date][-n:]
        if not ms:
            return [0.5, 0.0, 0.0, 0]
        w = 0; gf = 0; ga = 0
        for mm in ms:
            if mm['home'] == h:
                gf += mm['h_score']; ga += mm['a_score']
                w += 1 if mm['h_score']>mm['a_score'] else (0.5 if mm['h_score']==mm['a_score'] else 0)
            else:
                gf += mm['a_score']; ga += mm['h_score']
                w += 1 if mm['a_score']>mm['h_score'] else (0.5 if mm['a_score']==mm['h_score'] else 0)
        n_eff = len(ms)
        return [w/n_eff, gf/n_eff, ga/n_eff, n_eff]
    
    n = len(matches_sorted)
    X = np.zeros((n, 29), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)
    
    for i, m in enumerate(matches_sorted):
        h, a, d = m['home'], m['away'], m['date']
        try:
            eh = elo.get(h, 1500); ea = elo.get(a, 1500)
            lh, la = dc.predict_lambda(h, a, neutral=m.get('neutral', True))
            if lh is None: lh, la = 1.0, 1.0
            p = dc.predict_proba(h, a, neutral=m.get('neutral', True))
            
            # form5/f12 (复用同一函数, 实际12/5区别不强)
            fh = recent(h, d, 5)
            fa = recent(a, d, 5)
            fh12 = recent(h, d, 12)
            fa12 = recent(a, d, 12)
            h2h = h2h_full(h, a, d, 3)
            odds = elo_odds(eh, ea)
            
            tier_h = tier_map.get(h, 1)
            tier_a = tier_map.get(a, 1)
            neutral = 1.0 if m.get('neutral', True) else 0.0
            
            # 29维组装
            X[i] = [
                (eh - ea) / 400,           # 0: elo差归一
                lh, la, lh - la,           # 1-3: lambda
                math.log(max(lh, 0.01) / max(la, 0.01)),  # 4: log lambda比
                p[0], p[1], p[2],          # 5-7: DC probs
                fh[0], fa[0],              # 8-9: form5 胜率
                fh[1] - fa[2],             # 10: 攻-守
                fa[1] - fh[2],             # 11
                fh[1] - fa[1],             # 12: 攻vs攻
                fh[0] - fa[0],             # 13: 胜率差
                neutral,                   # 14: neutral
                h2h[1] - h2h[2],           # 15: h2h 净胜
                tier_h, tier_a,             # 16-17: tier
                fh12[1] - fa12[2],         # 18: form12 攻-守
                fa12[1] - fh12[0],         # 19
                odds[0], odds[1], odds[2],  # 20-22: Elo odds
                fh[1], fh[2],              # 23-24: form5 gf,ga
                fa[1], fa[2],              # 25-26: form5 gf,ga (客)
                fh[0]*3, fa[0]*3,          # 27-28: 胜率*3 (积分模拟)
            ]
            
            if m['h_score'] > m['a_score']: y[i] = 2
            elif m['h_score'] == m['a_score']: y[i] = 1
            else: y[i] = 0
            valid[i] = True
        except Exception as e:
            pass
    
    return X, y, valid

# ========== DC fit (快速, 关闭Stage5) ==========
def dc_fit_fast(dc, df, cutoff):
    import math as _math
    from scipy.optimize import minimize as _min
    from scipy.special import gammaln as _gammaln
    from collections import defaultdict as _dd
    all_teams = sorted(set(df['home'].unique()) | set(df['away'].unique()))
    dc.teams_ = all_teams
    dc.team_idx_ = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)
    hi = np.array([dc.team_idx_[t] for t in df['home']])
    ai = np.array([dc.team_idx_[t] for t in df['away']])
    hs = df['h_score'].values.astype(np.float64)
    as_ = df['a_score'].values.astype(np.float64)
    w = dc._weights(df['date'].values, cutoff=cutoff)
    home_adv = (~df['neutral'].values.astype(bool)).astype(np.float64)
    global_avg = np.mean(list(df['h_score']) + list(df['a_score']))
    dc.global_avg_ = global_avg
    n = len(df)
    team_gf, team_ga = _dd(list), _dd(list)
    for _, r in df.iterrows():
        team_gf[r['home']].append(r['h_score']); team_ga[r['home']].append(r['a_score'])
        team_gf[r['away']].append(r['a_score']); team_ga[r['away']].append(r['h_score'])
    init_att = np.array([_math.log(max(np.mean(team_gf[t]), 0.1)/global_avg) for t in all_teams])
    init_def = np.array([_math.log(max(np.mean(team_ga[t]), 0.1)/global_avg) for t in all_teams])
    init_att -= np.mean(init_att); init_def -= np.mean(init_def)
    def nll(x):
        a = x[:n_teams] - np.mean(x[:n_teams])
        d = x[n_teams:2*n_teams] - np.mean(x[n_teams:2*n_teams])
        g = max(0.01, x[2*n_teams])
        lh = np.clip(np.exp(a[hi] + d[ai] + g*home_adv), 0.01, 8.0)
        la = np.clip(np.exp(a[ai] + d[hi] + g*home_adv), 0.01, 8.0)
        ll = hs*np.log(lh) - lh - _gammaln(hs+1) + as_*np.log(la) - la - _gammaln(as_+1)
        return -np.sum(w*ll)
    def grad(x):
        a = x[:n_teams] - np.mean(x[:n_teams])
        d = x[n_teams:2*n_teams] - np.mean(x[n_teams:2*n_teams])
        g = max(0.01, x[2*n_teams])
        lh = np.clip(np.exp(a[hi] + d[ai] + g*home_adv), 0.01, 8.0)
        la = np.clip(np.exp(a[ai] + d[hi] + g*home_adv), 0.01, 8.0)
        vh = w*(hs - lh); va = w*(as_ - la)
        ga = np.zeros(n_teams); gd = np.zeros(n_teams)
        np.add.at(ga, hi, vh); np.add.at(ga, ai, va)
        np.add.at(gd, hi, va); np.add.at(gd, ai, vh)
        ga -= np.mean(ga); gd -= np.mean(gd)
        gg = np.sum(w*home_adv*(hs - lh)) + np.sum(w*home_adv*(as_ - la))
        return -np.concatenate([ga, gd, [gg]])
    x0 = np.zeros(2*n_teams + 1)
    x0[:n_teams] = init_att; x0[n_teams:2*n_teams] = init_def; x0[2*n_teams] = 0.12
    bounds = [(-3,3)]*(2*n_teams) + [(0,0.5)]
    r = _min(nll, x0, method='L-BFGS-B', jac=grad, bounds=bounds,
             options={'maxiter': 300, 'ftol': 1e-7, 'gtol': 1e-5})
    dc.rho_ = -0.1
    dc.attack_ = r.x[:n_teams] - np.mean(r.x[:n_teams])
    dc.defense_ = r.x[n_teams:2*n_teams] - np.mean(r.x[n_teams:2*n_teams])
    dc.gamma_ = max(0.01, r.x[2*n_teams])
    dc.fitted_ = True
    dc.host_bonus_ = 0.12
    return dc

# ========== 单届回测 ==========
def backtest_year(test_year, train_intl, test_matches, all_before, tier_map):
    log(f"\n{'='*70}")
    log(f"  📅 {test_year} WC | 训练 {len(train_intl)} | 测试 {len(test_matches)}")
    log(f"{'='*70}")
    t0 = time.time()
    
    if len(train_intl) < 100: return None
    
    elo = compute_elo(train_intl)
    dc = DixonColes(time_decay_hl=540 if test_year >= 1998 else 1080)
    dc_fit_fast(dc, pd.DataFrame(train_intl), f"{test_year}-01-01")
    log(f"  ⏱ Elo+DC: {time.time()-t0:.1f}s")
    
    # 训练特征 (29维)
    train_sorted = sorted(train_intl, key=lambda x: x['date'])
    X_train, y_train, valid = build_features_29(train_sorted, elo, dc, None, tier_map)
    X_train, y_train = X_train[valid], y_train[valid]
    log(f"  ⏱ Features: {X_train.shape} in {time.time()-t0:.1f}s")
    if len(X_train) < 200: return None
    
    # Train XGBoost
    split = int(len(X_train) * 0.9)
    xgb = XGBClassifier(
        max_depth=4, learning_rate=0.05, n_estimators=300,
        reg_alpha=2.0, reg_lambda=2.0, colsample_bytree=0.5,
        subsample=0.7, min_child_weight=5,
        n_jobs=-1, random_state=42, eval_metric='mlogloss',
        early_stopping_rounds=20, verbosity=0
    )
    t_xgb = time.time()
    xgb.fit(X_train[:split], y_train[:split],
            eval_set=[(X_train[split:], y_train[split:])], verbose=False)
    log(f"  ⏱ XGB: {time.time()-t_xgb:.1f}s (best_iter={xgb.best_iteration})")
    
    # Stacking: 在 train 上生成 out-of-fold 预测 (用 train 后 10% 作为 meta 训练集, 太短)
    # 简化: 直接用 train 最后 20% 做 meta fit
    meta_split = int(len(X_train) * 0.8)
    if meta_split > 1000:
        # XGB 在 meta_split 之后预测
        xgb_meta_train = xgb.predict_proba(X_train[meta_split:])  # [A, D, H]
        # DC prob
        dc_meta_train = []
        elo_meta_train = []
        for m in train_intl[meta_split:]:
            try:
                h, a, d = m['home'], m['away'], m['date']
                p_dc = dc.predict_proba(h, a, neutral=m.get('neutral', True))
                eh = elo.get(h, 1500); ea = elo.get(a, 1500)
                e_h = 1.0/(1+10**((ea-eh)/400))
                e_d = 0.26 * math.exp(-((eh-ea)/200)**2)
                p_elo = np.array([e_h*(1-e_d), e_d, (1-e_h)*(1-e_d)])
                p_elo /= p_elo.sum()
                dc_meta_train.append([p_dc[2], p_dc[1], p_dc[0]])  # [A,D,H]
                elo_meta_train.append([p_elo[2], p_elo[1], p_elo[0]])
            except: pass
        dc_meta_train = np.array(dc_meta_train) if dc_meta_train else None
        elo_meta_train = np.array(elo_meta_train) if elo_meta_train else None
        y_meta = y_train[meta_split:meta_split+len(xgb_meta_train)]
        
        if dc_meta_train is not None and len(dc_meta_train) == len(xgb_meta_train):
            meta_X = np.hstack([xgb_meta_train, dc_meta_train, elo_meta_train])  # 9维
            meta_y = y_meta
            meta_lr = LogisticRegression(max_iter=500, C=1.0)
            meta_lr.fit(meta_X, meta_y)
            log(f"  ⏱ Stacking LR fit: 9维meta, 样本{len(meta_y)}")
        else:
            meta_lr = None
    else:
        meta_lr = None
    
    # 预测
    history = sorted(all_before, key=lambda x: x['date'])
    team_games = defaultdict(list); h2h_pairs = defaultdict(list)
    for m in history:
        team_games[m['home']].append(m); team_games[m['away']].append(m)
        k = (m['home'], m['away']) if m['home'] < m['away'] else (m['away'], m['home'])
        h2h_pairs[k].append(m)
    
    def rform(team, date, n):
        gs = team_games.get(team, [])
        res = []
        for mm in reversed(gs):
            if mm['date'] < date:
                if mm['home'] == team:
                    res.append((mm['h_score'], mm['a_score']))
                else:
                    res.append((mm['a_score'], mm['h_score']))
                if len(res) >= n: break
        if not res: return [0.5, 0.0, 0.0]
        gf = sum(g for g,_ in res)/len(res); ga = sum(g for _,g in res)/len(res)
        w = sum(1 for g,g2 in res if g>g2) + sum(0.5 for g,g2 in res if g==g2)
        return [w/len(res), gf, ga]
    
    def predict_one(h, a, date, neutral=True):
        eh = elo.get(h, 1500); ea = elo.get(a, 1500)
        lh, la = dc.predict_lambda(h, a, neutral=neutral)
        if lh is None: lh, la = 1.0, 1.0
        p = dc.predict_proba(h, a, neutral=neutral)
        fh = rform(h, date, 5); fa = rform(a, date, 5)
        fh12 = rform(h, date, 12); fa12 = rform(a, date, 12)
        k = (h, a) if h < a else (a, h)
        ms = [mm for mm in h2h_pairs.get(k, []) if mm['date'] < date][-3:]
        if ms:
            w = gf = ga = 0
            for mm in ms:
                if mm['home'] == h:
                    gf += mm['h_score']; ga += mm['a_score']
                    w += 1 if mm['h_score']>mm['a_score'] else (0.5 if mm['h_score']==mm['a_score'] else 0)
                else:
                    gf += mm['a_score']; ga += mm['h_score']
                    w += 1 if mm['a_score']>mm['h_score'] else (0.5 if mm['a_score']==mm['h_score'] else 0)
            n = len(ms)
            h2h_w, h2h_gf, h2h_ga = w/n, gf/n, ga/n
        else:
            h2h_w, h2h_gf, h2h_ga = 0.5, 0.0, 0.0
        e_h = 1.0/(1+10**((ea-eh)/400))
        e_d = 0.26 * math.exp(-((eh-ea)/200)**2)
        odds = [e_h*(1-e_d), e_d, (1-e_h)*(1-e_d)]
        tier_h = tier_map.get(h, 1); tier_a = tier_map.get(a, 1)
        return np.array([
            (eh-ea)/400, lh, la, lh-la, math.log(max(lh,0.01)/max(la,0.01)),
            p[0], p[1], p[2],
            fh[0], fa[0], fh[1]-fa[2], fa[1]-fh[2], fh[1]-fa[1], fh[0]-fa[0],
            1.0 if neutral else 0.0,
            h2h_gf - h2h_ga, tier_h, tier_a,
            fh12[1] - fa12[2], fa12[1] - fh12[0],
            odds[0], odds[1], odds[2],
            fh[1], fh[2], fa[1], fa[2], fh[0]*3, fa[0]*3
        ], dtype=np.float32)
    
    n = len(test_matches)
    elo_preds = []; dc_preds = []; xgb_preds_list = []; stack_preds = []
    actuals = []; actuals_hcap = []; hcap_dc_preds = []; hcap_elo_preds = []
    
    for m in test_matches:
        h, a = m['home'], m['away']
        # Elo
        eh = elo.get(h, 1500); ea = elo.get(a, 1500)
        e_h = 1.0/(1+10**((ea-eh)/400))
        e_d = 0.26 * math.exp(-((eh-ea)/200)**2)
        p_elo = np.array([e_h*(1-e_d), e_d, (1-e_h)*(1-e_d)])
        p_elo /= p_elo.sum()
        # DC
        p_dc = dc.predict_proba(h, a, neutral=True)
        # XGB
        feat = predict_one(h, a, m['date'], True).reshape(1, -1)
        xp = xgb.predict_proba(feat)[0]
        # Stacking
        if meta_lr is not None:
            dc_ado = np.array([p_dc[2], p_dc[1], p_dc[0]])
            elo_ado = np.array([p_elo[2], p_elo[1], p_elo[0]])
            meta_feat = np.hstack([xp, dc_ado, elo_ado]).reshape(1, -1)
            sp = meta_lr.predict_proba(meta_feat)[0]
        else:
            sp = 0.6 * xp + 0.4 * np.array([p_dc[2], p_dc[1], p_dc[0]])
        # Actual
        act = 0 if m['h_score']<m['a_score'] else (1 if m['h_score']==m['a_score'] else 2)
        adj = m['h_score'] - m['a_score'] - 1
        act_hcap = 0 if adj>0 else (1 if adj==0 else 2)
        # 让球
        if h in dc.team_idx_ and a in dc.team_idx_:
            hi, ai = dc.team_idx_[h], dc.team_idx_[a]
            lh_v = max(0.1, min(5.0, math.exp(dc.attack_[hi]+dc.defense_[ai]+dc.gamma_)))
            la_v = max(0.1, min(5.0, math.exp(dc.attack_[ai]+dc.defense_[hi]+dc.gamma_)))
        else:
            lh_v, la_v = 1.0, 1.0
        hcap_dc = handicap_probs(lh_v, la_v, -1)
        lh_e = max(0.1, min(5.0, math.exp((eh-ea)/400/2+0.3)))
        la_e = max(0.1, min(5.0, math.exp((ea-eh)/400/2+0.3)))
        hcap_elo = handicap_probs(lh_e, la_e, -1)
        
        elo_preds.append(p_elo); dc_preds.append(p_dc)
        xgb_preds_list.append(xp); stack_preds.append(sp)
        actuals.append(act); actuals_hcap.append(act_hcap)
        hcap_dc_preds.append(hcap_dc); hcap_elo_preds.append(hcap_elo)
    
    actuals = np.array(actuals); actuals_hcap = np.array(actuals_hcap)
    elo_preds = np.array(elo_preds); dc_preds = np.array(dc_preds)
    xgb_preds_arr = np.array(xgb_preds_list); stack_preds = np.array(stack_preds)
    hcap_dc_preds = np.array(hcap_dc_preds); hcap_elo_preds = np.array(hcap_elo_preds)
    
    # 转 [A,D,H]
    elo_ado = np.array([elo_preds[:,2], elo_preds[:,1], elo_preds[:,0]]).T
    dc_ado = np.array([dc_preds[:,2], dc_preds[:,1], dc_preds[:,0]]).T
    xgb_ado = xgb_preds_arr.copy()
    stack_ado = stack_preds.copy()
    
    acc_elo = (elo_ado.argmax(1) == actuals).mean()
    acc_dc = (dc_ado.argmax(1) == actuals).mean()
    acc_xgb = (xgb_ado.argmax(1) == actuals).mean()
    acc_stack = (stack_ado.argmax(1) == actuals).mean()
    
    def brier(p, y):
        return sum(np.sum((p[i] - np.eye(3)[y[i]])**2) for i in range(len(y))) / (3*len(y))
    brier_elo = brier(elo_ado, actuals)
    brier_dc = brier(dc_ado, actuals)
    brier_xgb = brier(xgb_ado, actuals)
    brier_stack = brier(stack_ado, actuals)
    
    hcap_dc_lbl = hcap_dc_preds.argmax(1)
    hcap_elo_lbl = hcap_elo_preds.argmax(1)
    acc_hcap_dc = (hcap_dc_lbl == actuals_hcap).mean()
    acc_hcap_elo = (hcap_elo_lbl == actuals_hcap).mean()
    
    base = (actuals==2).mean(), (actuals==1).mean(), (actuals==0).mean()
    log(f"  📊 N={n}")
    log(f"     Elo:    {acc_elo*100:.1f}% br={brier_elo:.3f}")
    log(f"     DC:     {acc_dc*100:.1f}% br={brier_dc:.3f}")
    log(f"     XGB29:  {acc_xgb*100:.1f}% br={brier_xgb:.3f}")
    log(f"     Stack:  {acc_stack*100:.1f}% br={brier_stack:.3f}")
    log(f"     让-1: DC={acc_hcap_dc*100:.1f}% Elo={acc_hcap_elo*100:.1f}%")
    log(f"     基线 H/D/A={base[0]*100:.0f}/{base[1]*100:.0f}/{base[2]*100:.0f}")
    log(f"  ⏱ 总耗时 {time.time()-t0:.1f}s")
    
    return {
        'year': test_year, 'n': n,
        'elo': {'acc': float(acc_elo), 'brier': float(brier_elo)},
        'dc': {'acc': float(acc_dc), 'brier': float(brier_dc)},
        'xgb29': {'acc': float(acc_xgb), 'brier': float(brier_xgb)},
        'stack': {'acc': float(acc_stack), 'brier': float(brier_stack)},
        'hcap_dc': float(acc_hcap_dc), 'hcap_elo': float(acc_hcap_elo),
        'base': [float(b) for b in base],
    }

def main():
    log("="*70)
    log("  🌍 世界杯10届回测 v2 — 29维 + Stacking + 让球-1")
    log(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*70)
    
    wc_only = json.load(open(WC_HIST_PATH))
    intl = json.load(open(INTL_PATH))
    seen = set(); merged = []
    for m in intl:
        k = (m['date'], m['home'], m['away'])
        if k in seen: continue
        seen.add(k); merged.append(m)
    for m in wc_only:
        k = (m['date'], m['home'], m['away'])
        if k in seen: continue
        seen.add(k)
        m2 = dict(m); m2['tournament'] = 'FIFA World Cup'
        merged.append(m2)
    merged.sort(key=lambda x: x['date'])
    log(f"  加载: {len(merged)} 场")
    
    # Tier map (简化: WC常客=T1, 中等=T2, 弱队=T3) — 这里简化, 都设为1, 因tier数据难拿
    tier_map = {}
    
    t_start = time.time()
    results = []
    for yr in WC_YEARS:
        cutoff = f"{yr}-01-01"
        train = [m for m in merged if m['date'] < cutoff and m['tournament'] in A_MATCH_TOURNAMENTS]
        test = sorted([m for m in wc_only if m.get('year') == yr], key=lambda x: x['date'])
        before = [m for m in merged if m['date'] < cutoff]
        r = backtest_year(yr, train, test, before, tier_map)
        if r: results.append(r)
        log(f"  ⏱ 累计 {(time.time()-t_start):.0f}s\n")
    
    # 汇总
    log(f"\n\n{'='*70}")
    log(f"  📈 10届汇总 (29维 + Stacking)")
    log(f"{'='*70}")
    log(f"\n  {'届次':<7} {'N':>4} | Elo br | DC br | XGB29 br | STACK br | 让-1(DC) | 让-1(Elo)")
    log(f"  {'-'*80}")
    for r in results:
        e, d, x, s = r['elo'], r['dc'], r['xgb29'], r['stack']
        log(f"  {r['year']:<7} {r['n']:>4} | {e['acc']*100:>5.1f}% {e['brier']:.3f} | {d['acc']*100:>5.1f}% {d['brier']:.3f} | {x['acc']*100:>5.1f}% {x['brier']:.3f} | {s['acc']*100:>5.1f}% {s['brier']:.3f} | {r['hcap_dc']*100:>5.1f}%      | {r['hcap_elo']*100:>5.1f}%")
    
    if results:
        n_tot = sum(r['n'] for r in results)
        m_elo = np.mean([r['elo']['acc'] for r in results])
        m_dc = np.mean([r['dc']['acc'] for r in results])
        m_xgb = np.mean([r['xgb29']['acc'] for r in results])
        m_st = np.mean([r['stack']['acc'] for r in results])
        m_st_br = np.mean([r['stack']['brier'] for r in results])
        m_hc_dc = np.mean([r['hcap_dc'] for r in results])
        m_hc_elo = np.mean([r['hcap_elo'] for r in results])
        log(f"  {'-'*80}")
        log(f"  {'宏平均':<7} {n_tot:>4} | {m_elo*100:>5.1f}%     | {m_dc*100:>5.1f}%     | {m_xgb*100:>5.1f}%     | {m_st*100:>5.1f}%     | {m_hc_dc*100:>5.1f}%      | {m_hc_elo*100:>5.1f}%")
        log(f"\n  🎯 STACK Acc宏平均: {m_st*100:.2f}%  Brier宏平均: {m_st_br:.4f}")
        log(f"  🎯 XGB29 Acc宏平均: {m_xgb*100:.2f}%")
        log(f"  🎯 让球-1 DC宏平均: {m_hc_dc*100:.2f}%")
    
    out = {'ts': datetime.now().isoformat(), 'editions': WC_YEARS, 'results': results, 'note': '29 features + LR stacking meta-learner'}
    p = '/root/data/wc_10edition_backtest_v2.json'
    json.dump(out, open(p, 'w'), indent=2, default=str)
    log(f"\n  💾 {p}")
    log(f"  ⏱ 总耗时 {(time.time()-t_start):.0f}s")

if __name__ == '__main__':
    main()
