#!/usr/bin/env python3
"""
wc_2026_champ_runnerup.py — 冠军+亚军购买策略
==============================================
基于MC 200K模拟，同时记录冠军和亚军，输出EV/Kelly购买建议。
"""

import sys, os, json, math, random, pickle
import concurrent.futures
from datetime import datetime
from collections import defaultdict
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')

import numpy as np
import pandas as pd
from scipy.stats import poisson
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
import joblib

from wc_2026_phase1 import *
from mc_uncertainty_helper import jitter_prob, summarize_probs
from mc_market_weight_helper import market_weight_for_match

# make_odds_from_elo 已在 phase1 中定义

DATA_DIR = '/root/data'
MAX_GOALS = 6
DC_WEIGHT = 0.4
XGB_WEIGHT = 0.6
MARKET_WEIGHT = 0.40
MODEL_WEIGHT = 1.0 - MARKET_WEIGHT
HOST_TEAMS = {'United States', 'Mexico', 'Canada'}
HOST_BONUS = 0.1445

OPTUNA_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.03218571685398262,
    'n_estimators': 369,
    'reg_alpha': 3.0540401601028355,
    'reg_lambda': 2.694513099210833,
    'colsample_bytree': 0.4500553009276969,
    'subsample': 0.6426882590232543,
    'min_child_weight': 8.22712251093365,
}

def log(s=""): print(s, flush=True)

# ── Load saved models ──
log(" 📦 加载模型...")
xgb_model = joblib.load(os.path.join(DATA_DIR, 'xgb_model_20_3.pkl'))
dc = joblib.load(os.path.join(DATA_DIR, 'dc_model.pkl'))
elo = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))
if isinstance(elo, dict):
    elo = {k: float(v) for k, v in elo.items()}
log(f" DC: ρ={dc.rho_:.4f} γ={dc.gamma_:.4f}")
log(f" XGB: {type(xgb_model).__name__}")
log(f" Elo: {len(elo)} teams")

# ── Load market odds ──
market_path = os.path.join(DATA_DIR, 'theodds_api_data.json')
winner_odds = {}
if os.path.exists(market_path):
    with open(market_path) as f:
        md = json.load(f)
    winner_odds = md.get('winner_odds', {})
    log(f" 💰 市场赔率: {len(winner_odds)} 队")

# ── Load groups ──
with open(os.path.join(DATA_DIR, '2026_groups.json')) as f:
    GROUPS_2026 = json.load(f)

TEAMS_2026 = []
for g in GROUPS_2026.values():
    TEAMS_2026.extend(g)
log(f" 📋 {len(GROUPS_2026)} 组 × 4 = {len(TEAMS_2026)} 队")

# ── Build matchup cache (same logic as mc200k) ──
log(" 🔨 构建matchup缓存...")
mc_cache = {}

def dc_prob(h, a, neutral=True):
    """DC概率: 返回 (pa, pd, ph) — away/draw/home"""
    try:
        p = dc.predict_proba(h, a, neutral=neutral)
        return float(p[0]), float(p[1]), float(p[2])  # away, draw, home
    except:
        return 1/3, 1/3, 1/3

count = 0
for i, h in enumerate(TEAMS_2026):
    for a in TEAMS_2026:
        if h == a: continue
        dc_p = dc_prob(h, a, neutral=True)
        lam_h, lam_a = dc.predict_lambda(h, a, neutral=True)
        lam_h = max(0.1, min(5.0, lam_h))
        lam_a = max(0.1, min(5.0, lam_a))
        
        eh_elo = elo.get(h, 1500); ea_elo = elo.get(a, 1500)
        op = make_odds_from_elo(eh_elo, ea_elo)
        fh5 = [0.5, 0.0, 0.0, 0.0]; fa5 = [0.5, 0.0, 0.0, 0.0]
        b15 = [
            (eh_elo - ea_elo) / 400, lam_h, lam_a, lam_h - lam_a,
            math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
            dc_p[0], dc_p[1], dc_p[2],
            fh5[0], fa5[0],
            fh5[1] - fa5[2], fa5[1] - fh5[2],
            fh5[1] - fa5[1], fh5[0] - fa5[0],
            1,
        ]
        gold = [0.0, 1, 0, 0.0, 0.0]
        odds_feat = [op[0], op[1], op[2]]
        # 6 form features (placeholder for tournament sim)
        form_feat = [0.0, 0.0, 0.0, 0.0, 1.5, 1.5]
        feat = np.array([b15 + gold + odds_feat + form_feat])  # 29 dims
        xgb_p = xgb_model.predict_proba(feat)[0]
        
        dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])  # home,draw,away -> away,draw,home
        hybrid = DC_WEIGHT * dc_ado + XGB_WEIGHT * xgb_p
        
        # Market calibration
        final_hybrid = hybrid
        mh = winner_odds.get(h, 0)
        ma = winner_odds.get(a, 0)
        if mh > 0 and ma > 0:
            mw = MARKET_WEIGHT
            blended_h = hybrid[2] * MODEL_WEIGHT + (1/mh) / (1/mh + 1/ma + 0.01) * mw
            blended_a = hybrid[0] * MODEL_WEIGHT + (1/ma) / (1/mh + 1/ma + 0.01) * mw
            blended_d = max(0, 1 - blended_h - blended_a)
            final_hybrid = np.array([blended_a, blended_d, blended_h])
        
        samples = [jitter_prob(final_hybrid, epsilon=0.008, seed=(hash((h, a, i)) & 0xffffffff)) for i in range(8)]
        final_mean, final_std = summarize_probs(samples)
        final_hybrid = final_mean
        
        def make_cdf(lam):
            cdf = []; s = 0
            for k in range(MAX_GOALS + 1):
                s += poisson.pmf(k, lam)
                cdf.append(s)
            return cdf
        
        mc_cache[(h, a)] = (
            final_hybrid[0], final_hybrid[1], final_hybrid[2],
            final_std[0], final_std[1], final_std[2],
            lam_h, lam_a, make_cdf(lam_h), make_cdf(lam_a)
        )
        count += 1

log(f" ✅ {count} matchups cached")

# ── MC Worker: champion + runner_up ──
def _sim_worker_cr(mc_cache_dict, elo, seed, n_sims, teams, groups, host_teams, host_bonus):
    import random as _rnd
    _rnd.seed(seed)
    from collections import defaultdict as _dd
    import math as _math
    
    mc = {}
    for k, v in mc_cache_dict.items():
        parts = k.split('||')
        if len(parts) == 2:
            mc[(parts[0], parts[1])] = v
    
    if host_teams is None:
        host_teams = set()
    HOST_FACTOR = _math.exp(host_bonus)
    
    champ = _dd(int)
    runner = _dd(int)
    
    def _build_cdf(lam):
        s = 0; cdf = []
        for k in range(MAX_GOALS + 1):
            s += _math.exp(-lam) * (lam ** k) / _math.factorial(k)
            cdf.append(s)
        return cdf
    
    def _sim_match(mc, elo, h, a):
        if h not in host_teams and a in host_teams:
            h, a = a, h
        entry = mc.get((h, a))
        if entry is None:
            entry = mc.get((a, h))
        if entry is None:
            return 0, 0
        if (a, h) in mc and (h, a) not in mc:
            # swap
            if len(entry) >= 10:
                pa, pd_, ph, std_a, std_d, std_h, lam_a, lam_h, cdf_a, cdf_h = entry
            else:
                pa, pd_, ph, lam_a, lam_h, cdf_a, cdf_h = entry
        else:
            if len(entry) >= 10:
                pa, pd_, ph, std_a, std_d, std_h, lam_h, lam_a, cdf_h, cdf_a = entry
            else:
                pa, pd_, ph, lam_h, lam_a, cdf_h, cdf_a = entry
        
        if h in host_teams:
            lam_h *= HOST_FACTOR
            cdf_h = _build_cdf(lam_h)
        
        def _sample(cdf):
            r = _rnd.random()
            for k, cp in enumerate(cdf):
                if r <= cp: return k
            return MAX_GOALS
        
        hg, ag = _sample(cdf_h), _sample(cdf_a)
        r2 = _rnd.random()
        if r2 < ph:
            if hg <= ag: hg = ag + max(1, _rnd.randint(1, 3))
        elif r2 < ph + pd_:
            if hg != ag: sg = max(hg, ag); hg, ag = sg, sg
        else:
            if ag <= hg: ag = hg + max(1, _rnd.randint(1, 3))
        return hg, ag
    
    for _ in range(n_sims):
        pts_all = {}; gd_all = {}; gf_all = {}
        qualifiers = []
        
        for gname in sorted(groups.keys()):
            gt = groups[gname]
            pts = {t: 0 for t in gt}; gd_ = {t: 0 for t in gt}; gf_ = {t: 0 for t in gt}
            for i in range(4):
                for j in range(i+1, 4):
                    t1, t2 = gt[i], gt[j]
                    hg, ag = _sim_match(mc, elo, t1, t2)
                    gf_[t1] += hg; gf_[t2] += ag
                    gd_[t1] += hg - ag; gd_[t2] += ag - hg
                    if hg > ag: pts[t1] += 3
                    elif hg == ag: pts[t1] += 1; pts[t2] += 1
                    else: pts[t2] += 3
            ranked = sorted(gt, key=lambda t: (pts[t], gd_[t], gf_[t]), reverse=True)
            qualifiers.extend([ranked[0], ranked[1]])
            for t in gt:
                pts_all[t] = pts[t]; gd_all[t] = gd_[t]; gf_all[t] = gf_[t]
        
        thirds = []
        for gname in sorted(groups.keys()):
            gt = groups[gname]
            ranked = sorted(gt, key=lambda t: (pts_all[t], gd_all[t], gf_all[t]), reverse=True)
            thirds.append(ranked[2])
        best_thirds = sorted(thirds, key=lambda t: (pts_all[t], gd_all[t], gf_all[t]), reverse=True)[:8]
        qualifiers = qualifiers + best_thirds
        
        _rnd.shuffle(qualifiers)
        cur = [(qualifiers[i], qualifiers[i+1]) for i in range(0, 32, 2)]
        
        # R32→R16→QF→SF→Final, track finalists
        for rd in range(5):
            nxt = []
            for t1, t2 in cur:
                hg, ag = _sim_match(mc, elo, t1, t2)
                if hg == ag:
                    hg2, ag2 = _sim_match(mc, elo, t1, t2)
                    hg += hg2; ag += ag2
                    if hg == ag:
                        e1, e2 = elo.get(t1, 1500), elo.get(t2, 1500)
                        pp = 0.5 + (1 / (1 + 10**((e2 - e1) / 400)) - 0.5) * 0.3
                        winner = t1 if _rnd.random() < pp else t2
                    else:
                        winner = t1 if hg > ag else t2
                else:
                    winner = t1 if hg > ag else t2
                nxt.append(winner)
            
            if len(nxt) <= 1:
                break
            cur = [(nxt[i], nxt[i+1]) for i in range(0, len(nxt), 2)]
        
        # 决赛
        if len(cur) == 1 and len(cur[0]) == 2:
            t1, t2 = cur[0]
            hg, ag = _sim_match(mc, elo, t1, t2)
            if hg == ag:
                hg2, ag2 = _sim_match(mc, elo, t1, t2)
                hg += hg2; ag += ag2
                if hg == ag:
                    e1, e2 = elo.get(t1, 1500), elo.get(t2, 1500)
                    pp = 0.5 + (1 / (1 + 10**((e2 - e1) / 400)) - 0.5) * 0.3
                    winner = t1 if _rnd.random() < pp else t2
                else:
                    winner = t1 if hg > ag else t2
            else:
                winner = t1 if hg > ag else t2
            loser = t2 if winner == t1 else t1
            champ[winner] += 1
            runner[loser] += 1
    
    return dict(champ), dict(runner)

# ── Run MC ──
N = 200000
n_workers = 2
log(f"\n 🏃 MC {N:,} 冠军+亚军模拟 ({n_workers}进程)...")

mc_flat = {}
for (h, a), v in mc_cache.items():
    mc_flat[f"{h}||{a}"] = v

start = datetime.now()
with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
    sims_per = N // n_workers
    futures = []
    for w in range(n_workers):
        f = executor.submit(_sim_worker_cr, mc_flat, dict(elo), w*99999+42, sims_per, TEAMS_2026, GROUPS_2026, HOST_TEAMS, HOST_BONUS)
        futures.append(f)
    
    champ_total = defaultdict(int)
    runner_total = defaultdict(int)
    for i, f in enumerate(concurrent.futures.as_completed(futures)):
        c, r = f.result()
        for t, cnt in c.items(): champ_total[t] += cnt
        for t, cnt in r.items(): runner_total[t] += cnt
        log(f" worker {i+1}/{n_workers} done")

elapsed = (datetime.now() - start).total_seconds()
total = sum(champ_total.values())
log(f" ⏱ {elapsed:.1f}s ({total:,} 有效决赛)")

# ══════════════════════════════════════════
# 分析与策略输出
# ══════════════════════════════════════════

print(f"\n{'='*70}")
print(f" 🏆 2026 世界杯冠军+亚军概率 & 购买策略")
print(f"{'='*70}")

champs_sorted = sorted(champ_total.items(), key=lambda x: -x[1])
runners_sorted = sorted(runner_total.items(), key=lambda x: -x[1])

champ_prob = {t: c/total for t, c in champ_total.items()}
runner_prob = {t: r/total for t, r in runner_total.items()}

# ── 冠军EV表 ──
print(f"\n 📊 冠军EV分析")
print(f" {'─'*68}")
print(f" {'#':>3s} {'球队':<20s} {'模型%':>7s} {'赔率':>7s} {'隐含%':>7s} {'EV':>8s} {'Kelly':>7s}")
print(f" {'─'*68}")

kelly_results = []
for idx, (t, c) in enumerate(champs_sorted[:25], 1):
    p = c / total
    odds = winner_odds.get(t, 0)
    if odds > 0:
        implied = 1.0 / odds
        ev = p * odds - 1
        b = odds - 1
        kelly = max(0, (b * p - (1 - p)) / b) if b > 0 else 0
        print(f" {idx:>3d} {t:<20s} {p*100:>6.2f}% {odds:>6.1f} {implied*100:>6.1f}% {ev*100:>+7.1f}% {kelly*100:>6.2f}%")
        kelly_results.append((t, p, odds, ev, kelly))

# ── 亚军EV表 ──
print(f"\n 📊 亚军概率排名")
print(f" {'─'*50}")
print(f" {'#':>3s} {'球队':<20s} {'亚军%':>8s}")
print(f" {'─'*50}")
for idx, (t, r) in enumerate(runners_sorted[:15], 1):
    p_run = r / total
    print(f" {idx:>3d} {t:<20s} {p_run*100:>7.2f}%")

# ── 冠亚军同入 ──
print(f"\n 📊 冠军+亚军总入围率 (P(冠)+P(亚))")
print(f" {'─'*65}")
print(f" {'#':>3s} {'球队':<20s} {'冠军%':>7s} {'亚军%':>7s} {'总%':>7s} {'市场赔率':>9s}")
print(f" {'─'*65}")
combined = {}
for t in TEAMS_2026:
    cp = champ_prob.get(t, 0)
    rp = runner_prob.get(t, 0)
    combined[t] = cp + rp
combined_sorted = sorted(combined.items(), key=lambda x: -x[1])
for idx, (t, total_p) in enumerate(combined_sorted[:20], 1):
    cp = champ_prob.get(t, 0)
    rp = runner_prob.get(t, 0)
    odds = winner_odds.get(t, 0)
    print(f" {idx:>3d} {t:<20s} {cp*100:>6.2f}% {rp*100:>6.2f}% {total_p*100:>6.2f}% {odds:>8.1f}")

# ══════════════════════════════════════════
# 购买策略
# ══════════════════════════════════════════

print(f"\n{'='*70}")
print(f" 💰 购买策略推荐")
print(f"{'='*70}")

# Tier 1: 正EV冠军
tier1 = [(t, p, odds, ev, kelly) for t, p, odds, ev, kelly in kelly_results if ev > 0]
tier1.sort(key=lambda x: -x[3])

print(f"\n 🥇 Tier 1: 正EV冠军 (模型认为市场低估)")
if tier1:
    for t, p, odds, ev, kelly in tier1:
        stake = kelly * 100
        print(f"  ✓ {t:<20s} 赔率{odds:.1f} 模型{p*100:.1f}% vs 隐含{1/odds*100:.1f}% EV{ev*100:+.1f}% Kelly≈{stake:.1f}%")
else:
    print("  (无正EV标的)")

# Tier 2: 微亏但高赔率杠杆
tier2 = [(t, p, odds, ev, kelly) for t, p, odds, ev, kelly in kelly_results if -0.20 < ev <= 0]
tier2.sort(key=lambda x: -x[3])
print(f"\n 🥈 Tier 2: EV微亏但赔率杠杆大 (-20%<EV≤0)")
if tier2:
    for t, p, odds, ev, kelly in tier2:
        print(f"  △ {t:<20s} 赔率{odds:.1f} 模型{p*100:.1f}% vs 隐含{1/odds*100:.1f}% EV{ev*100:+.1f}%")

# Tier 3: 亚军标的
print(f"\n 🥉 Tier 3: 亚军高概率标的")
for t, r in runners_sorted[:8]:
    p_run = r / total
    cp = champ_prob.get(t, 0)
    total_entry = cp + p_run
    odds = winner_odds.get(t, 0)
    print(f"  ○ {t:<20s} 亚军{p_run*100:.1f}% 冠军{cp*100:.1f}% 总入决赛{total_entry*100:.1f}%")

# ── 具体投注方案 ──
print(f"\n{'='*70}")
print(f" 🎯 具体投注方案 (假设本金1000元)")
print(f"{'='*70}")

# 总仓位5-8% = 50-80元
bankroll = 1000
max_total_stake = bankroll * 0.08  # 8%

if tier1:
    total_kelly_sum = sum(k for _, _, _, _, k in tier1)
    if total_kelly_sum > 0:
        print(f"\n  【方案A: Kelly比例分配 (总仓位≤{max_total_stake:.0f}元)】")
        for t, p, odds, ev, kelly in tier1[:6]:
            alloc_pct = kelly / total_kelly_sum
            stake = min(max_total_stake * alloc_pct, max_total_stake * 0.35)  # 单注≤35%
            payout = stake * odds
            print(f"  冠军×{t:<18s} 投{stake:>6.0f}元 赔率{odds:.1f} 中奖返{payout:>7.0f}元")
        total_staked = sum(min(max_total_stake * (kelly/total_kelly_sum), max_total_stake*0.35) for _,_,_,_,kelly in tier1[:6])
        print(f"  总投注: {total_staked:.0f}元 ({total_staked/bankroll*100:.1f}%)")
else:
    print(f"\n  【无正EV标的 → 分散小额方案】")
    print(f"  建议总仓位≤5%(50元), 分散投: ")
    for t, p, odds, ev, kelly in tier2[:5]:
        stake = 10  # 每注10元小额
        payout = stake * odds
        print(f"  冠军×{t:<18s} 投{stake}元 赔率{odds:.1f} 中奖返{payout:>7.0f}元")

# ── 冠亚军组合 ──
print(f"\n  【方案B: 冠亚军组合对冲】")
print(f"  选前3热门的冠军+亚军双买, 保证决赛入场即赢:")
top3_champ = champs_sorted[:3]
for t, c in top3_champ:
    cp = c / total
    rp = runner_prob.get(t, 0) / total if runner_prob.get(t, 0) else 0
    odds = winner_odds.get(t, 0)
    if odds > 0:
        runner_odds_est = round(odds * 1.4, 1)
        total_final = cp + rp
        # 冠军投5元+亚军投5元
        stake_champ = 5; stake_runner = 5
        payout_champ = stake_champ * odds
        payout_runner = stake_runner * runner_odds_est
        print(f"  {t:<18s} 冠军×5元@{odds:.1f}+亚军×5元@~{runner_odds_est:.1f} → 入决赛概率{total_final*100:.1f}% 最多返{max(payout_champ,payout_runner):.0f}元")

# ── Save ──
output = {
    'type': 'wc2026_champ_runnerup_strategy',
    'ts': datetime.now().isoformat(),
    'sims': N,
    'champ_prob': {t: round(c/total, 6) for t, c in champ_total.items()},
    'runner_prob': {t: round(r/total, 6) for t, r in runner_total.items()},
    'winner_odds': winner_odds,
    'tier1': [(t, round(p,6), odds, round(ev,6), round(kelly,6)) for t,p,odds,ev,kelly in tier1],
    'tier2': [(t, round(p,6), odds, round(ev,6), round(kelly,6)) for t,p,odds,ev,kelly in tier2],
}
out_path = os.path.join(DATA_DIR, 'champ_runnerup_strategy.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
log(f"\n 💾 结果已保存: {out_path}")
