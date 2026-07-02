#!/usr/bin/env python3
"""
retrain_dc_model.py — Dixon-Coles 全量重训
================================================
读取 thestats_training_data.json, 拟合 Dixon-Coles 模型 (含时间衰减),
保存为 /root/data/dc_model.pkl (覆盖生产).

用法:
  python3 retrain_dc_model.py               # 全量重训 DC 模型
  python3 retrain_dc_model.py --dry-run     # 仅 1000 场验证
  python3 retrain_dc_model.py --half-life 720  # 自定义半衰期
"""

import os, sys, json, math
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
import joblib

from dc_model_definition import DixonColes

DATA_DIR = "/root/data"
TRAINING_DATA = f"{DATA_DIR}/thestats_training_data.json"
OUTPUT_MODEL = f"{DATA_DIR}/dc_model.pkl"

MAJOR_KEYWORDS = ['FIFA World Cup', 'EURO', 'Copa America',
                  'African Cup of Nations', 'AFC Asian Cup',
                  'Gold Cup', 'Confederations Cup', 'Oceania Nations Cup']

MIN_MATCHES = 5
MAX_TEAMS = 800


def log(msg):
    print(f"  {msg}")


def fit_dc_model(df, cutoff=None, half_life=540):
    """全量 Dixon-Coles MLE 拟合 (解析梯度).
    返回已拟合的 DixonColes 实例 (dc_model_definition.DixonColes)."""
    log("🧮 拟合 Dixon-Coles MLE (解析梯度)...")
    all_teams = sorted(set(df['home'].unique()) | set(df['away'].unique()))
    n_teams = len(all_teams)
    if n_teams > MAX_TEAMS:
        log(f"  ⚠️ 球队数 {n_teams} > {MAX_TEAMS}, 截断")
        team_counts = defaultdict(int)
        for t in df['home']: team_counts[t] += 1
        for t in df['away']: team_counts[t] += 1
        keep = {t for t, c in team_counts.items() if c >= MIN_MATCHES}
        df = df[df['home'].isin(keep) & df['away'].isin(keep)].copy()
        all_teams = sorted(set(df['home'].unique()) | set(df['away'].unique()))
        n_teams = len(all_teams)
        log(f"    截断后: {n_teams} 队, {len(df)} 场")

    # 初始化实例
    dc = DixonColes(time_decay_hl=half_life)
    dc.teams_ = all_teams
    dc.team_idx_ = {t: i for i, t in enumerate(all_teams)}
    n = len(df)
    log(f"    球队: {n_teams}, 比赛: {n}")

    hi = np.array([dc.team_idx_[t] for t in df['home']])
    ai = np.array([dc.team_idx_[t] for t in df['away']])
    hs = df['h_score'].values.astype(np.float64)
    as_ = df['a_score'].values.astype(np.float64)
    w = dc._weights(df['date'].values, cutoff=cutoff)
    home_adv = (~df['neutral'].values.astype(bool)).astype(np.float64)
    global_avg = float(np.mean(list(df['h_score']) + list(df['a_score'])))
    dc.global_avg_ = global_avg
    log(f"    场均进球: {global_avg:.3f}, 有效场次: {w.sum():.0f}")

    # 初始化攻防参数
    team_gf, team_ga = defaultdict(list), defaultdict(list)
    for _, r in df.iterrows():
        team_gf[r['home']].append(r['h_score']); team_ga[r['home']].append(r['a_score'])
        team_gf[r['away']].append(r['a_score']); team_ga[r['away']].append(r['h_score'])
    init_att = np.array([math.log(max(np.mean(team_gf[t]), 0.1)/global_avg) for t in all_teams])
    init_def = np.array([math.log(max(np.mean(team_ga[t]), 0.1)/global_avg) for t in all_teams])
    init_att -= np.mean(init_att); init_def -= np.mean(init_def)

    # Stage 1: Poisson MLE
    log("Stage 1: 泊松 MLE (解析梯度)...")
    def poisson_nll(x):
        a = x[:n_teams] - np.mean(x[:n_teams])
        d = x[n_teams:2*n_teams] - np.mean(x[n_teams:2*n_teams])
        g = max(0.01, x[2*n_teams])
        lh = np.clip(np.exp(a[hi] + d[ai] + g*home_adv), 0.01, 8.0)
        la = np.clip(np.exp(a[ai] + d[hi] + g*home_adv), 0.01, 8.0)
        ll = hs*np.log(lh) - lh - gammaln(hs+1) + as_*np.log(la) - la - gammaln(as_+1)
        return -np.sum(w*ll)

    def poisson_grad(x):
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
    bounds = [(-3, 3)]*(2*n_teams) + [(0, 0.5)]

    r = minimize(poisson_nll, x0, method='L-BFGS-B', jac=poisson_grad, bounds=bounds,
                 options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 200000})
    att_opt = r.x[:n_teams] - np.mean(r.x[:n_teams])
    def_opt = r.x[n_teams:2*n_teams] - np.mean(r.x[n_teams:2*n_teams])
    gam_opt = max(0.01, r.x[2*n_teams])
    log(f"    泊松 NLL={r.fun/n:.4f} γ={gam_opt:.4f} iter={r.nit} ok={r.success}")

    # Stage 2: Dixon-Coles rho
    log("Stage 2: Dixon-Coles ρ...")
    def dc_nll(rho, att, deff, gam):
        rho = max(-0.40, min(0.0, rho))
        ac = att - np.mean(att); dc2 = deff - np.mean(deff)
        lh = np.clip(np.exp(ac[hi] + dc2[ai] + gam*home_adv), 0.01, 8.0)
        la = np.clip(np.exp(ac[ai] + dc2[hi] + gam*home_adv), 0.01, 8.0)
        el = np.exp(-lh-la)
        tau = 1 + rho*(((hs==0)&(as_==0))*el + ((hs==1)&(as_==0))*lh*el +
                       ((hs==0)&(as_==1))*la*el + ((hs==1)&(as_==1))*lh*la*el)
        tau = np.maximum(tau, 1e-10)
        ll = hs*np.log(lh)-lh-gammaln(hs+1) + as_*np.log(la)-la-gammaln(as_+1) + np.log(tau)
        return -np.sum(w*ll)

    best_rho, best_nll = 0.0, float('inf')
    for rho_v in np.linspace(-0.30, 0.0, 61):
        nll = dc_nll(rho_v, att_opt, def_opt, gam_opt)
        if nll < best_nll: best_nll, best_rho = nll, rho_v
    log(f"    ρ 网格搜索: {best_rho:.4f} NLL={best_nll/n:.4f}")

    ref = minimize(lambda p: dc_nll(max(-0.40, min(0.0, p[0])), att_opt, def_opt, max(0.0, p[1])),
                   [best_rho, gam_opt], method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
    best_rho = max(-0.40, min(0.0, ref.x[0]))
    gam_final = max(0.0, ref.x[1])
    log(f"    ρ={best_rho:.4f} γ={gam_final:.4f} NLL={ref.fun/n:.4f}")

    # Stage 3: 精化攻防
    log("Stage 3: 精化攻防参数...")
    x0_3 = np.concatenate([att_opt, def_opt])
    r3 = minimize(lambda x: dc_nll(best_rho, x[:n_teams], x[n_teams:], gam_final),
                  x0_3, method='L-BFGS-B', bounds=[(-3, 3)]*(2*n_teams),
                  options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
    dc.attack_ = r3.x[:n_teams] - np.mean(r3.x[:n_teams])
    dc.defense_ = r3.x[n_teams:] - np.mean(r3.x[n_teams:])
    log(f"    精化后 NLL={r3.fun/n:.4f} iter={r3.nit}")

    # Stage 5: Tournament rho
    dc.rho_ = best_rho
    try:
        tourney_col = [c for c in df.columns if 'tournament' in c.lower() or 'comp' in c.lower()][0]
        tourney_names = list(df[tourney_col].astype(str))
        is_major = np.array([any(kw in tn for kw in MAJOR_KEYWORDS) for tn in tourney_names], dtype=bool)
    except (IndexError, AttributeError):
        is_major = np.ones(n, dtype=bool)
    low_mask = (hs + as_ <= 3) & (w > 0) & is_major
    if low_mask.sum() >= 200:
        hi_l = hi[low_mask]; ai_l = ai[low_mask]
        hs_l = hs[low_mask]; as_l = as_[low_mask]
        w_l = w[low_mask]; home_adv_l = home_adv[low_mask]
        def rho_nll_s5(rho_v):
            rho_c = max(-0.25, min(0.25, rho_v))
            ac = dc.attack_ - np.mean(dc.attack_); dc3 = dc.defense_ - np.mean(dc.defense_)
            lh = np.clip(np.exp(ac[hi_l] + dc3[ai_l] + gam_final*home_adv_l), 0.01, 8.0)
            la = np.clip(np.exp(ac[ai_l] + dc3[hi_l] + gam_final*home_adv_l), 0.01, 8.0)
            el = np.exp(-lh-la)
            tau = 1 + rho_c*(((hs_l==0)&(as_l==0))*el + ((hs_l==1)&(as_l==0))*lh*el +
                             ((hs_l==0)&(as_l==1))*la*el + ((hs_l==1)&(as_l==1))*lh*la*el)
            tau = np.maximum(tau, 1e-10)
            ll = hs_l*np.log(lh)-lh + as_l*np.log(la)-la + np.log(tau)
            return -np.sum(w_l*ll)
        best_r5, best_n5 = 0.0, float('inf')
        for rv in np.linspace(-0.25, 0.25, 101):
            nll = rho_nll_s5(rv)
            if nll < best_n5: best_n5, best_r5 = nll, rv
        if best_r5 != 0.0 and abs(best_r5) > 0.001:
            r_ref = minimize(lambda p: rho_nll_s5(p[0]), [best_r5],
                           method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6})
            dc.rho_ = max(-0.25, min(0.25, r_ref.x[0]))
            log(f"    🎯 ρ Stage5: {dc.rho_:+.4f} (from {low_mask.sum()} matches)")
        else:
            log(f"    ρ Stage5 skipped (best={best_r5:+.4f})")
    else:
        log(f"    ⚠ low-score matches {low_mask.sum()} (<200), ρ={best_rho:.4f}")

    dc.gamma_ = gam_final
    dc.fitted_ = True

    # Host bonus
    dc.host_bonus_ = 0.0
    HOST_TEAMS = ['Canada', 'Mexico', 'United States']
    host_mask = (df['home'].isin(HOST_TEAMS)) & (~df['neutral'].values.astype(bool))
    if host_mask.sum() >= 10:
        hi_h = hi[host_mask]; ai_h = ai[host_mask]; hs_h = hs[host_mask]
        as_h = as_[host_mask]; home_adv_h = home_adv[host_mask]; w_h = w[host_mask]
        def host_nll(b):
            b = max(-0.5, min(0.5, b[0]))
            ac = dc.attack_ - np.mean(dc.attack_); dch = dc.defense_ - np.mean(dc.defense_)
            lh = np.clip(np.exp(ac[hi_h] + dch[ai_h] + (gam_final + b)*home_adv_h), 0.01, 8.0)
            la = np.clip(np.exp(ac[ai_h] + dch[hi_h] + gam_final*home_adv_h), 0.01, 8.0)
            tau = 1 + dc.rho_*(((hs_h==0)&(as_h==0))*np.exp(-lh-la) +
                               ((hs_h==1)&(as_h==0))*lh*np.exp(-lh-la) +
                               ((hs_h==0)&(as_h==1))*la*np.exp(-lh-la) +
                               ((hs_h==1)&(as_h==1))*lh*la*np.exp(-lh-la))
            tau = np.maximum(tau, 1e-10)
            ll = hs_h*np.log(lh)-lh + as_h*np.log(la)-la + np.log(tau)
            return -np.sum(w_h*ll)
        r_host = minimize(host_nll, [0.05], method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-6})
        dc.host_bonus_ = max(-0.5, min(0.5, r_host.x[0]))
        log(f"    🏟️ Host Bonus: {dc.host_bonus_:+.4f}")

    return dc


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dixon-Coles 全量重训")
    parser.add_argument("--dry-run", action="store_true", help="仅用 1000 场验证")
    parser.add_argument("--half-life", type=int, default=540, help="衰减半衰期 (天)")
    args = parser.parse_args()
    DRY_RUN = args.dry_run
    HL = args.half_life

    print(f"{'='*60}")
    print(f"  ⚽ Dixon-Coles 全量重训")
    print(f"  半衰期: {HL} 天 | 模式: {'DRY-RUN' if DRY_RUN else '全量'}")
    print(f"{'='*60}")

    if not os.path.exists(TRAINING_DATA):
        print(f"\n❌ {TRAINING_DATA} 不存在, 先跑 pull_training_data.py")
        return 1

    print(f"\n📂 加载训练数据...")
    with open(TRAINING_DATA) as f:
        records = json.load(f)
    print(f"   总记录: {len(records)}")

    df = pd.DataFrame(records)
    df['neutral'] = df.get('neutral', False)
    if DRY_RUN:
        df = df.head(1000)
    df = df.dropna(subset=['h_score', 'a_score', 'date', 'home', 'away'])
    print(f"   有效: {len(df)} 场, {len(set(df['home'].unique())|set(df['away'].unique()))} 队")
    print(f"   跨度: {df['date'].min()} → {df['date'].max()}")

    print(f"\n{'─'*60}\n  🧠 拟合 DC 模型\n{'─'*60}")
    dc = fit_dc_model(df, cutoff=datetime.now().strftime('%Y-%m-%d'), half_life=HL)

    print(f"\n{'─'*60}\n  📊 结果\n{'─'*60}")
    print(f"  球队: {len(dc.teams_)} | λ̄={dc.global_avg_:.3f} | γ={dc.gamma_:.4f} | ρ={dc.rho_:+.4f} | 东道主={dc.host_bonus_:+.4f}")
    att_top = sorted(zip(dc.teams_, dc.attack_), key=lambda x: -x[1])[:10]
    print(f"  🔥 攻击 Top: {', '.join(f'{t}({v:.2f})' for t,v in att_top)}")

    if not DRY_RUN:
        # 确保 joblib 能找到 DixonColes 类
        dc.__class__.__module__ = 'dc_model_definition'
        joblib.dump(dc, OUTPUT_MODEL)
        print(f"\n  ✅ 已保存至 {OUTPUT_MODEL}")

    # 验证加载
    try:
        dc2 = joblib.load(OUTPUT_MODEL)
        test = dc2.predict_lambda('France', 'Belgium', neutral=True)
        print(f"  ✅ 验证加载: France vs Belgium λ=({test[0]:.3f}, {test[1]:.3f})")
    except Exception as e:
        print(f"  ⚠️ 加载验证失败: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
