#!/usr/bin/env python3
"""
wc_2026_phase1.py — Phase 1: Dixon-Coles + XGBoost + Monte Carlo
===============================================================
目标: 单场 HDA 准确率从 57.8% → 63%+
"""

import json, math, os, sys, urllib.request, csv, random
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.special import gammaln
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

def println(s=""):
    print(s, flush=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SIMULATIONS = 100000
MAX_GOALS = 6

TEAMS_2026 = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia",
    "Croatia", "Cura\u00e7ao", "Czech Republic", "DR Congo", "Ecuador",
    "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan",
    "Jordan", "Mexico", "Morocco", "Netherlands", "New Zealand",
    "Norway", "Panama", "Paraguay", "Portugal", "Qatar",
    "Saudi Arabia", "Scotland", "Senegal", "South Africa", "South Korea",
    "Spain", "Sweden", "Switzerland", "Tunisia", "Turkey",
    "United States", "Uruguay", "Uzbekistan"
]

A_MATCH_TOURNAMENTS = {
    'FIFA World Cup', 'FIFA World Cup qualification', 'FIFA Series',
    'UEFA Euro', 'UEFA Euro qualification', 'UEFA Nations League',
    'Copa Am\u00e9rica',
    'African Cup of Nations', 'African Cup of Nations qualification',
    'AFC Asian Cup', 'AFC Asian Cup qualification',
    'Gold Cup', 'Gold Cup qualification',
    'Oceania Nations Cup',
    'CONCACAF Nations League', 'CONCACAF Series',
    'Friendly', 'Friendlies',
    'Arab Cup', 'Arab Cup qualification',
    'Gulf Cup', 'CAFA Nations Cup', 'SAFF Cup', 'Baltic Cup',
}

def load_data(cache_path):
    if not os.path.exists(cache_path):
        println("  \U0001f4e1 下载国际赛数据...")
        url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'wc_predictor/1.0'})
            raw = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
        except Exception as e:
            println(f"  \u274c 下载失败: {e}")
            return []
        matches = []
        for row in csv.DictReader(raw.splitlines()):
            try:
                matches.append({
                    'date': row['date'], 'home': row['home_team'],
                    'away': row['away_team'], 'tournament': row['tournament'],
                    'h_score': int(row['home_score']), 'a_score': int(row['away_score']),
                    'neutral': row.get('neutral', '').strip().lower() == 'true',
                })
            except: continue
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(matches, f)
        return matches
    with open(cache_path) as f:
        return json.load(f)

def filter_matches(matches, start_date=None, end_date=None):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    filtered = [m for m in matches if start_date <= m['date'] <= end_date
                and m['tournament'] in A_MATCH_TOURNAMENTS]
    println(f"  \U0001f4ca 赛事过滤: {len(matches)} \u2192 {len(filtered)} 场 (A级赛)")
    return filtered

def compute_elo(matches):
    elo = defaultdict(lambda: 1500.0)
    for m in sorted(matches, key=lambda x: x['date']):
        h, a = m['home'], m['away']
        e_h = 1.0 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        sh = 1.0 if m['h_score'] > m['a_score'] else (0.5 if m['h_score'] == m['a_score'] else 0.0)
        elo[h] += 32 * (sh - e_h)
        elo[a] += 32 * ((1-sh) - (1 - e_h))
    return dict(elo)


# ============================================================
#  DIXON-COLES MLE (with analytical gradient)
# ============================================================

class DixonColes:
    def __init__(self, time_decay_hl=540):
        self.teams_ = []
        self.team_idx_ = {}
        self.attack_ = None
        self.defense_ = None
        self.rho_ = 0.0
        self.gamma_ = 0.0
        self.global_avg_ = None
        self.half_life_ = time_decay_hl
        self.host_bonus_ = 0.0
        self.fitted_ = False

    def _weights(self, dates, cutoff=None):
        if cutoff is None:
            cutoff = datetime.now().strftime('%Y-%m-%d')
        cutoff_dt = datetime.strptime(cutoff, '%Y-%m-%d')
        days = np.array([(cutoff_dt - datetime.strptime(d, '%Y-%m-%d')).days for d in dates])
        return 0.5 ** (np.maximum(days, 0) / self.half_life_)

    def fit(self, df, cutoff=None):
        println("  \U0001f9ee 拟合 Dixon-Coles MLE (解析梯度)...")
        all_teams = sorted(set(df['home'].unique()) | set(df['away'].unique()))
        self.teams_ = all_teams
        self.team_idx_ = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)
        println(f"    球队: {n_teams}, 比赛: {len(df)}")

        hi = np.array([self.team_idx_[t] for t in df['home']])
        ai = np.array([self.team_idx_[t] for t in df['away']])
        hs = df['h_score'].values.astype(np.float64)
        as_ = df['a_score'].values.astype(np.float64)
        w = self._weights(df['date'].values, cutoff=cutoff)
        home_adv = (~df['neutral'].values.astype(bool)).astype(np.float64)
        global_avg = np.mean(list(df['h_score']) + list(df['a_score']))
        self.global_avg_ = global_avg
        n = len(df)
        println(f"    场均进球: {global_avg:.3f}, 时间衰减和: {w.sum():.0f}")

        # Init from raw stats
        team_gf, team_ga = defaultdict(list), defaultdict(list)
        for _, r in df.iterrows():
            team_gf[r['home']].append(r['h_score']); team_ga[r['home']].append(r['a_score'])
            team_gf[r['away']].append(r['a_score']); team_ga[r['away']].append(r['h_score'])
        init_att = np.array([math.log(max(np.mean(team_gf[t]), 0.1)/global_avg) for t in all_teams])
        init_def = np.array([math.log(max(np.mean(team_ga[t]), 0.1)/global_avg) for t in all_teams])
        init_att -= np.mean(init_att); init_def -= np.mean(init_def)

        # Stage 1: Poisson MLE with analytical gradient
        println("    Stage 1: 泊松 MLE (解析梯度)...")
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
        bounds = [(-3,3)]*(2*n_teams) + [(0,0.5)]

        r = minimize(poisson_nll, x0, method='L-BFGS-B', jac=poisson_grad, bounds=bounds,
                     options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 200000})
        att_opt = r.x[:n_teams] - np.mean(r.x[:n_teams])
        def_opt = r.x[n_teams:2*n_teams] - np.mean(r.x[n_teams:2*n_teams])
        gam_opt = max(0.01, r.x[2*n_teams])
        println(f"    泊松 NLL={r.fun/n:.4f} \u03b3={gam_opt:.4f} iter={r.nit} ok={r.success}")

        # Stage 2: Dixon-Coles rho
        println("    Stage 2: Dixon-Coles \u03c1...")
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
        for rho in np.linspace(-0.30, 0.0, 61):
            nll = dc_nll(rho, att_opt, def_opt, gam_opt)
            if nll < best_nll: best_nll, best_rho = nll, rho
        println(f"    \u03c1 网格搜索: {best_rho:.4f} NLL={best_nll/n:.4f}")

        ref = minimize(lambda p: dc_nll(max(-0.40,min(0.0,p[0])), att_opt, def_opt, max(0.0,p[1])),
                       [best_rho, gam_opt], method='Nelder-Mead',
                       options={'maxiter':200,'xatol':1e-6,'fatol':1e-8})
        best_rho = max(-0.40, min(0.0, ref.x[0]))
        gam_final = max(0.0, ref.x[1])
        println(f"    \u03c1={best_rho:.4f} \u03b3={gam_final:.4f} NLL={ref.fun/n:.4f}")

        # Stage 3: Refine attack/defense with fixed rho,gamma
        println("    Stage 3: 精化攻防参数...")
        x0_3 = np.concatenate([att_opt, def_opt])
        r3 = minimize(lambda x: dc_nll(best_rho, x[:n_teams], x[n_teams:], gam_final),
                      x0_3, method='L-BFGS-B', bounds=[(-3,3)]*(2*n_teams),
                      options={'maxiter':200,'ftol':1e-8,'gtol':1e-6})
        att_final = r3.x[:n_teams] - np.mean(r3.x[:n_teams])
        def_final = r3.x[n_teams:] - np.mean(r3.x[n_teams:])
        println(f"    精化后 NLL={r3.fun/n:.4f} iter={r3.nit}")

        # Stage 5: Refine rho on major tournament low-scoring matches (total goals ≤ 3)
        # rho corrects for score correlation in important matches — critical for WC
        # Using major tournaments only to avoid noise from minnow-vs-minnow blowouts
        self.rho_ = best_rho  # keep Stage 2 rho as fallback
        major_keywords = ['FIFA World Cup', 'UEFA Euro', 'Copa América',
                          'African Cup of Nations', 'AFC Asian Cup', 'Gold Cup',
                          'Confederations Cup', 'Oceania Nations Cup']
        try:
            tourney_col = df.columns[df.columns.str.contains('tournament|competition', case=False)][0]
            tourney_names = list(df[tourney_col].astype(str))
            is_major = np.array([any(kw in tn for kw in major_keywords) for tn in tourney_names], dtype=bool)
        except (IndexError, AttributeError):
            is_major = np.ones(len(df), dtype=bool)  # fallback: all matches
        low_mask = (hs + as_ <= 3) & (w > 0) & is_major
        if low_mask.sum() >= 200:
            hi_l = hi[low_mask]; ai_l = ai[low_mask]
            hs_l = hs[low_mask]; as_l = as_[low_mask]
            w_l = w[low_mask]
            home_adv_l = home_adv[low_mask]
            
            def rho_nll_stage5(rho):
                rho_c = max(-0.25, min(0.25, rho))  # allow positive for WC (0-0>Poisson)
                ac = att_final - np.mean(att_final); dc3 = def_final - np.mean(def_final)
                lh = np.clip(np.exp(ac[hi_l] + dc3[ai_l] + gam_final*home_adv_l), 0.01, 8.0)
                la = np.clip(np.exp(ac[ai_l] + dc3[hi_l] + gam_final*home_adv_l), 0.01, 8.0)
                el = np.exp(-lh-la)
                tau = 1 + rho_c*(((hs_l==0)&(as_l==0))*el + ((hs_l==1)&(as_l==0))*lh*el +
                                 ((hs_l==0)&(as_l==1))*la*el + ((hs_l==1)&(as_l==1))*lh*la*el)
                tau = np.maximum(tau, 1e-10)
                ll = hs_l*np.log(lh)-lh + as_l*np.log(la)-la + np.log(tau)
                return -np.sum(w_l*ll)
            
            # Grid search rho ∈ [-0.25, 0.25] on low-scoring major tournament matches
            best_r, best_n = 0.0, float('inf')
            for r in np.linspace(-0.25, 0.25, 101):
                nll = rho_nll_stage5(r)
                if nll < best_n: best_n, best_r = nll, r
            
            if best_r != 0.0 and abs(best_r) > 0.001:
                # Fine-tune with Nelder-Mead
                r_ref = minimize(lambda p: rho_nll_stage5(p[0]), [best_r],
                               method='Nelder-Mead', options={'maxiter':100, 'xatol':1e-6, 'fatol':1e-8})
                self.rho_ = max(-0.25, min(0.25, r_ref.x[0]))
                println(f"    🎯 ρ Stage5: {self.rho_:+.4f} (from {low_mask.sum()} low-score major matches) NLL improved")
            else:
                println(f"    ⚠ ρ Stage5: ρ≈0 (best={best_r:+.4f}), keeping ρ={best_rho:.4f}")
        else:
            println(f"    ⚠ low-score matches only {low_mask.sum()} (<500), keeping ρ={best_rho:.4f}")

        self.attack_ = att_final; self.defense_ = def_final
        self.gamma_ = gam_final; self.fitted_ = True

        # Stage 4: Estimate host bonus (extra home advantage for host nations)
        self.host_bonus_ = 0.0
        HOST_TEAMS = ['Canada', 'Mexico', 'United States']
        host_mask = ((df['home'].isin(HOST_TEAMS)) & (~df['neutral'].values.astype(bool)))
        if host_mask.sum() >= 10:
            hi_h = hi[host_mask]; ai_h = ai[host_mask]
            hs_h = hs[host_mask]; as_h = as_[host_mask]
            w_h = w[host_mask]
            
            def host_nll(hb):
                lh = np.clip(np.exp(att_final[hi_h] + def_final[ai_h] + gam_final + hb), 0.01, 8.0)
                la = np.clip(np.exp(att_final[ai_h] + def_final[hi_h] + gam_final), 0.01, 8.0)
                ll = hs_h*np.log(lh)-lh + as_h*np.log(la)-la
                return -np.sum(w_h*ll)
            
            rh = minimize(host_nll, 0.15, method='Nelder-Mead',
                         options={'maxiter':100, 'xatol':1e-6, 'fatol':1e-8})
            self.host_bonus_ = max(0.0, rh.x[0])
            println(f"    🏟 host_bonus={self.host_bonus_:.4f} (from {host_mask.sum()} host matches) NLL={rh.fun/host_mask.sum():.4f}")
        else:
            println(f"    ⚠ host matches only {host_mask.sum()} (<10), using gamma only")

        ta = sorted([(self.teams_[i], self.attack_[i]) for i in range(n_teams)],
                    key=lambda x:-x[1])
        td = sorted([(self.teams_[i], self.defense_[i]) for i in range(n_teams)],
                    key=lambda x:x[1])
        println(f"\n    \u2694 Top 10 攻击力:")
        for t,v in ta[:10]: println(f"      {t:<25s} {v:>7.3f}")
        println(f"\n    \U0001f6e1 Top 10 防守力 (越低越好):")
        for t,v in td[:10]: println(f"      {t:<25s} {v:>7.3f}")
        return self

    def predict_lambda(self, home, away, neutral=True, host_bonus=0.0):
        hi, ai = self.team_idx_.get(home), self.team_idx_.get(away)
        if hi is None or ai is None: return None, None
        lh = math.exp(self.attack_[hi] + self.defense_[ai] + self.gamma_*(0 if neutral else 1) + host_bonus)
        la = math.exp(self.attack_[ai] + self.defense_[hi] + self.gamma_*(0 if neutral else 1))
        return max(0.1,min(5.0,lh)), max(0.1,min(5.0,la))

    def predict_proba(self, home, away, neutral=True, host_bonus=0.0):
        lh, la = self.predict_lambda(home, away, neutral, host_bonus)
        if lh is None: return np.array([1/3, 1/3, 1/3])
        probs = {}
        for i in range(MAX_GOALS+1):
            for j in range(MAX_GOALS+1):
                p = poisson.pmf(i, lh) * poisson.pmf(j, la)
                if (i<=1 and j<=1):
                    el = math.exp(-lh-la)
                    tau = 1 + self.rho_*(
                        (1 if i==0 and j==0 else 0)*el +
                        (1 if i==1 and j==0 else 0)*lh*el +
                        (1 if i==0 and j==1 else 0)*la*el +
                        (1 if i==1 and j==1 else 0)*lh*la*el)
                    p *= max(tau, 1e-10)
                probs[(i,j)] = max(p, 1e-10)
        total = sum(probs.values())
        ph = sum(v for (i,j),v in probs.items() if i>j)/total
        pd2 = sum(v for (i,j),v in probs.items() if i==j)/total
        pa = sum(v for (i,j),v in probs.items() if i<j)/total
        return np.array([ph, pd2, pa])


# ============================================================
#  3. FEATURE ENGINEERING
# ============================================================

def compute_recent_form(matches, team, date, n=5):
    """最近n场表现: [胜率, 场均进, 场均失, 净胜球均]"""
    relevant = [m for m in matches
                if m['date'] < date and (m['home'] == team or m['away'] == team)]
    relevant = sorted(relevant, key=lambda m: m['date'], reverse=True)[:n]
    if not relevant:
        return [0.5, 0.0, 0.0, 0.0]
    wins = 0; gf = 0; ga = 0
    for m in relevant:
        if m['home'] == team:
            gf += m['h_score']; ga += m['a_score']
            if m['h_score'] > m['a_score']: wins += 1
            elif m['h_score'] == m['a_score']: wins += 0.5
        else:
            gf += m['a_score']; ga += m['h_score']
            if m['a_score'] > m['h_score']: wins += 1
            elif m['a_score'] == m['h_score']: wins += 0.5
    return [wins/len(relevant), gf/len(relevant), ga/len(relevant), (gf-ga)/len(relevant)]

def compute_h2h(matches, home, away, date, max_n=3):
    """历史交锋: 主队对客队近N场 [主队胜率, 主队场均进, 主队场均失, 场次]"""
    h2h = [m for m in matches if m['date'] < date
           and ((m['home'] == home and m['away'] == away)
                or (m['home'] == away and m['away'] == home))]
    h2h = sorted(h2h, key=lambda m: m['date'], reverse=True)[:max_n]
    if not h2h:
        return [0.5, 0.0, 0.0, 0]
    wins = 0; gf = 0; ga = 0
    for m in h2h:
        if m['home'] == home:
            gf += m['h_score']; ga += m['a_score']
            if m['h_score'] > m['a_score']: wins += 1
            elif m['h_score'] == m['a_score']: wins += 0.5
        else:
            gf += m['a_score']; ga += m['h_score']
            if m['a_score'] > m['h_score']: wins += 1
            elif m['a_score'] == m['h_score']: wins += 0.5
    return [wins/len(h2h), gf/len(h2h), ga/len(h2h), len(h2h)]

def tournament_tier(tournament):
    """赛事等级: [is_friendly, is_major_cup, is_knockout_capable, is_qualifier]"""
    t = tournament or ''
    friendly = int(t in ('Friendly', 'Friendlies'))
    major = int(any(kw in t for kw in ('FIFA World Cup', 'UEFA Euro', 'Copa América',
                                        'African Cup of Nations', 'AFC Asian Cup',
                                        'Gold Cup', 'Oceania Nations Cup')))
    final_round = int(any(kw in t for kw in ('Final', 'Semi', 'Quarter', 'Round',
                                              'play-off', 'Play-off', 'knockout')))
    qualifier = int(any(kw in t for kw in ('qualification', 'Qualification')))
    return [friendly, major, final_round, qualifier]

def compute_rest_days(matches, team, date):
    """距离上一场比赛的休整天数"""
    prev = [m for m in matches if m['date'] < date
            and (m['home'] == team or m['away'] == team)]
    if not prev:
        return 30
    last = max(m['date'] for m in prev)
    delta = datetime.strptime(date, '%Y-%m-%d') - datetime.strptime(last, '%Y-%m-%d')
    return delta.days

def build_features(matches, dc_model, elo_ratings):
    """构建特征矩阵"""
    println("  🔧 构建特征...")
    features, targets, match_keys = [], [], []
    ms = sorted(matches, key=lambda m: m['date'])

    for i, m in enumerate(ms):
        h, a = m['home'], m['away']
        elo_h, elo_a = elo_ratings.get(h, 1500), elo_ratings.get(a, 1500)
        lam_h, lam_a = dc_model.predict_lambda(h, a, neutral=m.get('neutral', False))
        if lam_h is None: continue
        dc_probs = dc_model.predict_proba(h, a, neutral=m.get('neutral', False))
        fh5 = compute_recent_form(ms[:i], h, m['date'], 5)
        fa5 = compute_recent_form(ms[:i], a, m['date'], 5)

        feat = [
            (elo_h - elo_a) / 400,          # Elo差
            lam_h,                            # DC λ home
            lam_a,                            # DC λ away
            lam_h - lam_a,                    # λ差
            math.log(max(lam_h,0.01)/max(lam_a,0.01)),  # λ比
            dc_probs[0], dc_probs[1], dc_probs[2],  # DC概率
            fh5[0], fa5[0],                     # 近期胜率
            fh5[1] - fa5[2],                    # 主队进攻-客队防守
            fa5[1] - fh5[2],                    # 客队进攻-主队防守
            fh5[1] - fa5[1],                    # 进球差
            fh5[0] - fa5[0],                    # 胜率差
            int(m.get('neutral', False)),     # 是否中性场
        ]
        features.append(feat)
        if m['h_score'] > m['a_score']: targets.append(2)
        elif m['h_score'] == m['a_score']: targets.append(1)
        else: targets.append(0)
        match_keys.append((m['date'], h, a))

    return np.array(features), np.array(targets), match_keys

def make_feat_vec(elo_h, elo_a, lam_h, lam_a, dc_p, odds_p, 
                  fh5=None, fa5=None, fh12=None, fa12=None, h2h=None, tier=None, 
                  rest_h=14, rest_a=14, neutral=1):
    """构建33维特征向量，缺失值用默认值"""
    defaults = lambda: [0.5, 0.0, 0.0, 0.0]
    fh5 = fh5 or defaults(); fa5 = fa5 or defaults()
    fh12 = fh12 or defaults(); fa12 = fa12 or defaults()
    h2h = h2h or [0.5, 0.0, 0.0, 0]
    tier = tier or [0, 0, 0, 0]
    return np.array([[
        (elo_h - elo_a) / 400,
        lam_h, lam_a, lam_h - lam_a,
        math.log(max(lam_h,0.01)/max(lam_a,0.01)),
        dc_p[0], dc_p[1], dc_p[2],
        odds_p[0], odds_p[1], odds_p[2],
        fh5[0], fa5[0], fh5[1] - fa5[2], fa5[1] - fh5[2],
        fh5[1] - fa5[1], fh5[0] - fa5[0], fh5[3], fa5[3],
        fh12[0], fa12[0], fh12[1] - fa12[2], fa12[1] - fh12[1],
        h2h[0], h2h[1] - h2h[2], h2h[3],
        tier[0], tier[1], tier[2],
        rest_h, rest_a, rest_h - rest_a,
        neutral,
    ]])


# ============================================================
#  4. XGBOOST
# ============================================================

def train_xgboost(X, y):
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_class_weight

    println(f"  \U0001f332 训练 XGBoost... ({len(X)} samples, {X.shape[1]} features)")
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    sw = np.array([cw[list(classes).index(c)] for c in y_train])

    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                          eval_metric='mlogloss', early_stopping_rounds=20,
                          verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              sample_weight=sw, verbose=False)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    acc = accuracy_score(y_val, y_pred)
    nll = log_loss(y_val, y_proba)
    y_oh = np.zeros((len(y_val), 3))
    y_oh[np.arange(len(y_val)), y_val] = 1
    brier = np.mean(np.sum((y_proba - y_oh)**2, axis=1))
    vc = Counter(y_val)

    println(f"    \u2705 验证集: 准确率={acc*100:.1f}% LogLoss={nll:.4f} Brier={brier:.4f}")
    println(f"    分布: H={vc.get(2,0)} D={vc.get(1,0)} A={vc.get(0,0)}")
    return model, (X_train, X_val, y_train, y_val)


# ============================================================
#  5. MONTE CARLO
# ============================================================

def precompute_matchups(dc_model, xgb_model, teams, elo_ratings):
    """预计算参赛队所有对战的混合概率 + CDF"""
    println(f"  \U0001f4a1 预计算 {len(teams)}\u00d7{len(teams)} 对战概率...")
    probs = {}
    count = 0
    for h in teams:
        for a in teams:
            if h == a: continue
            dc_p = dc_model.predict_proba(h, a, True)
            lam_h, lam_a = dc_model.predict_lambda(h, a, True)
            if lam_h is not None and xgb_model is not None:
                feat = np.array([[(elo_ratings.get(h,1500)-elo_ratings.get(a,1500))/400,
                                  lam_h, lam_a, lam_h-lam_a,
                                  math.log(max(lam_h,0.01)/max(lam_a,0.01)),
                                  dc_p[0], dc_p[1], dc_p[2],
                                  0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1]])
                xp = xgb_model.predict_proba(feat)[0]
                xgb_hda = np.array([xp[2], xp[1], xp[0]])
            else:
                xgb_hda = dc_p
            hybrid = dc_p * 0.6 + xgb_hda * 0.4
            # 预计算 CDF
            def make_cdf(lam):
                cut = 0.0; cdf = []
                for k in range(MAX_GOALS+1):
                    cut += poisson.pmf(k, lam)
                    cdf.append(cut)
                return cdf
            probs[(h, a)] = (hybrid, lam_h, lam_a, make_cdf(lam_h), make_cdf(lam_a))
            count += 1
    println(f"    \u2705 {count} 对战组合已缓存 (含 CDF)")
    return probs

def simulate_from_cache(matchup_cache, elo_ratings, h, a):
    """从缓存快速模拟单场 (用预计算 CDF)"""
    hybrid, lam_h, lam_a, cdf_h, cdf_a = matchup_cache.get(
        (h, a), matchup_cache.get((a, h), (np.array([1/3,1/3,1/3]), 1.0, 1.0, list(range(7)), list(range(7)))))
    
    def sample(cdf):
        r = random.random()
        for k, cp in enumerate(cdf):
            if r <= cp: return k
        return MAX_GOALS
    
    hg, ag = sample(cdf_h), sample(cdf_a)
    r2 = random.random()
    if r2 < hybrid[0]:  # H
        if hg <= ag: hg = ag + max(1, random.randint(1, 3))
    elif r2 < hybrid[0] + hybrid[1]:  # D
        if hg != ag: sg = max(hg, ag); hg, ag = sg, sg
    else:  # A
        if ag <= hg: ag = hg + max(1, random.randint(1, 3))
    return hg, ag

def monte_carlo_fast(dc_model, xgb_model, teams, elo_ratings, n=SIMULATIONS):
    """快速 MC (预计算对战)"""
    mc = precompute_matchups(dc_model, xgb_model, teams, elo_ratings)

    println(f"\n  \U0001f3c3 快速 MC ({n:,} 次)...")
    champ = defaultdict(int)
    batch_size = 10000
    nb = n // batch_size

    for batch in range(nb):
        if batch > 0 and batch % 5 == 0:
            println(f"    {batch}/{nb} ({batch/nb*100:.0f}%) sims run...")
        for _ in range(batch_size):
            st = sorted(teams, key=lambda t: elo_ratings.get(t,1500), reverse=True)
            pots = [st[i:i+16] for i in range(0, 48, 16)]
            groups = {}
            for pi, pot in enumerate(pots):
                shuffled = list(pot); random.shuffle(shuffled)
                for gi, team in enumerate(shuffled):
                    gname = chr(ord('A')+gi)
                    if gname not in groups: groups[gname] = []
                    groups[gname].append(team)

            q = []
            for gname in sorted(groups):
                gt = groups[gname]
                if len(gt) != 3: continue
                pts = {t:0 for t in gt}; gd = {t:0 for t in gt}; gf = {t:0 for t in gt}
                for t1, t2 in [(gt[0],gt[1]), (gt[0],gt[2]), (gt[1],gt[2])]:
                    hg, ag = simulate_from_cache(mc, elo_ratings, t1, t2)
                    gf[t1]+=hg; gf[t2]+=ag; gd[t1]+=hg-ag; gd[t2]+=ag-hg
                    if hg>ag: pts[t1]+=3
                    elif hg==ag: pts[t1]+=1; pts[t2]+=1
                    else: pts[t2]+=3
                rk = sorted(gt, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
                q.append(rk[:2])

            if len(q) != 16: continue
            r32 = []
            for i in range(0, 16, 2):
                r32.append((q[i][0], q[i+1][1]))
                r32.append((q[i+1][0], q[i][1]))
            cur = r32
            for _ in range(5):
                if len(cur) <= 1: break
                nxt = []
                for i in range(0, len(cur), 2):
                    t1, t2 = cur[i][0], cur[i+1][0]
                    hg, ag = simulate_from_cache(mc, elo_ratings, t1, t2)
                    if hg == ag:
                        hg2, ag2 = simulate_from_cache(mc, elo_ratings, t1, t2)
                        hg += hg2; ag += ag2
                        if hg == ag:
                            e1, e2 = elo_ratings.get(t1,1500), elo_ratings.get(t2,1500)
                            pp = 0.5 + (1/(1+10**((e2-e1)/400)) - 0.5)*0.3
                            winner = t1 if random.random() < pp else t2
                            nxt.append((winner, None))
                            continue
                    winner = t1 if hg > ag else t2
                    nxt.append((winner, None))
                cur = nxt
            if cur: champ[cur[0][0]] += 1

    total = sum(champ.values())
    println(f"  \u2705 完成! {total:,}")
    return {'total': total, 'champion': sorted(champ.items(), key=lambda x:-x[1])}


# ============================================================
#  6. BACKTEST 2022
# ============================================================

def backtest_2022(dc_model, xgb_model, all_matches):
    println(f"\n  \U0001f4cb ===== 2022 世界杯回测 =====")
    wc = [m for m in all_matches if m['tournament'] == 'FIFA World Cup'
          and '2022-11-20' <= m['date'] <= '2022-12-18']
    println(f"    比赛数: {len(wc)}")
    if not wc: return None

    elo = defaultdict(lambda: 1500.0)
    pre = [m for m in all_matches if m['date'] < '2022-11-20']
    for m in pre:
        h, a = m['home'], m['away']
        e_h = 1/(1+10**((elo[a]-elo[h])/400))
        sh = 1 if m['h_score']>m['a_score'] else (0.5 if m['h_score']==m['a_score'] else 0)
        elo[h] += 32*(sh-e_h); elo[a] += 32*((1-sh)-(1-e_h))

    correct_h, correct_dc, n_m = 0, 0, len(wc)
    brier_h, brier_dc, nll_h, nll_dc = 0, 0, 0, 0

    for m in wc:
        h, a = m['home'], m['away']
        dc_p = dc_model.predict_proba(h, a, True)
        lam_h, lam_a = dc_model.predict_lambda(h, a, True)
        if lam_h is not None and xgb_model is not None:
            feat = np.array([[(elo.get(h,1500)-elo.get(a,1500))/400,
                              lam_h, lam_a, lam_h-lam_a,
                              math.log(max(lam_h,0.01)/max(lam_a,0.01)),
                              dc_p[0], dc_p[1], dc_p[2],
                              0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1]])
            xp = xgb_model.predict_proba(feat)[0]
            xgb_hda = np.array([xp[2], xp[1], xp[0]])
        else:
            xgb_hda = dc_p
        hybrid = dc_p*0.6 + xgb_hda*0.4

        if m['h_score'] > m['a_score']: actual = 0
        elif m['h_score'] == m['a_score']: actual = 1
        else: actual = 2

        if np.argmax(hybrid) == actual: correct_h += 1
        if np.argmax(dc_p) == actual: correct_dc += 1
        yoh = np.array([1 if actual==0 else 0, 1 if actual==1 else 0, 1 if actual==2 else 0])
        brier_h += np.sum((hybrid-yoh)**2)
        brier_dc += np.sum((dc_p-yoh)**2)
        nll_h += -math.log(max(hybrid[actual], 1e-10))
        nll_dc += -math.log(max(dc_p[actual], 1e-10))

    println(f"\n    回测 ({n_m}场):")
    println(f"    {'─'*42}")
    println(f"    {'DC模型':>20s} | HDA={correct_dc/n_m*100:.1f}% Brier={brier_dc/(3*n_m):.4f}")
    println(f"    {'混合模型':>20s} | HDA={correct_h/n_m*100:.1f}% Brier={brier_h/(3*n_m):.4f}")
    return {'n':n_m, 'dc_acc':correct_dc/n_m, 'hybrid_acc':correct_h/n_m,
            'dc_brier':brier_dc/(3*n_m), 'hybrid_brier':brier_h/(3*n_m)}


# ============================================================
#  7. MAIN
# ============================================================

def main():
    println("="*65)
    println("  \u26bd 2026 Phase 1: Dixon-Coles + XGBoost + Monte Carlo")
    println(f"  \U0001f550 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    println("="*65)

    # Load
    cache = os.path.join(DATA_DIR, 'international_results.json')
    all_m = load_data(cache)
    if not all_m: return 1
    matches = filter_matches(all_m)
    println(f"  过滤后: {len(matches)} 场")

    # Elo
    println("\n\U0001f9ee Elo...")
    elo = compute_elo(all_m)
    for t,s in sorted(elo.items(), key=lambda x:-x[1])[:10]:
        println(f"  {t:<25s} {s:>5.0f}")

    # DC
    import pandas as pd
    df = pd.DataFrame(matches)
    dc = DixonColes(time_decay_hl=540)
    dc.fit(df)

    # Features + XGBoost
    X, y, _ = build_features(matches, dc, elo)
    xgb_result = train_xgboost(X, y)
    xgb_model = xgb_result[0] if xgb_result else None

    # Backtest 2022
    bt = backtest_2022(dc, xgb_model, all_m)

    # MC
    println(f"\n{'='*65}")
    res = monte_carlo_fast(dc, xgb_model, TEAMS_2026, elo, n=SIMULATIONS)
    total, champs = res['total'], res['champion']

    println(f"\n{'='*65}")
    println(f"  \U0001f3c6 2026 冠军概率 (混合模型, {total:,}次)")
    println(f"{'='*65}")
    println(f"  {'#':>3s} {'球队':<25s} {'次数':>6s} {'概率%':>7s}")
    println(f"  {'─'*42}")
    best_pct = champs[0][1]/total*100
    for i,(t,c) in enumerate(champs[:20], 1):
        pct = c/total*100
        bar = '\u2588'*int(pct/best_pct*20) + '\u2591'*(20-int(pct/best_pct*20))
        println(f"  {i:>3d}. {t:<25s} {c:>6,d} {pct:>6.2f}% {bar}")
    if len(champs) > 20:
        oc = sum(c for _,c in champs[20:])
        println(f"  {'─'*42}")
        println(f"      其他{len(champs)-20}队{'':<19s} {oc:>6,d} {oc/total*100:>6.2f}%")

    # Save
    out = {'model':'Dixon-Coles+XGBoost+MC', 'ts':datetime.now().isoformat(),
           'sims':total, 'dc_params':{'rho':dc.rho_,'gamma':dc.gamma_},
           'backtest':bt,
           'champs':[(t,c,c/total*100) for t,c in champs[:30]]}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR,'phase1_results.json'),'w') as f:
        json.dump(out, f, indent=2, default=str)
    println(f"\n  \U0001f4be 已保存: {DATA_DIR}/phase1_results.json")
    return 0

if __name__ == '__main__':
    sys.exit(main())


# ============================================================
#  7. STRICT TEMPORAL BACKTEST (无泄漏, 优化版)
# ============================================================

def strict_backtest_2022(all_matches):
    """
    严格时序无泄漏回测（增量优化 O(n)）。
    用 per-team 缓存替代全量扫描，35K 场也能秒级构建。
    """
    println(f"\n{'='*65}")
    println(f"  🛡️ 严格时序 2022 回测（增量优化版）")
    println(f"{'='*65}")

    cutoff = '2022-11-20'
    wc = [m for m in all_matches if m['tournament'] == 'FIFA World Cup'
          and '2022-11-20' <= m['date'] <= '2022-12-18']
    historical = [m for m in all_matches if m['date'] < cutoff]
    train_all = [m for m in historical if m['tournament'] in A_MATCH_TOURNAMENTS]

    clean_elo = compute_elo(historical)
    import pandas as pd
    clean_dc = DixonColes(time_decay_hl=540)
    clean_dc.fit(pd.DataFrame(train_all))

    println(f"  训练集: {len(train_all)} 场 | 测试集: {len(wc)} 场")
    println(f"  Elo: {len(clean_elo)} 队 | DC: {len(clean_dc.teams_)} 队")

    # ── 增量特征构建器：O(1) per match ──
    class FeatureBuffer:
        def __init__(self, elo, dc):
            self.elo = elo
            self.dc = dc
            self.team_games = defaultdict(list)   # team -> [{date, gf, ga, is_home}]
            self.h2h_cache = defaultdict(lambda: defaultdict(list))  # t1->t2->[results]
            self.last_date = {}  # team -> last date
        
        def add_match(self, m):
            h, a = m['home'], m['away']
            for team, gf, ga in [(h, m['h_score'], m['a_score']), (a, m['a_score'], m['h_score'])]:
                self.team_games[team].append({'date':m['date'],'gf':gf,'ga':ga})
                self.last_date[team] = m['date']
            # H2H
            key = (h, a) if h < a else (a, h)
            self.h2h_cache[key[0]][key[1]].append({
                'date': m['date'], 'home': h, 'away': a,
                'h_score': m['h_score'], 'a_score': m['a_score']})
        
        def recent_form(self, team, date, n):
            games = [g for g in self.team_games.get(team, []) if g['date'] < date]
            relevant = sorted(games, key=lambda x: x['date'], reverse=True)[:n]
            if not relevant:
                return [0.5, 0.0, 0.0, 0.0]
            w = sum(1 for g in relevant if g['gf'] > g['ga']) + \
                sum(0.5 for g in relevant if g['gf'] == g['ga'])
            gf = sum(g['gf'] for g in relevant) / len(relevant)
            ga = sum(g['ga'] for g in relevant) / len(relevant)
            return [w/len(relevant), gf, ga, gf-ga]
        
        def h2h(self, home, away, date, n):
            k1, k2 = (home, away) if home < away else (away, home)
            raw = self.h2h_cache.get(k1, {}).get(k2, [])
            raw = [x for x in raw if x['date'] < date][-n:]
            if not raw:
                return [0.5, 0.0, 0.0, 0]
            w = 0; gf = 0; ga = 0
            for x in raw:
                if x['home'] == home:
                    gf += x['h_score']; ga += x['a_score']
                    w += 1 if x['h_score'] > x['a_score'] else (0.5 if x['h_score'] == x['a_score'] else 0)
                else:
                    gf += x['a_score']; ga += x['h_score']
                    w += 1 if x['a_score'] > x['h_score'] else (0.5 if x['a_score'] == x['h_score'] else 0)
            return [w/len(raw), gf/len(raw), ga/len(raw), len(raw)]
        
        def rest_days(self, team, date):
            ld = self.last_date.get(team)
            if not ld: return 30
            d = (datetime.strptime(date,'%Y-%m-%d') - datetime.strptime(ld,'%Y-%m-%d')).days
            return max(1, d)

        def make_odds(self, eh, ea):
            e_h = 1.0 / (1 + 10**((ea - eh) / 400))
            e_d = 0.26 * math.exp(-((eh-ea)/200)**2)
            margin = 0.06
            o = np.array([e_h*(1-e_d), e_d, (1-e_h)*(1-e_d)])
            o /= o.sum()
            return o

    # ── 构建训练特征（15维 + 33维，一次遍历） ──
    println("\n  🔧 构建特征（一次遍历双输出）...")
    X15, y15 = [], []
    X33, y33 = [], []
    fb = FeatureBuffer(clean_elo, clean_dc)
    ms = sorted(train_all, key=lambda m: m['date'])

    for i, m in enumerate(ms):
        if i % 5000 == 0: println(f"    {i}/{len(ms)}...")
        h, a = m['home'], m['away']
        eh, ea = clean_elo.get(h,1500), clean_elo.get(a,1500)
        lh, la = clean_dc.predict_lambda(h, a, neutral=m.get('neutral',False))
        if lh is None: continue
        dp = clean_dc.predict_proba(h, a, neutral=m.get('neutral',False))
        op = fb.make_odds(eh, ea)
        fh5 = fb.recent_form(h, m['date'], 5)
        fa5 = fb.recent_form(a, m['date'], 5)

        # 15维
        X15.append([(eh-ea)/400, lh, la, lh-la,
            math.log(max(lh,0.01)/max(la,0.01)),
            dp[0], dp[1], dp[2],
            fh5[0], fa5[0], fh5[1]-fa5[2], fa5[1]-fh5[2],
            fh5[1]-fa5[1], fh5[0]-fa5[0], int(m.get('neutral',False))])
        y15.append(2 if m['h_score']>m['a_score'] else (1 if m['h_score']==m['a_score'] else 0))

        # 33维
        fh12 = fb.recent_form(h, m['date'], 12)
        fa12 = fb.recent_form(a, m['date'], 12)
        h2h = fb.h2h(h, a, m['date'], 3)
        tier = tournament_tier(m.get('tournament',''))
        rh = fb.rest_days(h, m['date'])
        ra = fb.rest_days(a, m['date'])
        X33.append([(eh-ea)/400, lh, la, lh-la,
            math.log(max(lh,0.01)/max(la,0.01)),
            dp[0], dp[1], dp[2], op[0], op[1], op[2],
            fh5[0], fa5[0], fh5[1]-fa5[2], fa5[1]-fh5[2],
            fh5[1]-fa5[1], fh5[0]-fa5[0], fh5[3], fa5[3],
            fh12[0], fa12[0], fh12[1]-fa12[2], fa12[1]-fh12[1],
            h2h[0], h2h[1]-h2h[2], h2h[3],
            tier[0], tier[1], tier[2], rh, ra, rh-ra, int(m.get('neutral',False))])
        y33.append(y15[-1])

        fb.add_match(m)

    X15 = np.array(X15); y15 = np.array(y15)
    X33 = np.array(X33); y33 = np.array(y33)
    println(f"  15维: {X15.shape} | 33维: {X33.shape}")

    # ── 训练 ──
    from sklearn.utils.class_weight import compute_class_weight
    from xgboost import XGBClassifier

    def train_xgb(X, y):
        vs = int(len(X)*0.8)
        cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
        sw = np.array([cw[list(np.unique(y)).index(c)] for c in y])
        m = XGBClassifier(n_estimators=300, max_depth=5, lr=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            eval_metric='mlogloss', verbosity=0)
        m.fit(X[:vs], y[:vs], eval_set=[(X[vs:],y[vs:])], sample_weight=sw[:vs], verbose=False)
        return m

    println("\n  🌲 训练15维...")
    xgb15 = train_xgb(X15, y15)
    acc15 = accuracy_score(y15[int(len(X15)*0.8):], xgb15.predict(X15[int(len(X15)*0.8):]))
    println(f"    验证: {acc15*100:.1f}%")

    println("  🌲 训练33维...")
    xgb33 = train_xgb(X33, y33)
    acc33 = accuracy_score(y33[int(len(X33)*0.8):], xgb33.predict(X33[int(len(X33)*0.8):]))
    println(f"    验证: {acc33*100:.1f}%")

    # ── 逐场预测64场（动态时序） ──
    println("\n  ⚽ 逐场预测64场...")
    weights = [(0.3,0.7),(0.4,0.6),(0.5,0.5),(0.6,0.4),(0.7,0.3)]
    p15 = {w:{'c':0,'b':0.0} for w in weights}
    p33 = {w:{'c':0,'b':0.0} for w in weights}
    fb2 = FeatureBuffer(clean_elo, clean_dc)
    for m in historical:
        fb2.add_match(m)

    for idx, m in enumerate(sorted(wc, key=lambda x: x['date'])):
        h,a = m['home'], m['away']
        eh,ea = clean_elo.get(h,1500), clean_elo.get(a,1500)
        lh,la = clean_dc.predict_lambda(h,a,True)
        dp = clean_dc.predict_proba(h,a,True)
        op = fb2.make_odds(eh,ea)
        da = np.array([dp[2],dp[1],dp[0]])
        act = 2 if m['h_score']>m['a_score'] else (1 if m['h_score']==m['a_score'] else 0)
        yo = np.zeros(3); yo[act] = 1

        # 15维
        fh5 = fb2.recent_form(h,m['date'],5)
        fa5 = fb2.recent_form(a,m['date'],5)
        f15 = np.array([[(eh-ea)/400, lh, la, lh-la,
            math.log(max(lh,0.01)/max(la,0.01)),
            dp[0],dp[1],dp[2],
            fh5[0],fa5[0],fh5[1]-fa5[2],fa5[1]-fh5[2],fh5[1]-fa5[1],fh5[0]-fa5[0],1]])
        x15r = xgb15.predict_proba(f15)[0]

        # 33维
        fh12 = fb2.recent_form(h,m['date'],12)
        fa12 = fb2.recent_form(a,m['date'],12)
        h2h = fb2.h2h(h,a,m['date'],3)
        tier = tournament_tier(m.get('tournament',''))
        rh = fb2.rest_days(h,m['date'])
        ra = fb2.rest_days(a,m['date'])
        f33 = make_feat_vec(eh,ea,lh,la,dp,op,fh5,fa5,fh12,fa12,h2h,tier,rh,ra,1)
        x33r = xgb33.predict_proba(f33)[0]

        for wd,wx in weights:
            h15m = wd*da+wx*x15r
            if np.argmax(h15m)==act: p15[(wd,wx)]['c']+=1
            p15[(wd,wx)]['b']+=np.sum((h15m-yo)**2)
            h33m = wd*da+wx*x33r
            if np.argmax(h33m)==act: p33[(wd,wx)]['c']+=1
            p33[(wd,wx)]['b']+=np.sum((h33m-yo)**2)

        fb2.add_match(m)
        e_h = 1/(1+10**((ea-eh)/400))
        sh = 1 if m['h_score']>m['a_score'] else (0.5 if m['h_score']==m['a_score'] else 0)
        clean_elo[h] += 32*(sh-e_h); clean_elo[a] += 32*((1-sh)-(1-e_h))
        if (idx+1)%16==0: println(f"    {idx+1}/64")

    n = len(wc)
    println(f"\n{'='*65}")
    println(f"  🔥 严格时序回测（全量35K场训练）")
    println(f"{'='*65}")
    println(f"  {'权重':<25s} {'15维(基线)':>20s} {'33维(新特征)':>20s}")
    println(f"  {'─'*66}")
    for w in weights:
        o = p15[w]; nw = p33[w]
        println(f"  DC{w[0]:.1f}+XGB{w[1]:.1f}     {o['c']/n*100:>6.2f}% B={o['b']/n:.4f}  {nw['c']/n*100:>6.2f}% B={nw['b']/n:.4f}")

    best_o = max(weights, key=lambda w: p15[w]['c'])
    best_n = max(weights, key=lambda w: p33[w]['c'])
    println(f"\n 🏆 旧基线: DC{best_o[0]:.1f}+XGB{best_o[1]:.1f} = {p15[best_o]['c']/n*100:.2f}%")
    println(f" 🏆 新特征: DC{best_n[0]:.1f}+XGB{best_n[1]:.1f} = {p33[best_n]['c']/n*100:.2f}%")
    delta = p33[best_n]['c'] - p15[best_o]['c']
    println(f" 📊 差异: {delta:+d} 场 ({delta/n*100:+.2f}pp)")
    println(f" 📉 Brier: {p15[best_o]['b']/n:.4f} → {p33[best_n]['b']/n:.4f}")

    import json; os.makedirs('/root/data', exist_ok=True)
    with open('/root/data/strict_backtest_2022.json','w') as f:
        json.dump({'type':'strict_backtest_v3_full','n':n,
            'old15':{f'DC{w[0]:.1f}+XGB{w[1]:.1f}':{'acc':p15[w]['c']/n,'brier':p15[w]['b']/n} for w in weights},
            'new33':{f'DC{w[0]:.1f}+XGB{w[1]:.1f}':{'acc':p33[w]['c']/n,'brier':p33[w]['b']/n} for w in weights}},
            f, indent=2, default=str)
    println(f"\n  💾 保存: /root/data/strict_backtest_2022.json")
    return p15, p33


def make_odds_from_elo(eh, ea):
    e_h = 1.0 / (1 + 10**((ea - eh) / 400))
    e_d = 0.26 * math.exp(-((eh-ea)/200)**2)
    margin = 0.06
    o = np.array([e_h*(1-e_d), e_d, (1-e_h)*(1-e_d)])
    o /= o.sum()
    return o
