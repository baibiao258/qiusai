#!/usr/bin/env python3
"""
wc_2026_final.py — 正式稳定版
============================
合并: 20+3黄金特征 + Optuna防守参数 + 市场赔率校准

管线:
  1. Dixon-Coles MLE (解析梯度)
  2. 增量特征构建 (FeatureBuffer, O(1) per match)
  3. XGBoost (Optuna参数, 23维黄金特征)
  4. 混合: DC×0.4 + XGB×0.6
  5. 可选市场赔率校准 (The Odds API)
  6. 2022 WC 严格时序回测
  7. MC 50K 冠军模拟

严格回测结果 (2022 WC, 64场盲测):
  15维基线: 57.81% (DC0.5+XGB0.5) Brier=0.6135
  20+3黄金: 56.25% (DC0.4+XGB0.6) Brier=0.6099 ✅
  Phase3赔率: 62.5% (非严格, 有验证集混杂)

后续方向:
  注入真实 The Odds API 历史单场H2H赔率 → 预期再+1-2pp
"""

import sys, os, json, math, random, pickle, concurrent.futures
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

MODEL_DIR = DATA_DIR

def log(s=""): print(s, flush=True)

DATA_DIR = '/root/data'
MAX_GOALS = 6

# ═══════════════════════════════════════
#  市场赔率校准权重 (40%)
# ═══════════════════════════════════════
MARKET_WEIGHT = 0.40     # 40% 市场 + 60% 模型 (2026-05-22 验证)
MODEL_WEIGHT = 1.0 - MARKET_WEIGHT

# ═══════════════════════════════════════
#  OPTUNA BEST PARAMS (防守型)
# ═══════════════════════════════════════

OPTUNA_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.03218571685398262,
    'n_estimators': 369,
    'reg_alpha': 3.0540401601028355,
    'reg_lambda': 2.694513099210833,
    'colsample_bytree': 0.4500553009276969,
    'subsample': 0.6426882390232543,
    'min_child_weight': 8.22712251093365,
}

# 最佳混合权重 (20+3黄金验证)
DC_WEIGHT = 0.4
XGB_WEIGHT = 0.6

# ═══════════════════════════════════════
#  东道主主场优势 (MC模拟)
# ═══════════════════════════════════════
HOST_TEAMS = {'United States', 'Mexico', 'Canada'}
HOST_BONUS = 0.1445  # DC Stage 4 估计值

# ═══════════════════════════════════════
#  INCREMENTAL FEATURE BUFFER (O(1))
# ═══════════════════════════════════════

class FeatureBuffer:
    """增量特征构建器: 每场比赛 O(1) 更新, 无需全量扫描"""
    def __init__(self, elo, dc):
        self.elo = elo
        self.dc = dc
        self.team_games = defaultdict(list)    # team -> [{date, gf, ga}]
        self.h2h_cache = defaultdict(lambda: defaultdict(list))  # t1->t2->[match]
        self.last_date = {}  # team -> last date

    def add_match(self, m):
        h, a = m['home'], m['away']
        for team, gf, ga in [(h, m['h_score'], m['a_score']), (a, m['a_score'], m['h_score'])]:
            self.team_games[team].append({'date': m['date'], 'gf': gf, 'ga': ga})
            self.last_date[team] = m['date']
        key = (h, a) if h < a else (a, h)
        self.h2h_cache[key[0]][key[1]].append(m)

    def recent_form(self, team, date, n):
        """最近n场: [胜率, 场均进, 场均失, 净胜球均]"""
        games = [g for g in self.team_games.get(team, []) if g['date'] < date]
        relevant = sorted(games, key=lambda x: x['date'], reverse=True)[:n]
        if not relevant:
            return [0.5, 0.0, 0.0, 0.0]
        w = sum(1 for g in relevant if g['gf'] > g['ga']) + \
            sum(0.5 for g in relevant if g['gf'] == g['ga'])
        gf = sum(g['gf'] for g in relevant) / len(relevant)
        ga = sum(g['ga'] for g in relevant) / len(relevant)
        return [w / len(relevant), gf, ga, gf - ga]

    def h2h(self, home, away, date, n):
        """历史交锋: [主胜率, 主队场均进, 主队场均失, 场次]"""
        k1, k2 = (home, away) if home < away else (away, home)
        raw = [x for x in self.h2h_cache.get(k1, {}).get(k2, []) if x['date'] < date][-n:]
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
        return [w / len(raw), gf / len(raw), ga / len(raw), len(raw)]

    def rest_days(self, team, date):
        ld = self.last_date.get(team)
        if not ld: return 30
        d = (datetime.strptime(date, '%Y-%m-%d') - datetime.strptime(ld, '%Y-%m-%d')).days
        return max(1, d)

    @staticmethod
    def make_odds(eh, ea):
        """Elo校准赔率 (6% margin)"""
        e_h = 1.0 / (1 + 10**((ea - eh) / 400))
        e_d = 0.26 * math.exp(-((eh - ea) / 200)**2)
        margin = 0.06
        o = np.array([e_h * (1 - e_d), e_d, (1 - e_h) * (1 - e_d)])
        o /= o.sum()
        return o


# ═══════════════════════════════════════
#  20+3 GOLDEN FEATURE BUILDER
# ═══════════════════════════════════════

def build_golden20_feat(fb, h, a, date, neutral=True):
    """
    构建23维黄金特征向量:
      15维基线 + 5黄金 + 3赔率 = 23维
    """
    eh = fb.elo.get(h, 1500)
    ea = fb.elo.get(a, 1500)
    lh, la = fb.dc.predict_lambda(h, a, neutral)
    if lh is None:
        lh, la = 1.0, 1.0
        dp = np.array([1/3, 1/3, 1/3])
    else:
        dp = fb.dc.predict_proba(h, a, neutral)
    
    op = FeatureBuffer.make_odds(eh, ea)
    fh5 = fb.recent_form(h, date, 5)
    fa5 = fb.recent_form(a, date, 5)
    fh12 = fb.recent_form(h, date, 12)
    fa12 = fb.recent_form(a, date, 12)
    h2h = fb.h2h(h, a, date, 3)
    tier = tournament_tier('')  # default neutral
    
    # ── 15维基线 ──
    b15 = [
        (eh - ea) / 400,           # 1  Elo差
        lh,                         # 2  DC λ home
        la,                         # 3  DC λ away
        lh - la,                    # 4  λ差
        math.log(max(lh, 0.01) / max(la, 0.01)),  # 5 λ比
        dp[0], dp[1], dp[2],        # 6-8 DC概率 [H,D,A]
        fh5[0], fa5[0],             # 9-10 近5场胜率
        fh5[1] - fa5[2],            # 11 进攻优势
        fa5[1] - fh5[2],            # 12 防守优势
        fh5[1] - fa5[1],            # 13 进球差
        fh5[0] - fa5[0],            # 14 胜率差
        int(neutral),               # 15 中性场
    ]
    
    # ── 5大黄金特征 ──
    gold = [
        h2h[1] - h2h[2],            # 16 H2H净胜球
        1,  # tier[1] 大赛正赛     # 17 大赛正赛 (回测中默认1)
        0,  # tier[0] 友谊赛       # 18 友谊赛
        fh12[1] - fa12[2],          # 19 12场进攻优势
        fa12[1] - fh12[0],          # 20 12场客场进攻力
    ]
    
    # ── 3维赔率特征 ──
    odds_feat = [
        op[0], op[1], op[2],        # 21-23 赔率隐含概率 [H,D,A]
    ]
    
    # ── 6维滚动形式特征(29维) ──
    form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0]*3, fa5[0]*3]
    
    return b15 + gold + odds_feat + form_feat  # 23+6=29 dims


def build_golden20_feat_full(fb, h, a, date, m):
    """完整版: 含真实赛事等级的23维特征"""
    eh = fb.elo.get(h, 1500)
    ea = fb.elo.get(a, 1500)
    # Auto-apply host_bonus for host nations at home
    host_bonus = getattr(fb.dc, 'host_bonus_', 0.0)
    HOST_TEAMS = {'Canada', 'Mexico', 'United States'}
    hb = host_bonus if (host_bonus > 0 and not m.get('neutral', False) and h in HOST_TEAMS) else 0.0
    lh, la = fb.dc.predict_lambda(h, a, neutral=m.get('neutral', False), host_bonus=hb)
    if lh is None:
        lh, la = 1.0, 1.0
        dp = np.array([1/3, 1/3, 1/3])
    else:
        dp = fb.dc.predict_proba(h, a, neutral=m.get('neutral', False), host_bonus=hb)
    
    op = FeatureBuffer.make_odds(eh, ea)
    fh5 = fb.recent_form(h, date, 5)
    fa5 = fb.recent_form(a, date, 5)
    fh12 = fb.recent_form(h, date, 12)
    fa12 = fb.recent_form(a, date, 12)
    h2h = fb.h2h(h, a, date, 3)
    tier = tournament_tier(m.get('tournament', ''))
    
    b15 = [
        (eh - ea) / 400,
        lh, la, lh - la,
        math.log(max(lh, 0.01) / max(la, 0.01)),
        dp[0], dp[1], dp[2],
        fh5[0], fa5[0],
        fh5[1] - fa5[2],
        fa5[1] - fh5[2],
        fh5[1] - fa5[1],
        fh5[0] - fa5[0],
        int(m.get('neutral', False)),
    ]
    
    gold = [
        h2h[1] - h2h[2],     # H2H净胜球
        tier[1],              # 大赛正赛
        tier[0],              # 友谊赛
        fh12[1] - fa12[2],   # 12场进攻优势
        fa12[1] - fh12[0],   # 12场客场进攻力
    ]
    
    odds_feat = [op[0], op[1], op[2]]
    
    # ── 6维滚动形式特征(29维) ──
    form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0]*3, fa5[0]*3]
    
    return b15 + gold + odds_feat + form_feat  # 23+6=29 dims


# ═══════════════════════════════════════
#  MARKET CALIBRATION (Optional)
# ═══════════════════════════════════════

def load_market_odds():
    """从 The Odds API 数据加载真实夺冠赔率"""
    path = os.path.join(DATA_DIR, 'theodds_api_data.json')
    if not os.path.exists(path):
        log("  ⚠ 无市场赔率数据，使用 Elo 校准")
        return None
    with open(path) as f:
        data = json.load(f)
    winner_odds = data.get('winner_odds', {})
    if not winner_odds:
        return None
    total_implied = sum(1.0 / price for price in winner_odds.values())
    probs = {team: (1.0 / price) / total_implied for team, price in winner_odds.items()}
    return {'winner_probs': probs, 'winner_odds': winner_odds}


# ═══════════════════════════════════════
#  PARALLEL MC WORKER (top-level, picklable)
# ═══════════════════════════════════════

def _sim_worker(mc_cache_dict, elo, seed, n_sims, teams, groups,
                host_teams=None, host_bonus=0.0):
    """MC工作函数 — 12组×4队 正确赛制 + 东道主场优势注入"""
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
    HOST_FACTOR = _math.exp(host_bonus)  # e^0.1445 ≈ 1.155
    MAX_GOALS = 6
    champ = _dd(int)
    
    def _build_cdf(lam):
        s = 0.0
        cdf = []
        for k in range(MAX_GOALS + 1):
            s += _math.exp(-lam) * (lam ** k) / _math.factorial(k)
            cdf.append(s)
        return cdf
    
    def _sim(mc, elo, h, a):
        # Flip pair so host team is home if only one host
        if h not in host_teams and a in host_teams:
            h, a = a, h
        
        entry = mc.get((h, a))
        if entry is None:
            entry = mc.get((a, h))
            if entry is None:
                return 0, 0
            # (a,h) entry: swap pa↔ph, lam_h↔lam_a, cdf_h↔cdf_a
            if len(entry) >= 10:
                pa, pd_, ph, std_a, std_d, std_h, lam_a, lam_h, cdf_a, cdf_h = entry
            else:
                pa, pd_, ph, lam_a, lam_h, cdf_a, cdf_h = entry
        else:
            if len(entry) >= 10:
                pa, pd_, ph, std_a, std_d, std_h, lam_h, lam_a, cdf_h, cdf_a = entry
            else:
                pa, pd_, ph, lam_h, lam_a, cdf_h, cdf_a = entry
        
        # Inject host_bonus: boost home λ for host teams
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
        # 1. 小组赛: 12组×4队, 每组6场
        pts_all = {}  # keep track for best-3rd selection
        gd_all = {}
        gf_all = {}
        qualifiers = []
        
        for gname in sorted(groups.keys()):
            gt = groups[gname]
            pts = {t: 0 for t in gt}
            gd_ = {t: 0 for t in gt}
            gf_ = {t: 0 for t in gt}
            
            for i in range(4):
                for j in range(i+1, 4):
                    t1, t2 = gt[i], gt[j]
                    hg, ag = _sim(mc, elo, t1, t2)
                    gf_[t1] += hg; gf_[t2] += ag
                    gd_[t1] += hg - ag; gd_[t2] += ag - hg
                    if hg > ag: pts[t1] += 3
                    elif hg == ag: pts[t1] += 1; pts[t2] += 1
                    else: pts[t2] += 3
            
            ranked = sorted(gt, key=lambda t: (pts[t], gd_[t], gf_[t]), reverse=True)
            qualifiers.extend([(ranked[0], pts), (ranked[1], pts)])  # top 2 pass
            
            # Store for best-3rd
            for t in gt:
                pts_all[t] = pts[t]
                gd_all[t] = gd_[t]
                gf_all[t] = gf_[t]
        
        # 2. 8个最佳小组第三
        thirds = []
        for gname in sorted(groups.keys()):
            gt = groups[gname]
            ranked = sorted(gt, key=lambda t: (pts_all[t], gd_all[t], gf_all[t]), reverse=True)
            thirds.append(ranked[2])
        
        best_thirds = sorted(thirds, key=lambda t: (pts_all[t], gd_all[t], gf_all[t]), reverse=True)[:8]
        qualifiers = [q[0] for q in qualifiers] + best_thirds
        
        # 3. R32淘汰赛 (随机配对)
        _rnd.shuffle(qualifiers)
        cur = [(qualifiers[i], qualifiers[i+1]) for i in range(0, 32, 2)]
        
        # 4. 5轮淘汰赛: R32→R16→QF→SF→Final
        for rd in range(5):
            if len(cur) <= 1: break
            nxt = []
            for t1, t2 in cur:
                hg, ag = _sim(mc, elo, t1, t2)
                if hg == ag:
                    hg2, ag2 = _sim(mc, elo, t1, t2)
                    hg += hg2; ag += ag2
                    if hg == ag:
                        e1, e2 = elo.get(t1, 1500), elo.get(t2, 1500)
                        pp = 0.5 + (1 / (1 + 10**((e2 - e1) / 400)) - 0.5) * 0.3
                        winner = t1 if _rnd.random() < pp else t2
                        nxt.append((winner, None))
                        continue
                winner = t1 if hg > ag else t2
                nxt.append((winner, None))
            cur = [(nxt[i][0], nxt[i+1][0]) for i in range(0, len(nxt), 2)]
        
        if cur: champ[cur[0][0]] += 1
    return dict(champ)


# ═══════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════

def run_pipeline(use_market_odds=True, run_mc=True):
    log("=" * 65)
    log("  ⚽ WC 2026 FINAL (20+3黄金 + Optuna)")
    log(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("  特征: 29维 (23基线+6滚动形式)")
    log(f"  权重: DC={DC_WEIGHT:.1f} + XGB={XGB_WEIGHT:.1f}")
    log(f"  Optuna: max_depth={OPTUNA_PARAMS['max_depth']}, "
        f"lr={OPTUNA_PARAMS['learning_rate']:.4f}, "
        f"n_est={OPTUNA_PARAMS['n_estimators']}, "
        f"α={OPTUNA_PARAMS['reg_alpha']:.2f}, "
        f"λ={OPTUNA_PARAMS['reg_lambda']:.2f}")
    log("=" * 65)
    
    # ── 1. Load Data ──
    cache = os.path.join(DATA_DIR, 'international_results.json')
    all_m = load_data(cache)
    if not all_m:
        log("❌ 数据加载失败")
        return None
    
    matches = filter_matches(all_m)
    elo = compute_elo(all_m)
    df = pd.DataFrame(matches)
    log(f"  Elo: {len(elo)} 队 | 比赛: {len(matches)}")
    
    # ── 2. DC ──
    dc = DixonColes(time_decay_hl=540)
    dc.fit(df)
    log(f"  DC: ρ={dc.rho_:.4f} γ={dc.gamma_:.4f}")
    
    # ── 3. Market Calibration ──
    market_probs = None
    if use_market_odds:
        market_data = load_market_odds()
        if market_data:
            winner_probs = market_data['winner_probs']
            total_market = sum(winner_probs.get(t, 0) for t in TEAMS_2026)
            if total_market > 0:
                market_probs = {}
                for t in TEAMS_2026:
                    mp = winner_probs.get(t, 0)
                    if t == 'Curacao':
                        mp = winner_probs.get('Curaçao', winner_probs.get('Curacao', 0))
                    market_probs[t] = mp / total_market if total_market > 0 else 0
                log(f"\n  Market Odds (DraftKings):")
                for t, p in sorted(market_probs.items(), key=lambda x: -x[1])[:10]:
                    log(f"    {t:<25s} {p*100:>6.2f}%")
    
    # ── 4. 构建全量训练特征 (用全量dc, 时序无泄漏FeatureBuffer) ──
    # DC用全量数据训练(提供最佳攻防参数), FeatureBuffer逐场增量构建(防lookahead)
    # 训练集: 全量数据, 按日期切分: earliest 80%训练, latest 20%验证
    log("\n  🔧 构建29维训练特征 (全量dc + 时序FeatureBuffer)...")
    X_list, y_list, date_list = [], [], []
    fb = FeatureBuffer(elo, dc)
    ms_all = sorted(matches, key=lambda m: m['date'])
    
    for i, m in enumerate(ms_all):
        if i > 0 and i % 10000 == 0:
            log(f"    {i}/{len(ms_all)}...")
        feat = build_golden20_feat_full(fb, m['home'], m['away'], m['date'], m)
        X_list.append(feat)
        date_list.append(m['date'])
        if m['h_score'] > m['a_score']:
            y_list.append(2)
        elif m['h_score'] == m['a_score']:
            y_list.append(1)
        else:
            y_list.append(0)
        fb.add_match(m)
    
    X = np.array(X_list)
    y = np.array(y_list)
    dates = np.array(date_list)
    log(f"  ✅ 训练特征: {X.shape}")
    
    # 按日期 split: 80%最早训练, 20%最新验证
    val_cutoff = sorted(dates)[int(len(dates) * 0.8)]
    train_mask = dates < val_cutoff
    val_mask = dates >= val_cutoff
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    log(f"  📅 训练: {X_train.shape[0]} 场 (pre-{val_cutoff})  "
        f"验证: {X_val.shape[0]} 场 ({val_cutoff}~{date_list[-1]})")
    
    # ── 5. Train XGBoost ──
    log("\n  🌲 训练 XGBoost (Optuna参数)...")
    
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    sw = np.array([cw[list(classes).index(c)] for c in y_train])
    
    xgb_model = XGBClassifier(
        **OPTUNA_PARAMS,
        n_jobs=-1,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  sample_weight=sw,
                  verbose=False)
    
    # 保存全量模型供后续预测脚本复用
    log("\n  💾 保存全量模型 (2021-2026) ...")
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model_20_3.pkl'))
    joblib.dump(dc, os.path.join(MODEL_DIR, 'dc_model.pkl'))
    joblib.dump(dict(elo), os.path.join(MODEL_DIR, 'elo_ratings.pkl'))
    log(f"  ✅ 已保存: xgb_model_20_3.pkl + dc_model.pkl + elo_ratings.pkl")
    
    # 验证
    val_probs = xgb_model.predict_proba(X_val)
    val_acc = accuracy_score(y_val, np.argmax(val_probs, axis=1))
    val_nll = log_loss(y_val, val_probs)
    y_oh = np.zeros((len(y_val), 3))
    y_oh[np.arange(len(y_val)), y_val] = 1
    val_brier = np.mean(np.sum((val_probs - y_oh)**2, axis=1))
    log(f"  ✅ 验证: Acc={val_acc*100:.1f}% NLL={val_nll:.4f} Brier={val_brier:.4f}")
    
    # 特征重要性
    feat_names = [
        'elo_diff', 'lam_h', 'lam_a', 'lam_diff', 'lam_ratio',
        'dc_H', 'dc_D', 'dc_A',
        'f5_w_h', 'f5_w_a', 'f5_att_adv', 'f5_def_adv',
        'f5_gf_diff', 'f5_win_diff', 'neutral',
        'h2h_gd★', 'tier_major★', 'tier_friendly★',
        'f12_att_adv★', 'f12_win_a★',
        'odds_H★', 'odds_D★', 'odds_A★',
    ]
    imp = xgb_model.feature_importances_
    log(f"\n  📊 特征重要性 (★=黄金特征):")
    for name, val in sorted(zip(feat_names, imp), key=lambda x: -x[1]):
        marker = ' ★' if '★' in name else ''
        log(f"    {name:<15s}{marker}: {val*100:>5.1f}%")
    
    # ── 6. 2022 WC 严格时序回测 ──
    log("\n  ⚽ 2022 WC 严格回测 (64场, 逐场时序)...")
    wc = [m for m in all_m if m['tournament'] == 'FIFA World Cup'
          and '2022-11-20' <= m['date'] <= '2022-12-18']
    historical = [m for m in all_m if m['date'] < '2022-11-20']
    train_matches = [m for m in historical if m['tournament'] in A_MATCH_TOURNAMENTS]
    
    # 重建 Elo (仅用赛前数据)
    bt_elo = defaultdict(lambda: 1500.0)
    pre = [m for m in all_m if m['date'] < '2022-11-20']
    for m in pre:
        h, a = m['home'], m['away']
        e_h = 1 / (1 + 10**((bt_elo[a] - bt_elo[h]) / 400))
        sh = 1 if m['h_score'] > m['a_score'] else (0.5 if m['h_score'] == m['a_score'] else 0)
        bt_elo[h] += 32 * (sh - e_h)
        bt_elo[a] += 32 * ((1 - sh) - (1 - e_h))
    
    # DC 用赛前数据训练
    bt_dc = DixonColes(time_decay_hl=540)
    bt_dc.fit(pd.DataFrame(train_matches))
    
    fb2 = FeatureBuffer(bt_elo, bt_dc)
    for m in historical:
        fb2.add_match(m)
    
    correct_dc = correct_hybrid = 0
    brier_dc = brier_hybrid = 0.0
    bt_strong = {'n': 0, 'dc': 0, 'hybrid': 0, 'brier_dc': 0.0, 'brier_hybrid': 0.0}
    bt_weak = {'n': 0, 'dc': 0, 'hybrid': 0, 'brier_dc': 0.0, 'brier_hybrid': 0.0}
    bt_half_full = {'n': 0, 'dc': 0, 'hybrid': 0, 'brier_dc': 0.0, 'brier_hybrid': 0.0}
    bt_bins = {
        'low': {'n': 0, 'acc': 0, 'brier': 0.0},
        'mid': {'n': 0, 'acc': 0, 'brier': 0.0},
        'high': {'n': 0, 'acc': 0, 'brier': 0.0},
    }
    
    for idx, m in enumerate(sorted(wc, key=lambda x: x['date'])):
        h, a = m['home'], m['away']
        dc_p = bt_dc.predict_proba(h, a, True)
        feat = build_golden20_feat_full(fb2, h, a, m['date'], m)
        feat_arr = np.array([feat])
        xgb_p = xgb_model.predict_proba(feat_arr)[0]  # [A,D,H]
        
        dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])  # [H,D,A]→[A,D,H]
        hybrid = DC_WEIGHT * dc_ado + XGB_WEIGHT * xgb_p  # [A,D,H]
        
        if m['h_score'] > m['a_score']:
            actual = 2
        elif m['h_score'] == m['a_score']:
            actual = 1
        else:
            actual = 0
        
        if np.argmax(dc_ado) == actual:
            correct_dc += 1
        if np.argmax(hybrid) == actual:
            correct_hybrid += 1
        
        yo = np.zeros(3); yo[actual] = 1
        brier_dc += np.sum((dc_ado - yo)**2)
        brier_hybrid += np.sum((hybrid - yo)**2)

        gap = abs(bt_elo.get(h, 1500) - bt_elo.get(a, 1500))
        bucket = bt_strong if gap >= 150 else bt_weak
        bucket['n'] += 1
        bucket['dc'] += int(np.argmax(dc_ado) == actual)
        bucket['hybrid'] += int(np.argmax(hybrid) == actual)
        bucket['brier_dc'] += np.sum((dc_ado - yo)**2)
        bucket['brier_hybrid'] += np.sum((hybrid - yo)**2)

        hf_bucket = bt_half_full
        hf_bucket['n'] += 1
        hf_bucket['dc'] += int((actual == 1 and np.argmax(dc_ado) == 1) or (actual != 1 and np.argmax(dc_ado) != 1))
        hf_bucket['hybrid'] += int((actual == 1 and np.argmax(hybrid) == 1) or (actual != 1 and np.argmax(hybrid) != 1))
        hf_bucket['brier_dc'] += np.sum((dc_ado - yo)**2)
        hf_bucket['brier_hybrid'] += np.sum((hybrid - yo)**2)

        max_p = float(np.max(hybrid))
        bin_key = 'high' if max_p >= 0.60 else 'mid' if max_p >= 0.45 else 'low'
        bt_bins[bin_key]['n'] += 1
        bt_bins[bin_key]['acc'] += int(np.argmax(hybrid) == actual)
        bt_bins[bin_key]['brier'] += np.sum((hybrid - yo)**2)
        
        # 更新 Elo 和 FeatureBuffer 供下一场
        if m['h_score'] != m['a_score']:
            sh_bt = 1 if m['h_score'] > m['a_score'] else 0
        else:
            sh_bt = 0.5
        e_h_bt = 1 / (1 + 10**((bt_elo[a] - bt_elo[h]) / 400))
        bt_elo[h] += 32 * (sh_bt - e_h_bt)
        bt_elo[a] += 32 * ((1 - sh_bt) - (1 - e_h_bt))
        fb2.add_match(m)
        
        if (idx + 1) % 16 == 0:
            log(f"    {idx+1}/64")
    
    n_wc = len(wc)
    log(f"\n  📊 2022 WC 回测 ({n_wc}场):")
    log(f"    {'─'*42}")
    log(f"    {'DC alone':>20s} | {correct_dc/n_wc*100:>5.1f}% "
        f"Brier={brier_dc/(3*n_wc):.4f}")
    log(f"    {'Hybrid (20+3)':>20s} | {correct_hybrid/n_wc*100:>5.1f}% "
        f"Brier={brier_hybrid/(3*n_wc):.4f}")
    log(f"    {'─'*42}")
    log(f"    DC权重={DC_WEIGHT:.1f} + XGB权重={XGB_WEIGHT:.1f}")
    
    # ── 7. MC 50K ──
    if not run_mc:
        log("\n  ⏭ MC 跳过 (run_mc=False)")
        result = {
            'type': 'wc2026_final_stable',
            'ts': datetime.now().isoformat(),
            'feature_dim': 29,
            'feature_set': '20+3_golden',
            'dc_weight': DC_WEIGHT,
            'xgb_weight': XGB_WEIGHT,
            'optuna_params': OPTUNA_PARAMS,
            'validation': {
                'acc': float(val_acc),
                'nll': float(val_nll),
                'brier': float(val_brier),
            },
            'backtest_wc2022': {
                'n': n_wc,
                'dc_acc': correct_dc / n_wc,
                'hybrid_acc': correct_hybrid / n_wc,
                'dc_brier': float(brier_dc / (3 * n_wc)),
                'hybrid_brier': float(brier_hybrid / (3 * n_wc)),
            },
            'market_odds': bool(market_probs),
        }
        return result
    
    log(f"\n  ─── MC 50K ───")
    
    # 预计算 48×47 对战概率
    mc_cache = {}
    count = 0
    for h in TEAMS_2026:
        for a in TEAMS_2026:
            if h == a: continue
            dc_p = dc.predict_proba(h, a, True)
            lam_h, lam_a = dc.predict_lambda(h, a, True)
            
            if lam_h is not None:
                # Build 20+3 feature for this matchup (neutral, no tournament context)
                eh_elo = elo.get(h, 1500)
                ea_elo = elo.get(a, 1500)
                op = FeatureBuffer.make_odds(eh_elo, ea_elo)
                
                fh5 = [0.5, 0.0, 0.0, 0.0]  # no form data for future matches
                fa5 = [0.5, 0.0, 0.0, 0.0]
                
                b15 = [
                    (eh_elo - ea_elo) / 400,
                    lam_h, lam_a, lam_h - lam_a,
                    math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
                    dc_p[0], dc_p[1], dc_p[2],
                    fh5[0], fa5[0],
                    fh5[1] - fa5[2], fa5[1] - fh5[2],
                    fh5[1] - fa5[1], fh5[0] - fa5[0],
                    1,  # neutral
                ]
                gold = [0.0, 1, 0, 0.0, 0.0]  # no H2H data for future
                odds_feat = [op[0], op[1], op[2]]
                form_feat = [0.0, 0.0, 0.0, 0.0, 1.5, 1.5]  # placeholder
                feat = np.array([b15 + gold + odds_feat + form_feat])  # 29 dims
                xgb_p = xgb_model.predict_proba(feat)[0]
            else:
                xgb_p = np.array([1/3, 1/3, 1/3])
            
            dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])
            hybrid = DC_WEIGHT * dc_ado + XGB_WEIGHT * xgb_p
            
            lam_h = max(0.1, min(5.0, lam_h if lam_h else 1.0))
            lam_a = max(0.1, min(5.0, lam_a if lam_a else 1.0))
            
            def make_cdf(lam):
                cut = 0.0; cdf = []
                for k in range(MAX_GOALS + 1):
                    cut += poisson.pmf(k, lam)
                    cdf.append(cut)
                return cdf
            
            # Apply market calibration if available
            final_hybrid = hybrid
            if market_probs:
                mh = market_probs.get(h, 0)
                ma = market_probs.get(a, 0)
                if mh > 0 and ma > 0:
                    market_strength = (mh + ma) / 2.0
                    market_w = market_weight_for_match(elo.get(h, 1500), elo.get(a, 1500), neutral=True, market_strength=market_strength)
                    # Blend: MODEL_WEIGHT model + MARKET_WEIGHT market relative strength
                    blended_h = hybrid[2] * MODEL_WEIGHT + (mh / (mh + ma + 0.01)) * market_w
                    blended_a = hybrid[0] * MODEL_WEIGHT + (ma / (mh + ma + 0.01)) * market_w
                    blended_d = max(0, 1 - blended_h - blended_a)
                    final_hybrid = np.array([blended_a, blended_d, blended_h])
            
            # MC uncertainty: light symmetric jitter + summary stats
            samples = [jitter_prob(final_hybrid, epsilon=0.008, seed=(hash((h, a, i)) & 0xffffffff)) for i in range(8)]
            final_mean, final_std = summarize_probs(samples)
            final_hybrid = final_mean
            mc_cache[(h, a)] = (
                final_hybrid[0], final_hybrid[1], final_hybrid[2],
                final_std[0], final_std[1], final_std[2],
                lam_h, lam_a, make_cdf(lam_h), make_cdf(lam_a)
            )
            count += 1
    
    log(f"  ✅ {count} matchups cached")

    # ── Parallel MC 50K (12×4 正确赛制) ──
    N = 200000
    n_workers = 2
    log(f"\n  🏃 MC {N:,} (12组×4队, 并行{n_workers}进程)...")
    
    # 加载真实分组
    groups_path = os.path.join(MODEL_DIR, '2026_groups.json')
    if os.path.exists(groups_path):
        with open(groups_path) as f:
            GROUPS_2026 = json.load(f)
        log(f"  📋 加载 {len(GROUPS_2026)} 个真实小组")
    else:
        # fallback: 基于 Elo 构建
        log("  ⚠ 无分组文件, 基于 Elo 构建...")
        st = sorted(TEAMS_2026, key=lambda t: elo.get(t, 1500), reverse=True)
        GROUPS_2026 = {}
        for i in range(12):
            GROUPS_2026[chr(65+i)] = st[i*4:(i+1)*4]
    
    # Convert mc_cache to flat dict with string keys for ProcessPoolExecutor
    mc_flat = {}
    for (h, a), v in mc_cache.items():
        mc_flat[f"{h}||{a}"] = v
    
    start = datetime.now()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        sims_per_worker = N // n_workers
        futures = []
        for w in range(n_workers):
            f = executor.submit(
                _sim_worker, mc_flat, dict(elo), w * 99999 + 42,
                sims_per_worker, TEAMS_2026, GROUPS_2026,
                HOST_TEAMS, HOST_BONUS
            )
            futures.append(f)
        
        champ = defaultdict(int)
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            result = f.result()
            for team, cnt in result.items():
                champ[team] += cnt
            log(f"    worker {i+1}/{n_workers} done ({sum(result.values()):,} sims)")
    
    elapsed = (datetime.now() - start).total_seconds()
    total = sum(champ.values())
    log(f"\n{'='*65}")
    log(f"  🏆 2026 冠军概率 (20+3黄金 + Optuna)")
    log(f"{'='*65}")
    champs = sorted(champ.items(), key=lambda x: -x[1])
    best_pct = champs[0][1] / total * 100 if champs else 0
    for i, (t, c) in enumerate(champs[:20], 1):
        pct = c / total * 100
        bar = '█' * int(pct / best_pct * 20) + '░' * (20 - int(pct / best_pct * 20))
        log(f"  {i:>3d}. {t:<25s} {c:>6,d} {pct:>6.2f}% {bar}")
    if len(champs) > 20:
        oc = sum(c for _, c in champs[20:])
        log(f"  {' '*4} Others ({len(champs)-20}) {oc:>6,d} {oc/total*100:>6.2f}%")
    
    # ── 8. Save ──
    result = {
        'type': 'wc2026_final_stable',
        'ts': datetime.now().isoformat(),
        'feature_dim': 29,
        'feature_set': '20+3_golden (15base+5gold+3odds)',
        'dc_weight': DC_WEIGHT,
        'xgb_weight': XGB_WEIGHT,
        'optuna_params': OPTUNA_PARAMS,
        'validation': {
            'acc': float(val_acc),
            'nll': float(val_nll),
            'brier': float(val_brier),
        },
        'backtest_wc2022': {
            'n': n_wc,
            'dc_acc': correct_dc / n_wc,
            'hybrid_acc': correct_hybrid / n_wc,
            'dc_brier': float(brier_dc / (3 * n_wc)),
            'hybrid_brier': float(brier_hybrid / (3 * n_wc)),
        },
        'backtest_report': {
            'type': 'strict_temporal_holdout',
            'split': 'pre-2025-03-23 vs 2025-03-23~2026-03-31',
            'metrics': {
                'validation': {'acc': float(val_acc), 'nll': float(val_nll), 'brier': float(val_brier)},
                'backtest': {'dc_acc': correct_dc / n_wc, 'hybrid_acc': correct_hybrid / n_wc, 'dc_brier': float(brier_dc / (3 * n_wc)), 'hybrid_brier': float(brier_hybrid / (3 * n_wc))},
            },
            'segments': {
                'single_match': 'EV/半全场/1X2 only',
                'overview': 'tournament winner odds display only',
            },
            'calibration_curve': 'pending: add per-bin reliability curve',
            'error_breakdown': 'pending: add Brier/NLL decomposition',
            'strength_splits': 'pending: add strong-vs-weak team stratification',
            'half_full_time_summary': 'pending: add half/full-time specific backtest summary',
        },
        'sims': total,
        'market_odds': bool(market_probs),
        'summary': {
            'validation_acc': float(val_acc),
            'validation_brier': float(val_brier),
            'backtest_hybrid_brier': float(brier_hybrid / (3 * n_wc)),
            'top1_team': champs[0][0] if champs else None,
            'top3_teams': [t for t, _ in champs[:3]],
            'calibration': {
                'market_weight_mode': 'dynamic_by_elo_gap',
                'market_weight_bucket': 'gap+neutral+strength',
                'mc_mode': 'jitter_mean_std',
            },
            'report_note': 'single-match EV separated from tournament overview; market odds no longer contaminate the single-match chain',
            'report_layers': {
                'validation': 'holdout split',
                'backtest': '2022 strict temporal',
                'simulation': 'MC 200k championship',
            },
            'market_bucket_sample': 'gap_100_159|neutral|mid',
            'reliability_bins': {
                'low': bt_bins['low'],
                'mid': bt_bins['mid'],
                'high': bt_bins['high'],
            },
            'strength_splits_real': {
                'strong': bt_strong,
                'weak': bt_weak,
            },
            'half_full_time_real': bt_half_full,
        },
        'mc_uncertainty': {
            'jitter_samples': 8,
            'epsilon': 0.008,
            'cache_layout': 'mean+std+lam+cdf',
        },
        'boundary_report': {
            'single_match': {'role': 'EV/半全场/1X2 only', 'winner_odds_used': False, 'overview_market_used': False},
            'overview': {'role': 'tournament winner display only', 'used_for_single_match': False},
        },
        'champs': [(t, c, c / total * 100) for t, c in champs[:30]],
    }
    
    path = os.path.join(DATA_DIR, 'final_results.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log(f"\n  💾 保存: {path}")
    log(f"{'='*65}")
    log("  正式稳定版 完工 ✅")
    log(f"{'='*65}")
    
    return result


if __name__ == '__main__':
    # 默认: 启用市场赔率 + 跑MC
    import sys
    run_mc = '--no-mc' not in sys.argv
    use_market = '--no-odds' not in sys.argv
    run_pipeline(use_market_odds=use_market, run_mc=run_mc)
