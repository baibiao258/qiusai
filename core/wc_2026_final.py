#!/usr/bin/env python3
"""
wc_2026_final.py — 正式稳定版 (修复版)
============================
单⼀冠军管线：DC + XGBoost + 市场赔率校准 + MC 冠军/亚军模拟

修复清单:
  1. 淘汰赛签表: 按小组排名+Elo配对, 去掉 _rnd.shuffle
  2. 概率字段显式命名 (p_h/p_d/p_a 替换数组索引)
  3. 市场校准用标准去水公式, 删除 +0.01 hack
  4. 东道主加成按国家单独估计
  5. 统一输出冠军概率 + 亚军概率 + EV 表 (废弃旁路脚本)
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

MODEL_DIR = DATA_DIR = '/root/data'
MAX_GOALS = 6

def log(s=""): print(s, flush=True)

# ═══════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════

MARKET_WEIGHT = 0.40     # 40% 市场 + 60% 模型
MODEL_WEIGHT = 1.0 - MARKET_WEIGHT

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

# ── 基于熵的XGB动态权重 (替换静态 DC_WEIGHT/XGB_WEIGHT) ──
def compute_dynamic_xgb_weight(xgb_probs, alpha=0.30, beta=0.50):
    """根据XGB预测的香农熵自动分配XGB/DC权重.
    高置信度(尖锐分布) → XGB权重大; 低置信度(均匀分布) → DC权重大.
    xgb_probs: [3] array [away, draw, home] 来自 XGBoost predict_proba
    Returns: (xgb_w, dc_w, confidence) confidence∈[0,1]
    """
    p = np.asarray(xgb_probs, dtype=float)
    p = np.clip(p, 1e-10, 1.0)
    p = p / p.sum()
    e = -np.sum(p * np.log2(p))           # 香农熵
    max_e = np.log2(3)                     # ≈1.585 (均匀分布的熵)
    confidence = 1.0 - e / max_e           # 1=极自信, 0=完全不确定
    xgb_w = np.clip(alpha + beta * confidence, 0.10, 0.90)
    return xgb_w, 1.0 - xgb_w, confidence

# 东道主
HOST_TEAMS = {'United States', 'Mexico', 'Canada'}
# 国家独立 host_bonus 估计 (Phase5 可调)
HOST_BONUS_BY_TEAM = {
    'United States': 0.1445,  # 68场大样本: 70%得分率
    'Mexico': 0.10,           # 17场小样本: 79%得分率(0负), 保守下调
    'Canada': 0.07,           # 18场极小样本: 75%得分率, 大幅下调
}
# 淘汰赛衰减系数 (小组赛满值 * 该系数)
KO_HOST_DECAY = 0.5  # 晋级淘汰赛后减半
# 灵敏度测试范围
HOST_BONUS_SENSITIVITY = [0.0, 0.07, 0.1445]

# ═══════════════════════════════════════
#  OFFICIAL FIFA BRACKET (openfootball/worldcup)
# ═══════════════════════════════════════

# R32: (match_label, (pos, group), (pos, group))
# pos: '1'=group winner, '2'=runner-up, list=eligible group list for 3rd
R32_SPEC = [
    ('M73',  ('2','A'), ('2','B')),
    ('M74',  ('1','E'), ('3',['A','B','C','D','F'])),
    ('M75',  ('1','F'), ('2','C')),
    ('M76',  ('1','C'), ('2','F')),
    ('M77',  ('1','I'), ('3',['C','D','F','G','H'])),
    ('M78',  ('2','E'), ('2','I')),
    ('M79',  ('1','A'), ('3',['C','E','F','H','I'])),
    ('M80',  ('1','L'), ('3',['E','H','I','J','K'])),
    ('M81',  ('1','D'), ('3',['B','E','F','I','J'])),
    ('M82',  ('1','G'), ('3',['A','E','H','I','J'])),
    ('M83',  ('2','K'), ('2','L')),
    ('M84',  ('1','H'), ('2','J')),
    ('M85',  ('1','B'), ('3',['E','F','G','I','J'])),
    ('M86',  ('1','J'), ('2','H')),
    ('M87',  ('1','K'), ('3',['D','E','I','J','L'])),
    ('M88',  ('2','D'), ('2','G')),
]

# Third-placed team slot constraints
THIRD_SLOTS_SPEC = [
    ('M74', ['A','B','C','D','F']),
    ('M77', ['C','D','F','G','H']),
    ('M79', ['C','E','F','H','I']),
    ('M80', ['E','H','I','J','K']),
    ('M81', ['B','E','F','I','J']),
    ('M82', ['A','E','H','I','J']),
    ('M85', ['E','F','G','I','J']),
    ('M87', ['D','E','I','J','L']),
]

# Bracket tree: (round_name, [(next_match, prev_match_1, prev_match_2), ...])
BRACKET_TREE = [
    ('R16', [('M89','M74','M77'),('M90','M73','M75'),('M91','M76','M78'),
             ('M92','M79','M80'),('M93','M83','M84'),('M94','M81','M82'),
             ('M95','M86','M88'),('M96','M85','M87')]),
    ('QF',  [('M97','M89','M90'),('M98','M93','M94'),('M99','M91','M92'),
             ('M100','M95','M96')]),
    ('SF',  [('M101','M97','M98'),('M102','M99','M100')]),
]

# ═══════════════════════════════════════
#  FeatureBuffer (from mc200k)
# ═══════════════════════════════════════

class FeatureBuffer:
    def __init__(self, elo, dc):
        self.elo = elo
        self.dc = dc
        self.team_games = defaultdict(list)
        self.h2h_cache = defaultdict(lambda: defaultdict(list))
        self.last_date = {}

    def add_match(self, m):
        h, a = m['home'], m['away']
        for team, gf, ga in [(h, m['h_score'], m['a_score']), (a, m['a_score'], m['h_score'])]:
            self.team_games[team].append({'date': m['date'], 'gf': gf, 'ga': ga})
            self.last_date[team] = m['date']
        key = (h, a) if h < a else (a, h)
        self.h2h_cache[key[0]][key[1]].append(m)

    def recent_form(self, team, date, n):
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

    @staticmethod
    def make_odds(eh, ea):
        """Elo校准赔率"""
        e_h = 1.0 / (1 + 10**((ea - eh) / 400))
        e_d = 0.26 * math.exp(-((eh - ea) / 200)**2)
        margin = 0.06
        o = np.array([e_h * (1 - e_d), e_d, (1 - e_h) * (1 - e_d)])
        o /= o.sum()
        return o

def build_golden20_feat_full(fb, h, a, date, m):
    eh = fb.elo.get(h, 1500)
    ea = fb.elo.get(a, 1500)
    host_bonus = getattr(fb.dc, 'host_bonus_', 0.0)
    hb = host_bonus if (host_bonus > 0 and not m.get('neutral', False) and h in HOST_TEAMS) else 0.0
    lh, la = fb.dc.predict_lambda(h, a, neutral=m.get('neutral', False), host_bonus=hb)
    if lh is None:
        lh, la = 1.0, 1.0
        dp = {'home': 1/3, 'draw': 1/3, 'away': 1/3}
    else:
        p = fb.dc.predict_proba(h, a, neutral=m.get('neutral', False), host_bonus=hb)
        dp = {'home': p[0], 'draw': p[1], 'away': p[2]}

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
        dp['home'], dp['draw'], dp['away'],
        fh5[0], fa5[0],
        fh5[1] - fa5[2],
        fa5[1] - fh5[2],
        fh5[1] - fa5[1],
        fh5[0] - fa5[0],
        int(m.get('neutral', False)),
    ]
    gold = [
        h2h[1] - h2h[2],
        tier[1], tier[0],
        fh12[1] - fa12[2],
        fa12[1] - fh12[0],
    ]
    odds_feat = [op[0], op[1], op[2]]
    form_feat = [
        fh5[1], fh5[2],
        fa5[1], fa5[2],
        fh5[0] * 3, fa5[0] * 3,
    ]
    return b15 + gold + odds_feat + form_feat  # 29 dims

# ═══════════════════════════════════════
#  MARKET ODDS (standard devig)
# ═══════════════════════════════════════

def load_market_odds():
    """加载市场赔率 → 去水概率 (标准公式: p_i = (1/o_i)/sum(1/o))"""
    path = os.path.join(DATA_DIR, 'theodds_api_data.json')
    if not os.path.exists(path):
        log("  ⚠ 无市场赔率数据")
        return None
    with open(path) as f:
        data = json.load(f)
    winner_odds = data.get('winner_odds', {})
    if not winner_odds:
        return None
    total_implied = sum(1.0 / price for price in winner_odds.values())
    probs = {team: (1.0 / price) / total_implied for team, price in winner_odds.items()}
    return {'winner_probs': probs, 'winner_odds': winner_odds}


def market_probs_3way_from_outright(rel_h, rel_a, k=1.0, knockout=False, model_draw=None):
    """
    从两队相对夺冠强度估计单场三维市场概率。
    输出: [away_prob, draw_prob, home_prob] (对齐管线[A,D,H]约定)
    
    使用几何平均估计平局概率，修复旧版 market_vec = [rel_a, 0.0, rel_h] 丢弃平局信号。
    
    参数
    ----
    knockout : bool
        若为True, 调整 outright→90min 映射。
        淘汰赛夺冠(晋级)赔率反映的是"最终晋级"而非"90分钟获胜"。
    model_draw : float or None
        模型的90分钟平局概率估计, knockout=True 时必传。
    """
    inv_h = rel_h ** k
    inv_a = rel_a ** k
    inv_d = (rel_h * rel_a) ** (k / 2.0)
    total = inv_h + inv_d + inv_a
    if total <= 0:
        return np.array([1/3, 1/3, 1/3])
    
    # 返回 [A, D, H] 顺序
    p_h = inv_h / total
    p_d = inv_d / total
    p_a = inv_a / total
    
    if knockout:
        # 淘汰赛反解: 夺冠赔率→晋级概率→90分钟概率
        # 关系: P_qualify(H) = P(H_90) + 0.5 * P(D_90)
        # 推导: P(H_90) = P_qualify(H) - 0.5 * P(D_90)
        if model_draw is not None:
            p_qual_h = rel_h / (rel_h + rel_a) if (rel_h + rel_a) > 0 else 0.5
            p_qual_a = 1.0 - p_qual_h
            
            p_d_adj = model_draw  # 用模型的平局估计
            p_h_adj = p_qual_h - 0.5 * p_d_adj
            p_a_adj = p_qual_a - 0.5 * p_d_adj
            
            # 保护: 概率不越界
            if p_h_adj < 0 or p_a_adj < 0:
                max_draw = 2.0 * min(p_qual_h, p_qual_a)
                p_d_adj = max(0.01, min(p_d_adj, max_draw))
                p_h_adj = p_qual_h - 0.5 * p_d_adj
                p_a_adj = p_qual_a - 0.5 * p_d_adj
            
            p_h, p_d, p_a = p_h_adj, p_d_adj, p_a_adj
    
    return np.array([p_a, p_d, p_h])

# ═══════════════════════════════════════
#  MC WORKER — 修复版淘汰赛签表
def _sim_worker(mc_cache_dict, elo, seed, n_sims, teams, groups,
                host_teams=None, host_bonus_by_team=None, ko_decay=0.5,
                bracket_mode='ranked'):
    """
    MC工作函数:
    1. 淘汰赛不再 _rnd.shuffle
    2. 追踪冠军+亚军
    3. 概率字段用 dict 传递
    4. 每队独立 host_bonus (host_bonus_by_team dict)
    5. 淘汰赛衰减 ko_decay (默认 0.5 = 减半)
    6. bracket_mode: 'ranked' (Elo排名配对) 或 'official' (FIFA官方路书)
    """
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
    if host_bonus_by_team is None:
        host_bonus_by_team = {}
    # 预计算每个东道主的满值因子
    _host_factor_cache = {}
    for t in host_teams:
        bon = host_bonus_by_team.get(t, 0.0)
        _host_factor_cache[t] = _math.exp(bon)

    champ = _dd(int)
    runner = _dd(int)
    round_counts = _dd(lambda: _dd(int))  # team -> round -> count

    def _build_cdf(lam):
        s = 0.0; cdf = []
        for k in range(MAX_GOALS + 1):
            s += _math.exp(-lam) * (lam ** k) / _math.factorial(k)
            cdf.append(s)
        return cdf

    def _sim_match(mc, elo, h, a, ko=False):
        """模拟单场 (同原版)"""
        if h not in host_teams and a in host_teams:
            h, a = a, h
        entry = mc.get((h, a))
        if entry is None:
            entry = mc.get((a, h))
            if entry is None:
                return 0, 0
            if len(entry) >= 10:
                pa, pd_, ph, _s1, _s2, _s3, lam_a, lam_h, cdf_a, cdf_h = entry
            else:
                pa, pd_, ph, lam_a, lam_h, cdf_a, cdf_h = entry
        else:
            if len(entry) >= 10:
                pa, pd_, ph, _s1, _s2, _s3, lam_h, lam_a, cdf_h, cdf_a = entry
            else:
                pa, pd_, ph, lam_h, lam_a, cdf_h, cdf_a = entry
        if h in host_teams:
            hf = _host_factor_cache.get(h, 1.0)
            if ko:
                hf = 1.0 + (hf - 1.0) * ko_decay
            lam_h *= hf
            cdf_h = _build_cdf(lam_h)

        # ── 淘汰赛概率反解: outright→90min ──
        # 缓存概率是 model(90min) + market(outright) 的凸组合。
        # 在淘汰赛中 outright 反映的是"最终晋级"而非"90分钟胜",
        # 会高估热门方的90分钟胜率 (平局→加时→点球让弱队有二次机会).
        # 压缩 H/A 分布来修正此偏差.
        if ko:
            _de = 0.65  # de-sharpen factor
            ph = ph * _de + (1 - _de) * (1 - pd_) / 2
            pa = pa * _de + (1 - _de) * (1 - pd_) / 2
            _t = pa + pd_ + ph
            pa, pd_, ph = pa / _t, pd_ / _t, ph / _t

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

    def _play_ko(mc, elo, t1, t2):
        """淘汰赛单场(含加时+点球), 返回胜者"""
        hg, ag = _sim_match(mc, elo, t1, t2, ko=True)
        if hg == ag:
            hg2, ag2 = _sim_match(mc, elo, t1, t2, ko=True)
            hg += hg2; ag += ag2
            if hg == ag:
                e1, e2 = elo.get(t1, 1500), elo.get(t2, 1500)
                pp = 0.5 + (1 / (1 + 10**((e2 - e1) / 400)) - 0.5) * 0.3
                return t1 if _rnd.random() < pp else t2
            else:
                return t1 if hg > ag else t2
        else:
            return t1 if hg > ag else t2

    for _ in range(n_sims):
        pts_all = {}; gd_all = {}; gf_all = {}
        group_standings = {}
        qualifiers = {}

        for gname in sorted(groups.keys()):
            gt = groups[gname]
            pts = {t: 0 for t in gt}
            gd_ = {t: 0 for t in gt}
            gf_ = {t: 0 for t in gt}

            for i in range(4):
                for j in range(i+1, 4):
                    t1, t2 = gt[i], gt[j]
                    hg, ag = _sim_match(mc, elo, t1, t2)
                    gf_[t1] += hg; gf_[t2] += ag
                    gd_[t1] += hg - ag; gd_[t2] += ag - hg
                    if hg > ag: pts[t1] += 3
                    elif hg == ag: pts[t1] += 1; pts[t2] += 1
                    else: pts[t2] += 3

            for t in gt:
                pts_all[t] = pts[t]; gd_all[t] = gd_[t]; gf_all[t] = gf_[t]
                qualifiers[t] = {'pts': pts[t], 'gd': gd_[t], 'gf': gf_[t]}
            group_standings[gname] = {t: {'pts': pts[t], 'gd': gd_[t], 'gf': gf_[t]} for t in gt}

        # ── 淘汰赛签表 ──
        if bracket_mode == 'official':
            # Per-group ranked lists (with Elo tiebreaker)
            group_ranked = {}
            thirds_info = []
            for gname in sorted(groups.keys()):
                gt = groups[gname]
                ranked = sorted(gt, key=lambda t: (pts_all[t], gd_all[t], gf_all[t], elo.get(t, 1500)), reverse=True)
                group_ranked[gname] = ranked
                t3 = ranked[2]
                thirds_info.append((t3, gname, pts_all[t3], gd_all[t3], gf_all[t3]))

            # Top 8 thirds
            thirds_info.sort(key=lambda x: (-x[2], -x[3], -x[4], -elo.get(x[0], 1500)))
            top8 = thirds_info[:8]

            # Team-constrained-first assignment
            slot_by_group = _dd(list)
            for label, eg in THIRD_SLOTS_SPEC:
                for g in eg:
                    slot_by_group[g].append(label)

            teams_info = []
            for t, g, pt, gd, gf in top8:
                eligible = slot_by_group.get(g, [])
                teams_info.append({'team': t, 'group': g, 'pts': pt, 'gd': gd, 'gf': gf,
                                   'eligible': set(eligible), 'n_slots': len(eligible)})
            teams_info.sort(key=lambda x: (x['n_slots'], -x['pts'], -x['gd'], -x['gf']))

            third_map = {}
            used_slots = set()
            for ti in teams_info:
                remaining = ti['eligible'] - used_slots
                if not remaining:
                    remaining = set(s[0] for s in THIRD_SLOTS_SPEC) - used_slots
                if remaining:
                    slot_elo = {s: elo.get(group_ranked.get(s[1:] if len(s)>2 else '', [group_ranked.get('A',[''])[0]])[0] if len(s)>3 else '', 1500) for s in remaining}
                    # Simple approach: pick first available
                    chosen = sorted(remaining)[0]
                    third_map[chosen] = (ti['team'], ti['group'])
                    used_slots.add(chosen)

            # Build R32 from official bracket
            r32_pairs = []
            r32_labels = []
            for label, (hp, hg), (ap, ag) in R32_SPEC:
                home = group_ranked[hg][0] if hp == '1' else group_ranked[hg][1]
                if ap == '1':
                    away = group_ranked[ag][0]
                elif ap == '2':
                    away = group_ranked[ag][1]
                else:
                    away = third_map.get(label, (group_ranked[ag[0]][2],))[0]
                r32_pairs.append((home, away))
                r32_labels.append(label)

            # Play R32 → track winners
            r32_w = {}
            for lab, (h, a) in zip(r32_labels, r32_pairs):
                w = _play_ko(mc, elo, h, a)
                r32_w[lab] = w
                round_counts[w]['R16'] += 1

            # Follow bracket tree
            cur_pairs = None
            round_names = ['QF', 'SF', 'Final']
            for rd_idx, (rd_name, rd_matches) in enumerate(BRACKET_TREE):
                winners = []
                for nlab, m1, m2 in rd_matches:
                    w = _play_ko(mc, elo, r32_w[m1], r32_w[m2])
                    winners.append(w)
                    r32_w[nlab] = w
                    round_counts[w][round_names[rd_idx]] += 1
                cur_pairs = None
                if len(winners) >= 2:
                    cur_pairs = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]

            # Final → also count Champion/Runner
            if cur_pairs and len(cur_pairs) == 1:
                t1, t2 = cur_pairs[0]
                winner = _play_ko(mc, elo, t1, t2)
                loser = t2 if winner == t1 else t1
                round_counts[winner]['Champion'] += 1
                round_counts[loser]['Runner'] += 1
                champ[winner] += 1
                runner[loser] += 1
        else:
            # Elo-ranked bracket (original logic)
            thirds = []
            for gname in sorted(groups.keys()):
                gt = groups[gname]
                ranked = sorted(gt, key=lambda t: (pts_all[t], gd_all[t], gf_all[t]), reverse=True)
                thirds.append(ranked[2])
            best_thirds = sorted(thirds, key=lambda t: (pts_all[t], gd_all[t], gf_all[t]), reverse=True)[:8]

            gw_teams = []
            for gname in sorted(groups.keys()):
                gt = groups[gname]
                ranked = sorted(gt, key=lambda t: (pts_all[t], gd_all[t], gf_all[t]), reverse=True)
                gw_teams.append(ranked[0])
            gr_teams = []
            for gname in sorted(groups.keys()):
                gt = groups[gname]
                ranked = sorted(gt, key=lambda t: (pts_all[t], gd_all[t], gf_all[t]), reverse=True)
                gr_teams.append(ranked[1])

            all_teams = list(gw_teams) + list(gr_teams) + list(best_thirds)
            all_teams.sort(key=lambda t: elo.get(t, 1500), reverse=True)
            cur = [(all_teams[i], all_teams[len(all_teams)-1-i]) for i in range(len(all_teams)//2)]

            for rd in range(5):
                if len(cur) <= 1:
                    break
                nxt = []
                round_names = ['R16', 'QF', 'SF', 'Final', 'Champion'][rd]
                for t1, t2 in cur:
                    w = _play_ko(mc, elo, t1, t2)
                    nxt.append(w)
                    round_counts[w][round_names] += 1
                cur = [(nxt[i], nxt[i+1]) for i in range(0, len(nxt), 2)]

            if len(cur) == 1 and len(cur[0]) == 2:
                t1, t2 = cur[0]
                winner = _play_ko(mc, elo, t1, t2)
                loser = t2 if winner == t1 else t1
                round_counts[winner]['Champion'] += 1
                round_counts[loser]['Runner'] += 1
                champ[winner] += 1
                runner[loser] += 1

    return dict(champ), dict(runner), {t: dict(rc) for t, rc in round_counts.items()}


def _sim_worker_champ(mc_cache_dict, elo, seed, n_sims, teams, groups,
                      host_teams=None, host_bonus_by_team=None, ko_decay=0.5):
    """冠军-only worker (向后兼容)"""
    champ, _, _ = _sim_worker(mc_cache_dict, elo, seed, n_sims, teams, groups,
                            host_teams, host_bonus_by_team, ko_decay)
    return champ

# ═══════════════════════════════════════
#  HOST BONUS SENSITIVITY
# ═══════════════════════════════════════

def run_mc_for_bonus(mc_flat, elo, n_sims, teams, groups, host_teams, host_bonus_by_team, bracket_mode='ranked'):
    """用指定 host_bonus_by_team dict 跑 MC"""
    n_workers = 2
    sims_per = n_sims // n_workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for w in range(n_workers):
            f = executor.submit(
                _sim_worker, mc_flat, dict(elo), w * 99999 + 42,
                sims_per, teams, groups, host_teams, host_bonus_by_team,
                0.5, bracket_mode
            )
            futures.append(f)
        champ_total = defaultdict(int)
        runner_total = defaultdict(int)
        round_total = defaultdict(lambda: defaultdict(int))
        for f in concurrent.futures.as_completed(futures):
            c, r, rc = f.result()
            for t, cnt in c.items(): champ_total[t] += cnt
            for t, cnt in r.items(): runner_total[t] += cnt
            for t, rounds in rc.items():
                for rd, cnt in rounds.items():
                    round_total[t][rd] += cnt
    total = sum(champ_total.values())
    return champ_total, runner_total, dict(round_total), total

# ═══════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════

def run_pipeline(use_market_odds=True, run_mc=True, bracket_mode='ranked'):
    log("=" * 65)
    log("  ⚽ WC 2026 FINAL (修复版 v2.0)")
    log(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  特征: 29维 (23基线+6滚动形式)")
    log(f"  权重: 动态 (熵基, α=0.30 β=0.50 范围0.10-0.90)")
    log(f"  淘汰赛: {'FIFA官方路书' if bracket_mode=='official' else 'Elo排名配对'}")
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
    log(f"  DC: ρ={dc.rho_:.4f} γ={dc.gamma_:.4f} host_bonus={dc.host_bonus_:.4f}")

    # ── 3. Market Calibration ──
    market_probs = None
    winner_odds = {}
    if use_market_odds:
        market_data = load_market_odds()
        if market_data:
            winner_probs = market_data['winner_probs']
            winner_odds = market_data['winner_odds']
            total_market = sum(winner_probs.get(t, 0) for t in TEAMS_2026)
            if total_market > 0:
                market_probs = {}
                for t in TEAMS_2026:
                    mp = winner_probs.get(t, 0)
                    if t == 'Curacao':
                        mp = winner_probs.get('Curaçao', winner_probs.get('Curacao', 0))
                    market_probs[t] = mp / total_market
                log(f"\n  Market Odds (去水):")
                for t, p in sorted(market_probs.items(), key=lambda x: -x[1])[:10]:
                    log(f"    {t:<25s} {p*100:>6.2f}%")

    # ── 4. 严格时序切分: train/cal/val = 60/20/20 ──
    ms_all = sorted(matches, key=lambda m: m['date'])
    n_total = len(ms_all)
    train_end = int(n_total * 0.6)
    cal_end = int(n_total * 0.8)
    ms_train = ms_all[:train_end]
    ms_cal = ms_all[train_end:cal_end]
    ms_val = ms_all[cal_end:]
    log(f"    train: {len(ms_train)} 场 | cal: {len(ms_cal)} 场 | val: {len(ms_val)} 场")

    # Train features — 仅用train数据填充buffer
    fb_train = FeatureBuffer(elo, dc)
    X_train, y_train = [], []
    for m in ms_train:
        feat = build_golden20_feat_full(fb_train, m['home'], m['away'], m['date'], m)
        X_train.append(feat)
        if m['h_score'] > m['a_score']: y_train.append(2)
        elif m['h_score'] == m['a_score']: y_train.append(1)
        else: y_train.append(0)
        fb_train.add_match(m)

    # Cal/Val features — 仅用train buffer（不加入buffer，防止泄漏）
    X_cal, y_cal = [], []
    for m in ms_cal:
        feat = build_golden20_feat_full(fb_train, m['home'], m['away'], m['date'], m)
        X_cal.append(feat)
        if m['h_score'] > m['a_score']: y_cal.append(2)
        elif m['h_score'] == m['a_score']: y_cal.append(1)
        else: y_cal.append(0)

    X_val, y_val = [], []
    for m in ms_val:
        feat = build_golden20_feat_full(fb_train, m['home'], m['away'], m['date'], m)
        X_val.append(feat)
        if m['h_score'] > m['a_score']: y_val.append(2)
        elif m['h_score'] == m['a_score']: y_val.append(1)
        else: y_val.append(0)

    X_train, X_cal, X_val = np.array(X_train), np.array(X_cal), np.array(X_val)
    y_train, y_cal, y_val = np.array(y_train), np.array(y_cal), np.array(y_val)
    log(f"  ✅ 特征: train {X_train.shape} cal {X_cal.shape} val {X_val.shape}")

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
    xgb_model.fit(X_train, y_train, sample_weight=sw,
                  eval_set=[(X_cal, y_cal)],  # early stop on cal, not val
                  verbose=False)

    # Evaluate on val set (truly unseen)
    y_pred = xgb_model.predict(X_val)
    y_proba = xgb_model.predict_proba(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    val_nll = log_loss(y_val, y_proba)
    val_brier = np.mean((y_proba - np.eye(3)[y_val])**2)
    log(f"  ✅ Val: acc={val_acc:.4f} nll={val_nll:.4f} brier={val_brier:.4f}")

    # ── 5b. Isotonic Calibration (独立校准集: cal=20%, val=20%) ──
    log("  🌡 Isotonic 校准 hybrid 概率 (在 cal 集训练, val 集评估)...")
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import CalibratedClassifierCV

    def _build_hybrid_probs(match_list):
        """为比赛列表构建 hybrid [away,draw,home] 概率"""
        probs, labels = [], []
        buf = FeatureBuffer(elo, dc)
        for m in match_list:
            feat = build_golden20_feat_full(buf, m['home'], m['away'], m['date'], m)
            xp = xgb_model.predict_proba(np.array([feat]))[0]
            eh = elo.get(m['home'], 1500); ea = elo.get(m['away'], 1500)
            hb = getattr(dc, 'host_bonus_', 0.0)
            hb_val = hb if (not m.get('neutral', False) and m['home'] in HOST_TEAMS) else 0.0
            dp = dc.predict_proba(m['home'], m['away'], neutral=m.get('neutral', False), host_bonus=hb_val)
            dc_ado = np.array([dp[2], dp[1], dp[0]])
            xgb_w, dc_w, _conf = compute_dynamic_xgb_weight(xp)
            hybrid_prob = dc_w * dc_ado + xgb_w * xp
            probs.append(hybrid_prob)
            if m['h_score'] > m['a_score']: labels.append(2)
            elif m['h_score'] == m['a_score']: labels.append(1)
            else: labels.append(0)
        return np.array(probs), np.array(labels)

    # 1. Build hybrid probs for cal set (train isotonic here)
    X_cal_hybrid, y_cal_labels = _build_hybrid_probs(ms_cal)
    # 2. Build hybrid probs for val set (evaluate isotonic here, truly OOS)
    X_val_hybrid, y_val_labels = _build_hybrid_probs(ms_val)

    def _normalize_abc(probs):
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 1e-9, None)
        return probs / probs.sum()

    def _fit_platt_calibrator(x, y):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(
            solver='lbfgs',
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
        )
        lr.fit(np.asarray(x).reshape(-1, 1), np.asarray(y).astype(int))
        return lr

    def _predict_platt(calibrator, x):
        p = calibrator.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]
        return np.asarray(p, dtype=float)

    # 校准策略：样本不足或 isotonic 输出退化时，自动降级到 Platt
    MIN_ISOTONIC_SAMPLES = 100
    calibrators = []
    cal_modes = []
    cal_brier = 0.0
    val_brier_cal = 0.0
    for c, name in enumerate(('away', 'draw', 'home')):
        y_cal_bin = (y_cal_labels == c).astype(int)
        y_val_bin = (y_val_labels == c).astype(int)
        x_cal = X_cal_hybrid[:, c]
        x_val = X_val_hybrid[:, c]

        use_platt = int(y_cal_bin.sum()) < MIN_ISOTONIC_SAMPLES or len(np.unique(y_cal_bin)) < 2
        if use_platt:
            cal = _fit_platt_calibrator(x_cal, y_cal_bin)
            cal_modes.append('platt')
            p_cal = _predict_platt(cal, x_cal)
            p_val = _predict_platt(cal, x_val)
        else:
            cal = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
            cal.fit(x_cal, y_cal_bin)
            cal_modes.append('isotonic')
            p_cal = cal.predict(x_cal)
            p_val = cal.predict(x_val)

        calibrators.append(cal)
        cal_brier += np.mean((p_cal - y_cal_bin) ** 2)
        val_brier_cal += np.mean((p_val - y_val_bin) ** 2)

    cal_brier /= 3
    val_brier_cal /= 3

    # Compute val Brier BEFORE calibration
    val_brier_before = np.mean((X_val_hybrid - np.eye(3)[y_val_labels])**2)
    log(f"  ✅ Cal-set: 校准前 Brier {cal_brier:.4f}")
    log(f"  ✅ Val-set: 校准前 {val_brier_before:.4f} → 校准后 {val_brier_cal:.4f} (Δ={val_brier_before-val_brier_cal:+.4f})")
    log(f"  🔁 Calibrators: away={cal_modes[0]} draw={cal_modes[1]} home={cal_modes[2]}")

    # ── 保存校准器到磁盘 (与XGB/DC/Elo同源) ──
    cal_dict = {
        'away': calibrators[0],
        'draw': calibrators[1],
        'home': calibrators[2],
        'modes': {'away': cal_modes[0], 'draw': cal_modes[1], 'home': cal_modes[2]},
    }
    joblib.dump(cal_dict, os.path.join(DATA_DIR, 'calibrators.pkl'))
    log("  💾 校准器已保存 (与XGB/DC/Elo同源)")

    def apply_calibrators(hybrid_probs):
        """hybrid_probs: [3] array [away, draw, home] → 校准后。"""
        hybrid_probs = _normalize_abc(hybrid_probs)
        out = []
        for i, cal in enumerate(calibrators):
            if cal_modes[i] == 'platt':
                out.append(float(_predict_platt(cal, [hybrid_probs[i]])[0]))
            else:
                out.append(float(cal.predict([hybrid_probs[i]])[0]))
        out = np.asarray(out, dtype=float)
        return _normalize_abc(out)


    # 在后续 MC matchup 构建和回测中，对每场 hybrid 概率先 apply_calibrators()

    # ── 6. 2022 WC 回测 (严格时序) ──
    log("\n  🧪 2022 WC 回测 (严格时序)...")
    wc_results = []
    wc_matches = [m for m in matches
                  if m.get('tournament') == 'FIFA World Cup'
                  and m.get('date', '').startswith('2022')]
    wc_matches = sorted(wc_matches, key=lambda m: m['date'])
    bf = FeatureBuffer(elo, dc)

    correct_dc = 0; correct_hybrid = 0
    brier_dc = 0.0; brier_hybrid = 0.0
    n_wc = len(wc_matches)

    for m in wc_matches:
        feat = build_golden20_feat_full(bf, m['home'], m['away'], m['date'], m)
        xp = xgb_model.predict_proba(np.array([feat]))[0]
        eh = elo.get(m['home'], 1500); ea = elo.get(m['away'], 1500)
        hb = getattr(dc, 'host_bonus_', 0.0)
        hb_val = hb if (not m.get('neutral', False) and m['home'] in HOST_TEAMS) else 0.0
        dp = dc.predict_proba(m['home'], m['away'], neutral=False if not m.get('neutral', True) else True, host_bonus=hb_val)
        dc_ado = np.array([dp[2], dp[1], dp[0]])
        xgb_w, dc_w, _conf = compute_dynamic_xgb_weight(xp)
        hybrid = dc_w * dc_ado + xgb_w * xp
        # Apply isotonic calibration
        hybrid = apply_calibrators(hybrid)

        actual = 2 if m['h_score'] > m['a_score'] else (1 if m['h_score'] == m['a_score'] else 0)
        # 修复: dc_pred=np.argmax(dp) 返回 {0=主,1=平,2=客}，映射到 actual 的 {2=主,1=平,0=客}
        dc_pred = 2 - np.argmax(dp)
        hyb_pred = np.argmax([hybrid[0], hybrid[1], hybrid[2]])

        if dc_pred == actual: correct_dc += 1
        if hyb_pred == actual: correct_hybrid += 1

        actual_onehot = np.array([0, 0, 0])
        actual_onehot[2 - actual] = 1
        brier_dc += float(np.sum((np.array(dp) - actual_onehot)**2))
        # 修复: hybrid=[客,平,主], actual_onehot=[主,平,客]，需对齐
        hybrid_hda = np.array([hybrid[2], hybrid[1], hybrid[0]])  # [主,平,客]
        brier_hybrid += float(np.sum((hybrid_hda - actual_onehot)**2))
        bf.add_match(m)

    log(f"  📊 2022 WC ({n_wc}场):")
    log(f"    DC:      {correct_dc:>3d}/{n_wc} ({correct_dc/n_wc*100:.1f}%) Brier={brier_dc/(3*n_wc):.4f}")
    log(f"    Hybrid:  {correct_hybrid:>3d}/{n_wc} ({correct_hybrid/n_wc*100:.1f}%) Brier={brier_hybrid/(3*n_wc):.4f}")

    # ── 7. 保存模型 ──
    joblib.dump(xgb_model, os.path.join(DATA_DIR, 'xgb_model_29.pkl'))
    joblib.dump(dc, os.path.join(DATA_DIR, 'dc_model.pkl'))
    joblib.dump(elo, os.path.join(DATA_DIR, 'elo_ratings.pkl'))
    log("  💾 模型已保存")

    # ── 8. 构建 Matchup Cache ──
    log("\n  🔨 构建冠军模拟 matchup cache (48队×48队, 混合+市场校准)...")
    mc_cache = {}
    count = 0

    # 加载分组
    groups_path = os.path.join(MODEL_DIR, '2026_groups.json')
    if os.path.exists(groups_path):
        with open(groups_path) as f:
            GROUPS_2026 = json.load(f)
        log(f"  📋 加载 {len(GROUPS_2026)} 个小组")
    else:
        st = sorted(TEAMS_2026, key=lambda t: elo.get(t, 1500), reverse=True)
        GROUPS_2026 = {chr(65+i): st[i*4:(i+1)*4] for i in range(12)}

    # ── 预计算所有48队真实form数据 (修复train-serve skew) ──
    log("  🏃 预计算48队form/h2h特征...")
    fb_all = FeatureBuffer(elo, dc)
    for m in all_m:
        fb_all.add_match(m)
    TOURNAMENT_DATE = '2026-06-11'
    team_form_5 = {}; team_form_12 = {}; team_h2h_3 = {}
    for t in TEAMS_2026:
        team_form_5[t] = fb_all.recent_form(t, TOURNAMENT_DATE, 5)
        team_form_12[t] = fb_all.recent_form(t, TOURNAMENT_DATE, 12)
    for h in TEAMS_2026:
        for a in TEAMS_2026:
            if h == a: continue
            team_h2h_3[(h, a)] = fb_all.h2h(h, a, TOURNAMENT_DATE, 3)
    teams_with_form = sum(1 for v in team_form_5.values() if v[0] != 0.5)
    log(f"  ✅ 已预计算: {teams_with_form}/48 队有近5场form数据")
    for i, h in enumerate(TEAMS_2026):
        for a in TEAMS_2026:
            if h == a: continue
            # 预测 (中立)
            lh, la = dc.predict_lambda(h, a, neutral=True)
            if lh is None:
                lh, la = 1.0, 1.0
                dc_p = np.array([1/3, 1/3, 1/3])
            else:
                dc_p = dc.predict_proba(h, a, neutral=True)

            eh_elo = elo.get(h, 1500); ea_elo = elo.get(a, 1500)
            op = FeatureBuffer.make_odds(eh_elo, ea_elo)

            # ── 使用真实form/h2h替代占位符 ──
            fh5 = team_form_5.get(h, [0.5, 0.0, 0.0, 0.0])
            fa5 = team_form_5.get(a, [0.5, 0.0, 0.0, 0.0])
            fh12 = team_form_12.get(h, [0.5, 0.0, 0.0, 0.0])
            fa12 = team_form_12.get(a, [0.5, 0.0, 0.0, 0.0])
            h2h_data = team_h2h_3.get((h, a), [0.5, 0.0, 0.0, 0])
            b15 = [
                (eh_elo - ea_elo) / 400, lh, la, lh - la,
                math.log(max(lh, 0.01) / max(la, 0.01)),
                dc_p[0], dc_p[1], dc_p[2],
                fh5[0], fa5[0],
                fh5[1] - fa5[2], fa5[1] - fh5[2],
                fh5[1] - fa5[1], fh5[0] - fa5[0],
                1,
            ]
            gold = [
                h2h_data[1] - h2h_data[2],  # h2h avg_gf diff
                1, 0,  # major=1, final_round=0 (WC match)
                fh12[1] - fa12[2],  # 12g: home_gf - away_ga
                fa12[1] - fh12[0],  # 12g: away_gf - home_wr
            ]
            odds_feat = [op[0], op[1], op[2]]
            form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]
            feat = np.array([b15 + gold + odds_feat + form_feat])  # 29 dims
            xgb_p = xgb_model.predict_proba(feat)[0]

            # 混合: dc_p=[主,平,客], xgb_p=[客,平,主]
            dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])  # → [客,平,主]
            xgb_w, dc_w, _conf = compute_dynamic_xgb_weight(xgb_p)
            hybrid = dc_w * dc_ado + xgb_w * xgb_p
            # Apply isotonic calibration
            hybrid = apply_calibrators(hybrid)

            # 市场校准 — 凸组合: final = (1-mw) * model + mw * market_strength
            final_hybrid = hybrid
            if market_probs:
                mh = market_probs.get(h, 0)
                ma = market_probs.get(a, 0)
                if mh > 0 and ma > 0:
                    total = mh + ma
                    rel_h = mh / total  # market relative home strength
                    rel_a = ma / total  # market relative away strength
                    market_strength = total / 2.0 * (1 / 0.02)
                    mw = market_weight_for_match(eh_elo, ea_elo, neutral=True, market_strength=min(2.0, max(0.0, market_strength)))
                    # 市场概率 [A,D,H] 与模型概率 [A,D,H] 凸组合 (顺序已统一)
                    market_vec = market_probs_3way_from_outright(rel_h, rel_a)
                    blended = (1.0 - mw) * hybrid + mw * market_vec
                    # 确保概率和为1且每项非负
                    blended = np.clip(blended, 0.001, 0.999)
                    blended = blended / blended.sum()
                    final_hybrid = blended

            # MC uncertainty jitter
            samples = [jitter_prob(final_hybrid, epsilon=0.008, seed=(hash((h, a, i)) & 0xffffffff)) for i in range(8)]
            final_mean, final_std = summarize_probs(samples)
            final_hybrid = final_mean

            # 预计算 CDF
            def make_cdf(lam):
                cdf = []; s = 0
                for k in range(MAX_GOALS + 1):
                    s += poisson.pmf(k, lam)
                    cdf.append(s)
                return cdf

            mc_cache[(h, a)] = (
                float(final_hybrid[0]), float(final_hybrid[1]), float(final_hybrid[2]),
                float(final_std[0]), float(final_std[1]), float(final_std[2]),
                float(lh), float(la), make_cdf(lh), make_cdf(la)
            )
            count += 1

    log(f"  ✅ {count} matchups cached")

    # ── 8b. 东道主灵敏度分析 ──
    log("\n  📊 东道主灵敏度分析...")
    mc_flat = {}
    for (h, a), v in mc_cache.items():
        mc_flat[f"{h}||{a}"] = v

    for bonus_val in HOST_BONUS_SENSITIVITY:
        # 灵敏度: 所有东道主用同一值做基准对比
        uniform_dict = {t: bonus_val for t in HOST_TEAMS}
        ct, rt, _, tt = run_mc_for_bonus(
            mc_flat, elo, 50000, TEAMS_2026, GROUPS_2026,
            HOST_TEAMS, uniform_dict, bracket_mode=bracket_mode
        )
        log(f"    host_bonus={bonus_val:.4f}:")
        for t in sorted(ct, key=lambda x: -ct[x])[:5]:
            log(f"      {t:<20s} 冠军{ct[t]/tt*100:>6.2f}% 亚军{rt.get(t,0)/tt*100:>6.2f}%")

    # ── 9. 主 MC (200K) ──
    N = 200000
    n_workers = 2
    log(f"\n  🏃 MC {N:,} (12组×4队, 并行{n_workers}进程)...")

    start = datetime.now()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        sims_per = N // n_workers
        futures = []
        for w in range(n_workers):
            f = executor.submit(
                _sim_worker, mc_flat, dict(elo), w * 99999 + 42,
                sims_per, TEAMS_2026, GROUPS_2026,
                HOST_TEAMS, HOST_BONUS_BY_TEAM, KO_HOST_DECAY,
                bracket_mode
            )
            futures.append(f)

        champ_total = defaultdict(int)
        runner_total = defaultdict(int)
        round_total = defaultdict(lambda: defaultdict(int))
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            c, r, rc = f.result()
            for t, cnt in c.items(): champ_total[t] += cnt
            for t, cnt in r.items(): runner_total[t] += cnt
            for t, rounds in rc.items():
                for rd, cnt in rounds.items():
                    round_total[t][rd] += cnt
            log(f"    worker {i+1}/{n_workers} done")

    elapsed = (datetime.now() - start).total_seconds()
    total = sum(champ_total.values())
    log(f"  ⏱ {elapsed:.1f}s ({total:,} 有效决赛)")

    # ── 10. 输出 ──
    champs_sorted = sorted(champ_total.items(), key=lambda x: -x[1])
    runners_sorted = sorted(runner_total.items(), key=lambda x: -x[1])

    champ_prob = {t: c/total for t, c in champ_total.items()}
    runner_prob = {t: r/total for t, r in runner_total.items()}
    round_prob = {t: {rd: cnt/total for rd, cnt in rounds.items()} for t, rounds in round_total.items()}

    log(f"\n{'='*65}")
    log(f"  🏆 2026 冠军概率 (修复版)")
    log(f"{'='*65}")
    best_pct = champs_sorted[0][1] / total * 100 if champs_sorted else 0
    for i, (t, c) in enumerate(champs_sorted[:20], 1):
        pct = c / total * 100
        bar = '█' * int(pct / best_pct * 20) + '░' * (20 - int(pct / best_pct * 20))
        rp = runner_prob.get(t, 0) * 100
        odds = winner_odds.get(t, 0)
        log(f"  {i:>3d}. {t:<22s} 冠{pct:>5.2f}% 亚{rp:>5.2f}% 赔率{odds:>6.1f} {bar}")

    # EV 表
    log(f"\n{'='*65}")
    log(f"  💰 冠军 EV 分析")
    log(f"{'='*65}")
    kelly_results = []
    for idx, (t, c) in enumerate(champs_sorted[:25], 1):
        p = c / total
        odds = winner_odds.get(t, 0)
        if odds > 0:
            implied = 1.0 / odds
            ev = p * odds - 1
            b = odds - 1
            kelly = max(0, (b * p - (1 - p)) / b) if b > 0 else 0
            bar = '🟢' if ev > 0 else ('🟡' if ev > -0.2 else '🔴')
            log(f"  {idx:>3d}. {bar} {t:<22s} 模型{p*100:>6.2f}% 市场{implied*100:>6.2f}% 赔率{odds:>6.1f} EV{ev*100:+6.1f}%")
            kelly_results.append((t, p, odds, ev, kelly))

    # ═══════════════════════════════════
    #  购买策略
    # ═══════════════════════════════════
    log(f"\n{'='*65}")
    log(f"  🎯 购买策略")
    log(f"{'='*65}")

    # Tier 1: 正EV
    tier1 = [(t, p, odds, ev, kelly) for t, p, odds, ev, kelly in kelly_results if ev > 0]
    tier1.sort(key=lambda x: -x[3])
    log(f"\n  🥇 正EV标的 (市场低估):")
    if tier1:
        for t, p, odds, ev, kelly in tier1:
            log(f"    ✓ {t:<20s} 赔率{odds:.1f} 模型{p*100:.1f}% vs 市场{1/odds*100:.1f}% EV{ev*100:+.1f}%")
            rp = runner_prob.get(t, 0)
            log(f"      亚军{rp*100:.1f}% 决赛概率{(p+rp)*100:.1f}%")

    # Tier 2: 微亏但高赔率
    tier2 = [(t, p, odds, ev, kelly) for t, p, odds, ev, kelly in kelly_results if -0.20 < ev <= 0]
    tier2.sort(key=lambda x: -x[3])
    log(f"\n  🥈 微亏量(<20%)但赔率杠杆:")
    if tier2:
        for t, p, odds, ev, kelly in tier2[:8]:
            log(f"    △ {t:<20s} 赔率{odds:.1f} 模型{p*100:.1f}% EV{ev*100:+.1f}%")

    # 亚军高概率
    log(f"\n  🥉 亚军高概率:")
    runner_top = runners_sorted[:8]
    for t, r in runner_top:
        p_run = r / total
        cp = champ_prob.get(t, 0)
        log(f"    ○ {t:<20s} 亚军{p_run*100:.1f}% 冠军{cp*100:.1f}% 决赛{(cp+p_run)*100:.1f}%")

    # 每轮晋级概率 (Top 15) — 过滤已淘汰球队
    import json as _json
    _ts_path = '/root/data/tournament_state.json'
    _eliminated_teams = set()
    try:
        _ts = _json.load(open(_ts_path))
        _cn_to_en = {'日本':'Japan','德国':'Germany','荷兰':'Netherlands','巴西':'Brazil','法国':'France',
                      '西班牙':'Spain','英格兰':'England','葡萄牙':'Portugal','阿根廷':'Argentina','克罗地亚':'Croatia',
                      '比利时':'Belgium','瑞士':'Switzerland','瑞典':'Sweden','挪威':'Norway','波兰':'Poland',
                      '丹麦':'Denmark','意大利':'Italy','土耳其':'Turkey','乌克兰':'Ukraine','威尔士':'Wales',
                      '俄罗斯':'Russia','苏格兰':'Scotland','捷克':'Czech Republic','罗马尼亚':'Romania',
                      '奥地利':'Austria','匈牙利':'Hungary','斯洛文尼亚':'Slovenia','希腊':'Greece',
                      '塞尔维亚':'Serbia','斯洛伐克':'Slovakia','保加利亚':'Bulgaria','爱尔兰':'Ireland',
                      '北爱尔兰':'Northern Ireland','冰岛':'Iceland','黑山':'Montenegro','阿尔巴尼亚':'Albania',
                      '波黑':'Bosnia & Herzegovina','墨西哥':'Mexico','美国':'USA','加拿大':'Canada',
                      '哥斯达黎加':'Costa Rica','洪都拉斯':'Honduras','萨尔瓦多':'El Salvador','巴拉圭':'Paraguay',
                      '厄瓜多尔':'Ecuador','秘鲁':'Peru','智利':'Chile','乌拉圭':'Uruguay','哥伦比亚':'Colombia',
                      '委内瑞拉':'Venezuela','玻利维亚':'Bolivia','阿根廷':'Argentina',
                      '沙特阿拉伯':'Saudi Arabia','伊朗':'Iran','伊拉克':'Iraq','卡塔尔':'Qatar',
                      '阿联酋':'UAE','阿曼':'Oman','巴林':'Bahrain','科威特':'Kuwait','黎巴嫩':'Lebanon',
                      '约旦':'Jordan','叙利亚':'Syria','也门':'Yemen','巴勒斯坦':'Palestine',
                      '乌兹别克':'Uzbekistan','韩国':'South Korea','朝鲜':'North Korea','日本':'Japan',
                      '澳大利亚':'Australia','新西兰':'New Zealand','中国':'China',
                      '埃及':'Egypt','阿尔及利亚':'Algeria','突尼斯':'Tunisia','摩洛哥':'Morocco',
                      '塞内加尔':'Senegal','尼日利亚':'Nigeria','喀麦隆':'Cameroon','科特迪瓦':'Ivory Coast',
                      '加纳':'Ghana','民主刚果':'DR Congo','刚果(金)':'DR Congo','布基纳法索':'Burkina Faso',
                      '马里':'Mali','南非':'South Africa','佛得角':'Cape Verde','海地':'Haiti',
                      '库拉索':'Curaçao'}
        for cn_name, info in _ts.items():
            if info.get('eliminated', False):
                en_name = _cn_to_en.get(cn_name, cn_name)
                _eliminated_teams.add(en_name)
    except:
        pass
    log(f"  (已过滤 {len(_eliminated_teams)} 支淘汰球队:)")
    
    round_order = ['R16', 'QF', 'SF', 'Final', 'Champion']
    _displayed = 0
    for t, c in champs_sorted:
        if t in _eliminated_teams:
            continue
        if _displayed >= 15:
            break
        p_c = c / total * 100
        rounds_str = " | ".join(f"{rd}:{round_prob.get(t,{}).get(rd,0)*100:>5.1f}%" for rd in round_order)
        log(f"    {t:<22s} {rounds_str}")
        _displayed += 1

    # 风险提示
    log(f"\n{'='*65}")
    log(f"  ⚠ 风险声明")
    log(f"{'='*65}")
    log(f"  分组来源: {'正式分组' if os.path.exists(groups_path) else '基于Elo推断'}")
    bracket_label = 'FIFA官方路书(openfootball/worldcup)' if bracket_mode == 'official' else '按Elo排名配对'
    log(f'  淘汰赛签表: {bracket_label}')
    log(f"  东道主加成: {HOST_BONUS_BY_TEAM}")
    log(f"  模拟次数: {N}")
    log(f"  当前赔率缓存: {winner_odds.get('Spain', 'N/A')}")
    log(f"  同场比赛概率: 不从此表读取, 由独立单场预测脚本提供")

    # ── 保存 ──
    result = {
        'type': 'wc2026_final_stable',
        'ts': datetime.now().isoformat(),
        'feature_dim': 29,
        'feature_set': '20+3_golden',
        'dc_weight': 'dynamic_entropy',
        'xgb_weight': 'dynamic_entropy',
        'market_weight': MARKET_WEIGHT,
        'optuna_params': OPTUNA_PARAMS,
        'bracket_mode': bracket_mode,
        'validation': {'acc': float(val_acc), 'nll': float(val_nll), 'brier': float(val_brier), 'brier_calibrated': float(val_brier_cal)},
        'backtest_wc2022': {
            'n': n_wc,
            'dc_acc': correct_dc / n_wc,
            'hybrid_acc': correct_hybrid / n_wc,
            'dc_brier': float(brier_dc / (3 * n_wc)),
            'hybrid_brier': float(brier_hybrid / (3 * n_wc)),
        },
        'sims': total,
        'market_odds': bool(market_probs),
        'summary': {
            'validation_acc': float(val_acc),
            'validation_brier': float(val_brier),
            'backtest_hybrid_brier': float(brier_hybrid / (3 * n_wc)),
            'top1_team': champs_sorted[0][0] if champs_sorted else None,
            'top3_teams': [t for t, _ in champs_sorted[:3]],
            'bracket_info': f'{bracket_mode} (official={bracket_mode=="official"})',
            'host_bonus_info': f'{HOST_BONUS_BY_TEAM}',
        },
        'champ_prob': champ_prob,
        'runner_prob': runner_prob,
        'round_prob': round_prob,
        'winner_odds': winner_odds,
        'tier1': [(t, round(p,6), odds, round(ev,6), round(kelly,6)) for t,p,odds,ev,kelly in tier1],
        'tier2': [(t, round(p,6), odds, round(ev,6), round(kelly,6)) for t,p,odds,ev,kelly in tier2],
        'champs': [(t, c, c / total * 100) for t, c in champs_sorted[:30]],
        'runners': [(t, r, r / total * 100) for t, r in runners_sorted[:30]],
    }

    result_path = os.path.join(DATA_DIR, 'final_results.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log(f"\n  💾 保存: {result_path}")

    # 也输出人类可读副本
    txt_path = os.path.join(DATA_DIR, 'final_results.txt')
    with open(txt_path, 'w') as f:
        f.write(f"{'='*65}\n")
        f.write(f"  🏆 2026 冠军概率 (修复版)  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*65}\n")
        for i, (t, c) in enumerate(champs_sorted[:48], 1):
            pct = c / total * 100
            rp = runner_prob.get(t, 0) * 100
            odds = winner_odds.get(t, 999)
            ev_str = f"EV{(pct/100*odds-1)*100:+6.1f}%" if odds != 999 else ""
            in_tier1 = any(t == x[0] for x in tier1)
            marker = " 🟢" if in_tier1 else ""
            f.write(f"  {i:>3d}. {t:<22s} {pct:>6.2f}% 亚{rp:>5.2f}% 赔{odds:>6.1f} {ev_str}{marker}\n")
        f.write("\n")
        f.write(f"  分组: {len(GROUPS_2026)}个小组 | 淘汰赛: {'FIFA官方路书' if bracket_mode=='official' else 'Elo排名配对'}\n")
    log(f"\n  💾 保存: {txt_path}")
    log(f"\n{'='*65}")
    log("  ✅ 修复版管线 完工")
    log(f"{'='*65}")

    return result


if __name__ == '__main__':
    import sys
    run_mc = '--no-mc' not in sys.argv
    use_market = '--no-odds' not in sys.argv
    bracket_mode = 'official' if '--bracket=official' in sys.argv else 'ranked'
    run_pipeline(use_market_odds=use_market, run_mc=run_mc, bracket_mode=bracket_mode)
