#!/usr/bin/env python3
"""Single match predictor — DC + XGB + rolling form features.

Boundary: this file never loads tournament winner odds and never uses
overview-market inputs in single-match calibration.

Supports --home flag for per-team host_bonus (from HOST_BONUS_BY_TEAM).
When --home is active, DC AND XGB both use the boosted λ → full hybrid.

NOW with rolling form features (form_points, form_gf, form_ga).
"""
import sys, os, json, math, random
sys.path.insert(0, '/root')
import numpy as np
from scipy.stats import poisson
from scipy.special import softmax, logit
import joblib
from collections import defaultdict
from team_name_normalizer import normalize_match_pair, normalize_team_name
from feature_helper import build_gold_features, load_h2h_cache, load_form_12_cache

DATA_DIR = '/root/data'
MAX_GOALS = 6

# Load models
_dc = joblib.load(os.path.join(DATA_DIR, 'dc_model.pkl'))
_xgb_model = (
    joblib.load(os.path.join(DATA_DIR, 'xgb_model_29.pkl'))
    if os.path.exists(os.path.join(DATA_DIR, 'xgb_model_29.pkl'))
    else joblib.load(os.path.join(DATA_DIR, 'xgb_model_26.pkl'))
    if os.path.exists(os.path.join(DATA_DIR, 'xgb_model_26.pkl'))
    else joblib.load(os.path.join(DATA_DIR, 'xgb_model_20_3.pkl'))
)
_elo = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))

DC_WEIGHT = 0.4
XGB_WEIGHT = 0.6

# 与 wc_2026_final.py 同步的东道主加成
HOST_TEAMS = {'United States', 'Mexico', 'Canada'}
HOST_BONUS_BY_TEAM = {
    'United States': 0.1445,
    'Mexico': 0.10,
    'Canada': 0.07,
}

# 友谊赛折扣因子 — 改为从 friendly_calib.json 数据驱动加载
# 默认值 (无calib文件时使用保守折扣)
DEFAULT_FRIENDLY_DISCOUNT = 0.20  # 基于回测: 旗鼓相当match最优20%
_FRIENDLY_CALIB_CACHE = None

def _load_friendly_discount():
    """加载友谊赛自适应折扣参数."""
    global _FRIENDLY_CALIB_CACHE
    if _FRIENDLY_CALIB_CACHE is not None:
        return _FRIENDLY_CALIB_CACHE
    cal_path = os.path.join(DATA_DIR, 'friendly_calib.json')
    default = {'default_discount': 0.20, 'discount_by_strength': {'low_diff': 0.20, 'high_diff': 0.0}}
    if os.path.exists(cal_path):
        try:
            with open(cal_path) as f:
                _FRIENDLY_CALIB_CACHE = {**default, **json.load(f)}
            return _FRIENDLY_CALIB_CACHE
        except Exception:
            pass
    _FRIENDLY_CALIB_CACHE = default
    return _FRIENDLY_CALIB_CACHE


def _get_friendly_discount(strength_diff):
    """根据强度差异获取自适应折扣因子.

    Args:
        strength_diff: Elo差归一化 (eh-ea)/400, >0=主队更强

    Returns: discount [0, 1], 0=不折扣
    """
    cal = _load_friendly_discount()
    abs_diff = abs(strength_diff)
    disc_map = cal.get('discount_by_strength', {})
    if abs_diff >= 0.5:
        return disc_map.get('high_diff', 0.0)
    else:
        return disc_map.get('low_diff', cal.get('default_discount', 0.20))

# Isotonic 校准器已剥离 (2026-06-10 诊断为负优化, 生产Brier=0.2341 vs 训练0.2053)
# 国际赛/俱乐部赛均不再使用校准器, 统一回退 Temperature Scaling
_CALIBRATED_XGB = None
def _load_calibrators():
    """校准器已禁用，返回 None 强制走 Temperature Scaling 回退."""
    return (None, None)


def _temperature_scale_probs(probs, temp=1.2):
    """Temperature scaling on logits (not probabilities)."""
    probs_safe = np.clip(probs, 1e-15, 1 - 1e-15)
    probs_safe = probs_safe / probs_safe.sum()
    logits = np.log(probs_safe)
    scaled = logits / temp
    return softmax(scaled)


def _apply_calibration(probs, feat=None, n_cal_samples=None):
    """Apply calibration. probs = [p_away, p_draw, p_home].

    Priority:
    1. CalibratedClassifierCV (multi-class, best) — if feat provided and loaded
    2. Per-class Isotonic (legacy) — with renormalization
    3. Temperature scaling (fallback when sample size small or Isotonic not available)
    """
    kind, cals = _load_calibrators()
    if kind is None:
        # 无校准器时，使用温度缩放作为保守回退
        return _temperature_scale_probs(probs, temp=1.2)

    if kind == 'cv' and feat is not None:
        # Use full CalibratedClassifierCV (multi-class, no renorm drift)
        try:
            out = cals.predict_proba(np.array([feat]))[0]
            return np.array([out[0], out[1], out[2]])
        except Exception:
            pass

    if kind == 'dict':
        # Per-class Isotonic (legacy) — 必须重新归一化
        out = np.zeros_like(probs)
        for i, key in enumerate(['away', 'draw', 'home']):
            if key in cals:
                out[i] = float(np.clip(cals[key].transform([probs[i]])[0], 0.001, 0.999))
            else:
                out[i] = probs[i]
        s = out.sum()
        if s > 0:
            out = out / s
        return out

    # 未知 kind，使用温度缩放
    return _temperature_scale_probs(probs, temp=1.2)

# Feature dimension: 23 (old) or 29 (new with form)
# Auto-detect from model
_XGB_DIM = _xgb_model.n_features_in_ if hasattr(_xgb_model, 'n_features_in_') else 29


# ── Form state tracker (loaded from saved form_state.json) ──────────
_form_state = None
def _load_form_state():
    global _form_state
    if _form_state is not None:
        return _form_state
    path = os.path.join(DATA_DIR, 'form_state.json')
    if not os.path.exists(path):
        _form_state = {}
        return _form_state
    with open(path) as f:
        _form_state = json.load(f)
    return _form_state


def recent_form(team, n=5):
    """Get rolling form for a team. Returns [win_rate, avg_gf, avg_ga, avg_gd]."""
    fs = _load_form_state()
    if team not in fs or len(fs[team]) < 1:
        return [0.5, 0.0, 0.0, 0.0]
    games = fs[team][-n:]
    if not games:
        return [0.5, 0.0, 0.0, 0.0]
    w = sum(1 for g in games if g[0] > g[1]) + sum(0.5 for g in games if g[0] == g[1])
    gf = sum(g[0] for g in games) / len(games)
    ga = sum(g[1] for g in games) / len(games)
    return [w / len(games), gf, ga, gf - ga]


def make_odds(eh, ea):
    dh = ea - eh; da = eh - ea
    return [1 / (10 ** (-dh / 400) + 1), 1 / (10 ** (-da / 400) + 1), 0.0]


def predict_match(home, away, host_bonus=0.0, match_type='competitive'):
    """Return single-match probabilities with optional host_bonus.

    match_type: 'competitive' (默认, 正式比赛) | 'qualifier' (世预赛) | 'friendly' (友谊赛)
    友谊赛自动应用折扣: 30% 拉向均匀分布 + Isotonic 校准 (若已训练)
    """
    home, away = normalize_match_pair(home, away)

    is_host = host_bonus > 0 and home in HOST_TEAMS
    neutral = not is_host

    dc_p = _dc.predict_proba(home, away, neutral, host_bonus=host_bonus if is_host else 0.0)
    lam_h, lam_a = _dc.predict_lambda(home, away, neutral, host_bonus=host_bonus if is_host else 0.0)
    if lam_h is None:
        return None, "DC 模型未收敛"

    eh_elo = _elo.get(home, 1500)
    ea_elo = _elo.get(away, 1500)

    op = make_odds(eh_elo, ea_elo)

    # Rolling form features (last 5 games)
    fh5 = recent_form(home, 5)
    fa5 = recent_form(away, 5)

    fh5_ = [0.5, 0.0, 0.0, 0.0] if _XGB_DIM <= 23 else fh5
    fa5_ = [0.5, 0.0, 0.0, 0.0] if _XGB_DIM <= 23 else fa5

    b15 = [
        (eh_elo - ea_elo) / 400,
        lam_h, lam_a, lam_h - lam_a,
        math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
        dc_p[0], dc_p[1], dc_p[2],
        fh5_[0], fa5_[0],
        fh5_[1] - fa5_[2], fa5_[1] - fh5_[2],
        fh5_[1] - fa5_[1], fh5_[0] - fa5_[0],
        0 if is_host else 1,  # neutral flag
    ]
    # Gold features: [h2h_gd, tier_major, tier_friendly, fh12_attack_def, fa12_attack_wr]
    # P1: 用真实 H2H + 12场 form 替换占位符 (修复 train-serve skew)
    gold = build_gold_features(home, away, match_type=match_type)
    odds_feat = [op[0], op[1], op[2] if op[2] else 0.0]

    base_feat = b15 + gold + odds_feat  # 23 dims

    if _XGB_DIM > 23:
        # Add 6 rolling form raw features:
        form_feat = [
            fh5_[1],              # home avg GF last 5
            fh5_[2],              # home avg GA last 5
            fa5_[1],              # away avg GF last 5
            fa5_[2],              # away avg GA last 5
            fh5_[0] * 3,          # home form points
            fa5_[0] * 3,          # away form points
        ]
        feat = np.array([base_feat + form_feat])
    else:
        feat = np.array([base_feat])

    xgb_p = _xgb_model.predict_proba(feat)[0]  # [away, draw, home]
    dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])
    hybrid = DC_WEIGHT * dc_ado + XGB_WEIGHT * xgb_p

    # ── P2: Isotonic 校准 (可选, 传入feat以使用CalibratedClassifierCV) ──
    # TODO: 读取校准集样本数用于 Isotonic/Platt 选择
    hybrid_cal = _apply_calibration(hybrid, feat=feat[0], n_cal_samples=None)

    # ── P1: 友谊赛自适应折扣 (post-hoc smoothing, 数据驱动) ──
    smooth = 0.0
    if match_type == 'friendly':
        # 基于 Elo 差的强度差异决定折扣
        strength_diff = (eh_elo - ea_elo) / 400
        smooth = _get_friendly_discount(strength_diff)
    if smooth > 0:
        # smooth 比例拉向均匀分布, 降低模型置信度
        final = (1 - smooth) * hybrid_cal + smooth * np.array([1/3, 1/3, 1/3])
    else:
        final = hybrid_cal

    # ── P3: 友谊赛边际门控 (旁路输出, 不影响 final 概率) ──
    # 基于 76 场回测:
    #   < 5pp   → 41.7% 命中率 (接近随机, 不投)
    #   5-10pp  → 18.2% 命中率 (反向信号, 绝对避开)
    #   10-20pp → 50.0% 命中率 (微弱信号)
    #   20-30pp → 53.8% 命中率 (强信号)
    #   > 30pp  → 86.4% 命中率 (极强信号, 必投)
    # final 数组是 0-1 单位 (DC/XGB/calibration 都返回 0-1)
    p_h_fin = float(final[2])  # home win (0-1)
    p_d_fin = float(final[1])  # draw (0-1)
    p_a_fin = float(final[0])  # away win (0-1)
    probs_sorted = sorted([p_h_fin, p_d_fin, p_a_fin], reverse=True)
    margin_raw = probs_sorted[0] - probs_sorted[1]  # 0-1 单位
    margin_pp = margin_raw * 100  # 转为 pp (percentage points)
    best_label = ['H', 'D', 'A'][[p_h_fin, p_d_fin, p_a_fin].index(probs_sorted[0])]

    # ── S1: 次级门控 - 客队 form 数据缺失检查 ──
    # form_state.json 缺失的队 XGB 实际用默认值 (0.5/0/0/0), 是"客队神秘"信号
    # 例子: 'Ireland', 'Northern Macedonia' 在 22 场前向验证中 0 场 form 数据
    fs = _load_form_state()
    home_has_form = home in fs and len(fs[home]) >= 1
    away_has_form = away in fs and len(fs[away]) >= 1
    form_gap = (not home_has_form) or (not away_has_form)
    form_warn = []
    if not home_has_form:
        form_warn.append(f'主队{home}无form数据')
    if not away_has_form:
        form_warn.append(f'客队{away}无form数据')

    if match_type == 'friendly':
        if form_gap:
            # S1: 客队/主队 form 缺失 → 跳过 (避免"客队神秘"失真)
            bet_action = 'SKIP_DATA'
        elif margin_pp >= 20:
            bet_action = 'BET'  # 极强信号
        elif margin_pp >= 10:
            bet_action = 'BET'  # 强信号 (回测 66% 命中率, +28.8% ROI)
        else:
            bet_action = 'SKIP'  # 弱信号 (回测 18-42% 命中率, -ROI)
    else:
        # 正式比赛不做门控, 让 caller 决定
        bet_action = 'N/A'

    return {
        'home': home, 'away': away,
        'match_type': match_type,
        'friendly_discount': smooth,
        'calibrated': _load_calibrators() is not None,
        'host_bonus_applied': is_host,
        'host_bonus_val': round(host_bonus, 4) if is_host else 0,
        'lam_h': round(lam_h, 2),
        'lam_a': round(lam_a, 2),
        'dc_h': round(float(dc_p[0] * 100), 1),
        'dc_d': round(float(dc_p[1] * 100), 1),
        'dc_a': round(float(dc_p[2] * 100), 1),
        'xgb_a': round(float(xgb_p[0] * 100), 1),
        'xgb_d': round(float(xgb_p[1] * 100), 1),
        'xgb_h': round(float(xgb_p[2] * 100), 1),
        'hyb_a': round(float(hybrid[0] * 100), 1),
        'hyb_d': round(float(hybrid[1] * 100), 1),
        'hyb_h': round(float(hybrid[2] * 100), 1),
        'cal_a': round(float(hybrid_cal[0] * 100), 1),
        'cal_d': round(float(hybrid_cal[1] * 100), 1),
        'cal_h': round(float(hybrid_cal[2] * 100), 1),
        'fin_a': round(float(final[0] * 100), 1),
        'fin_d': round(float(final[1] * 100), 1),
        'fin_h': round(float(final[2] * 100), 1),
        'market_used': False,
        'elo_h': eh_elo,
        'elo_a': ea_elo,
        'feature_dim': _XGB_DIM,
        'form_used': _XGB_DIM > 23,
        'home_form_gf': round(fh5_[1], 2),
        'home_form_ga': round(fh5_[2], 2),
        'away_form_gf': round(fa5_[1], 2),
        'away_form_ga': round(fa5_[2], 2),
        'bet_recommendation': {
            'action': bet_action,         # 'BET' | 'SKIP' | 'SKIP_DATA' | 'N/A'
            'margin_pp': round(margin_pp, 1),
            'best_pick': best_label,      # 'H' | 'D' | 'A'
            'best_prob_pct': round(probs_sorted[0] * 100, 1),
            'form_warnings': form_warn,   # S1: form 缺失队名
            's1_triggered': form_gap,     # S1 门控是否触发
        },
        'total_goals_recommendation': _build_total_goals_recommendation(lam_h, lam_a, match_type, form_gap),
    }


def mc_score_dist(lam_h, lam_a, n=100000):
    hg = np.random.poisson(lam_h, n)
    ag = np.random.poisson(lam_a, n)
    scores = {}
    for i, j in zip(hg, ag):
        if i <= 6 and j <= 6:
            scores[(i, j)] = scores.get((i, j), 0) + 1
    total = sum(scores.values())
    return [(f"{i}:{j}", round(cnt / total * 100, 1)) for (i, j), cnt in sorted(scores.items(), key=lambda x: -x[1])[:5]]


# ── P4: 总进球 (大/小 2.5) 门控 (旁路) ──
# 基于 22 场真实友谊赛预研 (2026-05-29 ~ 2026-06-03 FIFA 窗口):
#   |p_over - 0.5| < 5pp   → 43% 命中率 (弱)
#   |p_over - 0.5| 5-15pp  → 80% 命中率 (中等)
#   |p_over - 0.5| 15-25pp → 88% 命中率 (强)
#   |p_over - 0.5| > 25pp  → 100% 命中率 (极强)
# λ_total < 2.0: 100% 命中小球 (3/3)
# λ_total > 3.5: 100% 命中大球 (5/5)
# 2.5-3.0: 43% 命中率 (陷阱区, 类似 HDA 5-10pp)
# 综合: 强信号 (15pp+) 9/10 = 90% 命中率, ROI +72.8%

def _build_total_goals_recommendation(lam_h, lam_a, match_type, form_gap):
    """P4: 总进球 (大/小 2.5) 门控 (旁路, 不影响 final 概率)

    返回 dict 含:
      - action: 'BET_OVER' | 'BET_UNDER' | 'SKIP' | 'SKIP_DATA' | 'N/A'
      - pick: '大' | '小' | '-'
      - p_over_2_5 / p_under_2_5 (% 形式)
      - lam_total
      - confidence_pp: |p_over - 0.5| * 100
      - reason: 决策原因
    """
    from scipy.stats import poisson as _poisson

    lam_total = lam_h + lam_a
    p_over_2_5 = 1 - sum(_poisson.pmf(k, lam_total) for k in range(3))
    p_under_2_5 = 1 - p_over_2_5
    confidence_pp = abs(p_over_2_5 - 0.5) * 100

    # 正式比赛不做门控, 让 caller 决定
    if match_type != 'friendly':
        return {
            'action': 'N/A',
            'pick': '-',
            'p_over_2_5_pct': round(p_over_2_5 * 100, 1),
            'p_under_2_5_pct': round(p_under_2_5 * 100, 1),
            'lam_total': round(lam_total, 2),
            'confidence_pp': round(confidence_pp, 1),
            'reason': '正式比赛 (非友谊赛), 门控未启用',
        }

    # S1: form 数据缺失 → 跳过
    if form_gap:
        return {
            'action': 'SKIP_DATA',
            'pick': '-',
            'p_over_2_5_pct': round(p_over_2_5 * 100, 1),
            'p_under_2_5_pct': round(p_under_2_5 * 100, 1),
            'lam_total': round(lam_total, 2),
            'confidence_pp': round(confidence_pp, 1),
            'reason': 'form 数据缺失, 信号不可靠',
        }

    # 强信号 (15pp+) 投
    if confidence_pp >= 25:
        return {
            'action': 'BET_OVER' if p_over_2_5 > 0.5 else 'BET_UNDER',
            'pick': '大' if p_over_2_5 > 0.5 else '小',
            'p_over_2_5_pct': round(p_over_2_5 * 100, 1),
            'p_under_2_5_pct': round(p_under_2_5 * 100, 1),
            'lam_total': round(lam_total, 2),
            'confidence_pp': round(confidence_pp, 1),
            'reason': f'极强信号 (|Δ|={confidence_pp:.1f}pp ≥ 25pp, 预研 100% 命中率)',
        }
    if confidence_pp >= 15:
        return {
            'action': 'BET_OVER' if p_over_2_5 > 0.5 else 'BET_UNDER',
            'pick': '大' if p_over_2_5 > 0.5 else '小',
            'p_over_2_5_pct': round(p_over_2_5 * 100, 1),
            'p_under_2_5_pct': round(p_under_2_5 * 100, 1),
            'lam_total': round(lam_total, 2),
            'confidence_pp': round(confidence_pp, 1),
            'reason': f'强信号 (|Δ|={confidence_pp:.1f}pp ≥ 15pp, 预研 88-100% 命中率)',
        }

    # 强 λ 信号: λ_total < 1.5 必小, > 3.8 必大
    if lam_total < 1.5:
        return {
            'action': 'BET_UNDER',
            'pick': '小',
            'p_over_2_5_pct': round(p_over_2_5 * 100, 1),
            'p_under_2_5_pct': round(p_under_2_5 * 100, 1),
            'lam_total': round(lam_total, 2),
            'confidence_pp': round(confidence_pp, 1),
            'reason': f'λ_total={lam_total:.2f}<1.5, 必小球 (预研 100% 命中)',
        }
    if lam_total > 3.8:
        return {
            'action': 'BET_OVER',
            'pick': '大',
            'p_over_2_5_pct': round(p_over_2_5 * 100, 1),
            'p_under_2_5_pct': round(p_under_2_5 * 100, 1),
            'lam_total': round(lam_total, 2),
            'confidence_pp': round(confidence_pp, 1),
            'reason': f'λ_total={lam_total:.2f}>3.8, 必大球 (预研 100% 命中)',
        }

    # 弱信号 (5-15pp): 预研 80% 命中率, 但样本仅 5 场, 暂不投
    # 2.5-3.0 陷阱区: 43% 命中率, 绝对避开
    return {
        'action': 'SKIP',
        'pick': '-',
        'p_over_2_5_pct': round(p_over_2_5 * 100, 1),
        'p_under_2_5_pct': round(p_under_2_5 * 100, 1),
        'lam_total': round(lam_total, 2),
        'confidence_pp': round(confidence_pp, 1),
        'reason': f'弱信号 (|Δ|={confidence_pp:.1f}pp < 15pp, 2.5-3.0陷阱区), 跳过',
    }


if __name__ == '__main__':
    home = sys.argv[1] if len(sys.argv) > 1 else 'Switzerland'
    away = sys.argv[2] if len(sys.argv) > 2 else 'Bosnia and Herzegovina'
    is_home = '--home' in sys.argv
    hb = HOST_BONUS_BY_TEAM.get(home, 0.0) if is_home else 0.0

    r = predict_match(home, away, host_bonus=hb)
    if not r:
        print(f"❌ 错误: {away}")
        sys.exit(1)
    print(json.dumps(r, ensure_ascii=False, indent=2))
