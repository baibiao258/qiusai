#!/usr/bin/env python3
"""
pipeline/probability.py — 纯数学函数（零 I/O，零模型依赖）

所有函数只接受数值参数，返回数值结果。不访问文件、不加载模型、
不调用外部 API。可直接 pytest 验证。

迁移来源: daily_jczq.py 中的同名函数。
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from config.settings import (
    HALF_FULL_MAX_FT,
    HALF_FULL_MAX_HT,
    HALF_FULL_R_HT,
    MAX_GOALS,
    XGB_WEIGHT_ALPHA,
    XGB_WEIGHT_BETA,
    XGB_WEIGHT_MAX as _XGB_WEIGHT_MAX,
    XGB_WEIGHT_MIN as _XGB_WEIGHT_MIN,
)

# ═══════════════════════════════════════════
#  泊松基础
# ═══════════════════════════════════════════


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson 概率质量函数 P(X=k | λ)."""
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def elo_expected(ra: float, rb: float) -> float:
    """Elo 期望得分 E = 1/(1+10^((rb-ra)/400))."""
    return 1.0 / (1 + 10 ** ((rb - ra) / 400))


# ═══════════════════════════════════════════
#  Dixon-Coles 校正
# ═══════════════════════════════════════════


def dc_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles tau 校正因子。只影响低比分 (x<=1, y<=1)。"""
    if x == 0 and y == 0:
        return 1 - rho * lam_h * lam_a
    elif x == 0 and y == 1:
        return 1 + rho * lam_h
    elif x == 1 and y == 0:
        return 1 + rho * lam_a
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _poisson_grid_dc(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    max_goals: int = MAX_GOALS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 (HG×AG) 概率矩阵及边际分布。

    返回 (matrix, home_marginal, away_marginal)，
    每个都是 (max_goals+1, max_goals+1) / (max_goals+1,) 的 ndarray。
    内部辅助函数，供 compute_rq_probs / compute_goals_distribution / compute_score_topn 共用。
    """
    size = max_goals + 1
    hp = np.array([poisson_pmf(k, lambda_home) for k in range(size)])
    ap = np.array([poisson_pmf(k, lambda_away) for k in range(size)])
    matrix = np.outer(hp, ap)

    if rho != 0.0:
        for hg in range(min(2, size)):
            for ag in range(min(2, size)):
                matrix[hg, ag] *= dc_tau(hg, ag, lambda_home, lambda_away, rho)

    total = matrix.sum()
    if total > 0:
        matrix /= total

    home_marginal = matrix.sum(axis=1)
    away_marginal = matrix.sum(axis=0)
    return matrix, home_marginal, away_marginal


# ═══════════════════════════════════════════
#  让球胜平负 (RQ)
# ═══════════════════════════════════════════


def rq_result_label(handicap: int, home_goals: int, away_goals: int) -> str:
    """handicap 下的结果标签。handicap > 0 表示主队受让。"""
    adj = home_goals + handicap - away_goals
    if adj > 0:
        return '让胜'
    if adj == 0:
        return '让平'
    return '让负'


def compute_rq_probs(
    lambda_home: float,
    lambda_away: float,
    handicap: int,
    rho: float = 0.0,
) -> Dict[str, float]:
    """让球胜平负概率分布。

    核心逻辑:
      1) Poisson×DC 枚举全场比分，按 handicap 归类
      2) 对 |handicap| 较大的结果向均匀分布做 shrinkage
         (回测: hcap=-1→33.3%, hcap=+1→42.9%, hcap=±2→0%, hcap=±3→0%)
    """
    probs = {'让胜': 0.0, '让平': 0.0, '让负': 0.0}
    shrink_factor = max(0.0, 1.0 - abs(handicap) * 0.15)

    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson_pmf(hg, lambda_home) * poisson_pmf(ag, lambda_away)
            if rho != 0.0:
                p *= dc_tau(hg, ag, lambda_home, lambda_away, rho)
            probs[rq_result_label(handicap, hg, ag)] += p

    total = sum(probs.values()) or 1.0
    raw = {k: v / total for k, v in probs.items()}

    if shrink_factor < 1.0:
        uniform = 1 / 3
        for k in raw:
            raw[k] = raw[k] * shrink_factor + uniform * (1 - shrink_factor)
        norm = sum(raw.values()) or 1.0
        for k in raw:
            raw[k] /= norm

    return raw


# ═══════════════════════════════════════════
#  赔率 → 隐含概率
# ═══════════════════════════════════════════


def implied_probs_from_odds(
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> Dict[str, float]:
    """欧赔 → 去除边际 (vig) 后的隐含概率。"""
    vals = []
    for odd in (odds_h, odds_d, odds_a):
        if odd and odd > 0:
            vals.append(1.0 / odd)
        else:
            vals.append(0.0)
    total = sum(vals) or 1.0
    return {
        'H': vals[0] / total,
        'D': vals[1] / total,
        'A': vals[2] / total,
    }


# ═══════════════════════════════════════════
#  总进球分布
# ═══════════════════════════════════════════


def compute_goals_distribution(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
) -> Dict[str, float]:
    """总进球概率分布。键为字符串 '0' ~ '12'。"""
    max_total = MAX_GOALS * 2
    dist: Dict[str, float] = {str(t): 0.0 for t in range(max_total + 1)}

    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson_pmf(hg, lambda_home) * poisson_pmf(ag, lambda_away)
            if rho != 0.0:
                p *= dc_tau(hg, ag, lambda_home, lambda_away, rho)
            total = hg + ag
            dist[str(total)] = dist.get(str(total), 0.0) + p

    return dist


# ═══════════════════════════════════════════
#  比分 Top-N
# ═══════════════════════════════════════════


def compute_score_topn(
    lambda_home: float,
    lambda_away: float,
    topn: int = 8,
    rho: float = 0.0,
) -> List[Tuple[str, float, int, int]]:
    """比分概率 Top-N。每项为 (比分标签, 概率, 主队进球, 客队进球)。

    示例: [('1:1', 0.15, 1, 1), ('2:1', 0.12, 2, 1), ...]
    """
    rows: List[Tuple[str, float, int, int]] = []
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson_pmf(hg, lambda_home) * poisson_pmf(ag, lambda_away)
            if rho != 0.0:
                p *= dc_tau(hg, ag, lambda_home, lambda_away, rho)
            rows.append((f'{hg}:{ag}', p, hg, ag))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:topn]


# ═══════════════════════════════════════════
#  半全场 (纯数学回退)
# ═══════════════════════════════════════════


def compute_htft_topn_math(
    lambda_home: float,
    lambda_away: float,
    topn: int = 6,
) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    """纯数学推导的半全场预测。

    将全场 λ 按节奏比 r_ht 拆分为上半场/下半场，
    分别枚举比分后交叉得到 9 宫格概率。
    这是原 compute_htft_topn 的 XGB 模型回退路径。
    外部依赖: half_full_model.predict_half_full_probs (纯数学)。
    """
    from half_full_model import predict_half_full_probs  # type: ignore[import-untyped]

    probs_cn = predict_half_full_probs(
        lambda_ft_home=lambda_home,
        lambda_ft_away=lambda_away,
        r_ht=HALF_FULL_R_HT,
        max_goals_ht=HALF_FULL_MAX_HT,
        max_goals_ft=HALF_FULL_MAX_FT,
    )
    rows = sorted(probs_cn.items(), key=lambda kv: kv[1], reverse=True)
    return rows[:topn], probs_cn


# ═══════════════════════════════════════════
#  动态 XGB/DC 融合权重 (基于熵)
# ═══════════════════════════════════════════


def compute_dynamic_xgb_weight(
    xgb_probs: Sequence[float],
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> Tuple[float, float, float]:
    """根据 XGB 预测的香农熵自动分配 XGB 与 DC 的权重。

    参数:
        xgb_probs: 长度为 3 的概率列表/元组 [p_h, p_d, p_a]。
        alpha: 基础权重 (默认 XGB_WEIGHT_ALPHA = 0.30)。
        beta: 置信度斜率 (默认 XGB_WEIGHT_BETA = 0.50)。

    返回:
        (xgb_w, dc_w, confidence)
        - xgb_w: XGB 模型的融合权重 [0.10, 0.90]
        - dc_w: DC 模型的融合权重 (1 - xgb_w)
        - confidence: 归一化置信度 [0, 1]
    """
    p = list(xgb_probs)
    total = sum(p)
    p = [v / total for v in p]
    e = -sum(v * math.log2(max(v, 1e-10)) for v in p)
    max_e = math.log2(3)
    confidence = 1.0 - e / max_e

    if alpha is None:
        alpha = XGB_WEIGHT_ALPHA
    if beta is None:
        beta = XGB_WEIGHT_BETA
    xgb_w = max(_XGB_WEIGHT_MIN, min(_XGB_WEIGHT_MAX, alpha + beta * confidence))
    return xgb_w, 1.0 - xgb_w, confidence


# ═══════════════════════════════════════════
#  模型验证指标
# ═══════════════════════════════════════════


def rps_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Ranked Probability Score (ordered 3-class H/D/A).

    y_true:  (N, 3) one-hot
    y_proba: (N, 3) predicted probabilities
    返回 mean RPS, lower is better.
    """
    cdf_true = np.cumsum(y_true, axis=1)
    cdf_pred = np.cumsum(y_proba, axis=1)
    n_cat = y_proba.shape[1] - 1
    return float(np.mean(np.sum((cdf_true - cdf_pred) ** 2, axis=1) / n_cat))


def brier_decomposition_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """One-vs-rest Brier decomposition averaged across classes.

    三个分量:
      - uncertainty:     climatology difficulty
      - resolution:      separation from the base rate
      - reliability:     calibration error
    check = uncertainty - resolution + reliability 应约等于 Brier.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.size == 0:
        return {'brier': 0.0, 'uncertainty': 0.0, 'resolution': 0.0, 'reliability': 0.0}

    n_samples, n_classes = y_true.shape
    brier = float(np.mean(np.sum((y_true - y_proba) ** 2, axis=1)))

    uncertainty = 0.0
    resolution = 0.0
    reliability = 0.0

    for c in range(n_classes):
        p = y_proba[:, c]
        o = y_true[:, c]
        base_rate = float(o.mean())
        uncertainty += base_rate * (1.0 - base_rate)

        if np.allclose(p, p[0]):
            obs = base_rate
            reliability += float(np.mean((obs - p) ** 2))
            continue

        quantiles = np.unique(
            np.quantile(p, np.linspace(0, 1, min(n_bins, n_samples) + 1))
        )
        if len(quantiles) <= 2:
            quantiles = np.array([p.min(), p.max()])

        quantiles[0] = quantiles[0] - 1e-12
        quantiles[-1] = quantiles[-1] + 1e-12
        bin_ids = np.digitize(p, quantiles[1:-1], right=True)

        for b in range(bin_ids.max() + 1):
            idx = bin_ids == b
            if not np.any(idx):
                continue
            w = float(idx.mean())
            p_bar = float(p[idx].mean())
            o_bar = float(o[idx].mean())
            reliability += w * (o_bar - p_bar) ** 2
            resolution += w * (o_bar - base_rate) ** 2

    uncertainty /= n_classes
    resolution /= n_classes
    reliability /= n_classes

    return {
        'brier': brier,
        'uncertainty': float(uncertainty),
        'resolution': float(resolution),
        'reliability': float(reliability),
        'check': float(uncertainty - resolution + reliability),
    }


def quick_validate(
    model_probs: List[Dict[str, float]],
    actual_results: List[str],
) -> Dict[str, Any]:
    """快速验证：模型概率 vs 实际结果。

    参数:
        model_probs:    每项含 {'H': p_h, 'D': p_d, 'A': p_a}
        actual_results: 每项为 'H' / 'D' / 'A'

    返回 Brier, RPS, 样本数, 及 Brier decomposition 分量。
    """
    y_true = np.zeros((len(actual_results), 3))
    y_proba = np.zeros((len(actual_results), 3))
    label_to_idx = {'A': 0, 'D': 1, 'H': 2}
    for i, (mp, act) in enumerate(zip(model_probs, actual_results)):
        y_proba[i] = [mp['A'], mp['D'], mp['H']]
        y_true[i, label_to_idx[act]] = 1

    brier = float(np.mean(np.sum((y_true - y_proba) ** 2, axis=1)))
    rps_val = rps_score(y_true, y_proba)
    decomp = brier_decomposition_multiclass(y_true, y_proba)
    return {
        'brier': brier,
        'rps': rps_val,
        'n': len(actual_results),
        **decomp,
    }


# ═══════════════════════════════════════════
#  格式化工具
# ═══════════════════════════════════════════


def format_pct(v: float) -> str:
    """小数 → 百分比字符串。例: 0.256 → '25.6%'"""
    return f'{v * 100:.1f}%'
