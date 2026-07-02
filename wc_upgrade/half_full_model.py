#!/usr/bin/env python3
"""
Phase 1 MVP - Half/Full Time 9-class probability engine.

Input: full-time expected goals (lambda_ft_home, lambda_ft_away) + r_ht.
Output: normalized 9-class probabilities in fixed order:
["HH", "HD", "HA", "DH", "DD", "DA", "AH", "AD", "AA"]

Design goals:
- Sidecar module, no impact on existing SPF/WC pipelines.
- Deterministic, auditable JSON output.
- Strict probability validation and normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, factorial
from typing import Dict, List, Tuple
import json

LABELS_9: List[str] = ["HH", "HD", "HA", "DH", "DD", "DA", "AH", "AD", "AA"]


@dataclass(frozen=True)
class MatchInput:
    match_id: str
    home_team: str
    away_team: str
    lambda_ft_home: float
    lambda_ft_away: float
    r_ht: float = 0.45
    kickoff_utc: str | None = None


def _poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam < 0:
        raise ValueError(f"lambda must be >= 0, got {lam}")
    return (lam ** k) * exp(-lam) / factorial(k)


def _result_probs_from_lambdas(
    lam_home: float,
    lam_away: float,
    max_goals: int = 10,
) -> Dict[str, float]:
    """
    Compute 1X2 probabilities from independent Poisson goals.
    States:
      H: home leads/wins
      D: draw
      A: away leads/wins
    """
    p_h = 0.0
    p_d = 0.0
    p_a = 0.0

    # Precompute PMF arrays for speed and reproducibility
    home_p = [_poisson_pmf(i, lam_home) for i in range(max_goals + 1)]
    away_p = [_poisson_pmf(j, lam_away) for j in range(max_goals + 1)]

    # Tail mass correction: put residual into max_goals bucket
    # so total mass remains close to 1 even with finite truncation.
    home_res = max(0.0, 1.0 - sum(home_p))
    away_res = max(0.0, 1.0 - sum(away_p))
    home_p[-1] += home_res
    away_p[-1] += away_res

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = home_p[i] * away_p[j]
            if i > j:
                p_h += p
            elif i == j:
                p_d += p
            else:
                p_a += p

    total = p_h + p_d + p_a
    if total <= 0:
        raise ValueError("invalid probability total in result-state computation")

    return {
        "H": p_h / total,
        "D": p_d / total,
        "A": p_a / total,
    }


def _normalize_probs_9(probs_9: Dict[str, float]) -> Dict[str, float]:
    for k in LABELS_9:
        if k not in probs_9:
            raise KeyError(f"missing label in probs_9: {k}")
        if probs_9[k] < 0:
            raise ValueError(f"negative probability for {k}: {probs_9[k]}")

    s = sum(probs_9[k] for k in LABELS_9)
    if s <= 0:
        raise ValueError("sum(probs_9) <= 0")
    return {k: probs_9[k] / s for k in LABELS_9}


def predict_half_full_probs(
    lambda_ft_home: float,
    lambda_ft_away: float,
    r_ht: float = 0.45,
    max_goals_ht: int = 8,
    max_goals_ft: int = 10,
    team_r_ht_home: float | None = None,
    team_r_ht_away: float | None = None,
) -> Dict[str, float]:
    """
    Core MVP engine:
      1) derive half-time lambdas from full-time lambdas via r_ht
         - team_r_ht_home/away: team-specific r_ht (overrides global r_ht)
      2) compute HT 1X2 and FT 1X2 probabilities
      3) approximate joint(HT,FT) with product of marginals
      4) map to 9 labels and normalize

    Note: independence approximation is intentional for Phase 1 speed.
    Phase 2 can upgrade to state-dependent transition model.
    """
    if lambda_ft_home < 0 or lambda_ft_away < 0:
        raise ValueError("lambda_ft must be non-negative")
    if not (0 < r_ht < 1):
        raise ValueError(f"r_ht must be in (0,1), got {r_ht}")

    # ── 球队级 r_ht ──
    # 防守型球队 (=对手全场进球少, 但半场进球比例可能不同)
    # 使用简单的加权: 主队用 team_r_ht_home, 客队用 team_r_ht_away
    r_ht_h = team_r_ht_home if team_r_ht_home is not None else r_ht
    r_ht_a = team_r_ht_away if team_r_ht_away is not None else r_ht
    # 钳位安全
    r_ht_h = max(0.1, min(0.9, r_ht_h))
    r_ht_a = max(0.1, min(0.9, r_ht_a))

    lam_ht_home = lambda_ft_home * r_ht_h
    lam_ht_away = lambda_ft_away * r_ht_a

    ht = _result_probs_from_lambdas(lam_ht_home, lam_ht_away, max_goals=max_goals_ht)
    ft = _result_probs_from_lambdas(lambda_ft_home, lambda_ft_away, max_goals=max_goals_ft)

    probs_9 = {
        "HH": ht["H"] * ft["H"],
        "HD": ht["H"] * ft["D"],
        "HA": ht["H"] * ft["A"],
        "DH": ht["D"] * ft["H"],
        "DD": ht["D"] * ft["D"],
        "DA": ht["D"] * ft["A"],
        "AH": ht["A"] * ft["H"],
        "AD": ht["A"] * ft["D"],
        "AA": ht["A"] * ft["A"],
    }

    return _normalize_probs_9(probs_9)


def build_protocol_output(
    payload: MatchInput,
    max_goals_ht: int = 8,
    max_goals_ft: int = 10,
) -> Dict:
    probs_9 = predict_half_full_probs(
        lambda_ft_home=payload.lambda_ft_home,
        lambda_ft_away=payload.lambda_ft_away,
        r_ht=payload.r_ht,
        max_goals_ht=max_goals_ht,
        max_goals_ft=max_goals_ft,
    )

    sum_check = sum(probs_9[k] for k in LABELS_9)
    top3 = sorted(probs_9.items(), key=lambda x: x[1], reverse=True)[:3]

    kickoff_utc = payload.kickoff_utc
    if kickoff_utc is None:
        kickoff_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    return {
        "protocol_version": "1.0.0",
        "match_id": payload.match_id,
        "kickoff_utc": kickoff_utc,
        "home_team": payload.home_team,
        "away_team": payload.away_team,
        "lambda_ft": {
            "home": float(payload.lambda_ft_home),
            "away": float(payload.lambda_ft_away),
        },
        "r_ht": float(payload.r_ht),
        "lambda_ht": {
            "home": float(payload.lambda_ft_home * payload.r_ht),
            "away": float(payload.lambda_ft_away * payload.r_ht),
        },
        "probs_9": {k: float(probs_9[k]) for k in LABELS_9},
        "sum_check": float(sum_check),
        "top3": [[k, float(v)] for k, v in top3],
    }


def _demo() -> None:
    sample = MatchInput(
        match_id="DEMO_PL_ARS_CHE",
        home_team="Arsenal",
        away_team="Chelsea",
        lambda_ft_home=1.62,
        lambda_ft_away=0.97,
        r_ht=0.45,
        kickoff_utc="2026-08-15T16:30:00Z",
    )
    out = build_protocol_output(sample)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _demo()
