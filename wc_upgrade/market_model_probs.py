#!/usr/bin/env python3
"""Model probability sidecar for market EV/Kelly filtering.

Inputs: lambda_ft_home / lambda_ft_away / optional 1X2 probs.
Outputs: score / totalgoals / htft market probabilities.
"""
from __future__ import annotations

from math import exp, factorial
from typing import Dict, List

from half_full_model import predict_half_full_probs


def poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * exp(-lam) / factorial(k)


def score_probs_from_lambdas(lam_h: float, lam_a: float, max_goals: int = 10) -> Dict[str, float]:
    probs: Dict[str, float] = {}
    total = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson_pmf(hg, lam_h) * poisson_pmf(ag, lam_a)
            probs[f'{hg}:{ag}'] = probs.get(f'{hg}:{ag}', 0.0) + p
            total += p
    if total <= 0:
        return probs
    return {k: v / total for k, v in probs.items()}


def totalgoals_probs_from_lambdas(lam_h: float, lam_a: float, max_goals: int = 10) -> Dict[str, float]:
    probs: Dict[str, float] = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson_pmf(hg, lam_h) * poisson_pmf(ag, lam_a)
            tg = hg + ag
            probs[str(tg)] = probs.get(str(tg), 0.0) + p
    s = sum(probs.values()) or 1.0
    return {k: v / s for k, v in probs.items()}


def htft_probs_from_lambdas(lam_h: float, lam_a: float, r_ht: float = 0.45) -> Dict[str, float]:
    return predict_half_full_probs(lambda_ft_home=lam_h, lambda_ft_away=lam_a, r_ht=r_ht)
