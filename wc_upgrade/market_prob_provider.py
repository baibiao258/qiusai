#!/usr/bin/env python3
"""Unified market probability provider for 500.com EV/Kelly filtering.

Provides model probabilities for main_play / htft / score / totalgoals.
This is a sidecar module and does not touch the main WC training pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from pathlib import Path
from typing import Dict, Optional
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
HF_DIR = Path('/usr/local/lib/hermes-agent/models')
if str(HF_DIR) not in sys.path:
    sys.path.insert(0, str(HF_DIR))

from half_full_model import predict_half_full_probs


@dataclass
class MarketProbInput:
    market: str
    lambda_ft_home: float
    lambda_ft_away: float
    r_ht: float = 0.45
    handicap: float = 0.0
    max_goals: int = 10
    odds_1x2: Optional[Dict[str, float]] = None


def poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * exp(-lam) / factorial(k)


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values()) or 1.0
    return {k: v / s for k, v in d.items()}


def main_play_probs_from_lambdas(lam_h: float, lam_a: float, max_goals: int = 10, handicap: float = 0.0) -> Dict[str, float]:
    h = 0.0
    d = 0.0
    a = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson_pmf(hg, lam_h) * poisson_pmf(ag, lam_a)
            adj_home = hg + handicap
            if adj_home > ag:
                h += p
            elif adj_home == ag:
                d += p
            else:
                a += p
    return _normalize({'H': h, 'D': d, 'A': a})


def htft_probs_from_lambdas(lam_h: float, lam_a: float, r_ht: float = 0.45) -> Dict[str, float]:
    return predict_half_full_probs(lambda_ft_home=lam_h, lambda_ft_away=lam_a, r_ht=r_ht)


def score_probs_from_lambdas(lam_h: float, lam_a: float, max_goals: int = 10) -> Dict[str, float]:
    probs: Dict[str, float] = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson_pmf(hg, lam_h) * poisson_pmf(ag, lam_a)
            probs[f"{hg}:{ag}"] = probs.get(f"{hg}:{ag}", 0.0) + p
    return _normalize(probs)


def totalgoals_probs_from_lambdas(lam_h: float, lam_a: float, max_goals: int = 10) -> Dict[str, float]:
    probs: Dict[str, float] = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson_pmf(hg, lam_h) * poisson_pmf(ag, lam_a)
            tg = hg + ag
            probs[str(tg)] = probs.get(str(tg), 0.0) + p
    return _normalize(probs)


def get_market_probs(inp: MarketProbInput) -> Dict[str, float]:
    m = inp.market
    if m == 'main_play':
        return main_play_probs_from_lambdas(
            inp.lambda_ft_home,
            inp.lambda_ft_away,
            inp.max_goals,
            handicap=inp.handicap,
        )
    if m == 'htft':
        return htft_probs_from_lambdas(inp.lambda_ft_home, inp.lambda_ft_away, inp.r_ht)
    if m == 'score':
        return score_probs_from_lambdas(inp.lambda_ft_home, inp.lambda_ft_away, inp.max_goals)
    if m == 'totalgoals':
        return totalgoals_probs_from_lambdas(inp.lambda_ft_home, inp.lambda_ft_away, inp.max_goals)
    raise ValueError(f'unsupported market: {m}')
