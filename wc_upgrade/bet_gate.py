from dataclasses import dataclass
from typing import List, Dict


@dataclass
class GateConfig:
    ev_threshold: float = 0.03
    prob_floor: float = 0.08
    kelly_frac: float = 0.25
    max_per_bet: float = 0.02
    max_daily_exposure: float = 0.08


def risk_level(ev: float) -> str:
    if ev >= 0.08:
        return "A"
    if ev >= 0.05:
        return "B"
    if ev >= 0.03:
        return "C"
    return "DROP"


def apply_gate(candidates: List[Dict], cfg: GateConfig) -> List[Dict]:
    out = []
    used = 0.0
    for x in sorted(candidates, key=lambda z: z["ev"], reverse=True):
        if x["ev"] < cfg.ev_threshold:
            continue
        if x["p_model"] < cfg.prob_floor:
            continue
        lvl = risk_level(x["ev"])
        if lvl == "DROP":
            continue
        stake = min(x["kelly"], cfg.max_per_bet)
        if used + stake > cfg.max_daily_exposure:
            continue
        used += stake
        y = dict(x)
        y["risk_level"] = lvl
        y["stake_ratio"] = stake
        out.append(y)
    return out
