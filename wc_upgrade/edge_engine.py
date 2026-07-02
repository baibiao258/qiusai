import numpy as np
from typing import List


def devig_decimal_odds(odds: List[float]) -> List[float]:
    inv = np.array([1.0 / x for x in odds], dtype=float)
    s = inv.sum()
    if s <= 0:
        return [0.0] * len(odds)
    return (inv / s).tolist()


def ev_decimal(p_model: float, odds: float) -> float:
    return p_model * (odds - 1.0) - (1.0 - p_model)


def kelly_fraction(p_model: float, odds: float, frac: float = 0.25) -> float:
    b = odds - 1.0
    q = 1.0 - p_model
    if b <= 0:
        return 0.0
    full = (b * p_model - q) / b
    return max(0.0, full) * frac
