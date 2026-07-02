from __future__ import annotations


def market_bucket(elo_h: float, elo_a: float, neutral: bool, market_strength: float) -> str:
    gap = abs(float(elo_h) - float(elo_a))
    if gap >= 220:
        g = 'gap_220+'
    elif gap >= 160:
        g = 'gap_160_219'
    elif gap >= 100:
        g = 'gap_100_159'
    elif gap >= 50:
        g = 'gap_50_99'
    else:
        g = 'gap_0_49'
    n = 'neutral' if neutral else 'nonneutral'
    s = 'strong' if market_strength >= 1.5 else 'weak' if market_strength <= 0.7 else 'mid'
    return f'{g}|{n}|{s}'


def market_weight_for_match(elo_h: float, elo_a: float, neutral: bool = True, market_strength: float = 1.0) -> float:
    """Return a conservative market blending weight.

    - Wider ELO gap => slightly more market reliance.
    - Neutral matches => slightly less market reliance.
    - Stronger market confidence => modestly more weight.
    """
    gap = abs(float(elo_h) - float(elo_a))
    base = 0.20
    if gap >= 220:
        base += 0.12
    elif gap >= 160:
        base += 0.09
    elif gap >= 100:
        base += 0.06
    elif gap >= 50:
        base += 0.03
    else:
        base -= 0.01

    if neutral:
        base -= 0.03
    else:
        base += 0.02

    strength = max(0.0, min(2.0, float(market_strength)))
    if strength >= 1.5:
        base += 0.03
    elif strength <= 0.7:
        base -= 0.03

    return max(0.10, min(0.42, base))
