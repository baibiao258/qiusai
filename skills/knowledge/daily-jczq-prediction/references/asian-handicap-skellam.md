# Asian Handicap Skellam Calculation (2026-06-08)

## Overview

Asian Handicap probabilities computed via the **Skellam distribution** (bivariate Poisson difference distribution). This converts the model's λ outputs — `λ_home` and `λ_away` — into exact win/push/lose probabilities for any handicap line between -3.0 and +3.0.

## Why Asian Handicap

- **Lower overround** than 竞彩 SPF (1.5-3% vs 13%)
- **Two-way market** compresses 1X2 → win/lose (half-win/loss for quarter balls)
- **Direct EV hunting**: model λ → AH probability → compare with bookmaker AH odds

## Mathematical Foundation

### Skellam PMF

```
P(净胜球 = k) = e^{-(λ₁+λ₂)} · (λ₁/λ₂)^{k/2} · I_{|k|}(2·√(λ₁·λ₂))
```

where `I_{|k|}` is the modified Bessel function of the first kind (`scipy.special.iv`).

### Handicap Types

| Type | Fraction | Win Condition | Push/Half Condition |
|---|---|---|---|
| Integer | 0.0 | net > h | net = h (full push/refund) |
| Half-ball | 0.5 | net > h (equivalently net > h_int+0.5) | No push |
| Quarter-ball | 0.25 | net ≥ h_int+1 | net = h_int (half loss) |
| Three-quarter | 0.75 | net ≥ h_int+2 + 0.5 × net = h_int+1 | net = h_int+1 (half win = half loss, mirror) |

### Effective Probability

For fair odds calculation:

- **Integer/half**: `eff = P(win) / (P(win) + P(lose))` (only 2 outcomes)
- **Quarter**: `eff = P(win) + 0.5 × P(push)` (push = half loss, reduces effective stake)

## Implementation

File: `/root/asian_handicap.py`

### Key Functions

```python
from asian_handicap import ah_probs, find_ah_odds, scan_ah_value

# 1. Basic probability for one handicap
probs = ah_probs(lam_home=2.0, lam_away=0.8, handicap=1.0)
# → {'handicap': 1.0, 'prob_win': 0.41, 'prob_push': 0.25,
#     'prob_lose': 0.35, 'effective_prob': 0.53, 'fair_odds': 1.89}

# 2. EV calculation against market odds
ev = find_ah_odds(lam_home, lam_away, handicap=1.0, market_odds=1.95)
# → {'handicap': 1.0, 'ev': 0.034, 'kelly_pct': 0.009, 'edge': 'positive'}

# 3. Batch scan across available lines
market = [(0.5, 1.85), (0.75, 2.05), (1.0, 1.95)]
results = scan_ah_value(lam_home, lam_away, market)
# → [values with EV>0, sorted by EV descending]
```

### Negative Handicaps

When `handicap < 0` (e.g. `-0.5` = away team favored by 0.5):
- The function internally flips to the away perspective
- Probabilities are always from the bettor's side (the team being backed with λ outputs)
- So for `ah_probs(1.5, 1.2, -0.5)`, the result is the probability of the **away team covering -0.5**

## Verification Checklist

- [ ] win + push + lose ≈ 1.0 (within rounding)
- [ ] Monotonicity: higher handicap → lower win probability
- [ ] Symmetry: `ah_probs(λ₁, λ₂, h)` ≈ 1 - `ah_probs(λ₂, λ₁, -h)`
- [ ] Integer handicap push prob = P(net = h) from raw Poisson
- [ ] No handicap (h=0): effective_prob ≥ win probability (due to push protection)

## Known Limitations

1. **Poisson independence assumption**: Skellam assumes independence of home/away goals, which is unrealistic for correlated outcomes. Bivariate Poisson (Dixon-Coles) corrects this with ρ parameter, but that's not available in pure Skellam.

2. **Discrete goal space truncation**: max_goals=15 cutoff may underestimate extreme tail probabilities for very high λ (e.g., λ_h=5.0). Bump to 25 for high-scoring leagues.

3. **Quarter-ball push symmetry**: The code treats half-win = half-loss for three-quarter balls, which matches standard AH settlement rules. Some bookmakers may use slightly different rounding.

4. **No dynamic overround adjustment**: `find_ah_odds()` defaults to 3% overround. Different markets (Pinnacle ~1%, SBOBET ~2%) should be adjusted.
