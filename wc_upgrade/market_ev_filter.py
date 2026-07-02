#!/usr/bin/env python3
"""Market-level EV/Kelly filter for 500.com markets.

Input: 500 unified schema JSON produced by match_market_adapter.py
Output: filtered value signals for main_play / htft / score / totalgoals
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from market_prob_provider import MarketProbInput, get_market_probs
from market_model_probs import htft_probs_from_lambdas, score_probs_from_lambdas, totalgoals_probs_from_lambdas


LABELS_MAP = {
    'main_play': ['nspf', 'spf'],
    'htft': None,
    'score': None,
    'totalgoals': None,
}


@dataclass
class Signal:
    match: str
    market: str
    label: str
    odds: float
    p_model: float
    p_implied: float
    edge: float
    ev: float
    kelly_full: float
    quarter_kelly: float


def _safe_odds(x: float) -> float:
    return max(1.01, float(x))


def _implied(odds: Dict[str, float]) -> Dict[str, float]:
    vals = {k: 1.0 / _safe_odds(v) for k, v in odds.items()}
    s = sum(vals.values()) or 1.0
    return {k: v / s for k, v in vals.items()}


def _kelly(p: float, odds: float) -> float:
    odds = _safe_odds(odds)
    b = odds - 1.0
    return max(0.0, (p * odds - 1.0) / b)


def _load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default='stdout', choices=['stdout', 'file'])
    ap.add_argument('--output-file', default='')
    ap.add_argument('--ev-threshold', type=float, default=0.02)
    ap.add_argument('--edge-threshold', type=float, default=0.01)
    args = ap.parse_args()

    data = _load(args.input)
    rows = []

    for market, payload in data.get('markets', {}).items():
        for row in payload.get('result', []):
            odds = row.get('odds', {})
            if not odds:
                continue
            lam_h = float(row.get('lambda_ft_home', row.get('lambda_home', 1.35)))
            lam_a = float(row.get('lambda_ft_away', row.get('lambda_away', 1.15)))
            r_ht = float(row.get('r_ht', 0.45))
            # For main_play, odds is nested nspf/spf. For other markets, odds is flat.
            if market == 'main_play':
                for submarket in ('nspf', 'spf'):
                    sub = odds.get(submarket, {})
                    if not sub:
                        continue
                    handicap = float(row.get('rangqiu', 0) or 0)
                    p_map = get_market_probs(MarketProbInput(
                        market='main_play',
                        lambda_ft_home=lam_h,
                        lambda_ft_away=lam_a,
                        r_ht=r_ht,
                        max_goals=int(row.get('max_goals', 10)),
                        handicap=0.0 if submarket == 'nspf' else handicap,
                    ))
                    imp = _implied({k: float(v) for k, v in sub.items()})
                    for label, p_imp in imp.items():
                        p_model = float(p_map.get({'3': 'H', '1': 'D', '0': 'A'}[label], p_imp))
                        o = _safe_odds(float(sub[label]))
                        edge = p_model - p_imp
                        ev = p_model * o - 1.0
                        kf = _kelly(p_model, o)
                        if ev < args.ev_threshold or edge < args.edge_threshold:
                            continue
                        rows.append({
                            'market': market,
                            'submarket': submarket,
                            'match': row.get('match', ''),
                            'num': row.get('num', ''),
                            'label': label,
                            'odds': o,
                            'p_model': p_model,
                            'p_implied': p_imp,
                            'edge': edge,
                            'ev': ev,
                            'kelly_full': kf,
                            'quarter_kelly': 0.25 * kf,
                            'source': row.get('source', '500.com'),
                            'source_url': row.get('source_url', ''),
                            'rangqiu': handicap,
                        })
                continue

            flat_odds = row.get('odds', {})
            if isinstance(flat_odds, dict) and flat_odds:
                lam_h = float(row.get('lambda_ft_home', row.get('lambda_home', 1.35)))
                lam_a = float(row.get('lambda_ft_away', row.get('lambda_away', 1.15)))
                r_ht = float(row.get('r_ht', 0.45))
                if market == 'htft':
                    p_map = htft_probs_from_lambdas(lam_h, lam_a, r_ht=r_ht)
                elif market == 'score':
                    p_map = score_probs_from_lambdas(lam_h, lam_a, max_goals=int(row.get('max_goals', 10)))
                elif market == 'totalgoals':
                    p_map = totalgoals_probs_from_lambdas(lam_h, lam_a, max_goals=int(row.get('max_goals', 10)))
                else:
                    p_map = get_market_probs(MarketProbInput(
                        market='main_play',
                        lambda_ft_home=lam_h,
                        lambda_ft_away=lam_a,
                        r_ht=r_ht,
                        max_goals=int(row.get('max_goals', 10)),
                    ))
                for label, o_raw in flat_odds.items():
                    if label not in p_map:
                        continue
                    p_model = float(p_map[label])
                    o = _safe_odds(float(o_raw))
                    p_imp = 1.0 / o
                    edge = p_model - p_imp
                    ev = p_model * o - 1.0
                    kf = _kelly(p_model, o)
                    if ev < args.ev_threshold or edge < args.edge_threshold:
                        continue
                    rows.append({
                        'market': market,
                        'match': row.get('match', ''),
                        'num': row.get('num', ''),
                        'label': label,
                        'odds': o,
                        'p_model': p_model,
                        'p_implied': p_imp,
                        'edge': edge,
                        'ev': ev,
                        'kelly_full': kf,
                        'quarter_kelly': 0.25 * kf,
                        'source': row.get('source', '500.com'),
                        'source_url': row.get('source_url', ''),
                        'lambda_ft_home': lam_h,
                        'lambda_ft_away': lam_a,
                        'r_ht': r_ht,
                    })

    out = {'ok': True, 'input': args.input, 'count': len(rows), 'signals': rows}
    if args.output == 'stdout':
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        out_file = Path(args.output_file) if args.output_file else Path(args.input).with_suffix('.signals.json')
        out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'ok': True, 'output': str(out_file), 'count': len(rows)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
