#!/usr/bin/env python3
"""Walk-forward backtest for 1X2 probabilities.

Rolling training windows:
- fit on historical prefix
- validate on next chunk
- repeat forward
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from calibration import fit_isotonic_multiclass, brier_multiclass
from edge_engine import devig_decimal_odds, ev_decimal, kelly_fraction
from bet_gate import GateConfig, apply_gate

CLASSES = [0, 1, 2]


@dataclass
class FoldResult:
    fold: int
    train_end: int
    test_start: int
    test_end: int
    brier_raw: float
    brier_cal: float
    acc_raw: float
    acc_cal: float
    bets: int
    roi: float


def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return brier_multiclass(p, y_true, CLASSES)


def run_walk_forward(proba_raw: np.ndarray, y_true: np.ndarray, odds_1x2: np.ndarray,
                     train_size: int = 800, test_size: int = 100, step_size: int = 100) -> Dict[str, Any]:
    n = len(y_true)
    if not (len(proba_raw) == len(odds_1x2) == n):
        raise ValueError("proba_raw, y_true, odds_1x2 must have same length")
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be > 0")

    cfg = GateConfig()
    folds: List[FoldResult] = []
    start = 0
    fold_id = 0

    while True:
        train_end = start + train_size
        test_start = train_end
        test_end = min(test_start + test_size, n)
        if test_end > n or train_end >= n:
            break

        p_train = proba_raw[start:train_end]
        y_train = y_true[start:train_end]
        p_test_raw = proba_raw[test_start:test_end]
        y_test = y_true[test_start:test_end]
        odds_test = odds_1x2[test_start:test_end]

        cal = fit_isotonic_multiclass(p_train, y_train, CLASSES)
        p_test = cal.predict_proba(p_test_raw)

        brier_raw = _brier(y_test, p_test_raw)
        brier_cal = _brier(y_test, p_test)
        acc_raw = float(np.mean(np.argmax(p_test_raw, axis=1) == y_test))
        acc_cal = float(np.mean(np.argmax(p_test, axis=1) == y_test))

        pnl = 0.0
        n_bets = 0
        for i in range(len(y_test)):
            fair = devig_decimal_odds(odds_test[i].tolist())
            cands = []
            for j, pick in enumerate(["H", "D", "A"]):
                pm = float(p_test[i, j])
                o = float(odds_test[i, j])
                cands.append({
                    "pick": pick,
                    "class_id": j,
                    "p_model": pm,
                    "p_market_fair": fair[j],
                    "edge": pm - fair[j],
                    "odds": o,
                    "ev": ev_decimal(pm, o),
                    "kelly": kelly_fraction(pm, o, frac=cfg.kelly_frac),
                })
            bets = apply_gate(cands, cfg)
            for b in bets:
                n_bets += 1
                stake = b["stake_ratio"]
                if y_test[i] == b["class_id"]:
                    pnl += stake * (b["odds"] - 1.0)
                else:
                    pnl -= stake

        folds.append(FoldResult(
            fold=fold_id,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            brier_raw=float(brier_raw),
            brier_cal=float(brier_cal),
            acc_raw=float(acc_raw),
            acc_cal=float(acc_cal),
            bets=n_bets,
            roi=float(pnl),
        ))

        fold_id += 1
        start += step_size
        if start + train_size + test_size > n:
            break

    if not folds:
        raise ValueError("No folds produced; increase data or decrease window sizes")

    out = {
        "ok": True,
        "folds": [f.__dict__ for f in folds],
        "summary": {
            "folds": len(folds),
            "brier_raw_mean": float(np.mean([f.brier_raw for f in folds])),
            "brier_cal_mean": float(np.mean([f.brier_cal for f in folds])),
            "acc_raw_mean": float(np.mean([f.acc_raw for f in folds])),
            "acc_cal_mean": float(np.mean([f.acc_cal for f in folds])),
            "bets": int(sum(f.bets for f in folds)),
            "roi": float(sum(f.roi for f in folds)),
        },
    }
    return out


if __name__ == "__main__":
    np.random.seed(42)
    n = 1200
    p = np.random.dirichlet([2, 2, 2], size=n)
    y = np.array([np.random.choice([0, 1, 2], p=row) for row in p])
    odds = np.random.uniform(1.6, 6.0, size=(n, 3))
    print(json.dumps(run_walk_forward(p, y, odds, train_size=800, test_size=100, step_size=100), ensure_ascii=False, indent=2))
