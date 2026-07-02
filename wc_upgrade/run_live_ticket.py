#!/usr/bin/env python3
# /root/wc_2026_upgrade/run_live_ticket.py
# 用法:
# python /root/wc_2026_upgrade/run_live_ticket.py \
#   --input /root/wc_2026_upgrade/matches_input.json \
#   --output /root/wc_2026_upgrade/ticket_output.json

import argparse
import json
from typing import Dict, List

from edge_engine import devig_decimal_odds, ev_decimal, kelly_fraction
from bet_gate import GateConfig, apply_gate

# class mapping: 0=H,1=D,2=A
PICKS = ["H", "D", "A"]


def build_candidates(match: Dict, cfg: GateConfig) -> List[Dict]:
    odds = match["odds_1x2"]  # [H,D,A]
    probs = match["proba_1x2"]  # calibrated model probs [H,D,A]

    if len(odds) != 3 or len(probs) != 3:
        raise ValueError(f"match {match.get('match_id')} odds/proba must be len=3")

    fair = devig_decimal_odds(odds)
    cands = []
    for j, p in enumerate(PICKS):
        p_m = float(probs[j])
        o = float(odds[j])
        ev = ev_decimal(p_m, o)
        kf = kelly_fraction(p_m, o, frac=cfg.kelly_frac)
        cands.append(
            {
                "match_id": match.get("match_id"),
                "home": match.get("home"),
                "away": match.get("away"),
                "pick": p,
                "class_id": j,
                "p_model": p_m,
                "p_market_fair": fair[j],
                "edge": p_m - fair[j],
                "odds": o,
                "ev": ev,
                "kelly": kf,
            }
        )
    return cands


def run_live(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg_dict = data.get("gate", {})
    cfg = GateConfig(
        ev_threshold=float(cfg_dict.get("ev_threshold", 0.03)),
        prob_floor=float(cfg_dict.get("prob_floor", 0.08)),
        kelly_frac=float(cfg_dict.get("kelly_frac", 0.25)),
        max_per_bet=float(cfg_dict.get("max_per_bet", 0.02)),
        max_daily_exposure=float(cfg_dict.get("max_daily_exposure", 0.08)),
    )

    all_candidates = []
    for m in data.get("matches", []):
        all_candidates.extend(build_candidates(m, cfg))

    selected = apply_gate(all_candidates, cfg)

    # 分级汇总
    buckets = {"A": [], "B": [], "C": []}
    for x in selected:
        buckets[x["risk_level"]].append(x)

    # 排序
    for k in buckets:
        buckets[k] = sorted(buckets[k], key=lambda z: z["ev"], reverse=True)

    out = {
        "ok": True,
        "config": {
            "ev_threshold": cfg.ev_threshold,
            "prob_floor": cfg.prob_floor,
            "kelly_frac": cfg.kelly_frac,
            "max_per_bet": cfg.max_per_bet,
            "max_daily_exposure": cfg.max_daily_exposure,
        },
        "summary": {
            "candidates": len(all_candidates),
            "selected": len(selected),
            "selected_A": len(buckets["A"]),
            "selected_B": len(buckets["B"]),
            "selected_C": len(buckets["C"]),
            "total_exposure": round(sum(x["stake_ratio"] for x in selected), 6),
        },
        "tickets": {
            "A": buckets["A"],
            "B": buckets["B"],
            "C": buckets["C"],
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": True, "output": output_path, "selected": out["summary"]["selected"]}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input json path")
    ap.add_argument("--output", required=True, help="output json path")
    args = ap.parse_args()
    run_live(args.input, args.output)


if __name__ == "__main__":
    main()
