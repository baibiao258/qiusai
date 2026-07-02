#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

HTFT_KEYS = ["HH", "HD", "HA", "DH", "DD", "DA", "AH", "AD", "AA"]


@dataclass
class HTFTParseResult:
    source: str
    market: str
    match: str
    odds: Dict[str, float]
    timestamp: str
    raw_snippet: str
    confidence: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_numbers(text: str) -> List[float]:
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            pass
    return out


def _pick_best_9(nums: List[float]) -> List[float] | None:
    if len(nums) < 9:
        return None
    best = nums[:9]
    return best


def _build_result(source: str, match: str, raw_snippet: str, best: List[float], confidence: float) -> HTFTParseResult:
    odds = {k: float(best[i]) for i, k in enumerate(HTFT_KEYS)}
    return HTFTParseResult(
        source=source,
        market="htft",
        match=match,
        odds=odds,
        timestamp="2026-05-31T06:15:00+00:00",
        raw_snippet=raw_snippet[:3000],
        confidence=confidence,
    )


def normalize_htft_result(res: HTFTParseResult) -> dict:
    return {
        "source": res.source,
        "market": res.market,
        "match": res.match,
        "odds": res.odds,
        "timestamp": res.timestamp,
        "raw_snippet": res.raw_snippet,
        "confidence": res.confidence,
    }


def parse_vegasinsider_htft(html: str, match: str, home: str, away: str) -> HTFTParseResult:
    text = _clean_text(re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I))
    idx = text.find(home)
    if idx < 0:
        idx = text.find(away)
    if idx < 0:
        idx = text.find(match)
    if idx < 0:
        return _build_result("vegasinsider", match, text[:3000], [0.0] * 9, 0.05)
    seg = text[max(0, idx - 2500): idx + 5000]
    nums = _extract_numbers(seg)
    best = _pick_best_9(nums)
    if best is None:
        return _build_result("vegasinsider", match, seg, [0.0] * 9, 0.05)
    return _build_result("vegasinsider", match, seg, best, 0.72)


def parse_betexplorer_htft(html: str, match: str, home: str, away: str) -> HTFTParseResult:
    text = _clean_text(re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I))
    idx = text.find(home)
    if idx < 0:
        idx = text.find(away)
    if idx < 0:
        idx = text.find(match)
    if idx < 0:
        return _build_result("betexplorer", match, text[:3000], [0.0] * 9, 0.05)
    seg = text[max(0, idx - 3000): idx + 6000]
    nums = _extract_numbers(seg)
    best = _pick_best_9(nums)
    if best is None:
        return _build_result("betexplorer", match, seg, [0.0] * 9, 0.05)
    return _build_result("betexplorer", match, seg, best, 0.68)


def _read_html(path_or_html: str) -> str:
    p = Path(path_or_html)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8", errors="ignore")
    return path_or_html


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse HTFT odds from free-source HTML")
    ap.add_argument("--source", choices=["vegasinsider", "betexplorer"], required=True)
    ap.add_argument("--html-file", required=True, help="HTML file path or raw HTML string")
    ap.add_argument("--match", required=True, help='e.g. "Spain vs France"')
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    args = ap.parse_args()

    html = _read_html(args.html_file)
    if args.source == "vegasinsider":
        res = parse_vegasinsider_htft(html, args.match, args.home, args.away)
    else:
        res = parse_betexplorer_htft(html, args.match, args.home, args.away)
    print(res.to_json())


if __name__ == "__main__":
    main()
