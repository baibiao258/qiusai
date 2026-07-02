#!/usr/bin/env python3
"""/root/wc_2026_upgrade/collect_odds.py

统一赔率采集器（最小可用版）
- 主源：VegasInsider
- 辅源：BetExplorer / FotMob
- 输出标准 JSON
- 含本地缓存和简单一致性检查

说明：
1) 这是最小可用版本，先支持冠军盘/单场通用抓取框架。
2) 页面结构会变，抓取失败时会在 raw_snippets 中保留可审计片段。
3) 对足球半全场(H/T/FT)场景，后续可扩展 source-specific extractor。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from htft_parser import parse_vegasinsider_htft, parse_betexplorer_htft, normalize_htft_result


DEFAULT_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


@dataclass
class SourceResult:
    source: str
    url: str
    ts: int
    ok: bool
    odds: Optional[Dict[str, float]] = None
    raw_snippet: Optional[str] = None
    notes: Optional[List[str]] = None


@dataclass
class ConsensusResult:
    market: str
    query: str
    ts: int
    consensus_odds: Optional[Dict[str, float]]
    source_results: List[Dict]
    consistency: Dict
    cache_key: str


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, dir_path: str, ttl_seconds: int):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def load(self, key: str):
        p = self.path(key)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if int(time.time()) - int(d.get("ts", 0)) <= self.ttl:
                return d
        except Exception:
            return None
        return None

    def save(self, key: str, data: dict):
        self.path(key).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class OddsCollector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_UA})

    def fetch(self, url: str) -> str:
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        return r.text

    def extract_vegasinsider(self, html: str, query: str, url: str) -> SourceResult:
        # 轻量级：找页面里所有看起来像赔率的数字；保留片段，不强行假装已精确识别
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n", strip=True)
        q = re.sub(r"\s+", "", query)
        idx = re.sub(r"\s+", "", text).find(q)
        if idx >= 0:
            clean = re.sub(r"\s+", "", text)
            seg = clean[max(0, idx - 2500): idx + 5000]
        else:
            seg = re.sub(r"\s+", " ", text[:12000])
        nums = re.findall(r"\b\d+(?:\.\d{1,3})\b", seg)
        odds = {}
        # 保守策略：只在明显的冠军盘页面中，尝试从前 20 个数中形成一个结构；否则只给 raw_snippet
        if len(nums) >= 3:
            # 这里不硬解析成错误赔率；只输出片段供上层规则/后续精抽
            return SourceResult(
                source="vegasinsider",
                url=url,
                ts=int(time.time()),
                ok=True,
                odds=None,
                raw_snippet=seg[:3000],
                notes=["extracted textual snippet; odds parsing deferred to market-specific parser"],
            )
        return SourceResult(source="vegasinsider", url=url, ts=int(time.time()), ok=False, raw_snippet=seg[:3000], notes=["no odds-like tokens found"])

    def extract_betexplorer(self, html: str, query: str, url: str) -> SourceResult:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n", strip=True)
        seg = re.sub(r"\s+", " ", text[:12000])
        return SourceResult(source="betexplorer", url=url, ts=int(time.time()), ok=True, odds=None, raw_snippet=seg[:3000], notes=["placeholder extractor"])

    def extract_fotmob(self, html: str, query: str, url: str) -> SourceResult:
        # FotMob 有 __NEXT_DATA__，先输出可审计片段
        m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S)
        if m:
            raw = m.group(1)
            return SourceResult(source="fotmob", url=url, ts=int(time.time()), ok=True, odds=None, raw_snippet=raw[:3000], notes=["__NEXT_DATA__ captured"])
        return SourceResult(source="fotmob", url=url, ts=int(time.time()), ok=False, raw_snippet=html[:3000], notes=["__NEXT_DATA__ not found"])

    def extract_htft(self, html: str, query: str, url: str) -> SourceResult:
        # HTFT 旁路：委托给专用 parser，返回标准 SourceResult
        # query 约定为 "HOME vs AWAY" 或任意可读比赛名
        parts = [x.strip() for x in query.split("vs")]
        home = parts[0] if parts else query
        away = parts[1] if len(parts) > 1 else query
        if "vegasinsider" in url:
            res = parse_vegasinsider_htft(html, query, home, away)
        else:
            res = parse_betexplorer_htft(html, query, home, away)
        norm = normalize_htft_result(res)
        return SourceResult(
            source=norm["source"],
            url=url,
            ts=int(time.time()),
            ok=True,
            odds=norm["odds"],
            raw_snippet=norm["raw_snippet"],
            notes=[f"htft:{norm['market']}", f"confidence={norm['confidence']}"]
        )

    def collect(self, query: str, market: str = "futures") -> ConsensusResult:
        cache_cfg = self.cfg.get("cache", {})
        cache = Cache(cache_cfg.get("dir", "/tmp/wc2026_cache"), int(cache_cfg.get("ttl_seconds", 900)))
        cache_key = sha1(json.dumps({"query": query, "market": market}, ensure_ascii=False, sort_keys=True))
        cached = cache.load(cache_key)
        if cached:
            return ConsensusResult(**cached)

        sources = self.cfg.get("sources", {})
        results: List[SourceResult] = []

        if market == "htft":
            urls = []
            if sources.get("vegasinsider", {}).get("enabled", True):
                urls.append(("vegasinsider", sources["vegasinsider"]["base_url"].rstrip("/") + "/soccer/odds/futures/"))
            if sources.get("betexplorer", {}).get("enabled", True):
                urls.append(("betexplorer", sources["betexplorer"]["base_url"].rstrip("/") + "/soccer/"))

            for src, url in urls:
                try:
                    html = self.fetch(url)
                    results.append(self.extract_htft(html, query, url))
                except Exception as e:
                    results.append(SourceResult(source=src, url=url, ts=int(time.time()), ok=False, notes=[str(e)]))

            consistency = {
                "min_sources_agree": 1,
                "max_relative_spread": 0.50,
                "agree_count": sum(1 for r in results if r.ok and r.odds),
                "passed": any(r.ok and r.odds for r in results),
            }
            out = ConsensusResult(
                market=market,
                query=query,
                ts=int(time.time()),
                consensus_odds=None,
                source_results=[asdict(r) for r in results],
                consistency=consistency,
                cache_key=cache_key,
            )
            cache.save(cache_key, asdict(out))
            return out

        if sources.get("vegasinsider", {}).get("enabled", True):
            url = sources["vegasinsider"]["base_url"].rstrip("/") + "/soccer/odds/futures/"
            try:
                html = self.fetch(url)
                results.append(self.extract_vegasinsider(html, query, url))
            except Exception as e:
                results.append(SourceResult(source="vegasinsider", url=url, ts=int(time.time()), ok=False, notes=[str(e)]))

        if sources.get("betexplorer", {}).get("enabled", True):
            url = sources["betexplorer"]["base_url"].rstrip("/") + "/soccer/"
            try:
                html = self.fetch(url)
                results.append(self.extract_betexplorer(html, query, url))
            except Exception as e:
                results.append(SourceResult(source="betexplorer", url=url, ts=int(time.time()), ok=False, notes=[str(e)]))

        if sources.get("fotmob", {}).get("enabled", True):
            url = sources["fotmob"]["base_url"].rstrip("/") + "/"
            try:
                html = self.fetch(url)
                results.append(self.extract_fotmob(html, query, url))
            except Exception as e:
                results.append(SourceResult(source="fotmob", url=url, ts=int(time.time()), ok=False, notes=[str(e)]))

        consistency = {
            "min_sources_agree": 2,
            "max_relative_spread": 0.18,
            "agree_count": 0,
            "passed": False,
        }
        out = ConsensusResult(
            market=market,
            query=query,
            ts=int(time.time()),
            consensus_odds=None,
            source_results=[asdict(r) for r in results],
            consistency=consistency,
            cache_key=cache_key,
        )
        cache.save(cache_key, asdict(out))
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/wc_2026_upgrade/odds_collector_config.json")
    ap.add_argument("--query", required=True, help="e.g. 'World Cup Winner' or match label")
    ap.add_argument("--market", default="futures")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    collector = OddsCollector(cfg)
    res = collector.collect(args.query, args.market)
    data = asdict(res)
    txt = json.dumps(data, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
