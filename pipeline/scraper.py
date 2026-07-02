"""500.com odds scraper – pure I/O layer, zero prediction logic.

Replaces _fetch_live_odds_map() and scrape_500_odds_today() previously
embedded in daily_jczq.py.

Public API
----------
scrape_500_odds_today() -> list[dict]
    Async-subprocess scraper for all 5 play-types.
    Returns [] on full circuit-breaker trip.

fetch_live_odds_map() -> dict | None
    live.500.com average euro odds fallback.
    Returns None on failure (non-fatal).

apply_euro_fallback(bundle, market_row) -> dict
    Enrich a prediction bundle with euro odds reference
    when nspf is absent. Pure dict mutation, no I/O.
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from datetime import date, datetime
from typing import Optional

from config.settings import DATA_DIR

# ── 内部常量 ────────────────────────────────────────────────────────────────

_ASYNC_SCRAPER = '/root/wc_2026_upgrade/async_500_scraper.py'
_BREAKER_LOG   = f'{DATA_DIR}/500breaker.log'
_SCRAPER_TIMEOUT = 45  # seconds

# 半全场 raw key → 中文标签
_HTFT_RAW_MAP = {
    '3-3': '胜胜', '3-1': '胜平', '3-0': '胜负',
    '1-3': '平胜', '1-1': '平平', '1-0': '平负',
    '0-3': '负胜', '0-1': '负平', '0-0': '负负',
}
_HTFT_ORDER = ['胜胜', '胜平', '胜负', '平胜', '平平', '平负', '负胜', '负平', '负负']


# ── 公开 API ─────────────────────────────────────────────────────────────────

def fetch_live_odds_map() -> Optional[dict]:
    """Fetch average euro odds from live.500.com as SPF fallback.

    Returns
    -------
    dict[code, {'h': float, 'd': float, 'a': float}]
        Keyed by match code like '周二201'.
    None
        On any network or parse failure (non-fatal).
    """
    try:
        req = urllib.request.Request(
            'https://live.500.com/',
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
        )
        html = urllib.request.urlopen(req, timeout=10).read().decode('gbk', errors='replace')

        m = re.search(r'var liveOddsList = ({.*?});', html, re.DOTALL)
        if not m:
            return None
        odds_by_fid: dict = json.loads(m.group(1))

        code_to_fid: dict[str, str] = {}
        for match in re.finditer(
            r'value="(\d+)"\s*/>\s*(周[一二三四五六七日]\d+)',
            html,
        ):
            code_to_fid[match.group(2)] = match.group(1)

        result: dict = {}
        for code, fid in code_to_fid.items():
            entry = odds_by_fid.get(fid, {})
            euro_avg = entry.get('0', [])
            if len(euro_avg) >= 3 and float(euro_avg[0]) > 1:
                result[code] = {
                    'h': float(euro_avg[0]),
                    'd': float(euro_avg[1]),
                    'a': float(euro_avg[2]),
                }

        if result:
            print(f'    🌐 live.500.com 平均欧赔兜底加载: {len(result)} 场')
            return result

    except Exception as exc:
        print(f'    ⚠️ live.500.com 加载失败: {exc}')

    return None


def scrape_500_odds_today() -> list[dict]:
    """Scrape today's 500.com JCZQ odds via async subprocess.

    Covers 5 play-types (SPF / NSPF / BF / JQS / BQC) for fixtures
    that are open for betting today.

    Returns
    -------
    list[dict]
        One dict per fixture; empty list on circuit-breaker trip.

    Notes
    -----
    Circuit-breaker conditions (all return []):
    - subprocess timeout > 45 s
    - non-zero exit code
    - JSON decode error
    - empty result set from scraper
    """
    date_str = date.today().isoformat()

    # ── 异步并发抓取 4 个玩法页面 ─────────────────────────────────────────
    try:
        proc = subprocess.run(
            ['python3', _ASYNC_SCRAPER, date_str, '269,270,271,272'],
            capture_output=True, text=True, timeout=_SCRAPER_TIMEOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError(f'exit code {proc.returncode}: {proc.stderr[:200]}')
        data = json.loads(proc.stdout)
        if not data.get('ok') or not data.get('result'):
            raise RuntimeError('返回空结果')
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            json.JSONDecodeError, RuntimeError) as exc:
        warn = f'[500BREAKER] 异步抓取失败: {exc}'
        print(f'    ⚠️ {warn}')
        with open(_BREAKER_LOG, 'a') as fh:
            fh.write(f'{datetime.now().isoformat()} {warn}\n')
        print('    🔴 500.com 全量熔断, 跳过市场校准')
        return []

    raw_matches: list[dict] = data['result']
    match_by_code = {m.get('no', ''): m for m in raw_matches if m.get('no')}

    if not match_by_code:
        print('    🔴 500.com 无有效赛事数据, 跳过市场校准')
        return []

    # ── 欧赔兜底 ──────────────────────────────────────────────────────────
    live_odds_map = fetch_live_odds_map()
    if live_odds_map:
        print(f'    🌐 live.500.com 平均欧赔兜底: {len(live_odds_map)} 场')

    return [_parse_row(code, row, live_odds_map) for code, row in match_by_code.items()]


def apply_euro_fallback(bundle: dict, market_row: Optional[dict]) -> dict:
    """Mark SPF as unavailable when nspf is empty; attach euro ref odds.

    No I/O. Mutates bundle in-place and returns it.
    When nspf is absent, the SPF odds are meaningless (the play wasn't
    even listed), so we record the euro reference without overwriting.
    """
    if not market_row or not market_row.get('nspf_empty'):
        return bundle

    euro = bundle.get('current_euro_odds_500', {})
    if euro and euro.get('home', 0) > 1:
        bundle['euro_odds_ref'] = euro
        bundle['model_note_append'] = '+SPF未开售(仅开让球)'
    return bundle


# ── 内部解析 ─────────────────────────────────────────────────────────────────

def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _parse_row(code: str, row: dict, live_odds_map: Optional[dict]) -> dict:
    """Parse one raw scraper row into a normalised market dict."""
    home_cn = re.sub(r'^\[\d+\]?', '', row.get('home', '').strip()).strip()
    away_raw = row.get('away', '').replace(' 单关', '').strip()
    away_cn = re.sub(r'\[\d+\]?$', '', away_raw).strip()

    odds   = row.get('odds', {})
    spf_raw  = odds.get('spf', {})
    nspf_raw = odds.get('nspf', {})

    handicap = int(row.get('handicap') or row.get('rangqiu') or 0)

    # ── SPF / NSPF 赔率映射规则 ─────────────────────────────────────────
    if handicap != 0 and nspf_raw and nspf_raw.get('3'):
        std_h = _to_float(nspf_raw.get('3'))
        std_d = _to_float(nspf_raw.get('1'))
        std_a = _to_float(nspf_raw.get('0'))
        rq_h  = _to_float(spf_raw.get('3'))
        rq_d  = _to_float(spf_raw.get('1'))
        rq_a  = _to_float(spf_raw.get('0'))
    else:
        std_h = _to_float(spf_raw.get('3'))
        std_d = _to_float(spf_raw.get('1'))
        std_a = _to_float(spf_raw.get('0'))
        rq_h  = _to_float(nspf_raw.get('3')) if nspf_raw else 0.0
        rq_d  = _to_float(nspf_raw.get('1')) if nspf_raw else 0.0
        rq_a  = _to_float(nspf_raw.get('0')) if nspf_raw else 0.0

    # ── nspf 为空时: 让球盘但未开售 SPF ─────────────────────────────────
    nspf_empty = handicap != 0 and not (nspf_raw and nspf_raw.get('3'))
    if nspf_empty:
        std_h = std_d = std_a = 0.0
        rq_h = _to_float(spf_raw.get('3'))
        rq_d = _to_float(spf_raw.get('1'))
        rq_a = _to_float(spf_raw.get('0'))
        print(f'    🌐 {code} nspf未开售(仅开让球{handicap}), SPF标记为不可用')

    # ── 其他玩法 ──────────────────────────────────────────────────────────
    bf_data  = odds.get('bf', {})
    jqs_data = odds.get('jqs', {})
    bqc_data = odds.get('bqc', {})

    bqc_labeled = {_HTFT_RAW_MAP.get(k, k): v for k, v in bqc_data.items()}
    htft_odds = {k: _to_float(bqc_labeled.get(k)) for k in _HTFT_ORDER
                 if _to_float(bqc_labeled.get(k)) > 0}
    hf9_odds  = [_to_float(bqc_labeled.get(k)) for k in _HTFT_ORDER
                 if _to_float(bqc_labeled.get(k)) > 0]
    bf_odds   = {str(k): _to_float(v) for k, v in bf_data.items() if _to_float(v) > 0}
    zjq_odds  = {f'{int(k)}球': _to_float(v) for k, v in jqs_data.items() if _to_float(v) > 0}

    # std_odds_source 标识
    if nspf_empty and live_odds_map and code in live_odds_map:
        std_odds_source = 'live_euro_avg'
    elif handicap != 0 and nspf_raw and nspf_raw.get('3'):
        std_odds_source = 'nspf'
    else:
        std_odds_source = 'spf'

    return {
        'code':      code,
        'home_cn':   home_cn,
        'away_cn':   away_cn,
        'time':      row.get('endtime', ''),
        'league':    row.get('league', '') or '',
        'odds_h':    std_h,
        'odds_d':    std_d,
        'odds_a':    std_a,
        'rq_h':      rq_h,
        'rq_d':      rq_d,
        'rq_a':      rq_a,
        'handicap':  handicap,
        'std_odds_source': std_odds_source,
        'nspf_empty':      nspf_empty,
        'hf9_odds':  hf9_odds,
        'htft_odds': htft_odds,
        'jqs_odds':  zjq_odds,
        'bf_odds':   bf_odds,
        'zjq_odds':  zjq_odds,
    }
