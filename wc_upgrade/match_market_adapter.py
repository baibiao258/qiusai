#!/usr/bin/env python3
"""统一汇总 500.com 赔率为模型可消费 schema"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

BASE = Path(__file__).resolve().parent

MARKET_SCRIPTS = {
    'main_play': BASE / 'fetch_500_market.py',
    'htft': BASE / 'fetch_500_market.py',
    'score': BASE / 'fetch_500_market.py',
    'totalgoals': BASE / 'fetch_500_market.py',
}

MARKET_PLAYIDS = {
    'main_play': ('269', '2'),
    'totalgoals': ('270', '2'),
    'score': ('271', '2'),
    'htft': ('272', '2'),
}


def run_script(script: Path, date: str, market: str) -> Dict[str, Any]:
    playid, g = MARKET_PLAYIDS[market]
    proc = subprocess.run([sys.executable, str(script), date, playid, g], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f'{script.name} failed')
    text = proc.stdout.strip()
    start = text.find('{')
    if start < 0:
        raise ValueError(f'{script.name}: no JSON found')
    return json.loads(text[start:])


def normalize_row(row: Dict[str, Any], market: str, date: str, source_url: str) -> Dict[str, Any]:
    num = row.get('num') or row.get('no')
    league = row.get('league', '')
    start = row.get('start') or row.get('date') or date
    end_time = row.get('endTime') or row.get('endtime') or ''
    match = row.get('match') or row.get('team') or ''
    raw_odds = row.get('odds', {})
    if isinstance(raw_odds, dict) and 'odds' in raw_odds and 'rangqiu' in raw_odds:
        odds = raw_odds.get('odds', {})
        rangqiu = raw_odds.get('rangqiu', row.get('rangqiu', row.get('data-rangqiu', 0)))
    else:
        odds = raw_odds
        rangqiu = row.get('rangqiu', row.get('data-rangqiu', 0))
    return {
        'market': market,
        'num': num,
        'league': league,
        'start': start,
        'endTime': end_time,
        'match': match,
        'odds': odds,
        'rangqiu': rangqiu,
        'source': '500.com',
        'source_url': source_url,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None)
    ap.add_argument('--markets', default='main_play,htft,score,totalgoals')
    ap.add_argument('--output', default='stdout', choices=['stdout', 'jsonl', 'file'])
    ap.add_argument('--output-file', default='', help='when --output=file, write JSON to this path')
    args = ap.parse_args()

    date = args.date or __import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
    markets = [m.strip() for m in args.markets.split(',') if m.strip()]

    out = {
        'ok': True,
        'source': '500.com',
        'date': date,
        'markets': {},
        'schema': ['market', 'num', 'league', 'start', 'endTime', 'match', 'odds', 'source', 'source_url'],
    }

    for market in markets:
        script = MARKET_SCRIPTS.get(market)
        if not script or not script.exists():
            out['markets'][market] = {'ok': False, 'error': 'script not found', 'count': 0, 'result': []}
            continue
        data = run_script(script, date, market)
        rows = [normalize_row(r, market, date, data.get('url', '')) for r in data.get('result', [])]
        out['markets'][market] = {
            'ok': bool(data.get('ok', True)),
            'count': len(rows),
            'source_url': data.get('url', ''),
            'result': rows,
        }

    if args.output == 'stdout':
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.output == 'jsonl':
        for market, payload in out['markets'].items():
            for row in payload.get('result', []):
                row2 = dict(row)
                row2['date'] = out['date']
                print(json.dumps(row2, ensure_ascii=False))
    else:
        out_path = Path(args.output_file) if args.output_file else BASE / f'500_odds_{date}.json'
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'ok': True, 'output': str(out_path), 'markets': list(out['markets'].keys())}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
