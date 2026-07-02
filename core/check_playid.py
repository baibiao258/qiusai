#!/usr/bin/env python3
"""检查500.com不同playid返回的数据"""
import json, subprocess, sys, os

date = sys.argv[1] if len(sys.argv) > 1 else '2026-06-09'
os.chdir('/root/wc_2026_upgrade')

for playid in ['265', '268', '269', '270', '271', '272']:
    try:
        out = subprocess.check_output(
            ['python3', 'fetch_500_market.py', date, playid, '1'],
            text=True, timeout=10, stderr=subprocess.DEVNULL
        )
        data = json.loads(out)
        ok = data.get('ok', False)
        count = data.get('count', 0)
        print(f'playid={playid}: ok={ok} count={count}')
        if ok and count > 0:
            r = data['result'][0]
            spf = r.get('odds', {}).get('spf', {})
            nspf = r.get('odds', {}).get('nspf', {})
            rq = r.get('rangqiu', '?')
            team = r.get('team', '')[:35]
            print(f'  team={team} rangqiu={rq}')
            print(f'  spf={spf}')
            print(f'  nspf={nspf}')
    except Exception as e:
        print(f'playid={playid}: ERROR {e}')
