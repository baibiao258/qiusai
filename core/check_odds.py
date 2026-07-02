import json, subprocess
from datetime import date

date_str = date.today().isoformat()
markets = {}
for playid in ('269', '270', '271', '272'):
    try:
        out = subprocess.check_output(['python3', '/root/wc_2026_upgrade/fetch_500_market.py', date_str, playid, '2'], text=True, timeout=30)
        data = json.loads(out)
        markets[playid] = {row['no']: row for row in data.get('result', [])}
        print(f'playid {playid}: {len(markets[playid])} 场')
    except Exception as e:
        print(f'playid {playid}: 抓取失败 {e}')
        markets[playid] = {}

main_rows = markets.get('269', {})
for code, row in sorted(main_rows.items()):
    home = row.get('home', '')
    away = row.get('away', '')
    rangqiu = row.get('rangqiu', '')
    endtime = row.get('endtime', '')
    spf = row.get('odds', {}).get('spf', {})
    nspf = row.get('odds', {}).get('nspf', {})
    bf = markets.get('271', {}).get(code, {}).get('odds', {})
    htft = markets.get('272', {}).get(code, {}).get('odds', {})
    zjq = markets.get('270', {}).get(code, {}).get('odds', {})
    
    spf_str = f'主胜{spf.get("3","-")} 平{spf.get("1","-")} 客胜{spf.get("0","-")}' if spf else '无'
    rq_str = f'让胜{nspf.get("3","-")} 让平{nspf.get("1","-")} 让负{nspf.get("0","-")}' if nspf else '无'
    zjq_str = ' '.join([f'{k}球{v}' for k,v in sorted(zjq.items())]) if zjq else '无'
    htft_str = ' '.join([f'{k}{v}' for k,v in list(htft.items())[:5]]) if htft else '无'
    bf_cnt = len(bf) if bf else 0
    
    print(f'{code} | {home} vs {away} | 让球{rangqiu} | {endtime}')
    print(f'  SPF: {spf_str}')
    print(f'  RQ:  {rq_str}')
    print(f'  ZJQ: {zjq_str}')
    print(f'  HTFT: {htft_str}... (共{len(htft)}项)')
    print(f'  BF:  {bf_cnt}项')
    print()