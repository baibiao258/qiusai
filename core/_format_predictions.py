#!/usr/bin/env python3
"""Format predictions from predictions_log.csv with sorted probabilities."""
import pandas as pd
import json

df = pd.read_csv('/root/data/predictions_log.csv')

today_codes = ['周三201','周三202','周四001','周四002','周五003','周五004',
               '周六005','周六006','周六007','周六008','周日009','周日010',
               '周日011','周日012','周一013','周一014','周一015','周一016',
               '周二017','周二018','周二019','周二020','周三021','周三022',
               '周三023','周三024']

htft_map = {'HH':'胜胜','DH':'平胜','AH':'负胜','HD':'胜平',
            'DD':'平平','AD':'负平','HA':'胜负','DA':'平负','AA':'负负'}

for code in today_codes:
    row = df[df['code']==code]
    if len(row)==0:
        continue
    row = row.iloc[-1]
    home = row['home_cn']
    away = row['away_cn']
    league = row.get('league','')
    rq = row.get('rq','')
    
    print('='*65)
    print(f'  {code}  {home} vs {away}')
    print(f'  联赛: {league}  |  让球: {rq}')
    print('='*65)
    
    # ① 胜平负
    spf = [(row['pred_h'],f'{home}胜'), (row['pred_d'],'平局'), (row['pred_a'],f'{away}胜')]
    spf.sort(key=lambda x: -x[0])
    oh, od, oa = row['odds_h'], row['odds_d'], row['odds_a']
    
    print('\n① 胜平负（按概率降序）')
    print('  '+'-'*50)
    for i,(p,name) in enumerate(spf):
        if name == f'{home}胜':
            odd = oh
        elif name == '平局':
            odd = od
        else:
            odd = oa
        ev = (p/100 * odd - 1)*100 if odd and odd>0 and odd<100 else 0
        label = '🏆' if i==0 else '  '
        if odd and odd>0 and odd<100:
            print(f'  {label} #{i+1} {name:<10s}  {p:.1f}%  @{odd:.2f}  EV={ev:+.1f}%')
        else:
            print(f'  {label} #{i+1} {name:<10s}  {p:.1f}%  未开售/异常赔率')
    
    # ② 让球
    rq_label = f'主让{rq}' if rq not in ['0','','0.0',0] else '不让球'
    try:
        rq_win = float(row['pred_rq_win'])
        rq_draw = float(row['pred_rq_draw'])
        rq_loss = float(row['pred_rq_loss'])
    except:
        rq_win = rq_draw = rq_loss = 0
    rq_items = [(rq_win,'让胜'), (rq_draw,'让平'), (rq_loss,'让负')]
    rq_items.sort(key=lambda x: -x[0])
    print(f'\n② 竞彩让球 ({rq_label})（按概率降序）')
    print('  '+'-'*50)
    for i,(p,name) in enumerate(rq_items):
        label = '🏆' if i==0 else '  '
        print(f'  {label} #{i+1} {name:<10s}  {p:.1f}%')
    
    # ③ 半全场
    try:
        hf_raw = row['htft_full']
        hf = json.loads(hf_raw) if isinstance(hf_raw,str) and hf_raw else {}
    except:
        hf = {}
    if hf:
        hf_sorted = sorted([(v, htft_map.get(k,k)) for k,v in hf.items()], key=lambda x: -x[0])
        print(f'\n③ 半全场（9项完整，按概率降序）')
        print('  '+'-'*50)
        cols = 3
        for i,(p,name) in enumerate(hf_sorted):
            label = '🏆' if i==0 else '  '
            print(f'  {label} #{i+1} {name:<8s}  {p*100:.1f}%')
    
    # ④ 比分
    try:
        sf_raw = row['score_full']
        sf = json.loads(sf_raw) if isinstance(sf_raw,str) and sf_raw else {}
    except:
        sf = {}
    if sf:
        sf_sorted = sorted(sf.items(), key=lambda x: -x[1])
        print(f'\n④ 比分（Top 15，按概率降序）')
        print('  '+'-'*50)
        for i,(score,prob) in enumerate(sf_sorted[:15]):
            label = '🏆' if i==0 else '  '
            print(f'  {label} #{i+1} {score:<6s}  {prob*100:.1f}%')
    
    # ⑤ 总进球
    try:
        gf_raw = row['goals_full']
        gf = json.loads(gf_raw) if isinstance(gf_raw,str) and gf_raw else {}
    except:
        gf = {}
    if gf:
        gf_sorted = sorted([(float(k),v) for k,v in gf.items()], key=lambda x: -x[1])
        print(f'\n⑤ 总进球（13档完整，按概率降序）')
        print('  '+'-'*50)
        for i,(goals,prob) in enumerate(gf_sorted):
            g = f'{int(goals)}球' if goals < 13 else f'{int(goals)}+球'
            if prob < 0.0005:
                continue
            label = '🏆' if i==0 else '  '
            print(f'  {label} #{i+1} {g:<6s}  {prob*100:.1f}%')
    
    print()
