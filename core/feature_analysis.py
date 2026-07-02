#!/usr/bin/env python3
"""Feature importance analysis for 33-dim model"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/root')
from wc_2026_phase1 import *
from collections import defaultdict
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath('/root/wc_2026_phase1.py')), 'data')
cache = os.path.join(DATA_DIR, 'international_results.json')
all_m = load_data(cache)

cutoff = '2022-11-20'
historical = [m for m in all_m if m['date'] < cutoff]
train = [m for m in historical if m['tournament'] in A_MATCH_TOURNAMENTS]
clean_elo = compute_elo(historical)
clean_dc = DixonColes(time_decay_hl=540)
clean_dc.fit(pd.DataFrame(train))

class FB:
    def __init__(s,elo,dc):
        s.elo=elo;s.dc=dc
        s.tg=defaultdict(list);s.h2h=defaultdict(lambda:defaultdict(list));s.ld={}
    def add(s,m):
        h,a=m['home'],m['away']
        for t,gf,ga in[(h,m['h_score'],m['a_score']),(a,m['a_score'],m['h_score'])]:
            s.tg[t].append({'d':m['date'],'gf':gf,'ga':ga});s.ld[t]=m['date']
        k=(h,a)if h<a else(a,h);s.h2h[k[0]][k[1]].append(m)
    def rf(s,team,date,n):
        g=[x for x in s.tg.get(team,[])if x['d']<date][-n:]
        if not g:return[.5,0,0,0]
        w=sum(1 for x in g if x['gf']>x['ga'])+sum(.5 for x in g if x['gf']==x['ga'])
        return[w/len(g),sum(x['gf']for x in g)/len(g),sum(x['ga']for x in g)/len(g),(sum(x['gf']for x in g)-sum(x['ga']for x in g))/len(g)]
    def get_h2h(s,home,away,date,n):
        k1,k2=(home,away)if home<away else(away,home)
        r=[x for x in s.h2h.get(k1,{}).get(k2,[])if x['date']<date][-n:]
        if not r:return[.5,0,0,0]
        w=0;gf=0;ga=0
        for x in r:
            if x['home']==home:gf+=x['h_score'];ga+=x['a_score'];w+=1 if x['h_score']>x['a_score']else(.5 if x['h_score']==x['a_score']else 0)
            else:gf+=x['a_score'];ga+=x['h_score'];w+=1 if x['a_score']>x['h_score']else(.5 if x['a_score']==x['h_score']else 0)
        return[w/len(r),gf/len(r),ga/len(r),len(r)]
    def rd(s,team,date):
        ld=s.ld.get(team);return 30 if not ld else max(1,(datetime.strptime(date,'%Y-%m-%d')-datetime.strptime(ld,'%Y-%m-%d')).days)

FEAT_NAMES = [
    'Elo_diff','lam_h','lam_a','lam_diff','lam_ratio',
    'DC_H','DC_D','DC_A','Odds_H','Odds_D','Odds_A',
    'f5_win_h','f5_win_a','f5_att_adv','f5_def_adv',
    'f5_gf_diff','f5_win_diff','f5_gd_h','f5_gd_a',
    'f12_win_h','f12_win_a','f12_att_adv','f12_def_adv',
    'h2h_win','h2h_gd','h2h_n',
    'tier_friendly','tier_major','tier_ko',
    'rest_h','rest_a','rest_diff','neutral'
]
NEW_IDX = [i for i,n in enumerate(FEAT_NAMES) if n in (
    'f5_gd_h','f5_gd_a','f12_win_h','f12_win_a','f12_att_adv','f12_def_adv',
    'h2h_win','h2h_gd','h2h_n','tier_friendly','tier_major','tier_ko',
    'rest_h','rest_a','rest_diff')]

print('Building 33-dim features...', flush=True)
X33,y33=[],[]
ms=sorted(train,key=lambda m:m['date'])
fb=FB(clean_elo,clean_dc)
for i,m in enumerate(ms):
    if i%10000==0:print(f'  {i}/{len(ms)}',flush=True)
    h,a=m['home'],m['away']
    eh,ea=clean_elo.get(h,1500),clean_elo.get(a,1500)
    lh,la=clean_dc.predict_lambda(h,a,neutral=m.get('neutral',False))
    if lh is None:continue
    dp=clean_dc.predict_proba(h,a,neutral=m.get('neutral',False))
    fh5=fb.rf(h,m['date'],5);fa5=fb.rf(a,m['date'],5)
    fh12=fb.rf(h,m['date'],12);fa12=fb.rf(a,m['date'],12)
    h2h=fb.get_h2h(h,a,m['date'],3);tier=tournament_tier(m.get('tournament',''))
    rh=fb.rd(h,m['date']);ra=fb.rd(a,m['date'])
    X33.append([(eh-ea)/400,lh,la,lh-la,math.log(max(lh,.01)/max(la,.01)),
        dp[0],dp[1],dp[2],1/3,1/3,1/3,
        fh5[0],fa5[0],fh5[1]-fa5[2],fa5[1]-fh5[2],fh5[1]-fa5[1],fh5[0]-fa5[0],fh5[3],fa5[3],
        fh12[0],fa12[0],fh12[1]-fa12[2],fa12[1]-fh12[1],
        h2h[0],h2h[1]-h2h[2],h2h[3],tier[0],tier[1],tier[2],rh,ra,rh-ra,int(m.get('neutral',0))])
    y33.append(2 if m['h_score']>m['a_score'] else(1 if m['h_score']==m['a_score'] else 0))
    fb.add(m)

X33=np.array(X33);y33=np.array(y33)
vs=int(len(X33)*.8)
cw=compute_class_weight('balanced',classes=np.unique(y33),y=y33)
sw=np.array([cw[list(np.unique(y33)).index(c)]for c in y33])

m=XGBClassifier(n_estimators=300,max_depth=5,lr=.05,subsample=.8,colsample_bytree=.8,
                reg_alpha=.1,reg_lambda=.1,random_state=42,eval_metric='mlogloss',verbosity=0)
m.fit(X33[:vs],y33[:vs],eval_set=[(X33[vs:],y33[vs:])],sample_weight=sw[:vs],verbose=False)

imp=m.feature_importances_
idx=sorted(range(len(imp)),key=lambda i:imp[i],reverse=True)

print()
SEP = '='*55
print(SEP)
print('  33维特征重要性全排名')
print(SEP)
print(f'  {"#":>3s} {"特征":<20s} {"重要性":>8s} {"累计":>8s} {"类型":>6s}')
print('  ' + '-'*50)
c=0
for rank,i in enumerate(idx,1):
    c+=imp[i]
    tag='旧' if i not in NEW_IDX else '新'
    print(f'  {rank:>3d}. {FEAT_NAMES[i]:<20s} {imp[i]:>8.4f} {c*100:>7.1f}% {tag:>6s}')

old_sum = sum(imp[i] for i in range(33) if i not in NEW_IDX)
new_sum = sum(imp[i] for i in NEW_IDX)
print()
print(f'  旧特征(11个)总权重: {old_sum*100:.1f}%')
print(f'  新特征(15个)总权重: {new_sum*100:.1f}%')
print(f'  Top 5累计: {sum(imp[i] for i in idx[:5]):.4f}')
print(f'  末尾5名累计: {sum(imp[i] for i in idx[-5:]):.4f}')
print()
print('  新特征个体排名:')
for rank,i in enumerate(idx,1):
    if i in NEW_IDX:
        print(f'    #{rank} {FEAT_NAMES[i]} = {imp[i]:.4f}')
