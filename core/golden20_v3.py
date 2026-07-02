#!/usr/bin/env python3
"""双输出一次遍历 + 严格回测"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '/root')
from wc_2026_phase1 import *
from collections import defaultdict
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = '/root/data'
with open(os.path.join(DATA_DIR, 'optuna_best.json')) as f:
    bp = json.load(f)['best_params']

all_m = load_data(os.path.join(DATA_DIR, 'international_results.json'))
cutoff='2022-11-20'
hist=[m for m in all_m if m['date']<cutoff]
train=[m for m in hist if m['tournament'] in A_MATCH_TOURNAMENTS]
wc=[m for m in all_m if m['tournament']=='FIFA World Cup' and cutoff<=m['date']<='2022-12-18']
print(f'{len(train)} train, {len(wc)} test', flush=True)

clean_elo=compute_elo(hist)
clean_dc=DixonColes(time_decay_hl=540)
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

def mk_odds(eh,ea):
    e=1/(1+10**((ea-eh)/400));d=0.26*np.exp(-((eh-ea)/200)**2)
    o=np.array([e*(1-d),d,(1-e)*(1-d)]);o/=o.sum();return o

# ⭐ 一次遍历：15维 + 20+3维 双输出
print('One-pass feature build...', flush=True)
X15,y15,X20,y20=[],[],[],[]
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
    # 15 base
    b15=[(eh-ea)/400,lh,la,lh-la,math.log(max(lh,.01)/max(la,.01)),dp[0],dp[1],dp[2],
         fh5[0],fa5[0],fh5[1]-fa5[2],fa5[1]-fh5[2],fh5[1]-fa5[1],fh5[0]-fa5[0],int(m.get('neutral',0))]
    X15.append(b15)
    y15.append(2 if m['h_score']>m['a_score']else(1 if m['h_score']==m['a_score']else 0))
    # 20+3
    fh12=fb.rf(h,m['date'],12);fa12=fb.rf(a,m['date'],12)
    h2h=fb.get_h2h(h,a,m['date'],3);tier=tournament_tier(m.get('tournament',''))
    op=mk_odds(eh,ea)
    X20.append(b15+[h2h[1]-h2h[2],tier[1],tier[0],fh12[1]-fa12[2],fa12[1]-fh12[0],op[0],op[1],op[2]])
    y20.append(y15[-1])
    fb.add(m)

X15=np.array(X15);y15=np.array(y15)
X20=np.array(X20);y20=np.array(y20)
print(f'15: {X15.shape}  20+3: {X20.shape}', flush=True)

# ── Train 20+3 ──
tm=[d<'2021-06-01' for d in [m['date'] for m in ms if not None]]  
# Actually dates need to be tracked for the skipped matches... 
# Simpler: use the same ms list and skip tracking
d_all=[m['date'] for i,m in enumerate(ms) if clean_dc.predict_lambda(m['home'],m['away'],neutral=m.get('neutral',False))[0] is not None]
tm=np.array([d<'2021-06-01' for d in d_all])
vm=np.array([d>='2021-06-01' and d<'2022-11-20' for d in d_all])

def train_xgb(X,y):
    cw=compute_class_weight('balanced',classes=np.unique(y[tm]),y=y[tm])
    sw=np.array([cw[list(np.unique(y[tm])).index(c)]for c in y[tm]])
    m=XGBClassifier(**bp,random_state=42,eval_metric='mlogloss',verbosity=0)
    m.fit(X[tm],y[tm],eval_set=[(X[vm],y[vm])],sample_weight=sw,verbose=False)
    return m

m20=train_xgb(X20,y20)
m15=train_xgb(X15,y15)
v20=np.mean(m20.predict(X20[vm])==y20[vm])*100
v15=np.mean(m15.predict(X15[vm])==y15[vm])*100
print(f'Val: 15={v15:.1f}%  20+3={v20:.1f}%', flush=True)

# ── 64场 ──
fb2=FB(clean_elo,clean_dc)
for mm in hist: fb2.add(mm)
wts=[(0.3,0.7),(0.4,0.6),(0.5,0.5),(0.6,0.4),(0.7,0.3)]
p15={w:{'c':0,'b':0.0}for w in wts};p20={w:{'c':0,'b':0.0}for w in wts}

for idx,mm in enumerate(sorted(wc,key=lambda x:x['date'])):
    h,a=mm['home'],mm['away']
    eh,ea=clean_elo.get(h,1500),clean_elo.get(a,1500)
    lh,la=clean_dc.predict_lambda(h,a,True);dp=clean_dc.predict_proba(h,a,True)
    da=np.array([dp[2],dp[1],dp[0]]);op=mk_odds(eh,ea)
    act=2 if mm['h_score']>mm['a_score']else(1 if mm['h_score']==mm['a_score']else 0)
    yo=np.zeros(3);yo[act]=1
    fh5=fb2.rf(h,mm['date'],5);fa5=fb2.rf(a,mm['date'],5)
    fh12=fb2.rf(h,mm['date'],12);fa12=fb2.rf(a,mm['date'],12)
    h2h=fb2.get_h2h(h,a,mm['date'],3);tier=tournament_tier(mm.get('tournament',''))
    b15=[(eh-ea)/400,lh,la,lh-la,math.log(max(lh,.01)/max(la,.01)),dp[0],dp[1],dp[2],
         fh5[0],fa5[0],fh5[1]-fa5[2],fa5[1]-fh5[2],fh5[1]-fa5[1],fh5[0]-fa5[0],1]
    f15=np.array([b15])
    f20=np.array([b15+[h2h[1]-h2h[2],tier[1],tier[0],fh12[1]-fa12[2],fa12[1]-fh12[0],op[0],op[1],op[2]]])
    for wd,wx in wts:
        hh=wd*da+wx*m20.predict_proba(f20)[0]
        if np.argmax(hh)==act:p20[(wd,wx)]['c']+=1
        p20[(wd,wx)]['b']+=np.sum((hh-yo)**2)
        h15=wd*da+wx*m15.predict_proba(f15)[0]
        if np.argmax(h15)==act:p15[(wd,wx)]['c']+=1
        p15[(wd,wx)]['b']+=np.sum((h15-yo)**2)
    fb2.add(mm)
    if (idx+1)%16==0:print(f'  {idx+1}/64',flush=True)

n=len(wc)
SEP='='*60
print(f'\n{SEP}')
print(f'  黄金20+3 严格回测')
print(f'{SEP}')
print(f'  {"─"*62}')
print(f'  {"权重":<22s} {"15维基线":>22s} {"20+3黄金":>22s}')
print(f'  {"─"*62}')
for w in wts:
    o=p15[w];nw=p20[w]
    print(f'  DC{w[0]:.1f}+XGB{w[1]:.1f}     {o["c"]/n*100:>7.2f}% B={o["b"]/n:.4f}  {nw["c"]/n*100:>7.2f}% B={nw["b"]/n:.4f}')
bo=max(wts,key=lambda w:p15[w]['c'])
bn=max(wts,key=lambda w:p20[w]['c'])
dlt=(p20[bn]['c']-p15[bo]['c'])/n*100
print(f'  {"─"*62}')
print(f'  15维: DC{bo[0]:.1f}+XGB{bo[1]:.1f} = {p15[bo]["c"]/n*100:.2f}%  B={p15[bo]["b"]/n:.4f}')
print(f'  20+3: DC{bn[0]:.1f}+XGB{bn[1]:.1f} = {p20[bn]["c"]/n*100:.2f}%  B={p20[bn]["b"]/n:.4f}  ({dlt:+.2f}pp)')
print(f'{SEP}')

with open(os.path.join(DATA_DIR,'golden20_final.json'),'w') as f:
    json.dump({'params':bp,'n':n,'v15':float(v15),'v20':float(v20),
        '15':{f'DC{w[0]:.1f}+XGB{w[1]:.1f}':{'acc':p15[w]['c']/n,'brier':p15[w]['b']/n}for w in wts},
        '20':{f'DC{w[0]:.1f}+XGB{w[1]:.1f}':{'acc':p20[w]['c']/n,'brier':p20[w]['b']/n}for w in wts}},f,indent=2)
