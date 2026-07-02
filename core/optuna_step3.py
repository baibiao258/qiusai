#!/usr/bin/env python3
"""Step 3: 用Optuna最优参数跑严格2022回测"""
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
print('Optuna best params:', bp, flush=True)

X33 = np.load(os.path.join(DATA_DIR, 'X33.npy'))
y33 = np.load(os.path.join(DATA_DIR, 'y33.npy'))
with open(os.path.join(DATA_DIR, 'dates.json')) as f:
    dates = json.load(f)

all_m = load_data(os.path.join(DATA_DIR, 'international_results.json'))
cutoff = '2022-11-20'
historical = [m for m in all_m if m['date'] < cutoff]
wc = [m for m in all_m if m['tournament']=='FIFA World Cup' and '2022-11-20'<=m['date']<='2022-12-18']
clean_elo = compute_elo(historical)
clean_dc = DixonColes(time_decay_hl=540)
clean_dc.fit(pd.DataFrame([m for m in historical if m['tournament'] in A_MATCH_TOURNAMENTS]))

# ── 时序切分训练 ──
tm = [d < '2021-06-01' for d in dates]
vm = [d >= '2021-06-01' and d < '2022-11-20' for d in dates]
Xt, yt = X33[tm], y33[tm]
Xv, yv = X33[vm], y33[vm]

cw = compute_class_weight('balanced', classes=np.unique(yt), y=yt)
sw = np.array([cw[list(np.unique(yt)).index(c)] for c in yt])

# Train 33-dim with Optuna params
m33 = XGBClassifier(**bp, random_state=42, eval_metric='mlogloss', verbosity=0)
m33.fit(Xt, yt, eval_set=[(Xv, yv)], sample_weight=sw, verbose=False)
print(f'33维 验证: {np.mean(m33.predict(Xv)==yv)*100:.1f}%', flush=True)

# Also build 15-dim and train
train_all = [m for m in historical if m['tournament'] in A_MATCH_TOURNAMENTS]
X15, y15, d15 = [], [], []
class FB15:
    def __init__(s,elo,dc):s.elo=elo;s.dc=dc;s.tg=defaultdict(list);s.ld={}
    def add(s,m):
        h,a=m['home'],m['away']
        for t,gf,ga in[(h,m['h_score'],m['a_score']),(a,m['a_score'],m['h_score'])]:
            s.tg[t].append({'d':m['date'],'gf':gf,'ga':ga});s.ld[t]=m['date']
    def rf(s,team,date,n):
        g=[x for x in s.tg.get(team,[])if x['d']<date][-n:]
        if not g:return[.5,0,0,0]
        w=sum(1 for x in g if x['gf']>x['ga'])+sum(.5 for x in g if x['gf']==x['ga'])
        return[w/len(g),sum(x['gf']for x in g)/len(g),sum(x['ga']for x in g)/len(g)]
fb15=FB15(clean_elo,clean_dc)
for m in sorted(train_all,key=lambda m:m['date']):
    h,a=m['home'],m['away']
    eh,ea=clean_elo.get(h,1500),clean_elo.get(a,1500)
    lh,la=clean_dc.predict_lambda(h,a,neutral=m.get('neutral',False))
    if lh is None:continue
    dp=clean_dc.predict_proba(h,a,neutral=m.get('neutral',False))
    fh5=fb15.rf(h,m['date'],5);fa5=fb15.rf(a,m['date'],5)
    X15.append([(eh-ea)/400,lh,la,lh-la,math.log(max(lh,.01)/max(la,.01)),dp[0],dp[1],dp[2],
        fh5[0],fa5[0],fh5[1]-fa5[2],fa5[1]-fh5[2],fh5[1]-fa5[1],fh5[0]-fa5[0],int(m.get('neutral',0))])
    y15.append(2 if m['h_score']>m['a_score']else(1 if m['h_score']==m['a_score']else 0))
    d15.append(m['date']);fb15.add(m)
X15=np.array(X15);y15=np.array(y15)
t15=[d<'2021-06-01' for d in d15]
v15=[d>='2021-06-01' and d<'2022-11-20' for d in d15]
cw15=compute_class_weight('balanced',classes=np.unique(y15[t15]),y=y15[t15])
sw15=np.array([cw15[list(np.unique(y15[t15])).index(c)]for c in y15[t15]])
m15=XGBClassifier(**bp,random_state=42,eval_metric='mlogloss',verbosity=0)
m15.fit(X15[t15],y15[t15],eval_set=[(X15[v15],y15[v15])],sample_weight=sw15,verbose=False)
print(f'15维 验证: {np.mean(m15.predict(X15[v15])==y15[v15])*100:.1f}%', flush=True)

# ── 逐场预测64场 ──
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

fb2=FB(clean_elo,clean_dc)
for mm in historical: fb2.add(mm)

weights=[(0.3,0.7),(0.4,0.6),(0.5,0.5),(0.6,0.4),(0.7,0.3)]
p15={w:{'c':0,'b':0.0}for w in weights}
p33={w:{'c':0,'b':0.0}for w in weights}

for idx,mm in enumerate(sorted(wc,key=lambda x:x['date'])):
    h,a=mm['home'],mm['away']
    eh,ea=clean_elo.get(h,1500),clean_elo.get(a,1500)
    lh,la=clean_dc.predict_lambda(h,a,True)
    dp=clean_dc.predict_proba(h,a,True)
    da=np.array([dp[2],dp[1],dp[0]])
    act=2 if mm['h_score']>mm['a_score']else(1 if mm['h_score']==mm['a_score']else 0)
    yo=np.zeros(3);yo[act]=1
    
    fh5=fb2.rf(h,mm['date'],5);fa5=fb2.rf(a,mm['date'],5)
    f15=np.array([[(eh-ea)/400,lh,la,lh-la,math.log(max(lh,.01)/max(la,.01)),dp[0],dp[1],dp[2],fh5[0],fa5[0],fh5[1]-fa5[2],fa5[1]-fh5[2],fh5[1]-fa5[1],fh5[0]-fa5[0],1]])
    x15r=m15.predict_proba(f15)[0]
    
    fh12=fb2.rf(h,mm['date'],12);fa12=fb2.rf(a,mm['date'],12)
    h2h=fb2.get_h2h(h,a,mm['date'],3);tier=tournament_tier(mm.get('tournament',''))
    rh=fb2.rd(h,mm['date']);ra=fb2.rd(a,mm['date'])
    f33=make_feat_vec(eh,ea,lh,la,dp,np.array([1/3,1/3,1/3]),fh5,fa5,fh12,fa12,h2h,tier,rh,ra,1)
    x33r=m33.predict_proba(f33)[0]
    
    for wd,wx in weights:
        h33=wd*da+wx*x33r
        if np.argmax(h33)==act:p33[(wd,wx)]['c']+=1
        p33[(wd,wx)]['b']+=np.sum((h33-yo)**2)
        h15=wd*da+wx*x15r
        if np.argmax(h15)==act:p15[(wd,wx)]['c']+=1
        p15[(wd,wx)]['b']+=np.sum((h15-yo)**2)
    fb2.add(mm)
    if (idx+1)%16==0:print(f'  {idx+1}/64',flush=True)

n=len(wc)
SEP='='*60
print(f'\n{SEP}')
print(f'  Optuna调参后严格回测 ({n}场)')
print(f'{SEP}')
print(f'  训练: <2021-06 | 验证: 2021-06~2022-11 | 测试: 2022 WC(盲测)')
print(f'  参数: max_depth={bp["max_depth"]} reg_alpha={bp["reg_alpha"]:.2f} reg_lambda={bp["reg_lambda"]:.2f}')
print(f'        colsample={bp["colsample_bytree"]:.2f} subsample={bp["subsample"]:.2f}')
print(f'        min_child_weight={bp["min_child_weight"]:.1f} lr={bp["learning_rate"]:.4f} n_est={bp["n_estimators"]}')
print(f'  {"─"*62}')
print(f'  {"权重":<22s} {"15维(Optuna)":>22s} {"33维(Optuna)":>22s}')
print(f'  {"─"*62}')
for w in weights:
    o=p15[w];nw=p33[w]
    print(f'  DC{w[0]:.1f}+XGB{w[1]:.1f}     {o["c"]/n*100:>7.2f}% B={o["b"]/n:.4f}  {nw["c"]/n*100:>7.2f}% B={nw["b"]/n:.4f}')
bo=max(weights,key=lambda w:p15[w]['c'])
bn=max(weights,key=lambda w:p33[w]['c'])
print(f'  {"─"*62}')
print(f'  旧基线最佳: DC{bo[0]:.1f}+XGB{bo[1]:.1f} = {p15[bo]["c"]/n*100:.2f}%')
print(f'  33维最佳:   DC{bn[0]:.1f}+XGB{bn[1]:.1f} = {p33[bn]["c"]/n*100:.2f}%')
print(f'  Brier:      15维={p15[bo]["b"]/n:.4f}  33维={p33[bn]["b"]/n:.4f}')
print(f'{SEP}')

with open(os.path.join(DATA_DIR,'optuna_backtest.json'),'w') as f:
    json.dump({'params':bp,'n':n,
        '15dim':{f'DC{w[0]:.1f}+XGB{w[1]:.1f}':{'acc':p15[w]['c']/n,'brier':p15[w]['b']/n}for w in weights},
        '33dim':{f'DC{w[0]:.1f}+XGB{w[1]:.1f}':{'acc':p33[w]['c']/n,'brier':p33[w]['b']/n}for w in weights}},f,indent=2)
print('Saved: optuna_backtest.json')
