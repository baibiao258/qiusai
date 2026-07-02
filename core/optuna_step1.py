#!/usr/bin/env python3
"""Step 1: 构建33维特征并保存，然后跑Optuna"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '/root')
from wc_2026_phase1 import *
from collections import defaultdict
import numpy as np, pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = '/root/data'
cache = os.path.join(DATA_DIR, 'international_results.json')
all_m = load_data(cache)
cutoff = '2022-11-20'
historical = [m for m in all_m if m['date'] < cutoff]
train = [m for m in historical if m['tournament'] in A_MATCH_TOURNAMENTS]
print(f'Train: {len(train)}', flush=True)

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

print('Building features...', flush=True)
X33,y33,dates=[],[],[]
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
    y33.append(2 if m['h_score']>m['a_score']else(1 if m['h_score']==m['a_score']else 0))
    dates.append(m['date']);fb.add(m)

X33=np.array(X33);y33=np.array(y33);dates=np.array(dates)
print(f'Matrix: {X33.shape}', flush=True)

# Save for later use
np.save(os.path.join(DATA_DIR,'X33.npy'),X33)
np.save(os.path.join(DATA_DIR,'y33.npy'),y33)
with open(os.path.join(DATA_DIR,'dates.json'),'w') as f:
    json.dump(dates.tolist(),f)
print('Saved to /root/data/', flush=True)

# ── 时序切分 ──
tm=dates<'2021-06-01'
vm=(dates>='2021-06-01')&(dates<'2022-11-20')
Xt,Xv=X33[tm],X33[vm];yt,yv=y33[tm],y33[vm]
print(f'Train: {len(Xt)} Val: {len(Xv)}', flush=True)
cw=compute_class_weight('balanced',classes=np.unique(yt),y=yt)
sw=np.array([cw[list(np.unique(yt)).index(c)]for c in yt])
yv_oh=np.zeros((len(yv),3));yv_oh[np.arange(len(yv)),yv]=1

# ── Optuna ──
def objective(trial):
    p={
        'max_depth': trial.suggest_int('max_depth',2,4),
        'learning_rate': trial.suggest_float('learning_rate',0.01,0.05,log=True),
        'n_estimators': trial.suggest_int('n_estimators',150,450),
        'reg_alpha': trial.suggest_float('reg_alpha',0.5,12.0,log=True),
        'reg_lambda': trial.suggest_float('reg_lambda',2.0,25.0,log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree',0.35,0.60),
        'subsample': trial.suggest_float('subsample',0.60,0.85),
        'min_child_weight': trial.suggest_float('min_child_weight',3.0,15.0),
        'random_state':42,'eval_metric':'mlogloss','verbosity':0}
    m=XGBClassifier(**p,early_stopping_rounds=25)
    m.fit(Xt,yt,eval_set=[(Xv,yv)],sample_weight=sw,verbose=False)
    yp=m.predict_proba(Xv)
    return float(np.mean(np.sum((yp-yv_oh)**2,axis=1)))

print('\nOptuna 50 trials...',flush=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study=optuna.create_study(direction='minimize')
study.optimize(objective,n_trials=50,show_progress_bar=True)

print(f'\nBest Brier: {study.best_value:.4f}')
print('Best params:')
for k,v in study.best_params.items():
    print(f"  '{k}': {v},")

with open(os.path.join(DATA_DIR,'optuna_best.json'),'w') as f:
    json.dump({'best_params':study.best_params,'best_brier':study.best_value},f,indent=2)
print(f'Saved: {DATA_DIR}/optuna_best.json')
print('DONE', flush=True)
