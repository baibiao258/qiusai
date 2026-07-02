#!/usr/bin/env python3
"""防守型 Optuna 调参 — 严格时序防泄漏，33维特征"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/root')
from wc_2026_phase1 import *
from collections import defaultdict
import numpy as np, pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

# ── 数据加载 ──
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath('/root/wc_2026_phase1.py')), 'data')
cache = os.path.join(DATA_DIR, 'international_results.json')
all_m = load_data(cache)
cutoff = '2022-11-20'
historical = [m for m in all_m if m['date'] < cutoff]
train = [m for m in historical if m['tournament'] in A_MATCH_TOURNAMENTS]

print(f'训练集: {len(train)} 场', flush=True)

# ── Elo + DC ──
clean_elo = compute_elo(historical)
clean_dc = DixonColes(time_decay_hl=540)
clean_dc.fit(pd.DataFrame(train))

# ── FeatureBuffer ──
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

# ── 构建33维特征 + 记录日期 ──
print('构建33维特征...', flush=True)
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
    y33.append(2 if m['h_score']>m['a_score'] else(1 if m['h_score']==m['a_score'] else 0))
    dates.append(m['date'])
    fb.add(m)

X33=np.array(X33);y33=np.array(y33);dates=np.array(dates)
print(f'特征矩阵: {X33.shape}', flush=True)

# ── 时序切分 ──
# 训练: < 2021-06-01
# 验证(Optuna目标): 2021-06-01 ~ 2022-11-19 (欧洲杯/美洲杯/世预赛)
train_mask=dates<'2021-06-01'
val_mask=(dates>='2021-06-01')&(dates<'2022-11-20')
Xt,Xv=X33[train_mask],X33[val_mask]
yt,yv=y33[train_mask],y33[val_mask]
print(f'训练: {len(Xt)} 验证: {len(Xv)}', flush=True)

classes=np.unique(yt)
cw=compute_class_weight('balanced',classes=classes,y=yt)
sw=np.array([cw[list(classes).index(c)]for c in yt])
yv_oh=np.zeros((len(yv),3))
yv_oh[np.arange(len(yv)),yv]=1

# ── Optuna ──
def objective(trial):
    params={
        'max_depth': trial.suggest_int('max_depth',2,4),
        'learning_rate': trial.suggest_float('learning_rate',0.01,0.05),
        'n_estimators': trial.suggest_int('n_estimators',150,450),
        'reg_alpha': trial.suggest_float('reg_alpha',0.5,12.0),
        'reg_lambda': trial.suggest_float('reg_lambda',2.0,25.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree',0.35,0.60),
        'subsample': trial.suggest_float('subsample',0.60,0.85),
        'min_child_weight': trial.suggest_float('min_child_weight',3.0,15.0),
        'random_state':42,'eval_metric':'mlogloss','verbosity':0}
    m=XGBClassifier(**params,early_stopping_rounds=25)
    m.fit(Xt,yt,eval_set=[(Xv,yv)],sample_weight=sw,verbose=False)
    yp=m.predict_proba(Xv)
    return np.mean(np.sum((yp-yv_oh)**2,axis=1))

print('\n🔥 Optuna 防守型调参 50 trials...',flush=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study=optuna.create_study(direction='minimize')
study.optimize(objective,n_trials=50,show_progress_bar=True)

print(f'\n🏆 最优Brier: {study.best_value:.4f}')
print('最优参数:')
for k,v in study.best_params.items():
    print(f"  '{k}': {v},")

# ── 用最优参数重新训练后做严格 2022 WC 回测 ──
print('\n⚽ 用最优参数做严格2022回测...',flush=True)
wc=[m for m in all_m if m['tournament']=='FIFA World Cup' and '2022-11-20'<=m['date']<='2022-12-18']

bp=study.best_params
m=XGBClassifier(**bp,random_state=42,eval_metric='mlogloss',verbosity=0)
m.fit(Xt,yt,eval_set=[(Xv,yv)],sample_weight=sw,verbose=False)
val_acc=np.mean(m.predict(Xv)==yv)
print(f'  验证集准确率: {val_acc*100:.1f}%',flush=True)

fb2=FB(clean_elo,clean_dc)
for mm in historical: fb2.add(mm)

weights=[(0.3,0.7),(0.4,0.6),(0.5,0.5),(0.6,0.4),(0.7,0.3)]
p33={w:{'c':0,'b':0.0}for w in weights}
p15={w:{'c':0,'b':0.0}for w in weights}

# 也训练一个15维基线做对比 (用相同参数但15维特征)
X15,y15=[],[]
fb15=FB(clean_elo,clean_dc)
ms_train=sorted(train,key=lambda m:m['date'])
for i,m in enumerate(ms_train):
    h,a=m['home'],m['away']
    eh,ea=clean_elo.get(h,1500),clean_elo.get(a,1500)
    lh,la=clean_dc.predict_lambda(h,a,neutral=m.get('neutral',False))
    if lh is None:continue
    dp=clean_dc.predict_proba(h,a,neutral=m.get('neutral',False))
    fh5=fb15.rf(h,m['date'],5);fa5=fb15.rf(a,m['date'],5)
    X15.append([(eh-ea)/400,lh,la,lh-la,math.log(max(lh,.01)/max(la,.01)),dp[0],dp[1],dp[2],
        fh5[0],fa5[0],fh5[1]-fa5[2],fa5[1]-fh5[2],fh5[1]-fa5[1],fh5[0]-fa5[0],int(m.get('neutral',0))])
    y15.append(2 if m['h_score']>m['a_score']else(1 if m['h_score']==m['a_score']else 0))
    fb15.add(m)
X15=np.array(X15);y15=np.array(y15)
cw15=compute_class_weight('balanced',classes=np.unique(y15),y=y15)
sw15=np.array([cw15[list(np.unique(y15)).index(c)]for c in y15])
m15=XGBClassifier(**bp,random_state=42,eval_metric='mlogloss',verbosity=0)
m15.fit(X15[train_mask],y15[train_mask],eval_set=[(X15[val_mask],y15[val_mask])],sample_weight=sw15[train_mask],verbose=False)

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
    x33r=m.predict_proba(f33)[0]
    
    for wd,wx in weights:
        h33=wd*da+wx*x33r
        if np.argmax(h33)==act:p33[(wd,wx)]['c']+=1
        p33[(wd,wx)]['b']+=np.sum((h33-yo)**2)
        h15=wd*da+wx*x15r
        if np.argmax(h15)==act:p15[(wd,wx)]['c']+=1
        p15[(wd,wx)]['b']+=np.sum((h15-yo)**2)
    fb2.add(mm)

n=len(wc)
print(f'\n{"="*60}')
print(f'  Optuna 调参后严格回测 ({n}场)')
print(f'{"="*60}')
print(f'  训练: <2021-06 ({len(Xt)}场)  验证: 2021-06~2022-11 ({len(Xv)}场)')
print(f'  2022 WC: 完全盲测')
print(f'{"─"*60}')
print(f'  {"权重":<22s} {"15维(Optuna参数)":>22s} {"33维(Optuna参数)":>22s}')
print(f'  {"─"*62}')
for w in weights:
    o=p15[w];nw=p33[w]
    print(f'  DC{w[0]:.1f}+XGB{w[1]:.1f}     {o["c"]/n*100:>7.2f}% B={o["b"]/n:.4f}  {nw["c"]/n*100:>7.2f}% B={nw["b"]/n:.4f}')
bo=max(weights,key=lambda w:p15[w]['c'])
bn=max(weights,key=lambda w:p33[w]['c'])
print(f'{"─"*62}')
print(f'  旧基线最佳: DC{bo[0]:.1f}+XGB{bo[1]:.1f} = {p15[bo]["c"]/n*100:.2f}%')
print(f'  33维最佳:   DC{bn[0]:.1f}+XGB{bn[1]:.1f} = {p33[bn]["c"]/n*100:.2f}%')
print(f'  Brier:      15维={p15[bo]["b"]/n:.4f}  33维={p33[bn]["b"]/n:.4f}')
print(f'{"="*60}')

import json
with open('/root/data/optuna_result.json','w') as f:
    json.dump({'best_params':study.best_params,'best_brier':study.best_value,
        'wc_test':{'old15':{f'DC{w[0]:.1f}+XGB{w[1]:.1f}':{'acc':p15[w]['c']/n,'brier':p15[w]['b']/n}for w in weights},
                   'new33':{f'DC{w[0]:.1f}+XGB{w[1]:.1f}':{'acc':p33[w]['c']/n,'brier':p33[w]['b']/n}for w in weights}}},
        f,indent=2,default=str)
print('已保存: /root/data/optuna_result.json')
