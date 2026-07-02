#!/usr/bin/env python3
"""Run 100k MC simulation for Canada vs Bosnia"""
import json, math
import numpy as np
from scipy.optimize import minimize

# --- Load data ---
with open('data/international_results.json') as f:
    raw = json.load(f)

results = []
for r in raw:
    if isinstance(r, dict) and 'home' in r:
        results.append(r)
    elif isinstance(r, list):
        results.extend(r)

print(f'Total matches: {len(results)}')

name_fix = {'United States': 'USA', 'Bosnia-Herzegovina': 'Bosnia',
            'Bosnia and Herzegovina': 'Bosnia',
            'Cape Verde Islands': 'Cape Verde', 'Curacao': 'Curaçao',
            'Czech Republic': 'Czechia'}

for r in results:
    r['home'] = name_fix.get(r['home'], r['home'])
    r['away'] = name_fix.get(r['away'], r['away'])

results = [r for r in results if int(r.get('date','2000')[:4]) >= 2021]
print(f'Recent matches (2021+): {len(results)}')

team_counts = {}
for r in results:
    team_counts[r['home']] = team_counts.get(r['home'], 0) + 1
    team_counts[r['away']] = team_counts.get(r['away'], 0) + 1

top_names = set(t for t,_ in sorted(team_counts.items(), key=lambda x:-x[1])[:60])
min_teams = {'Canada', 'Bosnia'}
for mt in min_teams:
    if mt in team_counts:
        top_names.add(mt)

team_list = sorted(top_names)
t2i = {t:i for i,t in enumerate(team_list)}
n = len(team_list)
filtered = [r for r in results if r['home'] in top_names and r['away'] in top_names]
print(f'Teams: {n}, matches: {len(filtered)}')
print(f'Canada: {team_counts.get("Canada",0)} matches')
print(f'Bosnia: {team_counts.get("Bosnia",0)} matches')

# Check if Bosnia has enough data
if team_counts.get('Bosnia',0) < 5:
    print('WARNING: Bosnia has very limited recent data!')

def dc_log_likelihood(params, data):
    attack = params[:n]; defense = params[n:2*n]; rho = params[-1]
    ll = 0.0
    for r in data:
        i = t2i.get(r['home'], -1); j = t2i.get(r['away'], -1)
        if i==-1 or j==-1: continue
        try: x=int(r['h_score']); y=int(r['a_score'])
        except: continue
        lam = np.exp(attack[i] + defense[j])
        mu = np.exp(attack[j] + defense[i])
        tau = 1.0
        if x==0 and y==0: tau = 1 - rho*lam*mu
        elif x==0 and y==1: tau = 1 + rho*lam
        elif x==1 and y==0: tau = 1 + rho*mu
        elif x==1 and y==1: tau = 1 - rho
        if tau<=0: tau=1e-10
        log_prob = (x*np.log(lam)-lam - np.sum(np.log(np.arange(1,x+1))) +
                    y*np.log(mu)-mu - np.sum(np.log(np.arange(1,y+1))) +
                    np.log(tau))
        ll += log_prob
    return -ll

init = np.zeros(2*n+1); init[-1]=0.0
print('Fitting Dixon-Coles...')
res = minimize(dc_log_likelihood, init, args=(filtered,),
               method='L-BFGS-B', options={'maxiter':20000,'disp':False})
attack=res.x[:n]; defense=res.x[n:2*n]; rho=res.x[-1]
print(f'Rho: {rho:.6f}')

can_i=t2i['Canada']; bos_i=t2i['Bosnia']
lam_h=np.exp(attack[can_i]+defense[bos_i])
lam_a=np.exp(attack[bos_i]+defense[can_i])
print(f'\n=== DC EXPECTED GOALS ===')
print(f'Canada: {lam_h:.4f}  Bosnia: {lam_a:.4f}')

# Attack/defense params
print(f'\nCanada attack: {attack[can_i]:.4f}, defense: {defense[can_i]:.4f}')
print(f'Bosnia attack: {attack[bos_i]:.4f}, defense: {defense[bos_i]:.4f}')

# DC probabilities
dc_p = np.zeros((9,9))
for i in range(9):
    pi = np.exp(-lam_h)*(lam_h**i)/math.factorial(i)
    for j in range(9):
        pj = np.exp(-lam_a)*(lam_a**j)/math.factorial(j)
        tau=1.0
        if i==0 and j==0: tau=1-rho*lam_h*lam_a
        elif i==0 and j==1: tau=1+rho*lam_h
        elif i==1 and j==0: tau=1+rho*lam_a
        elif i==1 and j==1: tau=1-rho
        dc_p[i,j]=pi*pj*max(tau,0)
dc_p/=dc_p.sum()
dc_h = np.sum(dc_p*(np.arange(9)[:,None]>np.arange(9)[None,:]))
dc_d = np.sum(np.diag(dc_p))
dc_a = np.sum(dc_p*(np.arange(9)[:,None]<np.arange(9)[None,:]))
print(f'DC HDA: {dc_h:.1%} / {dc_d:.1%} / {dc_a:.1%}')

# --- 100K MC ---
N=100000
np.random.seed(42)
wins=dr=loss=0
sc={}
for _ in range(N):
    h=np.random.poisson(lam_h); a=np.random.poisson(lam_a)
    if h>a: wins+=1
    elif h==a: dr+=1
    else: loss+=1
    k=f'{h}-{a}'
    sc[k]=sc.get(k,0)+1

hw=wins/N; drw=dr/N; aw=loss/N
print(f'\n{"="*55}')
print(f'   100,000 MONTE CARLO SIMULATIONS')
print(f'   Canada vs Bosnia')
print(f'{"="*55}')
print(f'\nRESULT:')
print(f'Canada Win:   {hw:.1%}  {"#"*int(hw*80)}')
print(f'Draw:         {drw:.1%}  {"#"*int(drw*80)}')
print(f'Bosnia Win:   {aw:.1%}  {"#"*int(aw*80)}')

print(f'\nTOP 15 SCORES:')
for s,c in sorted(sc.items(), key=lambda x:-x[1])[:15]:
    print(f'  {s:>5s}: {c/N:.1%}  {"#"*int(c/N*200)}')

bts=sum(c for k,c in sc.items() if int(k[0])>0 and int(k[2])>0)
ov25=sum(c for k,c in sc.items() if int(k[0])+int(k[2])>2)
un15=sum(c for k,c in sc.items() if int(k[0])+int(k[2])<=1)
print(f'\nMARKETS:')
print(f'  BTS:         {bts/N:.1%}')
print(f'  Over 2.5:    {ov25/N:.1%}')
print(f'  Under 1.5:   {un15/N:.1%}')

marg={}
for k,c in sc.items():
    d=int(k[0])-int(k[2])
    marg[d]=marg.get(d,0)+c
print(f'\nMARGINS:')
for d in sorted(marg):
    p=marg[d]/N
    if p>0.01:
        l=f'CAN+{d}' if d>0 else (f'BIH+{-d}' if d<0 else 'Draw')
        print(f'  {l:>10s}: {p:.1%}  {"#"*int(p*100)}')

bs=sorted(sc.items(),key=lambda x:-x[1])[0]
print(f'\nBEST PICK: Canada {bs[0]} ({bs[1]/N:.1%})')
print(f'MODEL CALL: {"HOME" if hw>aw+drw else "DRAW" if drw>max(hw,aw) else "AWAY"}')

w1=sum(c for k,c in sc.items() if int(k[0])-int(k[2])==1)/N
w2p=sum(c for k,c in sc.items() if int(k[0])-int(k[2])>=2)/N
l1=sum(c for k,c in sc.items() if int(k[2])-int(k[0])==1)/N
l2p=sum(c for k,c in sc.items() if int(k[2])-int(k[0])>=2)/N
print(f'\nDETAIL:')
print(f'  Canada win 2+: {w2p:.1%}')
print(f'  Canada win 1:  {w1:.1%}')
print(f'  Draw:          {drw:.1%}')
print(f'  Bosnia win 1:  {l1:.1%}')
print(f'  Bosnia win 2+: {l2p:.1%}')
print(f'\nDone: {N:,} simulations')
