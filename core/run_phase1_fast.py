#!/usr/bin/env python3
"""Phase 1 fast runner (precomputed CDF)"""
import sys, os, json, math, random
from datetime import datetime
from collections import defaultdict
sys.path.insert(0, '/root')
from wc_2026_phase1 import *
import pandas as pd

out = []
def log(s=""):
    print(s, flush=True)
    out.append(s)

log("="*65)
log(f"  Phase 1 快速模式: DC + XGB + MC (CDF)")
log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("="*65)

cache = os.path.join(DATA_DIR, 'international_results.json')
all_m = load_data(cache)
matches = filter_matches(all_m)
elo = compute_elo(all_m)

df = pd.DataFrame(matches)
dc = DixonColes(time_decay_hl=540)
dc.fit(df)
log(f"  rho={dc.rho_:.4f} gamma={dc.gamma_:.4f}")

X, y, _ = build_features(matches, dc, elo)
res = train_xgboost(X, y)
xgb_model = res[0]

bt = backtest_2022(dc, xgb_model, all_m)

mc = precompute_matchups(dc, xgb_model, TEAMS_2026, elo)

# Fast MC: 50K
N = 50000
log(f"\nMC {N:,} simulations...")
champ = defaultdict(int)
BATCH = 10000

for batch in range(N // BATCH):
    for _ in range(BATCH):
        st = sorted(TEAMS_2026, key=lambda t: elo.get(t,1500), reverse=True)
        pots = [st[i:i+16] for i in range(0, 48, 16)]
        groups = {}
        for pi, pot in enumerate(pots):
            sh = list(pot); random.shuffle(sh)
            for gi, tm in enumerate(sh):
                gn = chr(ord('A')+gi)
                if gn not in groups: groups[gn] = []
                groups[gn].append(tm)
        q = []
        for gn in sorted(groups):
            gt = groups[gn]
            if len(gt) != 3: continue
            pts = {t:0 for t in gt}; gd = {t:0 for t in gt}; gf = {t:0 for t in gt}
            for t1,t2 in [(gt[0],gt[1]),(gt[0],gt[2]),(gt[1],gt[2])]:
                hg,ag = simulate_from_cache(mc, elo, t1, t2)
                gf[t1]+=hg; gf[t2]+=ag; gd[t1]+=hg-ag; gd[t2]+=ag-hg
                if hg>ag: pts[t1]+=3
                elif hg==ag: pts[t1]+=1; pts[t2]+=1
                else: pts[t2]+=3
            rk = sorted(gt, key=lambda t:(pts[t],gd[t],gf[t]), reverse=True)
            q.append(rk[:2])
        if len(q)!=16: continue
        r32 = []
        for i in range(0,16,2):
            r32.append((q[i][0], q[i+1][1]))
            r32.append((q[i+1][0], q[i][1]))
        cur = r32
        for _ in range(5):
            if len(cur)<=1: break
            nxt = []
            for i in range(0,len(cur),2):
                t1,t2=cur[i][0],cur[i+1][0]
                hg,ag = simulate_from_cache(mc, elo, t1, t2)
                if hg==ag:
                    hg2,ag2 = simulate_from_cache(mc, elo, t1, t2)
                    hg+=hg2; ag+=ag2
                    if hg==ag:
                        e1,e2=elo.get(t1,1500),elo.get(t2,1500)
                        pp=0.5+(1/(1+10**((e2-e1)/400))-0.5)*0.3
                        winner=t1 if random.random()<pp else t2
                        nxt.append((winner,None))
                        continue
                winner=t1 if hg>ag else t2
                nxt.append((winner,None))
            cur=nxt
        if cur: champ[cur[0][0]]+=1
    log(f"  batch {batch+1}/{N//BATCH} done")

total = sum(champ.values())
log(f"\nDone! {total:,} simulations")
log(f"\n{'='*65}")
log(f"  TROPHY 2026 Champion Probabilities (Hybrid Model)")
log(f"{'='*65}")
champs = sorted(champ.items(), key=lambda x:-x[1])
best_pct = champs[0][1]/total*100 if champs else 0
for i,(t,c) in enumerate(champs[:20], 1):
    pct = c/total*100
    bar = chr(9608)*int(pct/best_pct*20) + chr(9617)*(20-int(pct/best_pct*20))
    log(f"  {i:>3d}. {t:<25s} {c:>6,d} {pct:>6.2f}% {bar}")
if len(champs)>20:
    oc = sum(c for _,c in champs[20:])
    log(f"  {' '*4} Others ({len(champs)-20} teams) {oc:>6,d} {oc/total*100:>6.2f}%")

result = {'model':'DC+XGB+MC-CDF','ts':datetime.now().isoformat(),
          'sims':total,'rho':dc.rho_,'gamma':dc.gamma_,
          'backtest':bt,
          'champs':[(t,c,c/total*100) for t,c in champs[:30]]}
os.makedirs(DATA_DIR, exist_ok=True)
with open(os.path.join(DATA_DIR,'phase1_results.json'),'w') as f:
    json.dump(result, f, indent=2, default=str)
log(f"\n  Saved to {DATA_DIR}/phase1_results.json")

# Write output to file
with open('/tmp/phase1_final.txt','w') as f:
    f.write('\n'.join(out))
