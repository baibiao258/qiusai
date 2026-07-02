#!/usr/bin/env python3
import json, os, itertools, math
from collections import defaultdict
from datetime import datetime
import concurrent.futures
import numpy as np
import joblib
import sys
sys.path.insert(0,'/root')
import wc_2026_final as wf

DATA_DIR='/root/data'

# ---------- helpers ----------
def load_json(p):
    with open(p,'r',encoding='utf-8') as f:
        return json.load(f)

def champ_top_map(champs_list, topn=10):
    out={}
    for x in champs_list[:topn]:
        # format from final_results: [team,count,pct]
        if isinstance(x,(list,tuple)) and len(x)>=3:
            out[x[0]]=float(x[2])
    return out

def team_prob_map(team_probs, key='qualify_r32_prob', topn=16):
    s=sorted(team_probs, key=lambda x:x[key], reverse=True)[:topn]
    return {x['team']:float(x[key])*100 for x in s}

# ---------- 200k champion simulation (same assumptions as wf) ----------
def build_mc_cache(dc, xgb, elo, teams, market_probs):
    mc_cache={}
    def make_cdf(lam, K=6):
        c=0.0; arr=[]
        for k in range(K+1):
            c += math.exp(-lam) * (lam**k) / math.factorial(k)
            arr.append(c)
        return arr

    for h in teams:
        for a in teams:
            if h==a: continue
            feat = wf.build_features(h,a,dc,elo,{}, {})
            xp = xgb.predict_proba(np.array([feat]))[0]  # [A,D,H]
            dp = dc.predict_proba(h,a,neutral=True)      # [H,D,A]
            hp = wf.DC_WEIGHT*np.array([dp[2],dp[1],dp[0]]) + wf.XGB_WEIGHT*xp
            lam_h, lam_a = dc.predict_lambda(h,a,neutral=True)
            # market calibration same as wf
            final_hybrid = hp
            if market_probs:
                mh = market_probs.get(h, 0)
                ma = market_probs.get(a, 0)
                if mh>0 and ma>0:
                    blended_h = hp[2]*wf.MODEL_WEIGHT + (mh/(mh+ma+0.01))*wf.MARKET_WEIGHT
                    blended_a = hp[0]*wf.MODEL_WEIGHT + (ma/(mh+ma+0.01))*wf.MARKET_WEIGHT
                    blended_d = max(0, 1-blended_h-blended_a)
                    final_hybrid = np.array([blended_a, blended_d, blended_h])
            mc_cache[(h,a)] = (
                float(final_hybrid[0]), float(final_hybrid[1]), float(final_hybrid[2]),
                float(lam_h), float(lam_a), make_cdf(float(lam_h)), make_cdf(float(lam_a))
            )
    return mc_cache


def run_champion_200k():
    xgb = joblib.load('/root/data/xgb_model_20_3.pkl')
    dc = joblib.load('/root/data/dc_model.pkl')
    elo = joblib.load('/root/data/elo_ratings.pkl')

    groups = load_json('/root/data/2026_groups.json')
    teams = sorted(set(t for g in groups.values() for t in g))

    market_data = wf.load_market_odds()
    market_probs = market_data['winner_probs'] if market_data else {}

    mc_cache = build_mc_cache(dc, xgb, elo, teams, market_probs)
    mc_flat={f"{h}||{a}":v for (h,a),v in mc_cache.items()}

    N=200000; n_workers=2
    sims_per=N//n_workers
    champ=defaultdict(int)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs=[]
        for w in range(n_workers):
            futs.append(ex.submit(
                wf._sim_worker, mc_flat, dict(elo), w*99999+4242,
                sims_per, teams, groups, wf.HOST_TEAMS, wf.HOST_BONUS
            ))
        for f in concurrent.futures.as_completed(futs):
            res=f.result()
            for k,v in res.items(): champ[k]+=v

    total=sum(champ.values())
    champs=sorted(champ.items(), key=lambda x:-x[1])
    top=[{'team':t,'count':c,'pct':100*c/total} for t,c in champs[:20]]
    return {'sims':total,'top20':top}

# ---------- 200k group advancement ----------
def run_advancement_200k():
    gp = load_json('/root/data/group_stage_predictions.json')['predictions']
    groups = load_json('/root/data/2026_groups.json')

    probs={}
    for r in gp:
        probs[(r['home'],r['away'])]=(r['prob_home'],r['prob_draw'],r['prob_away'])
        probs[(r['away'],r['home'])]=(r['prob_away'],r['prob_draw'],r['prob_home'])

    teams=sorted(set(t for g in groups.values() for t in g))
    stats={t:{'q':0,'1':0,'2':0,'3':0,'4':0} for t in teams}

    rng=np.random.default_rng(20260523)
    N=200000
    group_keys=sorted(groups.keys())

    for _ in range(N):
        rank_by_group={}
        tiebreak={}
        # group stage
        for gk in group_keys:
            ts=groups[gk]
            pts={t:0 for t in ts}; gd={t:0 for t in ts}; gf={t:0 for t in ts}
            for i in range(4):
                for j in range(i+1,4):
                    a,b=ts[i],ts[j]
                    ph,pd,pa=probs[(a,b)]
                    u=rng.random()
                    if u<ph:
                        pts[a]+=3; ga,gb=1,0
                    elif u<ph+pd:
                        pts[a]+=1; pts[b]+=1; ga,gb=1,1
                    else:
                        pts[b]+=3; ga,gb=0,1
                    gf[a]+=ga; gf[b]+=gb
                    gd[a]+=ga-gb; gd[b]+=gb-ga
            rk=sorted(ts,key=lambda t:(pts[t],gd[t],gf[t],rng.random()), reverse=True)
            rank_by_group[gk]=rk
            for t in ts: tiebreak[t]=(pts[t],gd[t],gf[t])

        thirds=[]
        for gk in group_keys:
            r=rank_by_group[gk]
            stats[r[0]]['1']+=1; stats[r[1]]['2']+=1; stats[r[2]]['3']+=1; stats[r[3]]['4']+=1
            thirds.append(r[2])

        best8=sorted(thirds,key=lambda t:(tiebreak[t][0],tiebreak[t][1],tiebreak[t][2],rng.random()), reverse=True)[:8]
        for gk in group_keys:
            r=rank_by_group[gk]
            stats[r[0]]['q']+=1; stats[r[1]]['q']+=1
            if r[2] in best8: stats[r[2]]['q']+=1

    out=[]
    for t,v in stats.items():
        out.append({
            'team':t,
            'qualify_r32_prob':v['q']/N,
            'group_1st':v['1']/N,
            'group_2nd':v['2']/N,
            'group_3rd':v['3']/N,
            'group_4th':v['4']/N,
        })
    out.sort(key=lambda x:x['qualify_r32_prob'], reverse=True)
    return {'sims':N,'team_probs':out}


def main():
    baseline50 = load_json('/root/data/final_results.json')
    baseline100 = load_json('/root/data/group_advancement_probs.json')

    ch200 = run_champion_200k()
    adv200 = run_advancement_200k()

    # comparisons
    top50 = champ_top_map(baseline50['champs'], topn=10)
    top200 = {x['team']:x['pct'] for x in ch200['top20'][:10]}
    champ_drift=[]
    for t,p in top200.items():
        old = top50.get(t,0.0)
        champ_drift.append({'team':t,'p200':p,'p50':old,'delta_pp':p-old})

    # hosts
    hosts=list(wf.HOST_TEAMS)
    host_drift=[]
    for h in hosts:
        p200=next((x['pct'] for x in ch200['top20'] if x['team']==h),0.0)
        p50=next((x[2] for x in baseline50['champs'] if x[0]==h),0.0)
        host_drift.append({'team':h,'p200':p200,'p50':p50,'delta_pp':p200-p50})

    t100 = team_prob_map(baseline100['team_probs'], 'qualify_r32_prob', 16)
    t200 = team_prob_map(adv200['team_probs'], 'qualify_r32_prob', 16)
    r32_drift=[]
    keys=sorted(set(t100)|set(t200), key=lambda k: -(t200.get(k,0)))[:20]
    for k in keys:
        r32_drift.append({'team':k,'q200':t200.get(k,0.0),'q100':t100.get(k,0.0),'delta_pp':t200.get(k,0.0)-t100.get(k,0.0)})

    # group stability compare (top2+3rd)
    grp=load_json('/root/data/2026_groups.json')
    team_to_group={t:g for g,ts in grp.items() for t in ts}
    st100={x['team']:x for x in baseline100['team_probs']}
    st200={x['team']:x for x in adv200['team_probs']}
    group_compare=[]
    for g in sorted(grp):
        rows=[]
        for t in grp[g]:
            a=st100[t]; b=st200[t]
            rows.append({
                'team':t,
                'q100':a['qualify_r32_prob']*100,
                'q200':b['qualify_r32_prob']*100,
                'dq_pp':(b['qualify_r32_prob']-a['qualify_r32_prob'])*100,
                'first100':a['group_1st']*100,
                'first200':b['group_1st']*100,
                'd1_pp':(b['group_1st']-a['group_1st'])*100,
                'second100':a['group_2nd']*100,
                'second200':b['group_2nd']*100,
                'd2_pp':(b['group_2nd']-a['group_2nd'])*100,
                'third100':a['group_3rd']*100,
                'third200':b['group_3rd']*100,
                'd3_pp':(b['group_3rd']-a['group_3rd'])*100,
            })
        max_abs=max(abs(r['dq_pp']) for r in rows)
        group_compare.append({'group':g,'max_abs_q_drift_pp':max_abs,'teams':rows})

    out={
      'generated_at':datetime.utcnow().isoformat()+'Z',
      'assumptions':'unchanged: DC/XGB weights, rho, host_bonus, market blend, matchup cache logic',
      'champion_200k':ch200,
      'advancement_200k':{'sims':adv200['sims']},
      'compare':{
        'champion_top10_200k_vs_50k':champ_drift,
        'hosts_champion_200k_vs_50k':host_drift,
        'r32_top16_200k_vs_100k':r32_drift,
        'group_stability_200k_vs_100k':group_compare,
      }
    }

    outp='/root/data/mc200k_compare.json'
    with open(outp,'w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print(outp)

if __name__=='__main__':
    main()
