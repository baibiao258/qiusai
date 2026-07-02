#!/usr/bin/env python3
"""
Knockout stage: full detail for ALL 31 matches along the expected bracket path.
Every match: score distribution (8), total goals (6), HT/FT (6), half-time probs.
No compression, no hiding.
"""
import sys, os, json, math
sys.path.insert(0, '/root')
import numpy as np
from collections import defaultdict
from team_name_normalizer import normalize_match_pair
import joblib

DATA_DIR = '/root/data'
_dc = joblib.load(os.path.join(DATA_DIR, 'dc_model.pkl'))
_xgb = joblib.load(os.path.join(DATA_DIR, 'xgb_model_20_3.pkl'))
_elo = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))
HOST_TEAMS = {'United States', 'Mexico', 'Canada'}
HOST_BONUS = {'United States': 0.1445, 'Mexico': 0.10, 'Canada': 0.07}
DC_WEIGHT = 0.4
XGB_WEIGHT = 0.6
MAX_GOALS = 7

with open(f'{DATA_DIR}/group_stage_predictions.json') as f:
    GROUP_MATCHES = json.load(f)
with open(f'{DATA_DIR}/2026_groups.json') as f:
    GROUPS = json.load(f)

def make_odds(eh, ea):
    dh = ea - eh; da = eh - ea
    return [1/(10**(-dh/400)+1), 1/(10**(-da/400)+1), 0.0]

def predict_match_detail(home, away, host_bonus=0.0):
    h, a = normalize_match_pair(home, away)
    is_host = host_bonus > 0 and home in HOST_TEAMS
    neutral = not is_host
    dc_p = _dc.predict_proba(h, a, neutral, host_bonus=host_bonus if is_host else 0.0)
    lam_h, lam_a = _dc.predict_lambda(h, a, neutral, host_bonus=host_bonus if is_host else 0.0)
    if lam_h is None: return None
    eh = _elo.get(home,1500); ea = _elo.get(away,1500)
    op = make_odds(eh, ea)
    fh5=[0.5,0,0,0]; fa5=[0.5,0,0,0]
    b15=[(eh-ea)/400, lam_h,lam_a, lam_h-lam_a, math.log(max(lam_h,.01)/max(lam_a,.01)),
         dc_p[0],dc_p[1],dc_p[2],fh5[0],fa5[0],fh5[1]-fa5[2],fa5[1]-fh5[2],
         fh5[1]-fa5[1],fh5[0]-fa5[0],0 if is_host else 1]
    gold=[0.0,1,0,0.0,0.0]; odds_feat=[op[0],op[1],op[2] if op[2] else 0.0]
    feat=np.array([b15+gold+odds_feat])
    xgb_p=_xgb.predict_proba(feat)[0]
    dc_ado=np.array([dc_p[2],dc_p[1],dc_p[0]])
    hybrid=DC_WEIGHT*dc_ado+XGB_WEIGHT*xgb_p

    n_sim=100000
    rng=np.random.RandomState()
    hg=rng.poisson(lam_h,n_sim); ag=rng.poisson(lam_a,n_sim)
    scores=defaultdict(int); htg_all=defaultdict(int)
    ht_lam_h=lam_h*0.45; ht_lam_a=lam_a*0.45
    htg=rng.poisson(ht_lam_h,n_sim); atg=rng.poisson(ht_lam_a,n_sim)
    for i in range(n_sim):
        sk=(min(hg[i],MAX_GOALS),min(ag[i],MAX_GOALS)); scores[sk]+=1
        htk=(min(htg[i],MAX_GOALS),min(atg[i],MAX_GOALS)); htg_all[htk]+=1
    score_pct=[{"score":f"{s[0]}:{s[1]}","pct":round(c/n_sim*100,1)} for s,c in sorted(scores.items(),key=lambda x:-x[1])[:8]]
    ht_score_pct=[{"score":f"{s[0]}:{s[1]}","pct":round(c/n_sim*100,1)} for s,c in sorted(htg_all.items(),key=lambda x:-x[1])[:8]]
    hhp=sum(c for s,c in htg_all.items() if s[0]>s[1])/n_sim*100
    hdp=sum(c for s,c in htg_all.items() if s[0]==s[1])/n_sim*100
    hap=sum(c for s,c in htg_all.items() if s[0]<s[1])/n_sim*100
    htft=defaultdict(int)
    ft_res=[(0 if hg[i]<ag[i] else(1 if hg[i]==ag[i] else 2)) for i in range(n_sim)]
    ht_res=[(0 if htg[i]<atg[i] else(1 if htg[i]==atg[i] else 2)) for i in range(n_sim)]
    for hr,fr in zip(ht_res,ft_res): htft[hr*3+fr]+=1
    htft_lbl=['负/负','负/平','负/胜','平/负','平/平','平/胜','胜/负','胜/平','胜/胜']
    htft_pct=[{"label":htft_lbl[i],"pct":round(c/n_sim*100,1)} for i,c in sorted(htft.items(),key=lambda x:-x[1])[:6]]
    tg_dist=defaultdict(int)
    for i in range(n_sim): tg_dist[min(hg[i]+ag[i],MAX_GOALS*2)]+=1
    tg_pct=[{"goals":g,"pct":round(c/n_sim*100,1)} for g,c in sorted(tg_dist.items(),key=lambda x:-x[0])]
    tg_pct.sort(key=lambda x:-x['pct'])
    # Determine winner: if ET/pen needed, mark it
    et_needed = False; pen_needed = False
    for i in range(n_sim):
        if hg[i] == ag[i]:
            # check ET
            ehg=rng.poisson(lam_h*0.3); eag=rng.poisson(lam_a*0.3)
            if ehg != eag: et_needed=True
            else: pen_needed=True
            break
    return {
        'home':home,'away':away,'lam_h':round(lam_h,2),'lam_a':round(lam_a,2),
        'elo_h':eh,'elo_a':ea,
        'prob_h':round(hybrid[2]*100,1),'prob_d':round(hybrid[1]*100,1),'prob_a':round(hybrid[0]*100,1),
        'dc_h':round(dc_p[0]*100,1),'dc_d':round(dc_p[1]*100,1),'dc_a':round(dc_p[2]*100,1),
        'xgb_h':round(xgb_p[2]*100,1),'xgb_d':round(xgb_p[1]*100,1),'xgb_a':round(xgb_p[0]*100,1),
        'ht_h':round(hhp,1),'ht_d':round(hdp,1),'ht_a':round(hap,1),
        'scores':score_pct,'ht_scores':ht_score_pct,'htft':htft_pct,'total_goals':tg_pct[:6],
        'host_bonus':host_bonus if is_host else 0,
    }

# ── Expected bracket ─────────────────────────────────────────────────
gp = {g:{t:{'pts':0.0} for t in GROUPS[g]} for g in GROUPS}
for m in GROUP_MATCHES:
    h,a=m['home'],m['away']
    g=next(grp for grp in GROUPS if h in GROUPS[grp])
    hp,dp,ap=m['home_win']/100,m['draw']/100,m['away_win']/100
    gp[g][h]['pts']+=hp*3+dp; gp[g][a]['pts']+=ap*3+dp

gw={g:sorted(GROUPS[g],key=lambda t:-gp[g][t]['pts'])[0] for g in GROUPS}
ru={g:sorted(GROUPS[g],key=lambda t:-gp[g][t]['pts'])[1] for g in GROUPS}
third=[]
for g in GROUPS:
    t3=sorted(GROUPS[g],key=lambda x:-gp[g][x]['pts'])[2]
    third.append((t3,g,gp[g][t3]['pts']))
third.sort(key=lambda x:(-x[2],-_elo.get(x[0],1500)))
bt_g={g:t for t,g,_ in third[:8]}

THIRD_SLOTS=[('M74',['A','B','C','D','F']),('M77',['C','D','F','G','H']),
             ('M79',['C','E','F','H','I']),('M80',['E','H','I','J','K']),
             ('M81',['B','E','F','I','J']),('M82',['A','E','H','I','J']),
             ('M85',['E','F','G','I','J']),('M87',['D','E','I','J','L'])]
# Assign third-placed teams to slots: simple greedy by rank
# Eligible groups per slot
slot_eligible={s[0]:s[1] for s in THIRD_SLOTS}
# Rank third-placed teams
third_ranked=sorted(third[:8], key=lambda x:(-x[2],-_elo.get(x[0],1500)))
tmap={}; used_slots=set()
# Assign best teams to most prestigious slots first
slot_order=sorted([s[0] for s in THIRD_SLOTS], key=lambda s:-_elo.get(gw[s[1][1]],1500) if s[0]=='M74' else -_elo.get(gw['G'],1500))
# Simpler: just iterate through ranked teams and assign each to first eligible slot
used_slots=set()
for team,group,pts in third_ranked:
    for slot,eg in THIRD_SLOTS:
        if group in eg and slot not in used_slots:
            tmap[slot]=(team,group)
            used_slots.add(slot)
            break
# Assign any remaining slots
for slot,eg in THIRD_SLOTS:
    if slot not in tmap:
        for team,group,pts in third_ranked:
            if slot not in used_slots and (slot,team) not in [(s,t) for s,t in tmap.items()]:
                tmap[slot]=(team,group)
                used_slots.add(slot)
                break

R32_SPEC=[
    ('M73',('2','A'),('2','B')),('M74',('1','E'),('3',['A','B','C','D','F'])),
    ('M75',('1','F'),('2','C')),('M76',('1','C'),('2','F')),
    ('M77',('1','I'),('3',['C','D','F','G','H'])),('M78',('2','E'),('2','I')),
    ('M79',('1','A'),('3',['C','E','F','H','I'])),('M80',('1','L'),('3',['E','H','I','J','K'])),
    ('M81',('1','D'),('3',['B','E','F','I','J'])),('M82',('1','G'),('3',['A','E','H','I','J'])),
    ('M83',('2','K'),('2','L')),('M84',('1','H'),('2','J')),
    ('M85',('1','B'),('3',['E','F','G','I','J'])),('M86',('1','J'),('2','H')),
    ('M87',('1','K'),('3',['D','E','I','J','L'])),('M88',('2','D'),('2','G')),
]
R16_PATH=[('M89','M74','M77'),('M90','M73','M75'),('M91','M76','M78'),
          ('M92','M79','M80'),('M93','M83','M84'),('M94','M81','M82'),
          ('M95','M86','M88'),('M96','M85','M87')]
QF_PATH=[('M97','M89','M90'),('M98','M93','M94'),('M99','M91','M92'),('M100','M95','M96')]
SF_PATH=[('M101','M97','M98'),('M102','M99','M100')]

def resolve(l,spec):
    p,g=spec
    if p=='1': return gw[g]
    if p=='2': return ru[g]
    if p=='3':
        if l in tmap: return tmap[l][0]
        return bt_g.get(g[0],'?')
    return '?'

# Build R32
r32_matches=[(l,resolve(l,h),resolve(l,a)) for l,h,a in R32_SPEC]

# Predict winners for R32 to build R16
print("="*70)
print("  2026 世界杯淘汰赛 — 72场逐一细项预测")
print("  路书: 官方FIFA路书 (openfootball/worldcup cup_finals.txt)")
print("  每场含: 8比分 | 6总进球 | 6半全场 | 半场概率")
print("="*70)

def fmt_match(r, title, label=""):
    h,a=r['home'],r['away']
    hw,dr,aw=r['prob_h'],r['prob_d'],r['prob_a']
    mx=max(hw,dr,aw)
    if mx==hw: pred=f"{h} 胜 ({hw}%)"
    elif mx==dr: pred=f"平局 ({dr}%)"
    else: pred=f"{a} 胜 ({aw}%)"
    # Confidence
    conf='🔴 高' if mx>60 else('🟡 中' if mx>45 else'🟢 低')
    host_str=f" (+{r['host_bonus']}东道主)" if r['host_bonus']>0 else ""
    lines=[]
    lines.append(f"\n{'─'*65}")
    lines.append(f"  {title} {label}")
    lines.append(f"{'─'*65}")
    lines.append(f"  {h:25s} vs {a}{host_str}")
    lines.append(f"  Elo {r['elo_h']:.0f} vs {r['elo_a']:.0f}  |  λ {r['lam_h']:.2f} : {r['lam_a']:.2f}")
    lines.append(f"  ───────────────────────────────────────")
    lines.append(f"  胜平负: 主 {hw:.1f}% | 平 {dr:.1f}% | 客 {aw:.1f}%  → {pred}  {conf}")
    lines.append(f"  DC模型: 主 {r['dc_h']:.1f}% 平 {r['dc_d']:.1f}% 客 {r['dc_a']:.1f}%")
    lines.append(f"  XGBoost: 主 {r['xgb_h']:.1f}% 平 {r['xgb_d']:.1f}% 客 {r['xgb_a']:.1f}%")
    lines.append(f"  ───────────────────────────────────────")
    # Half time
    lines.append(f"  半场概率: 主 {r['ht_h']:.1f}% | 平 {r['ht_d']:.1f}% | 客 {r['ht_a']:.1f}%")
    # Half time scores
    hs=" | ".join(f"{s['score']}({s['pct']}%)" for s in r['ht_scores'][:5])
    lines.append(f"  半场比分: {hs}")
    # Score distribution
    ss=" | ".join(f"{s['score']}({s['pct']}%)" for s in r['scores'])
    lines.append(f"  全场比赛比分分布：")
    lines.append(f"    {ss}")
    # Total goals
    tg=" | ".join(f"{t['goals']}球({t['pct']}%)" for t in r['total_goals'])
    lines.append(f"  总进球: {tg}")
    # HT/FT
    htfty=" | ".join(f"{h['label']}({h['pct']}%)" for h in r['htft'])
    lines.append(f"  半全场9向: {htfty}")
    return "\n".join(lines)

# Round 1: R32
print(f"\n{'#'*70}")
print(f"  R32  —  第1轮  × 16 场")
print(f"{'#'*70}")
r32_results={}
for i,(l,h,a) in enumerate(r32_matches,1):
    r=predict_match_detail(h,a)
    if a in HOST_TEAMS and a not in [gw[g] for g in GROUPS]:
        # Check if away is host
        pass
    if not r: continue
    print(fmt_match(r, f"R32#{i:2d}", f"({l})"))
    # Determine winner
    if r['prob_h']>r['prob_a']: winner=h
    elif r['prob_a']>r['prob_h']: winner=a
    else: winner=h if _elo.get(h,1500)>_elo.get(a,1500) else a
    r32_results[l]=winner
    print(f"  → 预测晋级: {winner}")
    r32_results[f"{l}_detail"]=r

# R16
print(f"\n{'#'*70}")
print(f"  R16  —  第2轮  × 8 场")
print(f"{'#'*70}")
r16_matches=[(nl,r32_results[m1],r32_results[m2]) for nl,m1,m2 in R16_PATH]
r16_results={}
for i,(l,h,a) in enumerate(r16_matches,1):
    r=predict_match_detail(h,a)
    if not r: continue
    print(fmt_match(r, f"R16#{i:2d}", f"({l})"))
    if r['prob_h']>r['prob_a']: winner=h
    elif r['prob_a']>r['prob_h']: winner=a
    else: winner=h if _elo.get(h,1500)>_elo.get(a,1500) else a
    r16_results[l]=winner
    print(f"  → 预测晋级: {winner}")

# QF
print(f"\n{'#'*70}")
print(f"  QF  —  第3轮  × 4 场")
print(f"{'#'*70}")
qf_matches=[(nl,r16_results[m1],r16_results[m2]) for nl,m1,m2 in QF_PATH]
qf_results={}
for i,(l,h,a) in enumerate(qf_matches,1):
    r=predict_match_detail(h,a)
    if not r: continue
    print(fmt_match(r, f"QF#{i:2d}", f"({l})"))
    if r['prob_h']>r['prob_a']: winner=h
    elif r['prob_a']>r['prob_h']: winner=a
    else: winner=h if _elo.get(h,1500)>_elo.get(a,1500) else a
    qf_results[l]=winner
    print(f"  → 预测晋级: {winner}")

# SF
print(f"\n{'#'*70}")
print(f"  SF  —  第4轮  × 2 场")
print(f"{'#'*70}")
sf_matches=[(nl,qf_results[m1],qf_results[m2]) for nl,m1,m2 in SF_PATH]
sf_results={}
for i,(l,h,a) in enumerate(sf_matches,1):
    r=predict_match_detail(h,a)
    if not r: continue
    print(fmt_match(r, f"SF#{i:2d}", f"({l})"))
    if r['prob_h']>r['prob_a']: winner=h
    elif r['prob_a']>r['prob_h']: winner=a
    else: winner=h if _elo.get(h,1500)>_elo.get(a,1500) else a
    sf_results[l]=winner
    print(f"  → 预测晋级: {winner}")

# Final + 3rd
print(f"\n{'#'*70}")
print(f"  🏆 决赛  +  季军战")
print(f"{'#'*70}")
final_match=(sf_results['M101'],sf_results['M102'])
r=predict_match_detail(*final_match)
if r:
    print(fmt_match(r, "🏆 决赛", "(M103)"))
    champ=sf_results['M101'] if r['prob_h']>r['prob_a'] else sf_results['M102']
    print(f"\n  🎉 预测冠军: {champ} 🎉")

print(f"\n{'='*70}")
print(f"  🏆 预测冠军路线")
print(f"{'='*70}")
# Print tree
def print_round(matches, label, results, level=0):
    for l,h,a in matches:
        w=results.get(l,'?')
        indent="  "*level
        print(f"{indent}{l}: {h:25s} vs {a:25s} → {w}")
print_round(r32_matches,"R32",r32_results,0)
print()
print_round(r16_matches,"R16",r16_results,1)
print()
print_round(qf_matches,"QF",qf_results,2)
print()
print_round(sf_matches,"SF",sf_results,3)
print(f"    🏆 决赛: {sf_results['M101']} vs {sf_results['M102']}")
print(f"    预测冠军: {sf_results['M101'] if r and r['prob_h']>r['prob_a'] else sf_results['M102']}")
