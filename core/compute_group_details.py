#!/usr/bin/env python3
"""
72 group stage matches — full detail: score, HT/FT, total goals, λ, probabilities.
"""
import sys, os, json, math, itertools
sys.path.insert(0, '/root')
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from team_name_normalizer import normalize_match_pair

DATA_DIR = '/root/data'
import joblib
_dc = joblib.load(os.path.join(DATA_DIR, 'dc_model.pkl'))
_xgb = (
    joblib.load(os.path.join(DATA_DIR, 'xgb_model_29.pkl'))
    if os.path.exists(os.path.join(DATA_DIR, 'xgb_model_29.pkl'))
    else joblib.load(os.path.join(DATA_DIR, 'xgb_model_20_3.pkl'))
)
_elo = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))

HOST_TEAMS = {'United States', 'Mexico', 'Canada'}
HOST_BONUS = {'United States': 0.1445, 'Mexico': 0.10, 'Canada': 0.07}
DC_WEIGHT = 0.4
XGB_WEIGHT = 0.6
MAX_GOALS = 6

with open(f'{DATA_DIR}/2026_groups.json') as f:
    GROUPS = json.load(f)
with open(f'{DATA_DIR}/group_stage_predictions.json') as f:
    GROUP_MATCHES = json.load(f)

def make_odds(eh, ea):
    dh = ea - eh; da = eh - ea
    return [1/(10**(-dh/400)+1), 1/(10**(-da/400)+1), 0.0]

def predict_match_detail(home, away, host_bonus=0.0):
    """Full match prediction returning probabilities, λ, and distributions."""
    h, a = normalize_match_pair(home, away)
    is_host = host_bonus > 0 and home in HOST_TEAMS
    neutral = not is_host
    
    dc_p = _dc.predict_proba(h, a, neutral, host_bonus=host_bonus if is_host else 0.0)
    lam_h, lam_a = _dc.predict_lambda(h, a, neutral, host_bonus=host_bonus if is_host else 0.0)
    if lam_h is None:
        return None
    
    eh = _elo.get(home, 1500); ea = _elo.get(away, 1500)
    op = make_odds(eh, ea)
    
    # Build feature vector for XGB
    fh5 = [0.5, 0.0, 0.0, 0.0]; fa5 = [0.5, 0.0, 0.0, 0.0]
    b15 = [(eh-ea)/400, lam_h, lam_a, lam_h-lam_a,
           math.log(max(lam_h,0.01)/max(lam_a,0.01)),
           dc_p[0], dc_p[1], dc_p[2],
           fh5[0], fa5[0], fh5[1]-fa5[2], fa5[1]-fh5[2],
           fh5[1]-fa5[1], fh5[0]-fa5[0],
           0 if is_host else 1]
    gold = [0.0, 1, 0, 0.0, 0.0]
    odds_feat = [op[0], op[1], op[2] if op[2] else 0.0]
    # 6 form features (placeholder for MC cache)
    form_feat = [0.0, 0.0, 0.0, 0.0, 1.5, 1.5]
    feat = np.array([b15 + gold + odds_feat + form_feat])  # 29 dims
    
    xgb_p = _xgb.predict_proba(feat)[0]  # [away, draw, home]
    dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])  # re-order to [away, draw, home]
    hybrid = DC_WEIGHT * dc_ado + XGB_WEIGHT * xgb_p
    
    # Poisson score distribution  
    n_sim = 50000
    hg = np.random.poisson(lam_h, n_sim)
    ag = np.random.poisson(lam_a, n_sim)
    
    scores = defaultdict(int)
    ht_scores = defaultdict(int)
    # Half-time λ ≈ 0.45 × full-time (empirical)
    ht_lam_h = lam_h * 0.45
    ht_lam_a = lam_a * 0.45
    
    htg = np.random.poisson(ht_lam_h, n_sim)
    atg = np.random.poisson(ht_lam_a, n_sim)
    
    for i in range(n_sim):
        sh, sa = hg[i], ag[i]
        # Cap at MAX_GOALS for display
        sk = (min(sh, MAX_GOALS), min(sa, MAX_GOALS))
        scores[sk] += 1
        
        hth, hta = htg[i], atg[i]
        htk = (min(hth, MAX_GOALS), min(hta, MAX_GOALS))
        ht_scores[htk] += 1
    
    # Score probabilities (top 8)
    score_list = sorted(scores.items(), key=lambda x: -x[1])[:8]
    score_pct = [{"score": f"{s[0]}:{s[1]}", "pct": round(c/n_sim*100, 1)} 
                 for s, c in score_list]
    
    # HT scores
    ht_list = sorted(ht_scores.items(), key=lambda x: -x[1])[:8]
    ht_pct = [{"score": f"{s[0]}:{s[1]}", "pct": round(c/n_sim*100, 1)}
              for s, c in ht_list]
    
    # HT result probabilities
    ht_home_pct = sum(c for s, c in ht_scores.items() if s[0] > s[1]) / n_sim * 100
    ht_draw_pct = sum(c for s, c in ht_scores.items() if s[0] == s[1]) / n_sim * 100
    ht_away_pct = sum(c for s, c in ht_scores.items() if s[0] < s[1]) / n_sim * 100
    
    # HT/FT 9-way
    htft = defaultdict(int)
    ft_results = [(0 if hg[i] < ag[i] else (1 if hg[i] == ag[i] else 2)) for i in range(n_sim)]
    ht_results = [(0 if htg[i] < atg[i] else (1 if htg[i] == atg[i] else 2)) for i in range(n_sim)]
    # Map: HH=0, HD=1, HA=2, DH=3, DD=4, DA=5, AH=6, AD=7, AA=8
    # where H=home win, D=draw, A=away win in each half
    for hr, fr in zip(ht_results, ft_results):
        idx = hr * 3 + fr  # hr: 0=away, 1=draw, 2=home; fr: same
        htft[idx] += 1
    
    htft_labels = ['负/负','负/平','负/胜','平/负','平/平','平/胜','胜/负','胜/平','胜/胜']
    htft_pct = [{"label": htft_labels[i], "pct": round(c/n_sim*100, 1)}
                for i, c in sorted(htft.items(), key=lambda x: -x[1])[:6]]
    
    # Total goals
    tg_dist = defaultdict(int)
    for i in range(n_sim):
        tg = min(hg[i] + ag[i], MAX_GOALS * 2)
        tg_dist[tg] += 1
    tg_pct = [{"goals": g, "pct": round(c/n_sim*100, 1)} 
              for g, c in sorted(tg_dist.items(), key=lambda x: -x[0])]
    tg_pct.sort(key=lambda x: -x['pct'])
    
    return {
        'home': home, 'away': away,
        'host_bonus_applied': is_host,
        'lam_h': round(lam_h, 2), 'lam_a': round(lam_a, 2),
        'elo_h': eh, 'elo_a': ea,
        'prob_h': round(hybrid[2]*100, 1),
        'prob_d': round(hybrid[1]*100, 1),
        'prob_a': round(hybrid[0]*100, 1),
        'dc_h': round(dc_p[0]*100, 1), 'dc_d': round(dc_p[1]*100, 1), 'dc_a': round(dc_p[2]*100, 1),
        'xgb_h': round(xgb_p[2]*100, 1), 'xgb_d': round(xgb_p[1]*100, 1), 'xgb_a': round(xgb_p[0]*100, 1),
        'ht_home_pct': round(ht_home_pct, 1),
        'ht_draw_pct': round(ht_draw_pct, 1),
        'ht_away_pct': round(ht_away_pct, 1),
        'scores': score_pct,
        'ht_scores': ht_pct,
        'htft': htft_pct,
        'total_goals': tg_pct[:6],
    }

def process_match(m):
    h, a = m['home'], m['away']
    hb = HOST_BONUS.get(h, 0.0)
    return predict_match_detail(h, a, host_bonus=hb)

def process_batch(batch):
    return [process_match(m) for m in batch]

def main():
    print(f"🔍 正在计算72场小组赛细项...")
    
    # Process in parallel
    results = []
    batch_size = 12
    batches = [GROUP_MATCHES[i:i+batch_size] for i in range(0, len(GROUP_MATCHES), batch_size)]
    
    for i, batch in enumerate(batches):
        print(f"  批次 {i+1}/{len(batches)} ({len(batch)}场)...")
        for m in batch:
            r = process_match(m)
            if r:
                results.append(r)
                print(f"    ✅ {r['home']:25s} vs {r['away']:25s} | {r['prob_h']}% / {r['prob_d']}% / {r['prob_a']}%")
            else:
                print(f"    ❌ {m['home']} vs {m['away']}")
    
    # Save with numpy type conversion
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(f'{DATA_DIR}/group_stage_details.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    
    print(f"\n✅ 全部完成! {len(results)}场, 已保存至 {DATA_DIR}/group_stage_details.json")
    
    # Generate compact report by group
    for g in sorted(GROUPS.keys()):
        teams = GROUPS[g]
        print(f"\n{'='*60}")
        print(f"  {g}组 — {' vs '.join(teams)}")
        print(f"{'='*60}")
        
        for r in results:
            if r['home'] not in teams and r['away'] not in teams:
                continue
            h, a = r['home'], r['away']
            hw, dr, aw = r['prob_h'], r['prob_d'], r['prob_a']
            
            # Determine direction
            mx = max(hw, dr, aw)
            if mx == hw: dir_ = f"主胜 {hw}%"
            elif mx == dr: dir_ = f"平局 {dr}%"
            else: dir_ = f"客胜 {aw}%"
            
            top_score = r['scores'][0]['score'] if r['scores'] else '?'
            top_tg = r['total_goals'][0] if r['total_goals'] else {}
            tg_str = ", ".join(f"{t['goals']}球({t['pct']}%)" for t in r['total_goals'][:3])
            
            print(f"\n  {h:25s} vs {a}")
            print(f"    λ {r['lam_h']:.2f}/{r['lam_a']:.2f} | 概率: {hw}/{dr}/{aw} → {dir_}")
            print(f"    比分: {top_score} | 总进球: {tg_str}")
            print(f"    半场: 主{r['ht_home_pct']}% 平{r['ht_draw_pct']}% 客{r['ht_away_pct']}%")
            if r['htft']:
                htft_top = " ".join(f"{h['label']}({h['pct']}%)" for h in r['htft'][:3])
                print(f"    半全场: {htft_top}")
    
    # Champion path summary
    print(f"\n{'='*60}")
    print(f"  📊 夺冠路线图 (按出线后Elo排名)")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: -max(x['prob_h'], x['prob_a'])):
        hw, dr, aw = r['prob_h'], r['prob_d'], r['prob_a']
        mx = max(hw, dr, aw)
        if mx > 60:
            tag = "🔴 稳" if mx > 75 else "🟡 可"
            print(f"  {tag} {r['home']:25s} vs {r['away']:25s} {hw:.0f}/{dr:.0f}/{aw:.0f}")

if __name__ == '__main__':
    main()
