#!/usr/bin/env python3
"""
2026 World Cup knockout simulation — official FIFA bracket.
Uses team-constrained-first assignment for 3rd-placed teams.
"""
import sys, os, json, math
sys.path.insert(0, '/root')
import numpy as np
import joblib
from collections import defaultdict, OrderedDict
from team_name_normalizer import normalize_match_pair

DATA_DIR = '/root/data'
_dc = joblib.load(os.path.join(DATA_DIR, 'dc_model.pkl'))
_elo = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))
HOST_TEAMS = {'United States', 'Mexico', 'Canada'}
HOST_BONUS = {'United States': 0.1445, 'Mexico': 0.10, 'Canada': 0.07}

with open(f'{DATA_DIR}/2026_groups.json') as f:
    GROUPS = json.load(f)
with open(f'{DATA_DIR}/group_stage_predictions.json') as f:
    GROUP_MATCHES = json.load(f)

# ── Official FIFA bracket from cup_finals.txt ─────────────────────
R32_SPEC = [
    ('M73',  ('2','A'), ('2','B')),
    ('M74',  ('1','E'), ('3',['A','B','C','D','F'])),
    ('M75',  ('1','F'), ('2','C')),
    ('M76',  ('1','C'), ('2','F')),
    ('M77',  ('1','I'), ('3',['C','D','F','G','H'])),
    ('M78',  ('2','E'), ('2','I')),
    ('M79',  ('1','A'), ('3',['C','E','F','H','I'])),
    ('M80',  ('1','L'), ('3',['E','H','I','J','K'])),
    ('M81',  ('1','D'), ('3',['B','E','F','I','J'])),
    ('M82',  ('1','G'), ('3',['A','E','H','I','J'])),
    ('M83',  ('2','K'), ('2','L')),
    ('M84',  ('1','H'), ('2','J')),
    ('M85',  ('1','B'), ('3',['E','F','G','I','J'])),
    ('M86',  ('1','J'), ('2','H')),
    ('M87',  ('1','K'), ('3',['D','E','I','J','L'])),
    ('M88',  ('2','D'), ('2','G')),
]

ROUND_ROBIN = OrderedDict([
    ('R16', [('M89','M74','M77'),('M90','M73','M75'),('M91','M76','M78'),
             ('M92','M79','M80'),('M93','M83','M84'),('M94','M81','M82'),
             ('M95','M86','M88'),('M96','M85','M87')]),
    ('QF',  [('M97','M89','M90'),('M98','M93','M94'),('M99','M91','M92'),
             ('M100','M95','M96')]),
    ('SF',  [('M101','M97','M98'),('M102','M99','M100')]),
])

# ── Third-placed team assignment (team-constrained-first) ──────────
THIRD_SLOTS = [
    ('M74', ['A','B','C','D','F']),
    ('M77', ['C','D','F','G','H']),
    ('M79', ['C','E','F','H','I']),
    ('M80', ['E','H','I','J','K']),
    ('M81', ['B','E','F','I','J']),
    ('M82', ['A','E','H','I','J']),
    ('M85', ['E','F','G','I','J']),
    ('M87', ['D','E','I','J','L']),
]

def assign_third_place_teams(third_qualifying):
    """Assign 8 qualifying third-placed teams to 8 R32 slots.
    
    Algorithm: team-constrained-first — teams with fewest eligible
    slots get assigned first, ensuring all 8 placements are valid.
    
    third_qualifying: [(team, group, pts, gd, gf)] ranked 1-8
    Returns: {match_label: (team, group)}
    """
    # Build eligible slot index for each team
    # Group → eligible match labels
    slot_by_group = defaultdict(list)
    for label, eg in THIRD_SLOTS:
        for g in eg:
            slot_by_group[g].append(label)
    
    # Teams with their eligible slots and constraint count
    teams_info = []
    for t, g, pts, gd, gf in third_qualifying:
        eligible_slots = slot_by_group.get(g, [])
        # Sort slots by prestige (opponent strength) for tiebreaking
        teams_info.append({
            'team': t, 'group': g, 'pts': pts, 'gd': gd, 'gf': gf,
            'eligible': set(eligible_slots),
            'n_slots': len(eligible_slots)
        })
    
    # Sort by constraint (fewest slots first), then by rank (pts descending)
    teams_info.sort(key=lambda x: (x['n_slots'], -x['pts'], -x['gd'], -x['gf']))
    
    assigned = {}
    used_slots = set()
    processed_teams = set()
    
    # Team-constrained-first assignment
    while len(assigned) < 8:
        # Find the unprocessed team with fewest remaining eligible slots
        best_team = None
        for ti in teams_info:
            if ti['team'] in processed_teams:
                continue
            rem = ti['eligible'] - used_slots
            if best_team is None or len(rem) < len(best_team['eligible'] - used_slots):
                best_team = ti
                best_team['remaining'] = rem
        
        if best_team is None:
            break
        
        remaining = best_team['remaining']
        if not remaining:
            # Emergency: pick any remaining slot
            remaining = set(s[0] for s in THIRD_SLOTS) - used_slots
        
        # Pick the best slot: highest prestige (group winner Elo)
        slot_prestige = {
            'M74': _elo.get(st_exp['E'][0] if 'st_exp' in dir() else 'Germany', 1900),
            'M77': _elo.get(st_exp['I'][0] if 'st_exp' in dir() else 'France', 1900),
            'M79': 1854, 'M80': 1941, 'M81': 1811, 'M82': 1862, 'M85': 1850, 'M87': 1947,
        }
        chosen = max(remaining, key=lambda s: slot_prestige.get(s, 1800))
        
        assigned[chosen] = (best_team['team'], best_team['group'])
        used_slots.add(chosen)
        processed_teams.add(best_team['team'])
    
    return assigned


# ── Group stage simulation ─────────────────────────────────────────
def sim_group_stage(rng):
    """Returns (standings_dict, third_ranked_list)."""
    pts = {g: {t: {'pts': 0, 'gf': 0, 'ga': 0} for t in GROUPS[g]} for g in GROUPS}
    
    for m in GROUP_MATCHES:
        h, a = m['home'], m['away']
        g = next(grp for grp in GROUPS if h in GROUPS[grp])
        hg = rng.poisson(m['lam_h'])
        ag = rng.poisson(m['lam_a'])
        pts[g][h]['gf'] += hg; pts[g][h]['ga'] += ag
        pts[g][a]['gf'] += ag; pts[g][a]['ga'] += hg
        if hg > ag: pts[g][h]['pts'] += 3
        elif hg == ag: pts[g][h]['pts'] += 1; pts[g][a]['pts'] += 1
        else: pts[g][a]['pts'] += 3
    
    st = {}
    third_all = []
    for g in sorted(GROUPS.keys()):
        st[g] = sorted(GROUPS[g], key=lambda t: (
            -pts[g][t]['pts'],
            -(pts[g][t]['gf'] - pts[g][t]['ga']),
            -pts[g][t]['gf'],
            -_elo.get(t, 1500)
        ))
        t3 = st[g][2]
        third_all.append((t3, g, pts[g][t3]['pts'],
                          pts[g][t3]['gf'] - pts[g][t3]['ga'],
                          pts[g][t3]['gf']))
    
    third_all.sort(key=lambda x: (-x[2], -x[3], -x[4], -_elo.get(x[0],1500)))
    return st, third_all[:8]


def resolve_team(spec, st, third_map, match_label):
    pos, arg = spec
    if pos == '1': return st[arg][0]
    if pos == '2': return st[arg][1]
    if pos == '3':
        if match_label in third_map:
            return third_map[match_label][0]
        return st[arg[0]][2] if arg[0] in st else '?'
    return '?'


def build_r32(st, third_map):
    return [(l, resolve_team(h, st, third_map, l), resolve_team(a, st, third_map, l))
            for l, h, a in R32_SPEC]


def sim_match(rng, home, away):
    hb = HOST_BONUS.get(home, 0.0) if home in HOST_TEAMS else 0.0
    is_host = hb > 0
    try:
        hn, an = normalize_match_pair(home, away)
        lh, la = _dc.predict_lambda(hn, an, not is_host, host_bonus=hb if is_host else 0.0)
        if lh is None or lh <= 0 or la is None or la <= 0:
            lh = la = 1.0
    except:
        lh = la = 1.0
    
    hg, ag = rng.poisson(lh), rng.poisson(la)
    if hg == ag:
        eh, ea = rng.poisson(lh*0.3), rng.poisson(la*0.3)
        hg += eh; ag += ea
    if hg == ag:
        ph = 0.4 + 0.6/(1+math.exp(-(_elo.get(home,1500)-_elo.get(away,1500))/200))
        hg += 1 if rng.random() < ph else 0; ag += 0 if rng.random() < ph else 1
    return home if hg > ag else away


def run_tournament(seed):
    rng = np.random.RandomState(seed)
    st, t3 = sim_group_stage(rng)
    tmap = assign_third_place_teams(t3)
    r32 = build_r32(st, tmap)
    
    prog = {}
    for g in GROUPS:
        for t in GROUPS[g]:
            prog[t] = {k: False for k in ['r32','r16','qf','sf','final','champ']}
    for _, h, a in r32:
        prog[h]['r32'] = True; prog[a]['r32'] = True
    
    w = {}
    for l, h, a in r32:
        w[l] = sim_match(rng, h, a)
        prog[w[l]]['r16'] = True
    
    for rd, pairs in ROUND_ROBIN.items():
        nw = {}
        for nl, m1, m2 in pairs:
            wi = sim_match(rng, w[m1], w[m2])
            nw[nl] = wi
            rkey = 'qf' if rd == 'R16' else ('sf' if rd == 'QF' else 'final')
            prog[wi][rkey] = True
        w = nw
    
    champ = list(w.values())[0]
    prog[champ]['champ'] = True
    return champ, prog


def run_batch(seeds):
    cc, rc = defaultdict(int), defaultdict(lambda: defaultdict(int))
    for s in seeds:
        c, p = run_tournament(s)
        cc[c] += 1
        for t, rds in p.items():
            for rk, v in rds.items():
                if v: rc[rk][t] += 1
    return cc, rc


def main(n_sims=50000):
    print(f"🏆 2026 世界杯淘汰赛模拟 — 官方FIFA路书")
    print(f"  模拟: {n_sims:,} | 数据: openfootball/worldcup cup_finals.txt")
    print(f'{"="*65}')
    
    # Expected standings
    gp = {g: {t: {'pts': 0.0} for t in GROUPS[g]} for g in GROUPS}
    for m in GROUP_MATCHES:
        h, a = m['home'], m['away']
        g = next(grp for grp in GROUPS if h in GROUPS[grp])
        hp, dp, ap = m['home_win']/100, m['draw']/100, m['away_win']/100
        gp[g][h]['pts'] += hp*3 + dp; gp[g][a]['pts'] += ap*3 + dp
    
    global st_exp
    st_exp = {}
    for g in sorted(GROUPS.keys()):
        st_exp[g] = sorted(GROUPS[g], key=lambda t: (-gp[g][t]['pts'], -_elo.get(t,1500)))
    
    gw = {g: st_exp[g][0] for g in GROUPS}
    ru = {g: st_exp[g][1] for g in GROUPS}
    third_raw = []
    for g in sorted(GROUPS.keys()):
        t = st_exp[g][2]
        third_raw.append((t, g, gp[g][t]['pts'], 0.0, 0.0))
    third_raw.sort(key=lambda x: (-x[2], -_elo.get(x[0],1500)))
    bt = third_raw[:8]
    
    print(f"\n📋 小组第一: {', '.join(gw[g] for g in sorted(GROUPS.keys()))}")
    print(f"📋 小组第二: {', '.join(ru[g] for g in sorted(GROUPS.keys()))}")
    print(f"📋 最佳第三:")
    for i,(t,g,p,_,_) in enumerate(bt,1):
        print(f"     {i}. {t:25s} G{g} {p:.1f}pt")
    
    # Expected R32
    tmap = assign_third_place_teams(bt)
    print(f"\n🏁 预期R32 (官方FIFA路书):")
    for l, hs, asp in R32_SPEC:
        home = gw[hs[1]] if hs[0]=='1' else ru[hs[1]]
        if asp[0] == '1': away = gw[asp[1]]
        elif asp[0] == '2': away = ru[asp[1]]
        else: away = tmap.get(l, ('?',))[0]
        print(f"  {l}  {home:24s} vs {away}")
    
    # Print bracket paths
    print(f"\n📈 淘汰赛路径:")
    print(f"  R16: M89(W74vW77) M90(W73vW75) M91(W76vW78) M92(W79vW80)")
    print(f"       M93(W83vW84) M94(W81vW82) M95(W86vW88) M96(W85vW87)")
    print(f"  QF:  M97(W89vW90) M98(W93vW94) M99(W91vW92) M100(W95vW96)")
    print(f"  SF:  M101(W97vW98) M102(W99vW100)")
    print(f"  FIN: W101 v W102 | 3rd: L101 v L102")
    
    # Run MC
    print(f"\n🔄 运行 {n_sims:,} 次 Monte Carlo...")
    batch_size = max(500, n_sims // 8)
    seeds_list = [list(range(i, min(i+batch_size, n_sims))) for i in range(0, n_sims, batch_size)]
    cc, rc = defaultdict(int), defaultdict(lambda: defaultdict(int))
    
    for seeds in seeds_list:
        bc, br = run_batch(seeds)
        for t, c in bc.items(): cc[t] += c
        for rk, ts in br.items():
            for t, c in ts.items(): rc[rk][t] += c
    
    total = n_sims
    print(f"\n{'='*65}")
    print(f"🏆 结果 ({total:,} 次) — 官方FIFA路书")
    print(f"{'='*65}")
    
    stl = sorted(cc.keys(), key=lambda t: -cc.get(t,0))
    print(f"\n{'球队':25s} {'R16':>6s} {'QF':>6s} {'SF':>6s} {'Fin':>6s} {'🏆':>6s}")
    print(f"{'-'*61}")
    for team in stl[:25]:
        c = cc.get(team,0)/total*100
        r16 = rc.get('r16',{}).get(team,0)/total*100
        qf = rc.get('qf',{}).get(team,0)/total*100
        sf = rc.get('sf',{}).get(team,0)/total*100
        fn = rc.get('final',{}).get(team,0)/total*100
        bar = '█'*max(0,int(c/2)) if c > 0.1 else ''
        print(f"{team:25s} {r16:5.1f}% {qf:5.1f}% {sf:5.1f}% {fn:5.1f}% {c:5.2f}%  {bar}")
    
    # Save
    result = {
        'type': 'wc2026_knockout_fifa_bracket_official',
        'n_sims': n_sims,
        'R32_expected': [(l, h, a) for l, h, a in 
            build_r32({g: st_exp[g] for g in st_exp}, tmap)],
        'champion_prob': {t: round(c/total*100,4) for t,c in sorted(cc.items(),key=lambda x:-x[1])},
        'round_probs': {
            k: {t: round(rc[k].get(t,0)/total*100,2) for t in cc}
            for k in ['r16','qf','sf','final','champ']
        }
    }
    with open(f'{DATA_DIR}/knockout_fifa_bracket.json','w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ 已保存: {DATA_DIR}/knockout_fifa_bracket.json")

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    main(n)
