#!/usr/bin/env python3
"""
Knockout stage simulation for 2026 World Cup.
Uses group stage predictions + predict_match.py for each knockout match.
Monte Carlo simulation to get round-by-round probabilities.
"""
import sys, os, json, math
sys.path.insert(0, '/root')
import numpy as np
import joblib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

DATA_DIR = '/root/data'

# Load groups
with open(f'{DATA_DIR}/2026_groups.json') as f:
    GROUPS = json.load(f)

# Load group stage predictions
with open(f'{DATA_DIR}/group_stage_predictions.json') as f:
    GROUP_MATCHES = json.load(f)

# Load Elo
_elo = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))

# Host teams
HOST_TEAMS = {'United States', 'Mexico', 'Canada'}
HOST_BONUS = {'United States': 0.1445, 'Mexico': 0.10, 'Canada': 0.07}

# --- Step 1: Determine group standings from predictions ---
def compute_group_standings():
    """Compute expected points and simulate group outcomes."""
    group_pts = {g: {t: {'pts': 0.0, 'gf': 0.0, 'ga': 0.0} for t in GROUPS[g]} for g in GROUPS}
    
    for m in GROUP_MATCHES:
        g = None
        for grp, teams in GROUPS.items():
            if m['home'] in teams:
                g = grp
                break
        if not g:
            continue
        
        h = m['home']
        a = m['away']
        hp = m['home_win'] / 100
        dp = m['draw'] / 100
        ap = m['away_win'] / 100
        
        # Points
        group_pts[g][h]['pts'] += hp * 3 + dp * 1
        group_pts[g][a]['pts'] += ap * 3 + dp * 1
        
        # Expected goals
        group_pts[g][h]['gf'] += m['lam_h'] * (hp + dp * 0.5)
        group_pts[g][a]['gf'] += m['lam_a'] * (ap + dp * 0.5)
        group_pts[g][h]['ga'] += m['lam_a'] * (ap + dp * 0.5)
        group_pts[g][a]['ga'] += m['lam_h'] * (hp + dp * 0.5)
    
    # Sort each group
    standings = {}
    for g in sorted(GROUPS.keys()):
        teams = sorted(GROUPS[g], key=lambda t: (
            -round(group_pts[g][t]['pts'], 2),
            -round(group_pts[g][t]['gf'] - group_pts[g][t]['ga'], 2),
            -round(group_pts[g][t]['gf'], 2),
            -_elo.get(t, 1500)
        ))
        standings[g] = [(t, round(group_pts[g][t]['pts'], 2)) for t in teams]
    return standings

# --- Step 2: Determine qualified teams ---
def get_qualified(standings):
    """Top 2 from each group + 8 best third-placed teams."""
    group_winners = []
    runners_up = []
    third_placed = []
    
    for g in sorted(standings.keys()):
        st = standings[g]
        group_winners.append((st[0][0], g, _elo.get(st[0][0], 1500)))
        runners_up.append((st[1][0], g, _elo.get(st[1][0], 1500)))
        third_placed.append((st[2][0], g, _elo.get(st[2][0], 1500), st[2][1]))
    
    # Best 8 third-placed teams (by points, then Elo)
    third_placed.sort(key=lambda x: (-x[3], -x[2]))
    best_third = third_placed[:8]
    
    return group_winners, runners_up, best_third

# --- Step 3: Build seeded bracket ---
def build_bracket(group_winners, runners_up, best_third):
    """Rank all 32 teams by Elo and pair 1v32, 2v31, etc."""
    all_teams = []
    
    # Group winners (seeded 1-12 by Elo)
    for t, g, elo in sorted(group_winners, key=lambda x: -x[2]):
        all_teams.append((t, g, 'W', elo))
    
    # Runners-up (seeded 13-24 by Elo)
    for t, g, elo in sorted(runners_up, key=lambda x: -x[2]):
        all_teams.append((t, g, 'RU', elo))
    
    # Best third-placed (seeded 25-32 by Elo)
    for t, g, elo, _ in sorted(best_third, key=lambda x: -x[2]):
        all_teams.append((t, g, '3rd', elo))
    
    # Sort all by Elo descending (1=highest Elo)
    all_teams.sort(key=lambda x: -x[3])
    
    # Pair 1v32, 2v31, 3v30, ... 16v17
    bracket_round32 = []
    n = len(all_teams)
    for i in range(n // 2):
        bracket_round32.append((all_teams[i][0], all_teams[n - 1 - i][0]))
    
    return bracket_round32, all_teams

# --- Step 4: Poisson match simulation ---
def poisson_result(lam_h, lam_a):
    """Simulate a match result using Poisson."""
    hg = np.random.poisson(lam_h)
    ag = np.random.poisson(lam_a)
    return hg, ag

def get_match_lambda(home, away, host_bonus=0.0):
    """Get match λ from DC model."""
    from team_name_normalizer import normalize_match_pair
    from predict_match import _dc
    
    home_n, away_n = normalize_match_pair(home, away)
    is_host = host_bonus > 0 and home in HOST_TEAMS
    neutral = not is_host
    
    lam_h, lam_a = _dc.predict_lambda(home_n, away_n, neutral, 
                                       host_bonus=host_bonus if is_host else 0.0)
    return lam_h, lam_a, home_n, away_n

# --- Step 5: Knockout tournament simulation ---
def simulate_tournament(seed=42):
    """Run one full tournament simulation. Returns the winner."""
    np.random.seed(seed)
    rng = np.random
    
    standings = compute_group_standings()
    gw, ru, bt = get_qualified(standings)
    bracket_round32, seeded_teams = build_bracket(gw, ru, bt)
    
    # Team Elo lookup
    team_elo = {t: elo for t, g, pos, elo in seeded_teams}
    
    current_round = bracket_round32
    round_num = 0
    team_in_round = defaultdict(set)
    
    # Track which teams are in which round
    for h, a in current_round:
        team_in_round[round_num].add(h)
        team_in_round[round_num].add(a)
    
    while len(current_round) > 0:
        next_round = []
        for home, away in current_round:
            # Determine host bonus (for Mexico/Canada/USA home games)
            # In knockout, only if the host is playing and it's "home" in bracket
            hb = HOST_BONUS.get(home, 0.0) if home in HOST_TEAMS else 0.0
            
            lam_h, lam_a, home_n, away_n = get_match_lambda(home, away, hb)
            if lam_h is None or lam_a is None:
                lam_h, lam_a = 1.0, 1.0
            
            # Simulate the match
            hg, ag = poisson_result(lam_h, lam_a)
            
            # Knockout: if draw, extra time + penalties
            if hg == ag:
                # Extra time: play another mini-period with lower λ
                et_hg = rng.poisson(lam_h * 0.3)
                et_ag = rng.poisson(lam_a * 0.3)
                if et_hg != et_ag:
                    hg += et_hg
                    ag += et_ag
                else:
                    # Penalties - slightly bias towards higher Elo
                    prob_h_win = 0.4 + 0.6 / (1 + math.exp(-(team_elo.get(home, 1500) - team_elo.get(away, 1500)) / 200))
                    if rng.random() < prob_h_win:
                        hg += 1
                    else:
                        ag += 1
            
            winner = home if hg > ag else away
            next_round.append(winner)
        
        round_num += 1
        
        # Track which teams reached this round
        for winner in next_round:
            team_in_round[round_num].add(winner)
        
        # Pair up for next round
        if len(next_round) >= 2:
            current_round = [(next_round[i], next_round[i+1]) for i in range(0, len(next_round), 2)]
        else:
            current_round = []
    
    # champion is the last remaining
    champion = next_round[0] if next_round else None
    
    return champion, team_in_round

def simulate_batch(seeds):
    """Run multiple simulations and return aggregated results."""
    results = []
    for s in seeds:
        champ, rounds = simulate_tournament(seed=s)
        results.append((champ, rounds))
    return results

# --- Main ---
def main(n_sims=50000):
    print(f"🏆 2026 世界杯淘汰赛模拟")
    print(f"模拟次数: {n_sims:,}")
    print(f"{'='*60}")
    
    # Get standings
    standings = compute_group_standings()
    gw, ru, bt = get_qualified(standings)
    
    # Print qualified teams
    print(f"\n📋 小组出线球队:")
    print(f"  小组第一 (12): {', '.join(t for t,g,_ in gw)}")
    print(f"  小组第二 (12): {', '.join(t for t,g,_ in ru)}")
    print(f"  最佳第三 (8): {', '.join(t for t,g,_,_ in bt)}")
    
    bracket_round32, seeded_teams = build_bracket(gw, ru, bt)
    print(f"\n📊 种子排序 (Elo):")
    for i, (t, g, pos, elo) in enumerate(seeded_teams, 1):
        print(f"  {i:2d}. {t:25s} Elo {elo:.0f} ({pos})")
    
    print(f"\n🏁 R32 配对:")
    for i, (h, a) in enumerate(bracket_round32, 1):
        print(f"  {i:2d}. {h:25s} vs {a}")
    
    # Run MC simulation
    print(f"\n🔄 运行 {n_sims:,} 次模拟...")
    
    # Use multiple workers
    n_workers = min(os.cpu_count() or 2, 4)
    batch_size = max(1000, n_sims // n_workers)
    
    champ_counts = defaultdict(int)
    round_counts = defaultdict(lambda: defaultdict(int))
    
    # Generate seed batches
    batches = []
    for i in range(0, n_sims, batch_size):
        actual = min(batch_size, n_sims - i)
        seeds = list(range(i, i + actual))
        batches.append(seeds)
    
    for batch_seeds in batches:
        batch_results = simulate_batch(batch_seeds)
        for champ, rounds in batch_results:
            if champ:
                champ_counts[champ] += 1
                for rnd, teams in rounds.items():
                    for t in teams:
                        round_counts[rnd][t] += 1
    
    # Results
    print(f"\n{'='*60}")
    print(f"🏆 模拟结果 ({n_sims:,} 次)")
    print(f"{'='*60}")
    
    total = n_sims
    
    # Round labels
    round_labels = {0: 'R32', 1: 'R16', 2: 'QF', 3: 'SF', 4: 'Final', 5: 'Champion'}
    
    print(f"\n📈 各轮晋级概率 Top15:")
    print(f"{'球队':25s} {'R16':>8s} {'QF':>8s} {'SF':>8s} {'Final':>8s} {'🏆冠军':>10s}")
    print(f"{'-'*70}")
    
    # Sort by champion probability
    sorted_teams = sorted(champ_counts.keys(), key=lambda t: -champ_counts.get(t, 0))
    
    for team in sorted_teams[:20]:
        champ_p = champ_counts[team] / total * 100
        r16_p = round_counts.get(1, {}).get(team, 0) / total * 100
        qf_p = round_counts.get(2, {}).get(team, 0) / total * 100
        sf_p = round_counts.get(3, {}).get(team, 0) / total * 100
        final_p = round_counts.get(4, {}).get(team, 0) / total * 100
        
        bar = '█' * int(champ_p / 2) if champ_p > 0.1 else ''
        print(f"{team:25s} {r16_p:7.2f}% {qf_p:7.2f}% {sf_p:7.2f}% {final_p:7.2f}% {champ_p:8.2f}%  {bar}")
    
    print(f"\n{'='*60}")
    
    # Save results
    result = {
        'type': 'wc2026_knockout_simulation',
        'n_sims': n_sims,
        'standings': standings,
        'bracket_r32': [(h, a) for h, a in bracket_round32],
        'champion_prob': {t: round(c/total*100, 4) for t, c in sorted(champ_counts.items(), key=lambda x: -x[1])},
        'round_probs': {
            'r16': {t: round(round_counts.get(1, {}).get(t, 0)/total*100, 2) for t in champ_counts.keys()},
            'qf': {t: round(round_counts.get(2, {}).get(t, 0)/total*100, 2) for t in champ_counts.keys()},
            'sf': {t: round(round_counts.get(3, {}).get(t, 0)/total*100, 2) for t in champ_counts.keys()},
            'final': {t: round(round_counts.get(4, {}).get(t, 0)/total*100, 2) for t in champ_counts.keys()},
            'champion': {t: round(c/total*100, 4) for t, c in sorted(champ_counts.items(), key=lambda x: -x[1])},
        }
    }
    
    with open(f'{DATA_DIR}/knockout_simulation.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ 结果已保存: {DATA_DIR}/knockout_simulation.json")

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    main(n)
