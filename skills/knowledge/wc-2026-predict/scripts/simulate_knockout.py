#!/usr/bin/env python3
"""
Knockout stage simulation for 2026 World Cup.
Uses group stage predictions + predict_match.py for each knockout match.
Monte Carlo simulation to get round-by-round probabilities.

Usage:
  python3 scripts/simulate_knockout.py [N]

  N = number of simulations (default: 50000)

Requires:
  - /root/data/2026_groups.json        (group composition)
  - /root/data/group_stage_predictions.json  (72 group match predictions)
  - /root/data/dc_model.pkl            (trained DC model)
  - /root/data/elo_ratings.pkl         (Elo ratings)
  - /root/predict_match.py             (DC+XGB single match predictor)

Output:
  - Prints R32 bracket, round-by-round probabilities, champion ranking
  - Saves /root/data/knockout_simulation.json

Bracket mode: currently uses Elo-seeded pairing (1v32, 2v31…).
  See references/2026-official-knockout-bracket.md for FIFA official bracket.
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

        group_pts[g][h]['pts'] += hp * 3 + dp * 1
        group_pts[g][a]['pts'] += ap * 3 + dp * 1
        group_pts[g][h]['gf'] += m['lam_h'] * (hp + dp * 0.5)
        group_pts[g][a]['gf'] += m['lam_a'] * (ap + dp * 0.5)
        group_pts[g][h]['ga'] += m['lam_a'] * (ap + dp * 0.5)
        group_pts[g][a]['ga'] += m['lam_h'] * (hp + dp * 0.5)

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

    third_placed.sort(key=lambda x: (-x[3], -x[2]))
    best_third = third_placed[:8]

    return group_winners, runners_up, best_third


def build_bracket(group_winners, runners_up, best_third):
    """Rank all 32 teams by Elo and pair 1v32, 2v31, etc."""
    all_teams = []
    for t, g, elo in sorted(group_winners, key=lambda x: -x[2]):
        all_teams.append((t, g, 'W', elo))
    for t, g, elo in sorted(runners_up, key=lambda x: -x[2]):
        all_teams.append((t, g, 'RU', elo))
    for t, g, elo, _ in sorted(best_third, key=lambda x: -x[2]):
        all_teams.append((t, g, '3rd', elo))

    all_teams.sort(key=lambda x: -x[3])

    bracket_round32 = []
    n = len(all_teams)
    for i in range(n // 2):
        bracket_round32.append((all_teams[i][0], all_teams[n - 1 - i][0]))

    return bracket_round32, all_teams


def poisson_result(lam_h, lam_a, rng):
    hg = rng.poisson(lam_h)
    ag = rng.poisson(lam_a)
    return hg, ag


def get_match_lambda(home, away, host_bonus=0.0):
    from team_name_normalizer import normalize_match_pair
    from predict_match import _dc
    home_n, away_n = normalize_match_pair(home, away)
    is_host = host_bonus > 0 and home in HOST_TEAMS
    neutral = not is_host
    lam_h, lam_a = _dc.predict_lambda(home_n, away_n, neutral,
                                      host_bonus=host_bonus if is_host else 0.0)
    return lam_h, lam_a, home_n, away_n


def simulate_tournament(seed=42):
    """Run one full tournament simulation. Returns the winner and round tracks."""
    rng = np.random.default_rng(seed)

    standings = compute_group_standings()
    gw, ru, bt = get_qualified(standings)
    bracket_round32, seeded_teams = build_bracket(gw, ru, bt)

    team_elo = {t: elo for t, g, pos, elo in seeded_teams}

    current_round = bracket_round32
    round_num = 0
    team_in_round = defaultdict(set)

    for h, a in current_round:
        team_in_round[round_num].add(h)
        team_in_round[round_num].add(a)

    while len(current_round) > 0:
        next_round = []
        for home, away in current_round:
            hb = HOST_BONUS.get(home, 0.0) if home in HOST_TEAMS else 0.0
            lam_h, lam_a, home_n, away_n = get_match_lambda(home, away, hb)
            if lam_h is None or lam_a is None:
                lam_h, lam_a = 1.0, 1.0

            hg, ag = poisson_result(lam_h, lam_a, rng)

            if hg == ag:
                et_hg = rng.poisson(lam_h * 0.3)
                et_ag = rng.poisson(lam_a * 0.3)
                if et_hg != et_ag:
                    hg += et_hg
                    ag += et_ag
                else:
                    prob_h_win = 0.4 + 0.6 / (1 + math.exp(-(team_elo.get(home, 1500) - team_elo.get(away, 1500)) / 200))
                    if rng.random() < prob_h_win:
                        hg += 1
                    else:
                        ag += 1

            winner = home if hg > ag else away
            next_round.append(winner)

        round_num += 1
        for winner in next_round:
            team_in_round[round_num].add(winner)

        if len(next_round) >= 2:
            current_round = [(next_round[i], next_round[i + 1]) for i in range(0, len(next_round), 2)]
        else:
            current_round = []

    champion = next_round[0] if next_round else None
    return champion, team_in_round


def simulate_batch(seeds):
    results = []
    for s in seeds:
        champ, rounds = simulate_tournament(seed=s)
        results.append((champ, rounds))
    return results


def main(n_sims=50000):
    print(f"\n🏆 2026 世界杯淘汰赛模拟 — {n_sims:,} 次")
    print(f"{'='*60}")

    standings = compute_group_standings()
    gw, ru, bt = get_qualified(standings)
    bracket_round32, seeded_teams = build_bracket(gw, ru, bt)

    print(f"\n📋 晋级的 32 强 (Elo 排序):")
    for i, (t, g, pos, elo) in enumerate(seeded_teams, 1):
        print(f"  {i:2d}. {t:25s} Elo {elo:.0f}")

    print(f"\n🏁 R32 对阵:")
    for i, (h, a) in enumerate(bracket_round32, 1):
        print(f"  {i:2d}. {h:25s} vs {a}")

    print(f"\n🔄 运行 {n_sims:,} 次模拟...")

    n_workers = min(os.cpu_count() or 2, 4)
    batch_size = max(1000, n_sims // n_workers)

    champ_counts = defaultdict(int)
    round_counts = defaultdict(lambda: defaultdict(int))

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

    print(f"\n📈 各轮晋级概率 Top20:")
    print(f"{'球队':25s} {'R16':>8s} {'QF':>8s} {'SF':>8s} {'Final':>8s} {'🏆冠军':>10s}")
    print(f"{'-'*70}")

    sorted_teams = sorted(champ_counts.keys(), key=lambda t: -champ_counts.get(t, 0))

    for team in sorted_teams[:20]:
        champ_p = champ_counts[team] / n_sims * 100
        r16_p = round_counts.get(1, {}).get(team, 0) / n_sims * 100
        qf_p = round_counts.get(2, {}).get(team, 0) / n_sims * 100
        sf_p = round_counts.get(3, {}).get(team, 0) / n_sims * 100
        final_p = round_counts.get(4, {}).get(team, 0) / n_sims * 100
        bar = '█' * int(champ_p / 2) if champ_p > 0.1 else ''
        print(f"{team:25s} {r16_p:7.2f}% {qf_p:7.2f}% {sf_p:7.2f}% {final_p:7.2f}% {champ_p:8.2f}%  {bar}")

    result = {
        'type': 'wc2026_knockout_simulation',
        'n_sims': n_sims,
        'standings': standings,
        'bracket_r32': [(h, a) for h, a in bracket_round32],
        'champion_prob': {t: round(c / n_sims * 100, 4) for t, c in
                          sorted(champ_counts.items(), key=lambda x: -x[1])},
        'round_probs': {
            'r16': {t: round(round_counts.get(1, {}).get(t, 0) / n_sims * 100, 2) for t in champ_counts.keys()},
            'qf': {t: round(round_counts.get(2, {}).get(t, 0) / n_sims * 100, 2) for t in champ_counts.keys()},
            'sf': {t: round(round_counts.get(3, {}).get(t, 0) / n_sims * 100, 2) for t in champ_counts.keys()},
            'final': {t: round(round_counts.get(4, {}).get(t, 0) / n_sims * 100, 2) for t in champ_counts.keys()},
            'champion': {t: round(c / n_sims * 100, 4) for t, c in
                         sorted(champ_counts.items(), key=lambda x: -x[1])},
        }
    }

    with open(f'{DATA_DIR}/knockout_simulation.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ 结果已保存: {DATA_DIR}/knockout_simulation.json")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    main(n)
