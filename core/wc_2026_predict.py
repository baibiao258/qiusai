#!/usr/bin/env python3
"""
wc_2026_predict.py — 2026 世界杯冠军预测
============================================
基于 v4 动态 Elo + 阶段切换模型的蒙特卡洛模拟

数据源: martj42/international_results (~1872-2026)
赛制: 48队, 16组×3队 → 32强淘汰赛
"""
import json, math, os, sys, urllib.request, csv, random
from datetime import datetime
from collections import defaultdict

MAX_GOALS = 6
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SIMULATIONS = 100000

# ── 2026 世界杯参赛队 (openfootball, 48队)
TEAMS_2026 = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia",
    "Croatia", "Curaçao", "Czech Republic", "DR Congo", "Ecuador",
    "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan",
    "Jordan", "Mexico", "Morocco", "Netherlands", "New Zealand",
    "Norway", "Panama", "Paraguay", "Portugal", "Qatar",
    "Saudi Arabia", "Scotland", "Senegal", "South Africa", "South Korea",
    "Spain", "Sweden", "Switzerland", "Tunisia", "Turkey",
    "United States", "Uruguay", "Uzbekistan"
]

# ── 模型 ──
def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def elo_expected(ra, rb):
    return 1.0 / (1 + 10 ** ((rb - ra) / 400))

# ── 数据加载 ──
def load_data(cache_path):
    if not os.path.exists(cache_path):
        print("  📡 下载国际赛数据...")
        url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'wc_predictor/1.0'})
            raw = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            return []
        matches = []
        for row in csv.DictReader(raw.splitlines()):
            try:
                matches.append({
                    'date': row['date'], 'home': row['home_team'],
                    'away': row['away_team'], 'tournament': row['tournament'],
                    'h_score': int(row['home_score']), 'a_score': int(row['away_score']),
                })
            except: continue
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(matches, f)
        return matches
    with open(cache_path) as f:
        return json.load(f)

# ── 球队强度 ──
def compute_team_strengths(matches, half_life=180):
    cutoff = '2026-06-11'
    stats = defaultdict(lambda: {'wg': 0.0, 'wc': 0.0, 'weight_sum': 0.0, 'matches': 0})
    for m in matches:
        if m['date'] >= cutoff: continue
        days_ago = (datetime.strptime(cutoff, '%Y-%m-%d') -
                    datetime.strptime(m['date'], '%Y-%m-%d')).days
        w = 0.5 ** (max(days_ago, 0) / half_life)
        for team, gf, ga in [(m['home'], m['h_score'], m['a_score']),
                             (m['away'], m['a_score'], m['h_score'])]:
            s = stats[team]
            s['wg'] += gf * w; s['wc'] += ga * w
            s['weight_sum'] += w; s['matches'] += 1
    total_wg = sum(s['wg'] for s in stats.values())
    total_ws = sum(s['weight_sum'] for s in stats.values())
    global_avg = total_wg / max(total_ws, 1)
    team_data = {}
    for team, s in stats.items():
        avg_gf = s['wg'] / max(s['weight_sum'], 0.001)
        avg_ga = s['wc'] / max(s['weight_sum'], 0.001)
        team_data[team] = {
            'attack': avg_gf / max(global_avg, 0.01),
            'defense': avg_ga / max(global_avg, 0.01),
            'matches': s['matches'],
        }
    return team_data, global_avg

# ── Elo 评分 ──
def compute_elo_ratings(matches, cutoff='2026-06-11'):
    elo = defaultdict(lambda: 1500.0)
    for m in matches:
        if m['date'] >= cutoff: continue
        h, a = m['home'], m['away']
        hs, az = m['h_score'], m['a_score']
        e_h = elo_expected(elo[h], elo[a])
        sh, sa = (1.0, 0.0) if hs > az else ((0.5, 0.5) if hs == az else (0.0, 1.0))
        elo[h] += 32 * (sh - e_h)
        elo[a] += 32 * (sa - (1 - e_h))
    return dict(elo)

# ── 模拟进球 ──
def simulate_goals(ts1, ts2, global_avg, group_stage=True, random_state=None):
    scale = 1.0 if group_stage else 0.85
    lam_h = global_avg * ts1['attack'] * ts2['defense'] * scale
    lam_a = global_avg * ts2['attack'] * ts1['defense'] * scale
    lam_h = max(0.1, min(5.0, lam_h))
    lam_a = max(0.1, min(5.0, lam_a))
    
    h_probs = [poisson_pmf(k, lam_h) for k in range(MAX_GOALS+1)]
    a_probs = [poisson_pmf(k, lam_a) for k in range(MAX_GOALS+1)]
    h_cum = [sum(h_probs[:i+1]) for i in range(MAX_GOALS+1)]
    a_cum = [sum(a_probs[:i+1]) for i in range(MAX_GOALS+1)]
    
    r = random.random()
    hg = next((i for i, c in enumerate(h_cum) if r <= c), MAX_GOALS)
    r = random.random()
    ag = next((i for i, c in enumerate(a_cum) if r <= c), MAX_GOALS)
    return hg, ag

# ── 分组 ──
def setup_groups(teams, elo_ratings):
    """分组: 48队 → 16组×3队 (按 Elo 分3档各16队)"""
    sorted_teams = sorted(teams, key=lambda t: elo_ratings.get(t, 1500), reverse=True)
    
    # 3 pots of 16 teams each
    pots = [sorted_teams[i:i+16] for i in range(0, 48, 16)]
    
    # 蛇形分配: A-P 共16组
    groups = {chr(ord('A')+i): [] for i in range(16)}
    for pot_idx, pot in enumerate(pots):
        shuffled = list(pot)
        random.shuffle(shuffled)
        for g_idx, team in enumerate(shuffled):
            groups[chr(ord('A')+g_idx)].append(team)
    
    return groups

# ── 小组赛 ──
def simulate_group_stage(groups, team_data, global_avg):
    """3队一组, 循环赛 → 前2名晋级"""
    qualifiers = []
    
    for g_name in sorted(groups.keys()):
        g_teams = groups[g_name]
        if len(g_teams) != 3:
            continue
        
        points = {t: 0 for t in g_teams}
        gd = {t: 0 for t in g_teams}
        gf = {t: 0 for t in g_teams}
        
        # 3场比赛: 0-1, 0-2, 1-2
        fixtures = [(g_teams[0], g_teams[1]),
                    (g_teams[0], g_teams[2]),
                    (g_teams[1], g_teams[2])]
        
        for t1, t2 in fixtures:
            ts1 = team_data.get(t1, {'attack': 1.0, 'defense': 1.0})
            ts2 = team_data.get(t2, {'attack': 1.0, 'defense': 1.0})
            hg, ag = simulate_goals(ts1, ts2, global_avg, group_stage=True)
            
            gf[t1] += hg; gf[t2] += ag
            gd[t1] += hg - ag; gd[t2] += ag - hg
            if hg > ag: points[t1] += 3
            elif hg == ag: points[t1] += 1; points[t2] += 1
            else: points[t2] += 3
        
        ranked = sorted(g_teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True)
        qualifiers.append(ranked[:2])
    
    return qualifiers

# ── 淘汰赛 ──
def simulate_knockout(qualifiers, team_data, global_avg, elo_ratings):
    """32强 → 冠军"""
    if len(qualifiers) != 16:
        return None
    
    # A1 vs B2, B1 vs A2, C1 vs D2, D1 vs C2 ...
    round32 = []
    for i in range(0, 16, 2):
        round32.append((qualifiers[i][0], qualifiers[i+1][1]))
        round32.append((qualifiers[i+1][0], qualifiers[i][1]))
    
    current = round32
    for _ in range(5):
        if len(current) <= 1:
            break
        next_round = []
        for i in range(0, len(current), 2):
            t1 = current[i][0]
            t2 = current[i+1][0]
            ts1 = team_data.get(t1, {'attack': 1.0, 'defense': 1.0})
            ts2 = team_data.get(t2, {'attack': 1.0, 'defense': 1.0})
            
            hg, ag = simulate_goals(ts1, ts2, global_avg, group_stage=False)
            
            if hg == ag:
                # 加时
                hg_et, ag_et = simulate_goals(ts1, ts2, global_avg, group_stage=False)
                hg += hg_et; ag += ag_et
                if hg == ag:
                    # 点球
                    e1 = elo_ratings.get(t1, 1500)
                    e2 = elo_ratings.get(t2, 1500)
                    penalty_p = 0.5 + (elo_expected(e1, e2) - 0.5) * 0.3
                    winner = t1 if random.random() < penalty_p else t2
                    next_round.append((winner, None))
                    continue
            
            winner = t1 if hg > ag else t2
            next_round.append((winner, None))
        
        current = next_round
    
    return current[0][0] if current else None

# ── 蒙特卡洛 ──
def run_monte_carlo(team_data, global_avg, elo_ratings, n=SIMULATIONS):
    print(f"\n  🏃 蒙特卡洛模拟 ({n:,} 次)...")
    
    target_teams = list(TEAMS_2026)
    champion_count = defaultdict(int)
    quarter_count = defaultdict(int)
    round16_count = defaultdict(int)
    
    batch_size = 10000
    num_batches = n // batch_size
    
    for batch in range(num_batches):
        if batch > 0 and batch % 5 == 0:
            print(f"    进度: {batch}/{num_batches} ({batch/num_batches*100:.0f}%)")
        
        for _ in range(batch_size):
            groups = setup_groups(target_teams, elo_ratings)
            qualifiers = simulate_group_stage(groups, team_data, global_avg)
            champion = simulate_knockout(qualifiers, team_data, global_avg, elo_ratings)
            if champion:
                champion_count[champion] += 1
    
    total = sum(champion_count.values())
    print(f"  ✅ 完成! {total:,} 次\n")
    
    sorted_champs = sorted(champion_count.items(), key=lambda x: x[1], reverse=True)
    return {'total': total, 'champion': sorted_champs}

# ══════════════════════════════════════
#  主入口
# ══════════════════════════════════════
def main():
    print("=" * 60)
    print("  ⚽ 2026 世界杯冠军预测")
    print(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  赛制: 48队 / 16组×3 → 32强淘汰赛")
    print("  模型: 泊松 + Elo + 时间衰减(hl=180d) + 阶段切换")
    print("  数据: martj42/international_results (1872-2026)")
    print("=" * 60)
    
    cache_path = os.path.join(DATA_DIR, 'international_results.json')
    matches = load_data(cache_path)
    if not matches:
        print("  ❌ 数据加载失败")
        return 1
    print(f"  📊 {len(matches):,} 场国际比赛")
    
    print(f"\n  🧠 球队强度 (hl=180d, 截止 2026-06-11)...")
    team_data, global_avg = compute_team_strengths(matches)
    print(f"    全球场均总进球 λ = {global_avg:.3f}")
    
    print(f"  🧠 Elo 评分...")
    elo_ratings = compute_elo_ratings(matches)
    
    # 参赛队排名
    team_elos = [(t, elo_ratings.get(t, 1500)) for t in TEAMS_2026]
    team_elos.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  🏆 参赛队 Elo/攻防排名:")
    print(f"  {'#':>2s} {'球队':<25s} {'Elo':>5s} {'⚔攻':>5s} {'🛡防':>5s}")
    print(f"  {'─'*42}")
    for i, (t, e) in enumerate(team_elos, 1):
        ts = team_data.get(t, {'attack': 1.0, 'defense': 1.0})
        print(f"  {i:>2d} {t:<25s} {e:>5.0f} {ts['attack']:>5.2f} {ts['defense']:>5.2f}")
        if i == 25:
            print(f"  {'─'*42}")
    
    # 蒙特卡洛
    result = run_monte_carlo(team_data, global_avg, elo_ratings, n=SIMULATIONS)
    
    total = result['total']
    champions = result['champion']
    
    print(f"{'='*60}")
    print(f"  🏆 2026 世界杯冠军概率 (蒙特卡洛 {total:,} 次)")
    print(f"{'='*60}")
    print(f"  {'排名':>4s} {'球队':<25s} {'冠军':>6s} {'概率%':>7s}  {'柱状图'}")
    print(f"  {'─'*55}")
    
    best_pct = champions[0][1] / total * 100 if champions else 0
    for i, (team, count) in enumerate(champions[:15], 1):
        pct = count / total * 100
        bar_len = int(pct / best_pct * 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"  {i:>3d}. {team:<25s} {count:>6,d} {pct:>6.2f}% {bar}")
    
    if len(champions) > 15:
        others = champions[15:]
        other_count = sum(c for _, c in others)
        other_pct = other_count / total * 100
        print(f"  {'─'*55}")
        print(f"  {'':>4s} 其他{len(others)}队{'':<19s} {other_count:>6,d} {other_pct:>6.2f}%")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
