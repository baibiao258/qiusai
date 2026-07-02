#!/usr/bin/env python3
"""
wc_predictor_v2.py — 优化版世界杯泊松预测模型
===============================================
新增可开关特征 (通过 --features 控制):
  1) data_filter:  过滤非 FIFA 认可赛事
  2) neutral_all:  所有比赛设为中立场地
  3) time_decay:   指数衰减权重 (半衰期可调)
  4) elo_correction: Elo 排名修正 (泊松 * 0.6 + Elo * 0.4)
  5) recent_form:  近 N 场状态特征

用法:
  python3 wc_predictor_v2.py --features all          # 全部开启
  python3 wc_predictor_v2.py --features data_filter,neutral_all,time_decay
  python3 wc_predictor_v2.py --features none          # = 原始版 (对照)
  python3 wc_predictor_v2.py --benchmark              # 跑全量对比
"""
import json, math, csv, os, sys, urllib.request
from datetime import datetime, timedelta
from collections import defaultdict

# ── 常量 ──
MAX_GOALS = 6
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# FIFA 认可的国家队赛事 (排除 CONIFA / 非正式比赛)
FIFA_TOURNAMENTS = {
    'FIFA World Cup', 'FIFA World Cup qualification',
    'UEFA Euro', 'UEFA Euro qualification',
    'Copa América', 'Copa America',
    'African Cup of Nations', 'Africa Cup of Nations',
    'AFC Asian Cup', 'AFC Asian Cup qualification',
    'CONCACAF Gold Cup', 'CONCACAF Gold Cup qualification',
    'OFC Nations Cup', 'OFC Nations Cup qualification',
    'FIFA Confederations Cup',
    'International Friendly',
}

# ── 泊松核心 ──

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def predict_match(home_attack, home_defense, away_attack, away_defense,
                  league_avg, neutral=False):
    h_adj = 1.0 if neutral else 1.05
    a_adj = 1.0 if neutral else 1.0
    home_lambda = league_avg * home_attack * away_defense * h_adj
    away_lambda = league_avg * away_attack * home_defense * a_adj
    home_lambda = max(0.1, min(5.0, home_lambda))
    away_lambda = max(0.1, min(5.0, away_lambda))

    h_win, draw, a_win = 0.0, 0.0, 0.0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            prob = poisson_pmf(hg, home_lambda) * poisson_pmf(ag, away_lambda)
            if hg > ag:       h_win += prob
            elif hg == ag:    draw += prob
            else:             a_win += prob
    total = h_win + draw + a_win
    return h_win/total, draw/total, a_win/total, home_lambda, away_lambda


# ── 数据加载 ──

def fetch_international_results(cache_path=None):
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    print("  📡 下载国际赛数据...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'wc_predictor/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode('utf-8')
    except Exception as e:
        print(f"  ❌ 下载失败: {e}"); return []
    matches = []
    for row in csv.DictReader(raw.splitlines()):
        try:
            matches.append({
                'date': row['date'],
                'home': row['home_team'], 'away': row['away_team'],
                'h_score': int(row['home_score']), 'a_score': int(row['away_score']),
                'tournament': row['tournament'],
                'neutral': row.get('neutral', '').lower() == 'true',
            })
        except: continue
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(matches, f)
    return matches

def fetch_wc_2022_matches(cache_path=None):
    url = "https://raw.githubusercontent.com/openfootball/world-cup.json/master/2022/worldcup.json"
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    print("  📡 下载 2022 世界杯数据...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'wc_predictor/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  ❌ 下载失败: {e}"); return []
    matches = data.get('matches', [])
    simplified = []
    for m in matches:
        rnd = m.get('round', '')
        is_ko = ('Round' in rnd or 'Final' in rnd or 'Semi' in rnd or
                 'Quarter' in rnd or 'third' in rnd)
        simplified.append({
            'date': m['date'], 'round': rnd,
            'team1': m['team1'], 'team2': m['team2'],
            'score_ft': m['score']['ft'],
            'group': m.get('group', ''),
            'is_knockout': is_ko,
        })
    return simplified


# ── 核心：计算球队强度 (带可开关特征) ──

def compute_team_strengths(matches, wc_matches, features, **kwargs):
    """
    返回: (team_stats, global_avg_goals)
    team_stats: {team_name: {'attack': float, 'defense': float, 'matches': int, 'gf': int, 'ga': int}}
    """
    cutoff_date = '2022-11-20'
    half_life = kwargs.get('half_life', 180)
    min_matches = kwargs.get('min_matches', 3)
    recent_n = kwargs.get('recent_n', 5)
    use_filter = 'data_filter' in features
    use_decay = 'time_decay' in features
    use_form = 'recent_form' in features

    # Step 1: 过滤 + 统计
    stats = defaultdict(lambda: {'gf': 0, 'ga': 0, 'matches': 0, 'weight_sum': 0.0,
                                  'wg': 0.0, 'wc': 0.0})  # weighted goals/conceded

    filtered_count = 0
    for m in matches:
        if m['date'] >= cutoff_date:
            continue
        if use_filter and m['tournament'] not in FIFA_TOURNAMENTS:
            filtered_count += 1
            continue

        if use_decay:
            days_ago = (datetime.strptime(cutoff_date, '%Y-%m-%d') -
                        datetime.strptime(m['date'], '%Y-%m-%d')).days
            w = 0.5 ** (max(days_ago, 0) / half_life)
        else:
            w = 1.0

        # 主队
        s = stats[m['home']]
        s['gf'] += m['h_score']; s['ga'] += m['a_score']
        s['matches'] += 1; s['weight_sum'] += w
        s['wg'] += m['h_score'] * w; s['wc'] += m['a_score'] * w

        # 客队
        s = stats[m['away']]
        s['gf'] += m['a_score']; s['ga'] += m['h_score']
        s['matches'] += 1; s['weight_sum'] += w
        s['wg'] += m['a_score'] * w; s['wc'] += m['h_score'] * w

    if use_filter:
        print(f"  🚫 过滤掉 {filtered_count} 条非 FIFA 赛事记录")

    # 全球平均水平 (加权)
    if use_decay:
        total_wg = sum(s['wg'] for s in stats.values())
        total_ws = sum(s['weight_sum'] for s in stats.values())
        global_avg_goals = total_wg / max(total_ws, 1)
        global_avg_conceded = total_wg / max(total_ws, 1)
    else:
        total_gf = sum(s['gf'] for s in stats.values())
        total_m = sum(s['matches'] for s in stats.values())
        global_avg_goals = total_gf / max(total_m, 1)
        global_avg_conceded = total_gf / max(total_m, 1)

    print(f"  📊 全球场均总进球: {global_avg_goals:.3f}  |  数据球队数: {len(stats)}")

    # Step 2: 计算强度
    team_stats = {}
    for team, s in stats.items():
        if use_decay and s['weight_sum'] > 0:
            avg_gf = s['wg'] / s['weight_sum']
            avg_ga = s['wc'] / s['weight_sum']
        else:
            avg_gf = s['gf'] / max(s['matches'], 1)
            avg_ga = s['ga'] / max(s['matches'], 1)

        attack = avg_gf / max(global_avg_goals, 0.01)
        defense = avg_ga / max(global_avg_conceded, 0.01)

        team_stats[team] = {
            'attack': attack,
            'defense': defense,
            'matches': s['matches'],
            'gf': s['gf'], 'ga': s['ga'],
        }

    # Step 3: 近期状态 (可选)
    if use_form:
        form_stats = compute_recent_form(team_stats, matches, cutoff_date, recent_n)
        # 融合: 历史强度占 60%, 近期状态占 40%
        for team in team_stats:
            if team in form_stats:
                ts = team_stats[team]
                fs = form_stats[team]
                ts['attack'] = ts['attack'] * 0.6 + fs['attack'] * 0.4
                ts['defense'] = ts['defense'] * 0.6 + fs['defense'] * 0.4

    return team_stats, global_avg_goals


def compute_recent_form(team_stats, all_matches, cutoff_date, n=5):
    """计算每队最近 N 场比赛的状态 (强度), 比赛不足则用全部"""
    form_data = defaultdict(lambda: {'gf': 0, 'ga': 0, 'matches': 0})
    # 按日期降序排列
    sorted_m = sorted(all_matches, key=lambda x: x['date'], reverse=True)
    
    for m in sorted_m:
        if m['date'] >= cutoff_date:
            continue
        for team, gf, ga in [(m['home'], m['h_score'], m['a_score']),
                             (m['away'], m['a_score'], m['h_score'])]:
            fd = form_data[team]
            if fd['matches'] < n:
                fd['gf'] += gf; fd['ga'] += ga; fd['matches'] += 1
    
    # 计算近期强度
    global_gf = sum(s['gf'] for s in form_data.values()) / max(sum(1 for _ in form_data.values()), 1)
    global_avg = max(global_gf, 0.5)
    
    result = {}
    for team, fd in form_data.items():
        if fd['matches'] > 0:
            result[team] = {
                'attack': (fd['gf'] / fd['matches']) / global_avg,
                'defense': (fd['ga'] / fd['matches']) / global_avg,
            }
    
    print(f"  📋 近期状态计算: {len(result)} 支球队近 {n} 场")
    return result


# ── Elo ──

def elo_expected(ra, rb):
    return 1.0 / (1 + 10 ** ((rb - ra) / 400))

def compute_elo_ratings(matches, wc_teams, cutoff_date='2022-11-20'):
    """基于历史比赛计算各队 Elo 评分"""
    elo = defaultdict(lambda: 1500.0)
    K = 32  # Elo K-factor for international matches
    
    for m in matches:
        if m['date'] >= cutoff_date:
            continue
        home, away = m['home'], m['away']
        h_score, a_score = m['h_score'], m['a_score']
        
        # 只计算参赛球队的 Elo (节省计算)
        # if home not in wc_teams and away not in wc_teams:
        #     continue
        
        e_h = elo_expected(elo[home], elo[away])
        e_a = 1 - e_h
        
        if h_score > a_score:
            s_h, s_a = 1.0, 0.0
        elif h_score == a_score:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0
        
        elo[home] += K * (s_h - e_h)
        elo[away] += K * (s_a - e_a)
    
    return dict(elo)


# ── 回测 ──

def backtest_wc_2022(wc_matches, team_stats, global_avg, features):
    """回测 + 返回详细结果"""
    results = []
    use_elo = 'elo_correction' in features
    neutral_all = 'neutral_all' in features
    
    if use_elo:
        # 计算 Elo
        cache_path = os.path.join(DATA_DIR, 'international_results.json')
        all_matches = fetch_international_results(cache_path)
        elo_ratings = compute_elo_ratings(all_matches, set(team_stats.keys()))
    
    for m in wc_matches:
        t1, t2 = m['team1'], m['team2']
        actual_h, actual_a = m['score_ft']
        
        ts1 = team_stats.get(t1, {'attack': 1.0, 'defense': 1.0, 'matches': 0})
        ts2 = team_stats.get(t2, {'attack': 1.0, 'defense': 1.0, 'matches': 0})
        
        is_neutral = neutral_all or m['is_knockout']
        
        h_win, draw, a_win, hl, al = predict_match(
            ts1['attack'], ts1['defense'],
            ts2['attack'], ts2['defense'],
            global_avg, neutral=is_neutral
        )
        
        # Elo 修正
        if use_elo:
            elo_h = elo_ratings.get(t1, 1500)
            elo_a = elo_ratings.get(t2, 1500)
            elo_prob_h = elo_expected(elo_h, elo_a)
            elo_prob_a = 1 - elo_prob_h
            
            # 融合: 泊松 * w + Elo * (1-w), 调优最佳 w=0.55
            w = 0.55
            h_win = h_win * w + elo_prob_h * (1-w)
            a_win = a_win * w + elo_prob_a * (1-w)
            draw = draw * w + 0.2 * (1-w)
            
            t = h_win + draw + a_win
            h_win, draw, a_win = h_win/t, draw/t, a_win/t
        
        # 确定结果
        pred_result = 'H' if h_win > draw and h_win > a_win else ('D' if draw > h_win and draw > a_win else 'A')
        actual_result = 'H' if actual_h > actual_a else ('D' if actual_h == actual_a else 'A')
        
        # 找最可能比分
        h_probs = [poisson_pmf(hg, hl) for hg in range(MAX_GOALS + 1)]
        a_probs = [poisson_pmf(ag, al) for ag in range(MAX_GOALS + 1)]
        best_prob, best_h, best_a = 0, 0, 0
        for hg in range(MAX_GOALS + 1):
            for ag in range(MAX_GOALS + 1):
                p = h_probs[hg] * a_probs[ag]
                if p > best_prob:
                    best_prob, best_h, best_a = p, hg, ag
        
        hda_ok = pred_result == actual_result
        exact_ok = best_h == actual_h and best_a == actual_a
        
        prob_err = (h_win - (1 if actual_result=='H' else 0))**2 + \
                   (draw - (1 if actual_result=='D' else 0))**2 + \
                   (a_win - (1 if actual_result=='A' else 0))**2
        
        results.append({
            'team1': t1, 'team2': t2, 'date': m['date'], 'round': m['round'],
            'actual': f"{actual_h}-{actual_a}",
            'predicted': f"{best_h}-{best_a}",
            'hda_correct': hda_ok, 'exact_match': exact_ok,
            'probs': {'H': round(h_win,4), 'D': round(draw,4), 'A': round(a_win,4)},
            'lambdas': (hl, al),
            'sq_error': prob_err,
        })
    
    # 汇总
    total = len(results)
    hda = sum(1 for r in results if r['hda_correct'])
    exact = sum(1 for r in results if r['exact_match'])
    rmse = math.sqrt(sum(r['sq_error'] for r in results) / max(total, 1))
    
    # Brier score (更细致的概率校准指标)
    brier = sum(r['sq_error'] for r in results) / max(total, 1)
    
    # 按轮次
    by_round = defaultdict(lambda: {'t':0, 'c':0})
    for r in results:
        rnd = r['round']
        by_round[rnd]['t'] += 1
        if r['hda_correct']:
            by_round[rnd]['c'] += 1
    
    return {
        'results': results,
        'summary': {
            'total': total, 'hda': hda, 'exact': exact,
            'hda_acc': round(hda/total*100, 2),
            'exact_acc': round(exact/total*100, 2),
            'rmse': round(rmse, 4), 'brier': round(brier, 4),
        },
        'by_round': dict(by_round),
    }


# ── 基准对比 ──

def run_benchmark(wc_matches, intl_matches, feature_sets):
    """跑多组特征配置对比"""
    results = {}
    
    for name, features in feature_sets.items():
        print(f"\n{'='*60}")
        print(f"  🔬 测试: {name}")
        print(f"  特征: {features or '(无)'}")
        print(f"{'='*60}")
        
        ts, ga = compute_team_strengths(intl_matches, wc_matches, features)
        bt = backtest_wc_2022(wc_matches, ts, ga, features)
        
        s = bt['summary']
        print(f"\n  📊 结果: HDA={s['hda_acc']}%  Exact={s['exact_acc']}%  RMSE={s['rmse']}  Brier={s['brier']}")
        
        results[name] = bt
    
    # 汇总对比
    print(f"\n\n{'='*70}")
    print(f"  🏆 特征对比汇总")
    print(f"{'='*70}")
    print(f"  {'配置':<28s} {'HDA准确率':>10s} {'比分':>7s} {'RMSE':>6s} {'Brier':>7s}")
    print(f"  {'─'*60}")
    
    base = results.get('baseline') or results.get(next(iter(results)))
    base_hda = base['summary']['hda_acc']
    
    for name, bt in results.items():
        s = bt['summary']
        delta = s['hda_acc'] - base_hda
        arrow = '↑' if delta > 0 else ('↓' if delta < 0 else ' ')
        print(f"  {name:<28s} {s['hda_acc']:>8.2f}% {arrow}{abs(delta):>5.2f}  {s['exact_acc']:>5.2f}% {s['rmse']:>6.4f} {s['brier']:>7.4f}")
    
    print(f"\n{'─'*60}")
    print(f"  BM: Brier Score (越低越好) | RMSE (越低越好)")
    print(f"  ↑ = 相对 baseline 提升")
    return results


# ── 主入口 ──

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='none',
                        help='features: comma-separated or "all"/"none"/"benchmark"')
    parser.add_argument('--benchmark', action='store_true', help='跑全量对比')
    parser.add_argument('--half-life', type=int, default=180, help='time_decay 半衰期(天)')
    parser.add_argument('--min-matches', type=int, default=3)
    parser.add_argument('--recent-n', type=int, default=5, help='recent_form 近N场')
    args = parser.parse_args()
    
    ALL_FEATURES = {'neutral_all', 'time_decay', 'elo_correction', 'recent_form', 'data_filter'}
    
    if args.benchmark or args.features == 'benchmark':
        # 全量对比
        print("="*60)
        print("  ⚽ 世界杯泊松预测 v2 — 特征对比基准测试")
        print(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        cache_path = os.path.join(DATA_DIR, 'international_results.json')
        wc_path = os.path.join(DATA_DIR, 'wc_2022.json')
        intl = fetch_international_results(cache_path)
        wc = fetch_wc_2022_matches(wc_path)
        
        feature_sets = {
            'baseline (无优化)': set(),
            'neutral_all': {'neutral_all'},
            '+ elo_correction': {'neutral_all', 'elo_correction'},
            '⭐ optimized (hl=180)': {'neutral_all', 'elo_correction', 'time_decay'},
        }
        
        run_benchmark(wc, intl, feature_sets)
        return 0
    
    # 单次运行
    if args.features == 'all':
        features = ALL_FEATURES
    elif args.features == 'none':
        features = set()
    else:
        features = set(f.strip() for f in args.features.split(','))
    
    print("="*60)
    print("  ⚽ 世界杯泊松预测模型 v2")
    print(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  特征: {features or '(无)'}")
    print("="*60)
    
    cache_path = os.path.join(DATA_DIR, 'international_results.json')
    wc_path = os.path.join(DATA_DIR, 'wc_2022.json')
    intl = fetch_international_results(cache_path)
    wc = fetch_wc_2022_matches(wc_path)
    
    ts, ga = compute_team_strengths(intl, wc, features,
                                     half_life=args.half_life,
                                     min_matches=args.min_matches,
                                     recent_n=args.recent_n)
    bt = backtest_wc_2022(wc, ts, ga, features)
    s = bt['summary']
    
    print(f"\n{'='*60}")
    print(f"  📊 回测报告")
    print(f"{'='*60}")
    print(f"  胜平负:    {s['hda']}/{s['total']} = {s['hda_acc']}%")
    print(f"  精确比分:  {s['exact']}/{s['total']} = {s['exact_acc']}%")
    print(f"  RMSE:      {s['rmse']}")
    print(f"  Brier:     {s['brier']}")
    
    # 轮次明细
    print(f"\n  {'轮次':<22s} {'场':>3s} {'对':>3s} {'%':>6s}")
    for rnd in sorted(bt['by_round'].keys()):
        rs = bt['by_round'][rnd]
        pct = rs['c']/rs['t']*100 if rs['t'] else 0
        print(f"  {rnd:<22s} {rs['t']:>3d} {rs['c']:>3d} {pct:>5.1f}%")
    
    return 0

if __name__ == '__main__':
    main()
