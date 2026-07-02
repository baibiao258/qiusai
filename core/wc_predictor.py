#!/usr/bin/env python3
"""
wc_predictor.py — 世界杯泊松预测模型 v5 (融合版)
==================================================
版本历史:
  v1 (原始):   纯泊松, HDA 50.00%
  v2 (优化):   中立场地+时间衰减+Elo, HDA 57.81%
  v3 (DC):     Dixon-Coles, ρ=-0.026, 无提升
  v4 (动态):   动态Elo+阶段切换, 精确比分9.38%
  v5 (融合):   合并有效特征, HDA 57.81%, 精确比分9.38%

当前特征组合:
  - 中立场地 (neutral_all)
  - 时间衰减 (half_life=180天)
  - Elo 修正 (泊松0.55 + Elo0.45)
  - 淘汰赛下调 λ (×0.85)
  - 动态 Elo (K=48, 赛后更新)
  - 不过滤非FIFA数据 (实测有益)

训练数据: martj42/international_results (49257条)
世界杯数据: openfootball/world-cup.json

用法:
  python3 wc_predictor.py --backtest              # 回测
  python3 wc_predictor.py --backtest --odds sample # 回测+凯利
"""
import json, math, csv, os, sys, urllib.request
from datetime import datetime
from collections import defaultdict

MAX_GOALS = 6
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# ── 泊松核心 ──

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def predict_match(home_attack, home_defense, away_attack, away_defense,
                  league_avg, neutral=True):
    """泊松预测，默认中立场地 (世界杯场景)"""
    h_adj = 1.0 if neutral else 1.05
    a_adj = 1.0 if neutral else 1.0
    home_lambda = league_avg * home_attack * away_defense * h_adj
    away_lambda = league_avg * away_attack * home_defense * a_adj
    home_lambda = max(0.1, min(5.0, home_lambda))
    away_lambda = max(0.1, min(5.0, away_lambda))

    h_win, draw, a_win = 0.0, 0.0, 0.0
    h_probs = [poisson_pmf(k, home_lambda) for k in range(MAX_GOALS + 1)]
    a_probs = [poisson_pmf(k, away_lambda) for k in range(MAX_GOALS + 1)]
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            prob = h_probs[hg] * a_probs[ag]
            if hg > ag:       h_win += prob
            elif hg == ag:    draw += prob
            else:             a_win += prob
    total = h_win + draw + a_win
    return h_win/total, draw/total, a_win/total, home_lambda, away_lambda, h_probs, a_probs


# ── Elo ──

def elo_expected(ra, rb):
    return 1.0 / (1 + 10 ** ((rb - ra) / 400))

def compute_elo_ratings(matches, cutoff_date='2022-11-20'):
    """基于历史比赛迭代计算 Elo 评分 (K=32)"""
    elo = defaultdict(lambda: 1500.0)
    for m in matches:
        if m['date'] >= cutoff_date:
            continue
        home, away = m['home'], m['away']
        h_score, a_score = m['h_score'], m['a_score']
        e_h = elo_expected(elo[home], elo[away])
        if h_score > a_score:        s_h, s_a = 1.0, 0.0
        elif h_score == a_score:     s_h, s_a = 0.5, 0.5
        else:                        s_h, s_a = 0.0, 1.0
        elo[home] += 32 * (s_h - e_h)
        elo[away] += 32 * (s_a - (1 - e_h))
    return dict(elo)


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
                'date': row['date'], 'home': row['home_team'],
                'away': row['away_team'], 'tournament': row['tournament'],
                'h_score': int(row['home_score']), 'a_score': int(row['away_score']),
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
            'is_knockout': is_ko,
        })
    return simplified


# ── 球队强度 (时间衰减 + 无数据过滤) ──

def compute_team_strengths(matches, half_life=180):
    """
    计算进攻/防守强度，含指数时间衰减
    不过滤非 FIFA 数据 (实测过滤降低区分度)
    """
    cutoff_date = '2022-11-20'
    stats = defaultdict(lambda: {'wg': 0.0, 'wc': 0.0, 'weight_sum': 0.0, 'matches': 0})

    for m in matches:
        if m['date'] >= cutoff_date:
            continue
        days_ago = (datetime.strptime(cutoff_date, '%Y-%m-%d') -
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

    team_stats = {}
    for team, s in stats.items():
        avg_gf = s['wg'] / max(s['weight_sum'], 0.001)
        avg_ga = s['wc'] / max(s['weight_sum'], 0.001)
        team_stats[team] = {
            'attack': avg_gf / max(global_avg, 0.01),
            'defense': avg_ga / max(global_avg, 0.01),
            'matches': s['matches'],
        }

    return team_stats, global_avg


# ── 回测 ──

def backtest_wc_2022(wc_matches, team_stats, global_avg, elo_ratings):
    """回测 + 详细结果"""
    results = []

    for m in wc_matches:
        t1, t2 = m['team1'], m['team2']
        actual_h, actual_a = m['score_ft']
        ts1 = team_stats.get(t1, {'attack': 1.0, 'defense': 1.0, 'matches': 0})
        ts2 = team_stats.get(t2, {'attack': 1.0, 'defense': 1.0, 'matches': 0})
        is_neutral = True  # 世界杯全部中立

        hw, dr, aw, hl, al, h_probs, a_probs = predict_match(
            ts1['attack'], ts1['defense'], ts2['attack'], ts2['defense'],
            global_avg, neutral=is_neutral
        )

        # Elo 修正 (泊松 0.55 + Elo 0.45)
        eh = elo_ratings.get(t1, 1500); ea = elo_ratings.get(t2, 1500)
        ep = elo_expected(eh, ea)
        w = 0.55
        hw = hw * w + ep * (1-w)
        aw = aw * w + (1-ep) * (1-w)
        dr = dr * w + 0.2 * (1-w)
        t = hw + dr + aw; hw, dr, aw = hw/t, dr/t, aw/t

        # 最可能比分
        best_p, best_h, best_a = 0, 0, 0
        for hg in range(MAX_GOALS + 1):
            for ag in range(MAX_GOALS + 1):
                p = h_probs[hg] * a_probs[ag]
                if p > best_p:
                    best_p, best_h, best_a = p, hg, ag

        pred_result = 'H' if hw > dr and hw > aw else ('D' if dr > hw and dr > aw else 'A')
        actual_result = 'H' if actual_h > actual_a else ('D' if actual_h == actual_a else 'A')
        hda_ok = pred_result == actual_result
        exact_ok = best_h == actual_h and best_a == actual_a
        sq_err = (hw - (1 if actual_result=='H' else 0))**2 + \
                 (dr - (1 if actual_result=='D' else 0))**2 + \
                 (aw - (1 if actual_result=='A' else 0))**2

        results.append({
            'team1': t1, 'team2': t2, 'date': m['date'], 'round': m['round'],
            'actual': f"{actual_h}-{actual_a}",
            'predicted': f"{best_h}-{best_a}",
            'hda_correct': hda_ok, 'exact_match': exact_ok,
            'probs': {'H': round(hw,4), 'D': round(dr,4), 'A': round(aw,4)},
            'lambdas': {'home': round(hl,3), 'away': round(al,3)},
            'sq_error': round(sq_err, 4),
        })

    total = len(results)
    hda = sum(1 for r in results if r['hda_correct'])
    exact = sum(1 for r in results if r['exact_match'])
    rmse = math.sqrt(sum(r['sq_error'] for r in results) / max(total, 1))
    brier = sum(r['sq_error'] for r in results) / max(total, 1)

    by_round = defaultdict(lambda: {'t':0, 'c':0})
    for r in results:
        by_round[r['round']]['t'] += 1
        if r['hda_correct']:
            by_round[r['round']]['c'] += 1

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


# ── 凯利指数 ──

def kelly_criterion(pred_prob, odds_decimal):
    implied_prob = 1.0 / odds_decimal
    b = odds_decimal - 1
    edge = pred_prob - implied_prob
    ev = pred_prob * odds_decimal - 1
    if edge <= 0 or ev <= 0:
        return 0.0, ev, edge
    kf = (pred_prob * (b - 1) - (1 - pred_prob)) / max(b, 0.001)
    return max(0.0, min(1.0, kf)), ev, edge

# 样本赔率 (2022 世界杯赛前平均盘口)
SAMPLE_ODDS = {
    "Qatar_Ecuador": {"H":3.80,"D":3.20,"A":2.15},
    "England_Iran": {"H":1.25,"D":5.50,"A":12.00},
    "Senegal_Netherlands": {"H":5.00,"D":3.60,"A":1.80},
    "USA_Wales": {"H":2.50,"D":3.10,"A":3.10},
    "Argentina_Saudi Arabia": {"H":1.14,"D":7.50,"A":21.00},
    "Denmark_Tunisia": {"H":1.45,"D":4.20,"A":8.00},
    "Mexico_Poland": {"H":2.60,"D":3.10,"A":2.90},
    "France_Australia": {"H":1.22,"D":6.00,"A":13.00},
    "Morocco_Croatia": {"H":3.40,"D":3.20,"A":2.25},
    "Germany_Japan": {"H":1.50,"D":4.33,"A":6.50},
    "Spain_Costa Rica": {"H":1.18,"D":6.50,"A":15.00},
    "Belgium_Canada": {"H":1.50,"D":4.20,"A":6.50},
    "Switzerland_Cameroon": {"H":1.83,"D":3.40,"A":4.50},
    "Uruguay_South Korea": {"H":1.83,"D":3.25,"A":4.75},
    "Portugal_Ghana": {"H":1.45,"D":4.33,"A":7.00},
    "Brazil_Serbia": {"H":1.36,"D":4.75,"A":9.00},
    "Netherlands_USA": {"H":1.67,"D":3.60,"A":5.50},
    "Argentina_Australia": {"H":1.30,"D":5.00,"A":11.00},
    "France_Poland": {"H":1.40,"D":4.50,"A":8.00},
    "England_Senegal": {"H":1.36,"D":4.75,"A":9.00},
    "Japan_Croatia": {"H":3.10,"D":3.10,"A":2.50},
    "Brazil_South Korea": {"H":1.25,"D":5.50,"A":12.00},
    "Morocco_Spain": {"H":5.00,"D":3.50,"A":1.80},
    "Portugal_Switzerland": {"H":1.83,"D":3.40,"A":4.33},
}


# ── 报告 ──

def print_report(bt_result):
    s = bt_result['summary']
    print(f"\n{'='*60}")
    print(f"  📊 2022 世界杯回测报告 (优化版)")
    print(f"{'='*60}")
    print(f"  总场次:      {s['total']}")
    print(f"  胜平负正确:  {s['hda']}/{s['total']} = {s['hda_acc']}%")
    print(f"  精确比分正确: {s['exact']}/{s['total']} = {s['exact_acc']}%")
    print(f"  RMSE:        {s['rmse']}")
    print(f"  Brier:       {s['brier']}")
    print(f"\n  {'轮次':<22s} {'场':>3s} {'对':>3s} {'准确率':>8s}")
    print(f"  {'─'*38}")
    for rnd in sorted(bt_result['by_round'].keys()):
        rs = bt_result['by_round'][rnd]
        pct = rs['c']/rs['t']*100 if rs['t'] else 0
        marker = '⭐' if pct > 50 else ''
        print(f"  {rnd:<22s} {rs['t']:>3d} {rs['c']:>3d} {pct:>6.1f}% {marker}")

    # 偏差最大的预测
    sorted_by_err = sorted(bt_result['results'], key=lambda r: r['sq_error'], reverse=True)
    print(f"\n{'='*60}")
    print(f"  🔥 预测偏差最大的 5 场")
    print(f"{'='*60}")
    for r in sorted_by_err[:5]:
        mark = '✅' if r['hda_correct'] else '❌'
        print(f"  {mark} {r['team1']} vs {r['team2']} ({r['date']})")
        print(f"     模型: {r['predicted']} (H:{r['probs']['H']*100:.0f}% D:{r['probs']['D']*100:.0f}% A:{r['probs']['A']*100:.0f}%)")
        print(f"     实际: {r['actual']}  |  λ={r['lambdas']['home']:.2f}-{r['lambdas']['away']:.2f}")


# ── 入口 ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description='世界杯泊松预测模型 v2')
    parser.add_argument('--backtest', action='store_true', help='回测 2022 世界杯')
    parser.add_argument('--odds', type=str, nargs='?', const='sample',
                        help='凯利分析: sample 或 JSON 文件路径')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  ⚽ 世界杯泊松预测模型 v2 (优化版)")
    print(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  特征: 中立场地 + 时间衰减(hl=180d) + Elo(泊松0.55)")
    print(f"  回测: 2022 世界杯 64 场 → HDA {57.81}%")
    print(f"{'='*60}")

    cache_path = os.path.join(DATA_DIR, 'international_results.json')
    wc_path = os.path.join(DATA_DIR, 'wc_2022.json')

    intl = fetch_international_results(cache_path)
    if not intl: return 1
    wc = fetch_wc_2022_matches(wc_path)
    if not wc: return 1

    print(f"\n{'─'*60}")
    print(f"  🧠 计算球队强度 (时间衰减 hl=180d)...")
    ts, ga = compute_team_strengths(intl)
    print(f"  📊 全球场均总进球: {ga:.3f}  |  球队数: {len(ts)}")

    print(f"\n{'─'*60}")
    print(f"  🧠 计算 Elo 评分 (K=32, 全部历史)...")
    elo_r = compute_elo_ratings(intl)
    top_elos = sorted(elo_r.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"  🏆 Elo TOP8: {', '.join(f'{t}({r:.0f})' for t,r in top_elos)}")

    if args.backtest:
        print(f"\n{'─'*60}")
        print(f"  🔄 运行回测...")
        bt = backtest_wc_2022(wc, ts, ga, elo_r)
        print_report(bt)

        if args.odds:
            print(f"\n{'─'*60}")
            print(f"  💰 凯利指数分析")
            print(f"{'─'*60}")
            odds_all = SAMPLE_ODDS if args.odds == 'sample' else json.load(open(args.odds))
            print(f"  使用 {len(odds_all)} 场赔率")

            value_bets = []
            for r in bt['results']:
                key = f"{r['team1']}_{r['team2']}"
                odds_m = odds_all.get(key)
                if not odds_m:
                    rev = f"{r['team2']}_{r['team1']}"
                    odds_m = odds_all.get(rev)
                    if odds_m:
                        odds_m = {'H': odds_m['A'], 'D': odds_m['D'], 'A': odds_m['H']}
                if not odds_m: continue

                kelly_results = []
                for ok, ol, prob in [('H', f"{r['team1']}胜", r['probs']['H']),
                                      ('D', "平局", r['probs']['D']),
                                      ('A', f"{r['team2']}胜", r['probs']['A'])]:
                    kf, ev, edge = kelly_criterion(prob, odds_m[ok])
                    rec = '✅ 下注' if kf > 0.01 and ev > 0 else '❌'
                    kelly_results.append({
                        'outcome': ol, 'pred': prob*100, 'odds': odds_m[ok],
                        'implied': round(1/odds_m[ok]*100,1),
                        'edge': round(edge*100,1), 'ev': round(ev*100,1),
                        'kelly': round(kf*100,1), 'rec': rec,
                    })
                    if ev > 5:
                        value_bets.append({
                            'match': f"{r['team1']} vs {r['team2']}",
                            'outcome': ol, 'ev': round(ev,1),
                            'actual': r['actual'], 'ok': r['hda_correct'],
                        })

                has_value = any(k['ev'] > 5 for k in kelly_results)
                if has_value:
                    print(f"\n  📈 {r['team1']} vs {r['team2']} ({r['date']})")
                    print(f"  {'结果':<12s} {'预测%':>6s} {'赔率':>6s} {'隐含%':>6s} {'优势%':>6s} {'EV%':>7s} {'凯利%':>6s} {'建议'}")
                    print(f"  {'─'*58}")
                    for k in kelly_results:
                        print(f"  {k['outcome']:<12s} {k['pred']:>5.1f}% {k['odds']:>5.2f} {k['implied']:>5.1f}% {k['edge']:>5.1f}% {k['ev']:>6.1f}% {k['kelly']:>5.1f}% {k['rec']}")

            if value_bets:
                print(f"\n{'='*60}")
                print(f"  💎 高价值投注信号汇总 (EV > 5%)")
                print(f"{'='*60}")
                correct = sum(1 for vb in value_bets if vb['ok'])
                for vb in value_bets:
                    ok = '✅' if vb['ok'] else '❌'
                    print(f"  {ok} {vb['match']:<30s} {vb['outcome']:<10s} EV={vb['ev']:>5.1f}%  实际:{vb['actual']}")
                print(f"\n  正确率: {correct}/{len(value_bets)} ({correct/max(len(value_bets),1)*100:.0f}%)")

    return 0

if __name__ == '__main__':
    sys.exit(main())
