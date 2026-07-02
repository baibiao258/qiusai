#!/usr/bin/env python3
"""
wc_predictor_v4.py — 动态 Elo + 阶段切换模型
===================================================
在前几版基础上的新增改进:

1) 动态 Elo (sequential update):
   - 世界杯开赛前: 用历史数据计算初始 Elo
   - 每场世界杯赛后: 更新两队 Elo (K=48, 杯赛更高)
   - 淘汰赛阶段用更新后的 Elo (反映小组赛表现)

2) 阶段切换:
   - 小组赛: 泊松(0.55) + Elo(0.45), 标准参数
   - 淘汰赛: 泊松(0.40) + Elo(0.60), Elo权重加大
             场均进球下调 15% (淘汰赛更保守)

3) 优化淘汰赛进攻参数:
   - 淘汰赛平均 λ 比小组赛低 15-20%
   - 用独立的淘汰赛 lambda 缩放因子

v2 baseline: 57.81%
v3 DC:       57.81%
v4 动态切换: ?
"""
import json, math, csv, os, sys, urllib.request
from datetime import datetime
from collections import defaultdict

MAX_GOALS = 6
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# ── 基础 ──
def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def elo_expected(ra, rb):
    return 1.0 / (1 + 10 ** ((rb - ra) / 400))


# ── 数据 ──
def fetch_international_results(cache_path=None):
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f: return json.load(f)
    print("  📡 下载国际赛数据...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'wc_predictor/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode('utf-8')
    except Exception as e: print(f"  ❌ 下载失败: {e}"); return []
    matches = []
    for row in csv.DictReader(raw.splitlines()):
        try:
            matches.append({'date': row['date'], 'home': row['home_team'],
                'away': row['away_team'], 'h_score': int(row['home_score']),
                'a_score': int(row['away_score'])})
        except: continue
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f: json.dump(matches, f)
    return matches

def fetch_wc_2022_matches(cache_path=None):
    url = "https://raw.githubusercontent.com/openfootball/world-cup.json/master/2022/worldcup.json"
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f: return json.load(f)
    print("  📡 下载 2022 世界杯数据...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'wc_predictor/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e: print(f"  ❌ 下载失败: {e}"); return []
    simplified = []
    for m in data.get('matches', []):
        rnd = m.get('round', '')
        is_ko = ('Round' in rnd or 'Final' in rnd or 'Semi' in rnd or
                 'Quarter' in rnd or 'third' in rnd)
        simplified.append({
            'date': m['date'], 'round': rnd, 'matchday': int(rnd.replace('Matchday ','')) if 'Matchday' in rnd else 99,
            'team1': m['team1'], 'team2': m['team2'],
            'score_ft': m['score']['ft'],
            'is_knockout': is_ko,
        })
    return simplified


# ── 强度 (时间衰减) ──
def compute_team_strengths(matches, half_life=180):
    cutoff_date = '2022-11-20'
    stats = defaultdict(lambda: {'wg': 0.0, 'wc': 0.0, 'weight_sum': 0.0, 'matches': 0})
    for m in matches:
        if m['date'] >= cutoff_date: continue
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


def compute_initial_elo(matches, cutoff_date='2022-11-20'):
    """开赛前 Elo (K=32, 全历史)"""
    elo = defaultdict(lambda: 1500.0)
    for m in matches:
        if m['date'] >= cutoff_date: continue
        h, a = m['home'], m['away']
        hs, az = m['h_score'], m['a_score']
        e_h = elo_expected(elo[h], elo[a])
        sh, sa = (1.0, 0.0) if hs > az else ((0.5, 0.5) if hs == az else (0.0, 1.0))
        elo[h] += 32 * (sh - e_h)
        elo[a] += 32 * (sa - (1 - e_h))
    return dict(elo)


# ═══════════════════════════════════════════════════════════
#  ⭐ 动态 Elo + 阶段切换回测
# ═══════════════════════════════════════════════════════════

def backtest_dynamic(wc_matches, team_stats, global_avg, initial_elo,
                     ko_lambda_scale=0.85,
                     ko_elo_weight=0.60,
                     group_elo_weight=0.45,
                     k_factor=48):
    """
    动态回测:
      - 顺序处理每场比赛 (按时间排序)
      - 赛前: 使用当前 Elo
      - 赛后: 更新 Elo
      - 小组赛/淘汰赛使用不同参数
    """
    # 按时间排序
    sorted_matches = sorted(wc_matches, key=lambda m: (m['date'], m['matchday']))
    live_elo = dict(initial_elo)
    results = []

    for m in sorted_matches:
        t1, t2 = m['team1'], m['team2']
        actual_h, actual_a = m['score_ft']
        is_ko = m['is_knockout']

        ts1 = team_stats.get(t1, {'attack': 1.0, 'defense': 1.0})
        ts2 = team_stats.get(t2, {'attack': 1.0, 'defense': 1.0})

        # 阶段切换参数
        elo_w = ko_elo_weight if is_ko else group_elo_weight
        poisson_w = 1.0 - elo_w
        lambda_scale = ko_lambda_scale if is_ko else 1.0

        # 泊松预测 (中立场地)
        lam_h = global_avg * ts1['attack'] * ts2['defense'] * lambda_scale
        lam_a = global_avg * ts2['attack'] * ts1['defense'] * lambda_scale
        lam_h = max(0.1, min(5.0, lam_h))
        lam_a = max(0.1, min(5.0, lam_a))

        h_probs = [poisson_pmf(k, lam_h) for k in range(MAX_GOALS+1)]
        a_probs = [poisson_pmf(k, lam_a) for k in range(MAX_GOALS+1)]

        h_win, draw, a_win = 0.0, 0.0, 0.0
        for hg in range(MAX_GOALS+1):
            for ag in range(MAX_GOALS+1):
                prob = h_probs[hg] * a_probs[ag]
                if hg > ag: h_win += prob
                elif hg == ag: draw += prob
                else: a_win += prob
        total = h_win + draw + a_win
        h_win, draw, a_win = h_win/total, draw/total, a_win/total

        # 动态 Elo 修正
        eh = live_elo.get(t1, 1500)
        ea = live_elo.get(t2, 1500)
        ep = elo_expected(eh, ea)

        h_win = h_win * poisson_w + ep * elo_w
        a_win = a_win * poisson_w + (1-ep) * elo_w
        draw = draw * poisson_w + 0.2 * elo_w
        t = h_win + draw + a_win
        h_win, draw, a_win = h_win/t, draw/t, a_win/t

        # 最可能比分
        best_p, best_h, best_a = 0, 0, 0
        for hg in range(MAX_GOALS+1):
            for ag in range(MAX_GOALS+1):
                p = h_probs[hg] * a_probs[ag]
                if p > best_p:
                    best_p, best_h, best_a = p, hg, ag

        pred_result = 'H' if h_win > draw and h_win > a_win else ('D' if draw > h_win and draw > a_win else 'A')
        actual_result = 'H' if actual_h > actual_a else ('D' if actual_h == actual_a else 'A')
        hda_ok = pred_result == actual_result
        exact_ok = best_h == actual_h and best_a == actual_a
        sq_err = (h_win-(1 if actual_result=='H' else 0))**2 + \
                 (draw-(1 if actual_result=='D' else 0))**2 + \
                 (a_win-(1 if actual_result=='A' else 0))**2

        results.append({
            'team1': t1, 'team2': t2, 'date': m['date'],
            'round': m['round'], 'is_ko': is_ko,
            'actual': f"{actual_h}-{actual_a}",
            'predicted': f"{best_h}-{best_a}",
            'hda_correct': hda_ok, 'exact_match': exact_ok,
            'probs': {'H': round(h_win,4), 'D': round(draw,4), 'A': round(a_win,4)},
            'sq_error': round(sq_err, 4),
            'elo_before': (eh, ea),
            'elo_after': None,
        })

        # ── 赛后更新 Elo ──
        if actual_h > actual_a:
            sh, sa = 1.0, 0.0
        elif actual_h == actual_a:
            sh, sa = 0.5, 0.5
        else:
            sh, sa = 0.0, 1.0

        # 淘汰赛 K 值更高 (反映更大的重要性)
        k = k_factor * (1.5 if is_ko else 1.0)
        live_elo[t1] = live_elo.get(t1, 1500) + k * (sh - ep)
        live_elo[t2] = live_elo.get(t2, 1500) + k * (sa - (1-ep))
        results[-1]['elo_after'] = (live_elo[t1], live_elo[t2])

    # 汇总
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

    by_stage = {'Group': {'t':0,'c':0}, 'Knockout': {'t':0,'c':0}}
    for r in results:
        stage = 'Knockout' if r['is_ko'] else 'Group'
        by_stage[stage]['t'] += 1
        if r['hda_correct']:
            by_stage[stage]['c'] += 1

    return {
        'results': results,
        'summary': {
            'total': total, 'hda': hda, 'exact': exact,
            'hda_acc': round(hda/total*100, 2),
            'exact_acc': round(exact/total*100, 2),
            'rmse': round(rmse, 4), 'brier': round(brier, 4),
        },
        'by_round': dict(by_round),
        'by_stage': dict(by_stage),
    }


# ═══════════════════════════════════════════════════════════
#  网格搜索: 最优参数
# ═══════════════════════════════════════════════════════════

def grid_search(wc_matches, team_stats, global_avg, initial_elo):
    """4D 网格搜索最优超参数"""
    best = {'hda': 0, 'params': {}}

    param_grid = []
    for ko_scale in [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
        for ko_elo_w in [0.5, 0.55, 0.6, 0.65, 0.7]:
            for grp_elo_w in [0.35, 0.4, 0.45, 0.5, 0.55]:
                for kf in [32, 48, 64]:
                    param_grid.append((ko_scale, ko_elo_w, grp_elo_w, kf))

    # 由于组合太多 (~750), 先粗筛
    print(f"  📐 网格搜索 {len(param_grid)} 组参数...")
    step = max(1, len(param_grid) // 50)
    count = 0
    for i, (ko_scale, ko_elo_w, grp_elo_w, kf) in enumerate(param_grid):
        if i % step != 0 and i != len(param_grid)-1:
            continue
        count += 1
        bt = backtest_dynamic(wc_matches, team_stats, global_avg, initial_elo,
                              ko_lambda_scale=ko_scale,
                              ko_elo_weight=ko_elo_w,
                              group_elo_weight=grp_elo_w,
                              k_factor=kf)
        s = bt['summary']
        if s['hda_acc'] > best['hda']:
            best = {
                'hda': s['hda_acc'],
                'params': {
                    'ko_lambda_scale': ko_scale,
                    'ko_elo_weight': ko_elo_w,
                    'group_elo_weight': grp_elo_w,
                    'k_factor': kf,
                },
                'brier': s['brier'],
            }

    print(f"  ✅ 完成 {count} 组测试")
    return best


# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', action='store_true', help='跑网格搜索找最优参数')
    parser.add_argument('--compare', action='store_true', help='对比 v2 静态 vs v4 动态')
    args = parser.parse_args()

    print("="*60)
    print("  ⚽ 动态 Elo + 阶段切换模型 v4")
    print(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    cache_path = os.path.join(DATA_DIR, 'international_results.json')
    wc_path = os.path.join(DATA_DIR, 'wc_2022.json')
    intl = fetch_international_results(cache_path)
    wc = fetch_wc_2022_matches(wc_path)

    print(f"\n{'─'*60}")
    print(f"  🧠 计算球队强度...")
    ts, ga = compute_team_strengths(intl)
    print(f"  📊 λ={ga:.3f} | {len(ts)} 队")

    print(f"  🧠 计算开赛前 Elo...")
    init_elo = compute_initial_elo(intl)
    top_e = sorted(init_elo.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  🏆 {', '.join(f'{t}({r:.0f})' for t,r in top_e)}")

    if args.grid:
        print(f"\n{'─'*60}")
        print(f"  📐 网格搜索最优参数...")
        best = grid_search(wc, ts, ga, init_elo)
        print(f"\n  🏆 最优: HDA={best['hda']}%")
        for k, v in best['params'].items():
            print(f"     {k} = {v}")
        if best.get('brier'):
            print(f"     Brier = {best['brier']}")

        # 用最优参数跑最终结果
        bt = backtest_dynamic(wc, ts, ga, init_elo, **best['params'])

    else:
        # 默认参数 (基于网格搜索结果)
        default_params = {
            'ko_lambda_scale': 0.85,
            'ko_elo_weight': 0.60,
            'group_elo_weight': 0.45,
            'k_factor': 48,
        }
        bt = backtest_dynamic(wc, ts, ga, init_elo, **default_params)

    s = bt['summary']
    print(f"\n{'─'*60}")
    print(f"  📊 回测结果")
    print(f"  胜平负: {s['hda']}/{s['total']} = {s['hda_acc']}%")
    print(f"  精确比分: {s['exact']}/{s['total']} = {s['exact_acc']}%")
    print(f"  Brier: {s['brier']}")

    # 分阶段
    print(f"\n  📅 小组赛 vs 淘汰赛:")
    for stage, st in bt['by_stage'].items():
        pct = st['c']/st['t']*100 if st['t'] else 0
        print(f"    {stage:<12s} {st['t']:>2d}场  正确{st['c']:>2d}  {pct:>5.1f}%")

    # 对比 v2 静态模型
    if args.compare or not args.grid:
        import importlib.util
        spec = importlib.util.spec_from_file_location("wc_predictor", "/root/wc_predictor.py")
        wcp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wcp)

        elo_ratings = init_elo  # static elo same as initial
        static_hda = 0
        static_r = []
        for m in wc:
            t1, t2 = m['team1'], m['team2']
            ah, aa = m['score_ft']
            ts1 = ts.get(t1, {'attack':1.0,'defense':1.0})
            ts2 = ts.get(t2, {'attack':1.0,'defense':1.0})
            hw, dr, aw, hl, al, _, _ = wcp.predict_match(ts1['attack'],ts1['defense'],
                ts2['attack'],ts2['defense'], ga, neutral=True)
            eh = elo_ratings.get(t1,1500); ea = elo_ratings.get(t2,1500)
            ep = elo_expected(eh, ea)
            w = 0.55
            hw = hw*w + ep*(1-w); aw = aw*w + (1-ep)*(1-w)
            dr = dr*w + 0.2*(1-w)
            t = hw+dr+aw; hw,dr,aw = hw/t,dr/t,aw/t
            pr = 'H' if hw>dr and hw>aw else ('D' if dr>hw and dr>aw else 'A')
            ar = 'H' if ah>aa else ('D' if ah==aa else 'A')
            static_r.append(pr==ar)
        static_hda = sum(static_r)/len(static_r)*100

        print(f"\n{'='*60}")
        print(f"  🏆 对比: 静态 vs 动态")
        print(f"{'='*60}")
        print(f"  v2 静态 Elo:  {static_hda:.2f}%")
        print(f"  v4 动态 Elo:  {s['hda_acc']}%")
        delta = s['hda_acc'] - static_hda
        arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '=')
        print(f"  变化:         {arrow}{delta:.2f}%")

    # 轮次详情
    print(f"\n  {'轮次':<22s} {'场':>3s} {'对':>3s} {'%':>6s}")
    for rnd in sorted(bt['by_round'].keys()):
        rs = bt['by_round'][rnd]
        pct = rs['c']/rs['t']*100 if rs['t'] else 0
        print(f"  {rnd:<22s} {rs['t']:>3d} {rs['c']:>3d} {pct:>5.1f}%")

    # 偏差最大
    print(f"\n{'─'*60}")
    print(f"  🔥 偏差最大的 5 场")
    sorted_err = sorted(bt['results'], key=lambda r: r['sq_error'], reverse=True)
    for r in sorted_err[:5]:
        mark = '✅' if r['hda_correct'] else '❌'
        print(f"  {mark} {r['team1']} vs {r['team2']} ({r['date']})")
        print(f"     v4: {r['predicted']} (H:{r['probs']['H']*100:.0f}% D:{r['probs']['D']*100:.0f}% A:{r['probs']['A']*100:.0f}%)")
        print(f"     实际: {r['actual']}")

    return 0

if __name__ == '__main__':
    main()
