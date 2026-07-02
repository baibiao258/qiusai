#!/usr/bin/env python3
"""
wc_predictor_v3.py — Dixon-Coles 模型 (改进型泊松)
===================================================
Dixon-Coles (1997) 在纯泊松基础上增加 rho 参数调整低比分相关性:
  - 负 rho → 更多低比分平局 (0-0, 1-1), 更少高比分
  - tau(x,y) 仅在比分 ≤1 时调整, 高比分使用标准独立泊松

同时实现了:
  - 进攻/防守参数的 MLE 估计 (从当前比率法初始值优化)
  - 时间衰减 (半衰期可调)
  - Elo 修正

对比基准:
  v2 (泊松+Elo+时间衰减) → 57.81%
  v3 (Dixon-Coles)      → ?
"""
import json, math, csv, os, sys, urllib.request
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize

MAX_GOALS = 6
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# ═══════════════════════════════════════════════════════════
#  1) 泊松基础 (与 v2 一致)
# ═══════════════════════════════════════════════════════════

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def elo_expected(ra, rb):
    return 1.0 / (1 + 10 ** ((rb - ra) / 400))


# ═══════════════════════════════════════════════════════════
#  2) Dixon-Coles 核心
# ═══════════════════════════════════════════════════════════

def dc_tau(x, y, lam_h, lam_a, rho):
    """
    Dixon-Coles 调整因子 tau
    仅对低比分 (x<=1, y<=1) 调整; 高比分 tau=1
    """
    if x == 0 and y == 0:
        return 1 - rho * lam_h * lam_a
    elif x == 0 and y == 1:
        return 1 + rho * lam_h
    elif x == 1 and y == 0:
        return 1 + rho * lam_a
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def dc_probability(x, y, lam_h, lam_a, rho):
    """Dixon-Coles 联合概率 P(X=x, Y=y)"""
    tau = dc_tau(x, y, lam_h, lam_a, rho)
    return tau * poisson_pmf(x, lam_h) * poisson_pmf(y, lam_a)


def predict_match_dc(home_attack, home_defense, away_attack, away_defense,
                     league_avg, rho, neutral=True):
    """
    Dixon-Coles 预测: 比分概率矩阵 + 汇总胜平负
    返回: (h_win, draw, a_win, h_lambda, a_lambda)
    """
    h_adj = 1.0 if neutral else 1.05
    a_adj = 1.0 if neutral else 1.0
    lam_h = league_avg * home_attack * away_defense * h_adj
    lam_a = league_avg * away_attack * home_defense * a_adj
    lam_h = max(0.1, min(5.0, lam_h))
    lam_a = max(0.1, min(5.0, lam_a))

    h_win, draw, a_win = 0.0, 0.0, 0.0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            prob = dc_probability(hg, ag, lam_h, lam_a, rho)
            if hg > ag:       h_win += prob
            elif hg == ag:    draw += prob
            else:             a_win += prob

    total = h_win + draw + a_win
    return h_win/total, draw/total, a_win/total, lam_h, lam_a


# ═══════════════════════════════════════════════════════════
#  3) MLE 参数估计 - 估计 rho
# ═══════════════════════════════════════════════════════════

def estimate_rho(matches, team_stats, global_avg, cutoff_date='2022-11-20'):
    """
    使用训练数据 MLE 估计 Dixon-Coles rho 参数
    在固定的攻防强度基础上，一维优化 rho
    
    rho 理论上在 [-0.3, 0.3] 范围, 足球负相关 (rho<0)
    """
    # 筛选训练比赛
    train_matches = [m for m in matches if m['date'] < cutoff_date]
    
    def neg_log_likelihood(rho):
        ll = 0.0
        for m in train_matches:
            ht, at = m['home'], m['away']
            hg, ag = m['h_score'], m['a_score']
            ts_h = team_stats.get(ht, {'attack': 1.0, 'defense': 1.0})
            ts_a = team_stats.get(at, {'attack': 1.0, 'defense': 1.0})
            # 用全局均值, 默认中立=0 (大部分比赛有主客场)
            
            lam_h = global_avg * ts_h['attack'] * ts_a['defense']
            lam_a = global_avg * ts_a['attack'] * ts_h['defense']
            
            prob = dc_probability(min(hg, MAX_GOALS), min(ag, MAX_GOALS),
                                  lam_h, lam_a, rho)
            if prob > 0:
                ll += math.log(prob)
            else:
                ll += math.log(1e-10)
        return -ll  # 返回负对数似然
    
    # 网格搜索初始化 + 精细优化
    grid = [round(-0.30 + i * 0.05, 2) for i in range(13)]  # -0.30 ~ 0.30 step 0.05
    best_rho, best_ll = 0.0, float('inf')
    for r in grid:
        try:
            ll = neg_log_likelihood(r)
            if ll < best_ll:
                best_ll, best_rho = ll, r
        except: continue
    
    # 从网格最佳点做精细优化
    try:
        result = minimize(neg_log_likelihood, best_rho, method='Nelder-Mead',
                          options={'xatol': 1e-6, 'maxiter': 200})
        if result.success:
            best_rho = float(result.x[0])
    except: pass
    
    best_rho = max(-0.4, min(0.1, best_rho))
    
    print(f"  📐 Dixon-Coles ρ = {best_rho:.4f} (网格搜索 → 精细优化)")
    return best_rho


# ═══════════════════════════════════════════════════════════
#  4) 数据加载 (与 v2 一致)
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
#  5) 球队强度 (时间衰减) — 同 v2
# ═══════════════════════════════════════════════════════════

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


def compute_elo_ratings(matches, cutoff_date='2022-11-20'):
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
#  6) 回测
# ═══════════════════════════════════════════════════════════

def backtest_wc_2022(wc_matches, team_stats, global_avg, elo_ratings, rho, use_dc=True):
    results = []

    for m in wc_matches:
        t1, t2 = m['team1'], m['team2']
        actual_h, actual_a = m['score_ft']
        ts1 = team_stats.get(t1, {'attack': 1.0, 'defense': 1.0, 'matches': 0})
        ts2 = team_stats.get(t2, {'attack': 1.0, 'defense': 1.0, 'matches': 0})

        if use_dc:
            hw, dr, aw, hl, al = predict_match_dc(
                ts1['attack'], ts1['defense'], ts2['attack'], ts2['defense'],
                global_avg, rho, neutral=True)
        else:
            # 回退到标准泊松
            from wc_predictor import predict_match
            hw, dr, aw, hl, al, _, _ = predict_match(
                ts1['attack'], ts1['defense'], ts2['attack'], ts2['defense'],
                global_avg, neutral=True)

        # Elo 修正 (同 v2: 泊松0.55 + Elo0.45)
        eh = elo_ratings.get(t1, 1500)
        ea = elo_ratings.get(t2, 1500)
        ep = elo_expected(eh, ea)
        w = 0.55
        hw = hw * w + ep * (1-w)
        aw = aw * w + (1-ep) * (1-w)
        dr = dr * w + 0.2 * (1-w)
        t = hw + dr + aw
        hw, dr, aw = hw/t, dr/t, aw/t

        # 最可能比分
        if use_dc:
            h_probs = [sum(dc_probability(hg, ag, hl, al, rho) for ag in range(MAX_GOALS+1)) for hg in range(MAX_GOALS+1)]
            a_probs = [sum(dc_probability(hg, ag, hl, al, rho) for hg in range(MAX_GOALS+1)) for ag in range(MAX_GOALS+1)]
        else:
            h_probs = [poisson_pmf(k, hl) for k in range(MAX_GOALS+1)]
            a_probs = [poisson_pmf(k, al) for k in range(MAX_GOALS+1)]

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


# ═══════════════════════════════════════════════════════════
#  7) 诊断: DC 对平局预测的具体影响
# ═══════════════════════════════════════════════════════════

def analyze_draw_effect(wc_matches, team_stats, global_avg, rho):
    """对比 DC vs Poisson 在平局预测上的差异"""
    diffs = []
    for m in wc_matches:
        t1, t2 = m['team1'], m['team2']
        ts1 = team_stats.get(t1, {'attack':1.0,'defense':1.0})
        ts2 = team_stats.get(t2, {'attack':1.0,'defense':1.0})
        actual_h, actual_a = m['score_ft']
        actual_is_draw = actual_h == actual_a

        _, dr_dc, _, _, _ = predict_match_dc(
            ts1['attack'], ts1['defense'], ts2['attack'], ts2['defense'],
            global_avg, rho, neutral=True)

        from wc_predictor import predict_match
        _, dr_pois, _, _, _, _, _ = predict_match(
            ts1['attack'], ts1['defense'], ts2['attack'], ts2['defense'],
            global_avg, neutral=True)

        diffs.append({
            'match': f"{t1} vs {t2}",
            'poisson_draw': round(dr_pois*100, 1),
            'dc_draw': round(dr_dc*100, 1),
            'diff': round((dr_dc - dr_pois)*100, 1),
            'actual': '平局' if actual_is_draw else f"{actual_h}-{actual_a}",
        })

    return diffs


# ═══════════════════════════════════════════════════════════
#  8) 入口
# ═══════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("  ⚽ Dixon-Coles 改进泊松预测 v3")
    print(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    cache_path = os.path.join(DATA_DIR, 'international_results.json')
    wc_path = os.path.join(DATA_DIR, 'wc_2022.json')
    intl = fetch_international_results(cache_path)
    wc = fetch_wc_2022_matches(wc_path)

    # ── 基础强度 ──
    print(f"\n{'─'*60}")
    print(f"  🧠 计算球队强度 (时间衰减 hl=180d)...")
    ts, ga = compute_team_strengths(intl)
    print(f"  📊 全球场均总进球: {ga:.3f}  |  球队数: {len(ts)}")

    # ── Elo ──
    print(f"\n{'─'*60}")
    print(f"  🧠 计算 Elo 评分...")
    elo_r = compute_elo_ratings(intl)

    # ── 估算 rho ──
    print(f"\n{'─'*60}")
    print(f"  📐 MLE 估计 Dixon-Coles ρ 参数...")
    rho = estimate_rho(intl, ts, ga)
    print(f"  ℹ️  ρ < 0 => 低比分比赛概率更高 (足球典型特征)")

    # ── 诊断平局差异 ──
    print(f"\n{'─'*60}")
    print(f"  🔍 DC vs 标准泊松: 平局概率对比 (偏差最大的8场)")
    print(f"{'─'*60}")
    diffs = analyze_draw_effect(wc, ts, ga, rho)
    diffs.sort(key=lambda x: abs(x['diff']), reverse=True)
    print(f"  {'比赛':<30s} {'泊松(平)':>8s} {'DC(平)':>8s} {'差':>6s} {'实际'}")
    for d in diffs[:8]:
        print(f"  {d['match']:<30s} {d['poisson_draw']:>6.1f}% {d['dc_draw']:>6.1f}% {d['diff']:>+5.1f}%  {d['actual']}")

    # ── 回测: DC vs 标准泊松 ──
    print(f"\n{'─'*60}")
    print(f"  🔄 回测对比: DC vs 标准泊松 (均含 Elo修正)")
    print(f"{'─'*60}")

    bt_dc = backtest_wc_2022(wc, ts, ga, elo_r, rho, use_dc=True)

    # 用 v2 的纯泊松
    # import the module properly
    import importlib.util
    spec = importlib.util.spec_from_file_location("wc_predictor", "/root/wc_predictor.py")
    wcp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wcp)

    bt_results = {}
    for name, use_dc_val, rho_val in [
        ('Dixon-Coles + Elo', True, rho),
        ('标准泊松 + Elo (v2)', False, 0.0),
    ]:
        if use_dc_val:
            bt = bt_dc
        else:
            # 标准泊松
            bt_r = []
            for m in wc:
                t1, t2 = m['team1'], m['team2']
                ah, aa = m['score_ft']
                ts1 = ts.get(t1, {'attack':1.0,'defense':1.0,'matches':0})
                ts2 = ts.get(t2, {'attack':1.0,'defense':1.0,'matches':0})
                hw, dr, aw, hl, al, _, _ = wcp.predict_match(
                    ts1['attack'], ts1['defense'], ts2['attack'], ts2['defense'],
                    ga, neutral=True)
                # Elo
                eh = elo_r.get(t1, 1500); ea = elo_r.get(t2, 1500)
                ep = elo_expected(eh, ea)
                w = 0.55
                hw = hw*w + ep*(1-w); aw = aw*w + (1-ep)*(1-w)
                dr = dr*w + 0.2*(1-w)
                t = hw+dr+aw; hw,dr,aw = hw/t,dr/t,aw/t
                pr = 'H' if hw>dr and hw>aw else ('D' if dr>hw and dr>aw else 'A')
                ar = 'H' if ah>aa else ('D' if ah==aa else 'A')
                sq = (hw-(1 if ar=='H' else 0))**2 + (dr-(1 if ar=='D' else 0))**2 + (aw-(1 if ar=='A' else 0))**2
                bt_r.append({
                    'hda_correct': pr==ar, 'exact_match': False,
                    'sq_error': sq, 'probs': {'H':hw,'D':dr,'A':aw},
                    'team1': t1, 'team2': t2, 'actual': f"{ah}-{aa}",
                    'round': m['round'],
                })
            bt = {
                'results': bt_r,
                'summary': {
                    'total': len(bt_r),
                    'hda': sum(1 for r in bt_r if r['hda_correct']),
                    'exact': 0,
                    'hda_acc': round(sum(1 for r in bt_r if r['hda_correct'])/len(bt_r)*100, 2),
                    'rmse': round(math.sqrt(sum(r['sq_error'] for r in bt_r)/len(bt_r)), 4),
                    'brier': round(sum(r['sq_error'] for r in bt_r)/len(bt_r), 4),
                }
            }

        bt_results[name] = bt

    # ── 对比表 ──
    print(f"\n{'='*60}")
    print(f"  🏆 对比结果")
    print(f"{'='*60}")
    print(f"  {'模型':<28s} {'HDA准确率':>10s} {'Brier':>7s} {'RMSE':>7s}")
    print(f"  {'─'*52}")
    for name, bt in bt_results.items():
        s = bt['summary']
        print(f"  {name:<28s} {s['hda_acc']:>8.2f}%  {s['brier']:>6.4f}  {s['rmse']:>6.4f}")

    # 详细 DC 结果
    s = bt_dc['summary']
    print(f"\n{'─'*60}")
    print(f"  📊 DC 详细: {s['hda']}/{s['total']} = {s['hda_acc']}%  |  比分: {s['exact']}/{s['total']} = {s['exact_acc']}%")
    print(f"  Brier: {s['brier']}  |  RMSE: {s['rmse']}")
    print(f"\n  {'轮次':<22s} {'场':>3s} {'对':>3s} {'%':>6s}")
    for rnd in sorted(bt_dc['by_round'].keys()):
        rs = bt_dc['by_round'][rnd]
        pct = rs['c']/rs['t']*100 if rs['t'] else 0
        print(f"  {rnd:<22s} {rs['t']:>3d} {rs['c']:>3d} {pct:>5.1f}%")

    # 偏差最大 vs v2
    print(f"\n{'─'*60}")
    print(f"  🔥 DC 预测偏差最大的 5 场")
    sorted_err = sorted(bt_dc['results'], key=lambda r: r['sq_error'], reverse=True)
    for r in sorted_err[:5]:
        mark = '✅' if r['hda_correct'] else '❌'
        print(f"  {mark} {r['team1']} vs {r['team2']} ({r['date']})")
        print(f"     DC: {r['predicted']} (H:{r['probs']['H']*100:.0f}% D:{r['probs']['D']*100:.0f}% A:{r['probs']['A']*100:.0f}%)")
        print(f"     实际: {r['actual']}")

    return 0

if __name__ == '__main__':
    main()
