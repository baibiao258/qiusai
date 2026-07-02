#!/usr/bin/env python3
"""
Backtest Suite for daily_jczq hybrid model
============================================
Loads completed match results, runs model predictions,
computes Brier decomposition, RPS, EV, and simulated ROI.

Usage:
  python3 backtest_runner.py
"""
import csv, json, os, sys, math, glob
from datetime import datetime

import numpy as np

sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')

# ── helpers ──
def rps_score(y_true, y_proba):
    cdf_true = np.cumsum(y_true, axis=1)
    cdf_pred = np.cumsum(y_proba, axis=1)
    return np.mean(np.sum((cdf_true - cdf_pred)**2, axis=1) / (y_proba.shape[1] - 1))

def brier_decomposition(y_true, y_proba):
    """
    Brier = Uncertainty - Resolution + Reliability
    y_true: (N, K) one-hot
    y_proba: (N, K) predicted probs
    Returns dict with components.
    """
    N, K = y_true.shape
    # base rate per class
    base = y_true.mean(axis=0)  # (K,)
    # Uncertainty = base * (1 - base).sum() / K ? No.
    # Actually: Brier is per-sample sum((y-p)^2) averaged.
    # Decomposition:
    # Brier = Uncertainty - Resolution + Reliability
    # Uncertainty = (1/K) * sum_k base_k * (1 - base_k)
    # Actually the correct formula:
    # Uncertainty = mean of (1 - sum_k base_k^2) ... no
    # Proper decomposition (Murphy 1973):
    # Uncertainty = mean_base = base.dot(1 - base)
    # Resolution = mean of (prob_in_bin - base)^2 * bin_weight
    # Reliability = mean of (observed_freq_in_bin - mean_pred_in_bin)^2 * bin_weight
    
    # Simplified approach for probability forecasts:
    # Brier = 1/N * sum_i sum_k (y_ik - p_ik)^2
    # Uncertainty = 1/K * sum_k base_k * (1 - base_k) -- actually for multi-class
    
    # For 3-class:
    # Uncertainty = sum_k base_k * (1 - base_k) / K ? Let's use proper formula.
    # The standard:
    # Brier = Uncertainty - Resolution + Reliability
    # where:
    # Uncertainty = (1/N) * sum_i sum_k y_ik*(1 - y_ik) = (1/N)*sum_i (1 - sum_k y_ik) = 0 since y is one-hot
    # Hmm no. Actually:
    # For one-hot:
    # Uncertainty = mean of variance = (1/N) * sum_i [1 - (1/K)]... 
    # Let me use the proper multi-class decomposition from Murphy (1973)
    # The common approach:
    # BS = (1/N) * sum_i sum_k (y_ik - p_ik)^2
    # BS_ref = sum_k base_k * (1 - base_k)  [uncertainty of the climatology]
    # BS_res = (1/N) * sum_i sum_k (p_ik - base_k)^2  [resolution]
    # BS_rel = (1/N) * sum_i sum_k (y_ik - p_ik)^2 - BS_ref + BS_res ... not right
    
    # Actually let me just implement the standard 3-component decomposition properly.
    # For multi-class, the standard approach by Murphy (1973):
    # BS = UNC - RES + REL
    # 
    # Let me use the sklearn convention:
    # For each class k:
    #   - Sort predictions and outcomes by predicted probability
    #   - Group into bins (e.g. 10 bins)
    #   - For each bin: mean_pred, observed_freq
    #   - Reliability = sum over bins of weight * (observed_freq - mean_pred)^2
    #   - Resolution = sum over bins of weight * (observed_freq - base_rate_k)^2  
    #   - Uncertainty = base_rate_k * (1 - base_rate_k)
    # Total = mean over classes
    
    # Let me do per-class then average
    bs = np.mean(np.sum((y_true - y_proba)**2, axis=1))
    
    # Per-class decomposition
    rel_sum, res_sum, unc_sum = 0, 0, 0
    n_bins = 10
    
    for k in range(K):
        base_k = base[k]
        unc_k = base_k * (1 - base_k)
        
        p_k = y_proba[:, k]
        y_k = y_true[:, k]
        
        # Sort by predicted probability
        idx = np.argsort(p_k)
        p_sorted = p_k[idx]
        y_sorted = y_k[idx]
        
        # Bin edges (quantile-based or uniform)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(p_sorted, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            w = mask.sum() / N
            mean_pred = p_sorted[mask].mean()
            obs_freq = y_sorted[mask].mean()
            rel_sum += w * (obs_freq - mean_pred)**2
            res_sum += w * (obs_freq - base_k)**2
        
        unc_sum += unc_k
    
    reliability = rel_sum / K
    resolution = res_sum / K
    uncertainty = unc_sum / K
    
    return {
        'brier': float(bs),
        'uncertainty': float(uncertainty),
        'resolution': float(resolution),
        'reliability': float(reliability),
        'brier_decomp_check': float(uncertainty - resolution + reliability),
    }

# ── Load model ──
from daily_jczq import _try_hybrid_predict, compute_dynamic_xgb_weight, poisson_pmf, MAX_GOALS

def predict_hybrid(team_home, team_away):
    """Run hybrid model on a pair of team names."""
    try:
        result = _try_hybrid_predict(team_home, team_away)
        if result is None:
            return None
        probs = result['probs']
        # Ensure sum to 1
        total = probs['H'] + probs['D'] + probs['A']
        if total > 0:
            for k in ['H','D','A']:
                probs[k] /= total
        return probs
    except Exception as e:
        return None

# ── Load results data ──
HDA_MAP = {'胜': 'H', '平': 'D', '负': 'A'}

def load_results():
    """Load all completed matches from result JSONs."""
    matches = []
    for f in sorted(glob.glob('/root/data/results/2026-06-*.json')):
        date = os.path.basename(f).replace('.json', '')
        with open(f) as fh:
            data = json.load(fh)
        for m in data:
            if m.get('hda_result') and m.get('score_full'):
                matches.append({
                    'date': date,
                    'code': m.get('code', ''),
                    'home': m.get('home', ''),
                    'away': m.get('away', ''),
                    'score': m.get('score_full', ''),
                    'hda_cn': m.get('hda_result', ''),
                    'hda': HDA_MAP.get(m.get('hda_result', ''), ''),
                    'league': m.get('league', ''),
                })
    return matches

# ── Team name mapping (results → model names) ──
# Since results use Chinese names and model uses internal normalized names,
# we try a few variants.
TEAM_ALIASES = {
    '克罗地亚': 'Croatia', '比利时': 'Belgium', '威尔士': 'Wales', '加纳': 'Ghana',
    '格鲁吉亚': 'Georgia', '罗马尼亚': 'Romania', '卢森堡': 'Luxembourg', '意大利': 'Italy',
    '波兰': 'Poland', '尼日利亚': 'Nigeria', '荷兰': 'Netherlands', '阿尔及利亚': 'Algeria',
    '丹麦': 'Denmark', '民主刚果': 'Congo DR', '刚果(金)': 'Congo DR', '刚果（金）': 'Congo DR',
    '墨西哥': 'Mexico', '塞尔维亚': 'Serbia', '法国': 'France', '科特迪瓦': 'Ivory Coast',
    '西班牙': 'Spain', '伊拉克': 'Iraq', '瑞典': 'Sweden', '希腊': 'Greece',
    '斯洛文尼亚': 'Slovenia', '塞浦路斯': 'Cyprus', '加拿大': 'Canada', '爱尔兰': 'Ireland',
    '匈牙利': 'Hungary', '芬兰': 'Finland', '斯洛伐克': 'Slovakia', '黑山': 'Montenegro',
    '新加坡': 'Singapore', '中国': 'China', '阿根廷': 'Argentina', '洪都拉斯': 'Honduras',
    '委内瑞拉': 'Venezuela', '土耳其': 'Turkey', '巴西': 'Brazil', '埃及': 'Egypt',
    '玻利维亚': 'Bolivia', '苏格兰': 'Scotland', '英格兰': 'England', '新西兰': 'New Zealand',
    '巴拿马': 'Panama', '波黑': 'Bosnia', '澳大利亚': 'Australia', '瑞士': 'Switzerland',
    '美国': 'USA', '德国': 'Germany', '葡萄牙': 'Portugal', '智利': 'Chile',
    '突尼斯': 'Tunisia', 
    '川崎前锋': 'Kawasaki Frontale', '广岛三箭': 'Sanfrecce Hiroshima',
    '柏太阳神': 'Kashiwa Reysol', '京都不死鸟': 'Kyoto Sanga',
    '横滨水手': 'Yokohama F. Marinos', '清水心跳': 'Shimizu S-Pulse',
    '浦和红钻': 'Urawa Reds', '冈山绿雉': 'Fagiano Okayama',
    '町田泽维亚': 'FC Machida Zelvia', '名古屋鲸鱼': 'Nagoya Grampus',
    '鹿岛鹿角': 'Kashima Antlers', '神户胜利船': 'Vissel Kobe',
}

def main():
    matches = load_results()
    print(f"📥 加载 {len(matches)} 场已完成比赛\n")
    
    predictions = []
    failures = []
    
    for i, m in enumerate(matches):
        home = m['home']
        away = m['away']
        
        # Try Chinese names directly (model's _try_hybrid_predict calls normalize_match_pair internally)
        probs = predict_hybrid(home, away)
        
        if probs is None:
            failures.append(m)
            continue
        
        predictions.append({**m, 'probs': probs})
    
    n_run = len(predictions)
    n_fail = len(failures)
    
    print(f"🎯 模型可预测: {n_run}/{len(matches)}")
    if n_fail > 0:
        print(f"⚠️ 无法预测 (无form数据/球队不在模型库): {n_fail} 场")
        for f in failures:
            print(f"   ✗ [{f['date']}] {f['home']} vs {f['away']} ({f['score']})")
    print()
    
    if n_run == 0:
        print("❌ 无可用预测，退出")
        return
    
    # ── Probability vectors & actuals ──
    y_true = np.zeros((n_run, 3))
    y_proba = np.zeros((n_run, 3))
    label_idx = {'A': 0, 'D': 1, 'H': 2}
    
    results_detail = []
    for i, p in enumerate(predictions):
        probs = p['probs']
        y_proba[i] = [probs['A'], probs['D'], probs['H']]
        y_true[i, label_idx[p['hda']]] = 1
        
        # Predicted outcome
        pred_idx = np.argmax(y_proba[i])
        pred_label = ['A', 'D', 'H'][pred_idx]
        correct = pred_label == p['hda']
        
        results_detail.append({
            'date': p['date'],
            'code': p['code'],
            'home': p['home'],
            'away': p['away'],
            'score': p['score'],
            'actual': p['hda'],
            'pred': pred_label,
            'probs': {'H': probs['H'], 'D': probs['D'], 'A': probs['A']},
            'correct': correct,
        })
    
    # ── Core Metrics ──
    brier = float(np.mean(np.sum((y_true - y_proba)**2, axis=1)))
    rps = float(rps_score(y_true, y_proba))
    accuracy = sum(r['correct'] for r in results_detail) / n_run
    
    decomp = brier_decomposition(y_true, y_proba)
    
    # ── EV Analysis ──
    # Simulate flat stake betting at fair odds (1/prob * 0.95 for margin)
    # Use actual 500.com odds from results? No — results don't have odds.
    # Use model-implied fair odds
    bankroll = 100.0
    stake = 1.0  # 1 unit per bet
    bets_placed = 0
    bets_won = 0
    roi_total = 0.0
    
    ev_detail = []
    for r in results_detail:
        probs = r['probs']
        total_ev = 0
        best_ev = 0
        best_bet = ''
        
        for label, p in [('H', probs['H']), ('D', probs['D']), ('A', probs['A'])]:
            fair_odds = 1.0 / max(p, 0.001)
            # Simulate 5% margin bookmaker odds
            market_odds = fair_odds * 0.92
            ev = p * market_odds - 1
            if ev > best_ev:
                best_ev = ev
                best_bet = label
        
        ev_detail.append({
            **r,
            'best_ev': best_ev,
            'best_bet': best_bet,
            'fair_odds_H': 1.0/max(probs['H'], 0.001),
            'fair_odds_D': 1.0/max(probs['D'], 0.001),
            'fair_odds_A': 1.0/max(probs['A'], 0.001),
        })
        
        if best_ev > 0.05:  # Only bet when >5% edge
            bets_placed += 1
            if best_bet == r['actual']:
                bets_won += 1
                market_odds = 1.0 / max({'H': probs['H'], 'D': probs['D'], 'A': probs['A']}[best_bet], 0.001) * 0.92
                roi_total += (market_odds - 1) * stake
            else:
                roi_total -= stake
    
    # ── Per-class calibration ──
    print("=" * 70)
    print("  📊 历史滚动回测报告")
    print(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  📅 数据范围: {matches[0]['date']} ~ {matches[-1]['date']} ({len(matches)}场)")
    print("=" * 70)
    print()
    print(f"  🎯 核心概率指标")
    print(f"  ─────────────────────────────────────")
    print(f"  Brier Score:        {brier:.4f}  (0=完美, 越小越好)")
    print(f"  RPS:                {rps:.4f}  (0=完美, 越小越好)")
    print(f"  HDA 准确率:         {accuracy*100:.1f}% ({sum(r['correct'] for r in results_detail)}/{n_run})")
    print()
    print(f"  🔬 Brier Score 分解")
    print(f"  ─────────────────────────────────────")
    print(f"  Uncertainty(基础难度):  {decomp['uncertainty']:.4f}")
    print(f"  Resolution(区分能力):   {decomp['resolution']:.4f}  ↑越高越好")
    print(f"  Reliability(校准度):    {decomp['reliability']:.4f}  ↓越低越好")
    print(f"  Brier = UNC - RES + REL: {decomp['brier_decomp_check']:.4f}")
    print(f"  (直接计算 Brier:       {decomp['brier']:.4f})")
    print()
    
    # Per-class stats
    print(f"  📈 各分类统计")
    print(f"  ─────────────────────────────────────")
    for label, name in [('H', '主胜'), ('D', '平局'), ('A', '客胜')]:
        li = label_idx[label]
        mask = y_true[:, li] == 1
        n_class = mask.sum()
        if n_class == 0:
            print(f"  {name}({label}): 无样本")
            continue
        avg_prob = y_proba[mask, li].mean()
        class_brier = np.mean(np.sum((y_true[mask] - y_proba[mask])**2, axis=1))
        print(f"  {name}({label}): {n_class}场 平均预测概率={avg_prob*100:.1f}% Brier={class_brier:.4f}")
    
    print()
    
    # ── Betting simulation ──
    print(f"  💰 模拟投注 (EV > 5%触发, 1单位/注)")
    print(f"  ─────────────────────────────────────")
    if bets_placed > 0:
        win_rate = bets_won / bets_placed * 100
        net_roi = roi_total
        roi_pct = roi_total / bets_placed * 100
        print(f"  投注场次:     {bets_placed}/{n_run}")
        print(f"  命中场次:     {bets_won}")
        print(f"  命中率:       {win_rate:.1f}%")
        print(f"  净收益:       {net_roi:+.2f} 单位")
        print(f"  ROI:          {roi_pct:+.1f}%")
    else:
        print(f"  无满足 EV>5% 的投注标的")
    print()
    
    # ── EV 标的分页 ──
    print(f"  🎲 EV 价值标的明细")
    print(f"  ─────────────────────────────────────")
    has_ev = [e for e in ev_detail if e['best_ev'] > 0.05]
    has_ev.sort(key=lambda x: -x['best_ev'])
    if has_ev:
        for e in has_ev:
            mark = '✅' if e['best_bet'] == e['actual'] else '❌'
            probs = e['probs']
            print(f"  {mark} [{e['date']}] {e['home']} vs {e['away']}")
            print(f"     预测: H={probs['H']*100:.0f}% D={probs['D']*100:.0f}% A={probs['A']*100:.0f}%")
            print(f"     实际: {e['score']} ({e['actual']}) | 推荐: {e['best_bet']} EV={e['best_ev']*100:.1f}%")
            print(f"     公平赔率: {e['fair_odds_H']:.2f}/{e['fair_odds_D']:.2f}/{e['fair_odds_A']:.2f}")
    else:
        print(f"  (无满足 EV>5% 的标的)")
    print()
    
    # ── Full match list ──
    print(f"  📋 全部预测明细")
    print(f"  ─────────────────────────────────────")
    for r in results_detail:
        mark = '✅' if r['correct'] else '❌'
        print(f"  {mark} [{r['date']}] {r['home']:6s} vs {r['away']:6s} | "
              f"预测={r['pred']} 实际={r['actual']} 比分={r['score']} | "
              f"H={r['probs']['H']*100:.0f}% D={r['probs']['D']*100:.0f}% A={r['probs']['A']*100:.0f}%")
    
    print()

if __name__ == '__main__':
    main()