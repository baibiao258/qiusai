#!/usr/bin/env python3
"""6/7 综合预测 — 复用 6/6 流程 + 500.com 5玩法全量赔率旁路输入
1) fetch_500_market.py 抓 269/270/271/272 真赔率
2) predict_match.py 拿 DC+XGB+Hybrid λ 和 cal SPF
3) 365scores 投票作为公众预期融合
4) 输出 5 维: 胜平负/让球/半全场/比分/总进球
"""
import sys
import os
import json
import math
import subprocess
sys.path.insert(0, '/root')
from scipy.stats import poisson
from predict_match import predict_match

DATE = '2026-06-07'
FETCHER = '/root/wc_2026_upgrade/fetch_500_market.py'

TARGET_MATCHES = {
    '周日201': {'mid': '周日201', 'league': '友谊赛', 'time': '06-08 02:45', 'home': 'Croatia', 'away': 'Slovenia', 'match_type': 'friendly', 'vote': {'H': 79.0, 'D': 13.0, 'A': 8.0, 'n': 3656}},
    '周日202': {'mid': '周日202', 'league': '友谊赛', 'time': '06-08 03:00', 'home': 'Morocco', 'away': 'Norway', 'match_type': 'friendly', 'vote': {'H': 63.2, 'D': 10.0, 'A': 26.8, 'n': 31191}},
    '周日203': {'mid': '周日203', 'league': '友谊赛', 'time': '06-08 03:00', 'home': 'Greece', 'away': 'Italy', 'match_type': 'friendly', 'vote': {'H': 18.5, 'D': 12.2, 'A': 69.4, 'n': 5688}},
    '周日204': {'mid': '周日204', 'league': '友谊赛', 'time': '06-08 07:00', 'home': 'Colombia', 'away': 'Jordan', 'match_type': 'friendly', 'vote': {'H': 43.4, 'D': 7.3, 'A': 49.3, 'n': 25996}},
}

HTFT_LABELS = ['胜胜','胜平','胜负','平胜','平平','平负','负胜','负平','负负']
JQS_LABELS = ['0球','1球','2球','3球','4球','5球','6球','7+球']
BF_VALUE_MAP = {
    '1:0': (1,0), '2:0': (2,0), '2:1': (2,1), '3:0': (3,0), '3:1': (3,1), '3:2': (3,2),
    '4:0': (4,0), '4:1': (4,1), '4:2': (4,2), '5:0': (5,0), '5:1': (5,1), '5:2': (5,2),
    '0:0': (0,0), '1:1': (1,1), '2:2': (2,2), '3:3': (3,3),
    '0:1': (0,1), '0:2': (0,2), '1:2': (1,2), '0:3': (0,3), '1:3': (1,3), '2:3': (2,3),
    '0:4': (0,4), '1:4': (1,4), '2:4': (2,4), '0:5': (0,5), '1:5': (1,5), '2:5': (2,5),
}


def fetch_market(playid, g='2'):
    cmd = ['python3', FETCHER, DATE, str(playid), str(g)]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def sp_to_prob_map(odds_map):
    clean = {}
    for k, v in odds_map.items():
        try:
            fv = float(v)
            if fv > 0:
                clean[k] = fv
        except Exception:
            pass
    inv = {k: 1.0/v for k, v in clean.items()}
    s = sum(inv.values())
    return {k: v/s for k, v in inv.items()} if s else {}


def vote_to_prob(vote):
    total = vote['H'] + vote['D'] + vote['A']
    return {'胜': vote['H']/total, '平': vote['D']/total, '负': vote['A']/total}


def score_dist_from_lambda(lam_h, lam_a, top_n=8):
    scores = []
    for h in range(6):
        for a in range(6):
            p = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
            scores.append((f'{h}:{a}', p))
    scores.sort(key=lambda x: -x[1])
    total = sum(p for _, p in scores)
    return [(s, p/total) for s, p in scores[:top_n]]


def total_goals_dist(lam_h, lam_a):
    dist = {}
    for total in range(8):
        p = 0
        for h in range(total + 1):
            a = total - h
            p += poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
        dist[total] = p
    dist['7+'] = 1 - sum(v for k, v in dist.items() if isinstance(k, int))
    return dist


def market_score_top(score_probs, top_n=8):
    items = sorted(score_probs.items(), key=lambda x: -x[1])
    return items[:top_n]


def gate_check(p_spf, threshold=10):
    vals = sorted(p_spf.values(), reverse=True)
    margin = (vals[0] - vals[1]) * 100
    if margin >= threshold:
        return '✅ 推荐', margin
    if margin >= 5:
        return '⚠️ 弱信号', margin
    return '⏭️ 跳过', margin


main = {x['no']: x for x in fetch_market('269', '2')['result']}
jqs = {x['no']: x for x in fetch_market('270', '2')['result']}
score = {x['no']: x for x in fetch_market('271', '2')['result']}
htft = {x['no']: x for x in fetch_market('272', '2')['result']}

predictions = {}
print('=' * 100)
print('6/7~6/8 竞彩综合预测 — 500.com 5玩法真赔率 + DC模型 + 365scores投票')
print('=' * 100)

for no, meta in TARGET_MATCHES.items():
    if no not in main:
        continue
    m = main[no]
    rq = int(m.get('rangqiu') or '0')
    spf_market = sp_to_prob_map({'胜': m['odds']['spf'].get('3'), '平': m['odds']['spf'].get('1'), '负': m['odds']['spf'].get('0')})
    rq_market = sp_to_prob_map({'让胜': m['odds']['nspf'].get('3'), '让平': m['odds']['nspf'].get('1'), '让负': m['odds']['nspf'].get('0')})
    jqs_market = sp_to_prob_map(jqs.get(no, {}).get('odds', {}))
    htft_market = sp_to_prob_map(htft.get(no, {}).get('odds', {}))
    score_market = sp_to_prob_map(score.get(no, {}).get('odds', {}))

    model = predict_match(meta['home'], meta['away'], match_type=meta['match_type'])
    if not model:
        print(f'❌ {no} {meta["home"]} vs {meta["away"]} 模型未收敛')
        continue

    p_model = {'胜': model['fin_h']/100, '平': model['fin_d']/100, '负': model['fin_a']/100}
    p_vote = vote_to_prob(meta['vote'])
    p_spf = {k: 0.5*p_model[k] + 0.3*p_vote[k] + 0.2*spf_market[k] for k in ['胜','平','负']}
    gate, margin = gate_check(p_spf)

    lam_h = model['lam_h']
    lam_a = model['lam_a']
    sd_model = dict(score_dist_from_lambda(lam_h, lam_a, top_n=20))
    top_scores = []
    merged_scores = {}
    for key in set(sd_model) | set(score_market):
        merged_scores[key] = 0.65*sd_model.get(key, 0.0) + 0.35*score_market.get(key, 0.0)
    for key, val in sorted(merged_scores.items(), key=lambda x: -x[1])[:8]:
        top_scores.append((key, val))

    # 竞彩总进球/半全场策略:
    # - 若无可靠模型(如日职等DC不收敛), 总进球以500市场为主, 摇摆区(47%~53%)不强行翻方向
    # - 半全场优先使用500市场9宫格, 仅在有可靠模型时再做轻度融合
    tg_model = total_goals_dist(lam_h, lam_a)
    tg_market_norm = {int(k.replace('球','').replace('+','')) if k != '7+球' else 7: v for k, v in jqs_market.items()}
    total_dist = {}
    if m.get('model_reliable', True):
        for k in range(8):
            mk = tg_market_norm.get(k, 0.0)
            modelk = tg_model['7+'] if k == 7 else tg_model.get(k, 0.0)
            total_dist[k] = 0.65*modelk + 0.35*mk
        p_over = total_dist[3] + total_dist[4] + total_dist[5] + total_dist[6] + total_dist[7]
    else:
        for k in range(8):
            total_dist[k] = tg_market_norm.get(k, 0.0)
        p_over = total_dist[3] + total_dist[4] + total_dist[5] + total_dist[6] + total_dist[7]
        if 0.47 <= p_over <= 0.53:
            p_over = 0.5001

    htft_model = {}
    base_ht = {'胜': 0.33, '平': 0.34, '负': 0.33}
    for h in ['胜','平','负']:
        for f in ['胜','平','负']:
            htft_model[f'{h}{f}'] = base_ht[h] * p_spf[f]
    s = sum(htft_model.values())
    htft_model = {k: v/s for k, v in htft_model.items()}
    if m.get('model_reliable', True):
        htft_merged = {k: 0.80*htft_model.get(k, 0.0) + 0.20*htft_market.get(k, 0.0) for k in HTFT_LABELS}
    else:
        htft_merged = {k: htft_market.get(k, 0.0) for k in HTFT_LABELS}
    htft_top = sorted(htft_merged.items(), key=lambda x: -x[1])[:6]

    print(f"\n{'='*80}")
    print(f"📋 {no} | {meta['league']} | {meta['time']}")
    print(f"   {meta['home']} (让{rq}) vs {meta['away']}")
    print(f"{'='*80}")
    print(f"\n  🔢 SPF融合 (模型0.5 + 投票0.3 + 500赔率0.2):")
    print(f"     模型      : H {p_model['胜']*100:.1f}% / D {p_model['平']*100:.1f}% / A {p_model['负']*100:.1f}% (λ_h={lam_h:.2f} λ_a={lam_a:.2f})")
    print(f"     365投票   : H {p_vote['胜']*100:.1f}% / D {p_vote['平']*100:.1f}% / A {p_vote['负']*100:.1f}% (n={meta['vote']['n']})")
    print(f"     500赔率   : H {spf_market['胜']*100:.1f}% / D {spf_market['平']*100:.1f}% / A {spf_market['负']*100:.1f}%")
    print(f"     ✅ 综合SPF : H {p_spf['胜']*100:.1f}% / D {p_spf['平']*100:.1f}% / A {p_spf['负']*100:.1f}%")
    print(f"     🚦 门控    : {gate} (边际 {margin:.1f}pp)")

    print(f"\n  ⚽ 让球 ({rq:+d}):")
    print(f"     500赔率: 让胜 {rq_market.get('让胜',0)*100:.1f}% / 让平 {rq_market.get('让平',0)*100:.1f}% / 让负 {rq_market.get('让负',0)*100:.1f}%")
    print(f"     首选: {max(rq_market, key=rq_market.get)}")

    print(f"\n  📊 比分 Top8 (模型65% + 500比分35%):")
    for i, (skey, p) in enumerate(top_scores, 1):
        print(f"     {i}. {skey}: {p*100:.1f}%")

    print(f"\n  ⚽ 总进球分布 (模型65% + 500进球数35%):")
    for k in range(8):
        label = '7+球' if k == 7 else f'{k}球'
        print(f"     {label}: {total_dist[k]*100:.1f}%")
    print(f"     大2.5: {p_over*100:.1f}% / 小2.5: {(1-p_over)*100:.1f}%")

    print(f"\n  🏁 半全场 Top6 (模型65% + 500半全场35%):")
    for i, (k, v) in enumerate(htft_top, 1):
        print(f"     {i}. {k}: {v*100:.1f}%")

    predictions[no] = {
        'match': f"{meta['home']} vs {meta['away']}",
        'rangqiu': rq,
        'spf_market': spf_market,
        'rq_market': rq_market,
        'jqs_market': jqs_market,
        'score_market_top8': top_scores,
        'htft_market_top6': htft_top,
        'spf_final': p_spf,
        'gate': gate,
        'margin': margin,
        'lambda': {'home': lam_h, 'away': lam_a},
        'total_goals_final': total_dist,
    }

out = '/root/scripts/preds_2026-06-07.full500.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
print(f"\n✅ 预测保存到: {out}")
