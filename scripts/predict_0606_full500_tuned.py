#!/usr/bin/env python3
import json
import subprocess
import sys
sys.path.insert(0, '/root')
from scipy.stats import poisson
from predict_match import predict_match

DATE = '2026-06-06'
FETCHER = '/root/wc_2026_upgrade/fetch_500_market.py'
TARGET_MATCHES = {
    '周六201': {'home': 'Kashima Antlers', 'away': 'Vissel Kobe', 'match_type': 'competitive'},
    '周六202': {'home': 'Machida Zelvia', 'away': 'Nagoya Grampus', 'match_type': 'competitive'},
    '周六203': {'home': 'Urawa Red Diamonds', 'away': 'Fagiano Okayama', 'match_type': 'competitive'},
    '周六204': {'home': 'Yokohama F. Marinos', 'away': 'Shimizu S-Pulse', 'match_type': 'competitive'},
    '周六205': {'home': 'Kashiwa Reysol', 'away': 'Kyoto Sanga', 'match_type': 'competitive'},
    '周六206': {'home': 'Kawasaki Frontale', 'away': 'Sanfrecce Hiroshima', 'match_type': 'competitive'},
}
HTFT_LABELS = ['胜胜','胜平','胜负','平胜','平平','平负','负胜','负平','负负']

def fetch_market(playid, g='2'):
    out = subprocess.check_output(['python3', FETCHER, DATE, str(playid), str(g)], text=True)
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

def score_dist_from_lambda(lam_h, lam_a, top_n=20):
    scores = []
    for h in range(6):
        for a in range(6):
            p = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
            scores.append((f'{h}:{a}', p))
    scores.sort(key=lambda x: -x[1])
    total = sum(p for _, p in scores)
    return {s: p/total for s, p in scores[:top_n]}

def total_goals_dist(lam_h, lam_a):
    dist = {}
    for total in range(8):
        p = 0
        for h in range(total + 1):
            a = total - h
            p += poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
        dist[total] = p
    return dist

main = {x['no']: x for x in fetch_market('269', '2')['result']}
jqs = {x['no']: x for x in fetch_market('270', '2')['result']}
score = {x['no']: x for x in fetch_market('271', '2')['result']}
htft = {x['no']: x for x in fetch_market('272', '2')['result']}

predictions = {}
for no, meta in TARGET_MATCHES.items():
    if no not in main:
        continue
    row = main[no]
    spf_market = sp_to_prob_map({'胜': row['odds']['spf'].get('3'), '平': row['odds']['spf'].get('1'), '负': row['odds']['spf'].get('0')})
    rq_market = sp_to_prob_map({'让胜': row['odds']['nspf'].get('3'), '让平': row['odds']['nspf'].get('1'), '让负': row['odds']['nspf'].get('0')})
    jqs_market_raw = sp_to_prob_map(jqs.get(no, {}).get('odds', {}))
    score_market = sp_to_prob_map(score.get(no, {}).get('odds', {}))
    htft_market = sp_to_prob_map(htft.get(no, {}).get('odds', {}))

    model = predict_match(meta['home'], meta['away'], match_type=meta['match_type'])
    model_ok = isinstance(model, dict)
    if model_ok:
        p_model = {'胜': model['fin_h']/100, '平': model['fin_d']/100, '负': model['fin_a']/100}
        lam_h = model['lam_h']
        lam_a = model['lam_a']
    else:
        p_model = spf_market
        jqs_mean = 0.0
        for label, prob in jqs_market_raw.items():
            goals = 7 if label == '7+球' else int(label.replace('球', ''))
            jqs_mean += goals * prob
        if jqs_mean <= 0:
            jqs_mean = 2.6
        home_share = p_model['胜'] + 0.5 * p_model['平']
        away_share = p_model['负'] + 0.5 * p_model['平']
        total_share = max(home_share + away_share, 1e-6)
        lam_h = max(0.2, jqs_mean * (home_share / total_share))
        lam_a = max(0.2, jqs_mean * (away_share / total_share))

    p_spf = {k: 0.8*p_model[k] + 0.2*spf_market.get(k, p_model[k]) for k in ['胜','平','负']}
    rq_pick = max(rq_market, key=rq_market.get) if rq_market else None

    sd_model = score_dist_from_lambda(lam_h, lam_a)
    merged_scores = {k: 0.65*sd_model.get(k, 0.0) + 0.35*score_market.get(k, 0.0) for k in set(sd_model) | set(score_market)}
    top_scores = sorted(merged_scores.items(), key=lambda x: -x[1])[:8]

    tg_market = {7 if k == '7+球' else int(k.replace('球','')): v for k, v in jqs_market_raw.items()}
    if model_ok:
        tg_model = total_goals_dist(lam_h, lam_a)
        total_dist = {k: 0.65*tg_model.get(k, 0.0) + 0.35*tg_market.get(k, 0.0) for k in range(8)}
        p_over = sum(total_dist[k] for k in [3,4,5,6,7])
    else:
        total_dist = {k: tg_market.get(k, 0.0) for k in range(8)}
        p_over = sum(total_dist[k] for k in [3,4,5,6,7])
        if 0.47 <= p_over <= 0.53:
            p_over = 0.5001

    base_ht = {'胜': 0.33, '平': 0.34, '负': 0.33}
    htft_model = {}
    for h in ['胜','平','负']:
        for f in ['胜','平','负']:
            htft_model[f'{h}{f}'] = base_ht[h] * p_spf[f]
    s = sum(htft_model.values())
    htft_model = {k: v/s for k, v in htft_model.items()}
    if model_ok:
        htft_final = {k: 0.80*htft_model.get(k, 0.0) + 0.20*htft_market.get(k, 0.0) for k in HTFT_LABELS}
    else:
        htft_final = {k: htft_market.get(k, 0.0) for k in HTFT_LABELS}
    htft_top = sorted(htft_final.items(), key=lambda x: -x[1])

    predictions[no] = {
        'spf': max(p_spf, key=p_spf.get),
        'spf_probs': p_spf,
        'rq': rq_pick,
        'rq_probs': rq_market,
        'scores': [k for k, _ in top_scores],
        'score_probs': {k: v for k, v in top_scores},
        'size': '大2.5' if p_over > 0.5 else '小2.5',
        'size_probs': {'大2.5': p_over, '小2.5': 1-p_over},
        'htft': htft_top[0][0] if htft_top else None,
        'htft_probs': {k: v for k, v in htft_top[:6]},
    }

with open('/root/scripts/preds_0606_full500_tuned.json', 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
print('/root/scripts/preds_0606_full500_tuned.json')
