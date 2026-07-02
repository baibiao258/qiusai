#!/usr/bin/env python3
"""Predict all 72 group stage matches of WC 2026."""
import sys, os, json, math
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')
import numpy as np
import joblib
from team_name_normalizer import normalize_match_pair

DATA_DIR = '/root/data'

dc = joblib.load(os.path.join(DATA_DIR, 'dc_model.pkl'))
xgb = (
    joblib.load(os.path.join(DATA_DIR, 'xgb_model_29.pkl'))
    if os.path.exists(os.path.join(DATA_DIR, 'xgb_model_29.pkl'))
    else joblib.load(os.path.join(DATA_DIR, 'xgb_model_20_3.pkl'))
)
elo = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))

HOST_TEAMS = {'United States', 'Mexico', 'Canada'}
HOST_BONUS_BY_TEAM = {'United States': 0.1445, 'Mexico': 0.10, 'Canada': 0.07}
DC_W = 0.4; XGB_W = 0.6; MAX_GOALS = 6


def predict(h, a):
    h, a = normalize_match_pair(h, a)
    is_host = h in HOST_TEAMS
    hb = HOST_BONUS_BY_TEAM.get(h, 0.0) if is_host else 0.0
    neutral = not is_host

    dc_p = dc.predict_proba(h, a, neutral, host_bonus=hb if is_host else 0.0)
    lh, la = dc.predict_lambda(h, a, neutral, host_bonus=hb if is_host else 0.0)
    if lh is None:
        return None

    eh_e = elo.get(h, 1500); ea_e = elo.get(a, 1500)
    b15 = [(eh_e - ea_e) / 400, lh, la, lh - la,
           math.log(max(lh, 0.01) / max(la, 0.01)),
           dc_p[0], dc_p[1], dc_p[2],
           0.5, 0.5, 0, 0, 0, 0,
           0 if is_host else 1]
    gold = [0, 1, 0, 0, 0]
    odds_f = [1 / (1 + 10 ** ((ea_e - eh_e) / 400)),
              1 / (1 + 10 ** ((eh_e - ea_e) / 400)),
              0.0]
    # 6 form features (placeholder for tournament sim)
    form_feat = [0.0, 0.0, 0.0, 0.0, 1.5, 1.5]
    feat = np.array([b15 + gold + odds_f + form_feat])  # 29 dims
    xgb_p = xgb.predict_proba(feat)[0]

    dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])
    hybrid = DC_W * dc_ado + XGB_W * xgb_p

    return {
        'home': h, 'away': a,
        'host_bonus': round(hb, 4),
        'lam_h': round(lh, 2), 'lam_a': round(la, 2),
        'home_win': round(float(hybrid[2] * 100), 1),
        'draw': round(float(hybrid[1] * 100), 1),
        'away_win': round(float(hybrid[0] * 100), 1),
    }


# Load schedule
with open(os.path.join(DATA_DIR, 'wc2026_official_schedule.json')) as f:
    sched = json.load(f)

results = []
for m in sched['matches']:
    r = predict(m['home'], m['away'])
    if r:
        r['date'] = m['date']
        r['time_utc'] = m['time_utc']
        r['venue'] = m['venue']
        results.append(r)

print(f'TOTAL:{len(results)}')
print(json.dumps(results, ensure_ascii=False))

# Also save
with open(os.path.join(DATA_DIR, 'group_stage_predictions.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
