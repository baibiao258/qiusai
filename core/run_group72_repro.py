#!/usr/bin/env python3
import json
import itertools
import math
import sys
from datetime import datetime

import joblib
import numpy as np

sys.path.insert(0, '/root')

DC_WEIGHT = 0.4
XGB_WEIGHT = 0.6

XGB_MODEL_PATH = '/root/data/xgb_model_29.pkl' if os.path.exists('/root/data/xgb_model_29.pkl') else '/root/data/xgb_model_20_3.pkl'
DC_MODEL_PATH = '/root/data/dc_model.pkl'
ELO_PATH = '/root/data/elo_ratings.pkl'
GROUPS_PATH = '/root/data/2026_groups.json'
OUT_JSON = '/root/data/group_stage_predictions.json'
OUT_TXT = '/root/data/group_stage_predictions.txt'


def make_odds_from_elo(eh: float, ea: float):
    e_h = 1.0 / (1 + 10 ** ((ea - eh) / 400))
    e_d = 0.26 * math.exp(-((eh - ea) / 200) ** 2)
    o = np.array([e_h * (1 - e_d), e_d, (1 - e_h) * (1 - e_d)], dtype=float)
    o /= o.sum()
    return float(o[0]), float(o[1]), float(o[2])  # H, D, A


def build_feature_23(dc, elo, home: str, away: str):
    eh = float(elo.get(home, 1500))
    ea = float(elo.get(away, 1500))

    lh, la = dc.predict_lambda(home, away, neutral=True)
    if lh is None or la is None:
        lh, la = 1.0, 1.0
        dp = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        dp = dc.predict_proba(home, away, neutral=True)  # H,D,A

    op_h, op_d, op_a = make_odds_from_elo(eh, ea)

    # 23维：15基线 + 5黄金 + 3赔率
    b15 = [
        (eh - ea) / 400,
        lh,
        la,
        lh - la,
        math.log(max(lh, 0.01) / max(la, 0.01)),
        float(dp[0]),
        float(dp[1]),
        float(dp[2]),
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        1,  # neutral
    ]

    gold5 = [
        0.0,  # h2h_gd placeholder
        1,    # tier_major
        0,    # tier_friendly
        0.0,  # f12_att_adv
        0.0,  # f12_win_a proxy
    ]

    odds3 = [op_h, op_d, op_a]
    # 6 form features (placeholder for tournament sim)
    form_feat = [0.0, 0.0, 0.0, 0.0, 1.5, 1.5]
    feat = np.array([b15 + gold5 + odds3 + form_feat], dtype=float)  # 29 dims
    return feat, lh, la


def normalize_groups(raw):
    if isinstance(raw, dict):
        return sorted(raw.items(), key=lambda x: x[0])

    items = []
    for g in raw:
        name = g.get('group') or g.get('name') or g.get('id')
        teams = g.get('teams') or g.get('members')
        items.append((name, teams))
    return sorted(items, key=lambda x: x[0])


def main():
    xgb = joblib.load(XGB_MODEL_PATH)
    dc = joblib.load(DC_MODEL_PATH)
    elo = joblib.load(ELO_PATH)

    with open(GROUPS_PATH, 'r', encoding='utf-8') as f:
        groups = normalize_groups(json.load(f))

    rows = []
    for gname, teams in groups:
        for home, away in itertools.combinations(teams, 2):
            feat, lh, la = build_feature_23(dc, elo, home, away)

            # xgb class order: [A, D, H]
            xgb_p = xgb.predict_proba(feat)[0]

            # dc to [A, D, H]
            dc_h, dc_d, dc_a = dc.predict_proba(home, away, neutral=True)
            dc_p = np.array([dc_a, dc_d, dc_h], dtype=float)

            hybrid = DC_WEIGHT * dc_p + XGB_WEIGHT * xgb_p
            p_away, p_draw, p_home = map(float, hybrid)

            s = p_home + p_draw + p_away
            p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s

            if p_home >= p_draw and p_home >= p_away:
                pick = home
            elif p_away >= p_home and p_away >= p_draw:
                pick = away
            else:
                pick = 'Draw'

            rows.append({
                'group': gname,
                'home': home,
                'away': away,
                'prob_home': round(p_home, 4),
                'prob_draw': round(p_draw, 4),
                'prob_away': round(p_away, 4),
                'lambda_home': round(float(lh), 3),
                'lambda_away': round(float(la), 3),
                'pick': pick,
            })

    rows.sort(key=lambda x: (x['group'], x['home'], x['away']))

    out = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'model': 'DC0.4+XGB0.6 (29 features, neutral venue)',
        'total_matches': len(rows),
        'predictions': rows,
    }

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = []
    cur = None
    for r in rows:
        if r['group'] != cur:
            cur = r['group']
            lines.append(f'\n=== Group {cur} ===')
        lines.append(
            f"{r['home']} vs {r['away']} | H {r['prob_home']*100:.1f}% D {r['prob_draw']*100:.1f}% A {r['prob_away']*100:.1f}% | "
            f"λ=({r['lambda_home']:.3f},{r['lambda_away']:.3f}) | Pick: {r['pick']}"
        )

    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).lstrip() + '\n')

    assert len(rows) == 72, f'Expected 72 matches, got {len(rows)}'
    print(json.dumps({'ok': True, 'total_matches': len(rows), 'json': OUT_JSON, 'txt': OUT_TXT}, ensure_ascii=False))


if __name__ == '__main__':
    main()
