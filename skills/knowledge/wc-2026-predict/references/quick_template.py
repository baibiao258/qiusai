#!/usr/bin/env python3
"""
快速执行模板：DC+XGB+MarketOdds 单场预测
复制这段代码，修改 MATCHES 列表即可用于新比赛
"""
import sys, os, json, math, random
sys.path.insert(0, '/root')
import numpy as np
import pandas as pd
from scipy.stats import poisson
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from wc_2026_phase1 import *

# ─── 配置 ───
MATCHES = [
    ('TeamA_DataName', 'TeamB_DataName'),  # 替换为实际队名
]

NAME_TO_DISPLAY = {
    'United States': 'USA',
    'Bosnia and Herzegovina': 'Bosnia',
}

# ─── 加载 ───
cache = os.path.join(DATA_DIR, 'international_results.json')
all_m = load_data(cache)
matches = filter_matches(all_m)
elo = compute_elo(all_m)
df = pd.DataFrame(matches)

# ─── DC ───
dc = DixonColes(time_decay_hl=540)
dc.fit(df)

# ─── 赔率 ───
match_odds = {}
odds_path = '/root/data/theodds_api_data.json'
if os.path.exists(odds_path):
    with open(odds_path) as f:
        raw = json.load(f)
    for m in raw['upcoming']['soccer_fifa_world_cup']:
        h, a = m['home'], m['away']
        ti = sum(1.0/v for v in m['odds'].values())
        probs = {t: (1.0/p)/ti for t,p in m['odds'].items()}
        match_odds[(h,a)] = probs
        match_odds[(a,h)] = probs

# ─── 特征 ───
def elo_odds(eh, ea):
    e_h = 1.0/(1+10**((ea-eh)/400))
    e_d = 0.26*math.exp(-((eh-ea)/200)**2)
    e_hm = e_h*(1-e_d); e_aw = (1-e_h)*(1-e_d)
    t = e_hm+e_d+e_aw; m=0.06
    return np.array([(1/((e_hm/t)*(1-m)))/sum(1/((x/t)*(1-m)) for x in [e_hm,e_d,e_aw]) for _ in [0]])

# Simplified - full feature build follows predict_matches_full.py pattern
print("See: /root/predict_matches_full.py for complete pipeline")
