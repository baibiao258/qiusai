#!/usr/bin/env python3
import sys, os, json, math
from datetime import date, timedelta

import numpy as np

def rps_score(y_true, y_proba):
    cdf_true = np.cumsum(y_true, axis=1)
    cdf_pred = np.cumsum(y_proba, axis=1)
    n_cat = y_proba.shape[1] - 1
    return np.mean(np.sum((cdf_true - cdf_pred) ** 2, axis=1) / n_cat)

def quick_validate_decomposition(model_probs, actual_results):
    y_true = np.zeros((len(actual_results), 3))
    y_proba = np.zeros((len(actual_results), 3))
    label_to_idx = {'A': 0, 'D': 1, 'H': 2}
    
    for i, (mp, act) in enumerate(zip(model_probs, actual_results)):
        y_proba[i] = [mp['A'], mp['D'], mp['H']]
        y_true[i, label_to_idx[act]] = 1
        
    brier = np.mean(np.sum((y_true - y_proba) ** 2, axis=1))
    rps = rps_score(y_true, y_proba)
    
    base_rates = np.mean(y_true, axis=0) 
    uncertainty = np.sum(base_rates * (1 - base_rates))
    
    reliability_mc = 0.0
    resolution_mc = 0.0
    
    pred_class = np.argmax(y_proba, axis=1)
    for c in range(3):
        idx = (pred_class == c)
        nk = np.sum(idx)
        if nk == 0: continue
        fk = np.mean(y_proba[idx], axis=0)
        ok = np.mean(y_true[idx], axis=0)
        reliability_mc += (nk / len(actual_results)) * np.sum((fk - ok)**2)
        resolution_mc += (nk / len(actual_results)) * np.sum((ok - base_rates)**2)
        
    print(f"Evaluated {len(actual_results)} matches.")
    print(f"Brier Score: {brier:.4f}")
    print(f"RPS: {rps:.4f}")
    print(f"Decomposition:")
    print(f"  Uncertainty (Base Difficulty): {uncertainty:.4f}")
    print(f"  Reliability (Calibration Error, lower is better): {reliability_mc:.4f}")
    print(f"  Resolution  (Discrimination, higher is better): {resolution_mc:.4f}")
    print(f"Check sum: U - Res + Rel = {uncertainty - resolution_mc + reliability_mc:.4f} (should match Brier {brier:.4f})")

try:
    with open('/root/data/backtest_results.json', 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} results from backtest_results.json")
    if len(data) > 0:
        probs = [{'H': d['prob_h'], 'D': d['prob_d'], 'A': d['prob_a']} for d in data]
        actual = [d['actual'] for d in data]
        quick_validate_decomposition(probs, actual)
except Exception as e:
    print(f"Error: {e}")
