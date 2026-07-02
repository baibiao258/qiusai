#!/usr/bin/env python3
import pandas as pd
import numpy as np

def rps_score(y_true, y_proba):
    cdf_true = np.cumsum(y_true, axis=1)
    cdf_pred = np.cumsum(y_proba, axis=1)
    n_cat = y_proba.shape[1] - 1
    return np.mean(np.sum((cdf_true - cdf_pred) ** 2, axis=1) / n_cat)

def quick_validate_decomposition(df):
    """
    Brier Score Decomposition:
    Brier = Uncertainty - Resolution + Reliability
    Uncertainty = \sum p_k (1 - p_k) where p_k is the base rate of outcome k
    Reliability (Calibration) = \sum_{bin} \frac{n_{bin}}{N} \sum_{k} (pred_{bin,k} - actual_{bin,k})^2
    Resolution = \sum_{bin} \frac{n_{bin}}{N} \sum_{k} (actual_{bin,k} - p_k)^2
    """
    if 'actual_hda' not in df.columns or df['actual_hda'].isna().all() or (df['actual_hda'] == '').all():
        print("No actual results available for evaluation in the log.")
        return

    # Filter rows with actual results
    df = df[df['actual_hda'].isin(['H', 'D', 'A'])].copy()
    
    if len(df) == 0:
        print("No matches with actual_hda found.")
        return
        
    y_true = np.zeros((len(df), 3))
    y_proba = np.zeros((len(df), 3))
    label_to_idx = {'A': 0, 'D': 1, 'H': 2}
    
    for i, (_, row) in enumerate(df.iterrows()):
        y_proba[i] = [row['pred_a']/100, row['pred_d']/100, row['pred_h']/100]
        y_true[i, label_to_idx[row['actual_hda']]] = 1
        
    brier = np.mean(np.sum((y_true - y_proba) ** 2, axis=1))
    rps = rps_score(y_true, y_proba)
    
    # 1. Uncertainty: Based solely on the prior (base rate) of outcomes
    base_rates = np.mean(y_true, axis=0) # [P(A), P(D), P(H)]
    uncertainty = np.sum(base_rates * (1 - base_rates))
    
    # Simple binning for Reliability and Resolution
    # We bin based on the predicted probability of the *true* class or just bin across all predicted probabilities.
    # Standard multi-class Brier decomposition uses bins on the simplex, but we can do a marginal approach or bin by predicted max prob.
    # To keep it rigorous, we bin probabilities into deciles (0-1) across all N*3 scalar predictions.
    
    y_proba_flat = y_proba.flatten()
    y_true_flat = y_true.flatten()
    
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_proba_flat, bins) - 1
    # clamp to valid bin indices
    bin_indices = np.clip(bin_indices, 0, 9)
    
    reliability = 0
    resolution = 0
    N = len(y_proba_flat)
    
    # Base rate for each class flatten? Actually, overall marginal base rate is not 1/3 but exact mean.
    # Let's compute proper multi-class.
    
    reliability_mc = 0.0
    resolution_mc = 0.0
    
    # Let's partition the sample based on the predicted outcome (H, D, A)
    # This is a simplified partition for multi-class.
    pred_class = np.argmax(y_proba, axis=1)
    for c in range(3):
        idx = (pred_class == c)
        nk = np.sum(idx)
        if nk == 0: continue
        
        # average forecast in this bin
        fk = np.mean(y_proba[idx], axis=0)
        # average observation in this bin
        ok = np.mean(y_true[idx], axis=0)
        
        # Reliability: distance between average forecast and average observation
        reliability_mc += (nk / len(df)) * np.sum((fk - ok)**2)
        
        # Resolution: distance between average observation and global base rate
        resolution_mc += (nk / len(df)) * np.sum((ok - base_rates)**2)
        
    # Notice: Brier_mc = Uncertainty - Resolution_mc + Reliability_mc
    
    print(f"Evaluated {len(df)} matches.")
    print(f"Brier Score: {brier:.4f}")
    print(f"RPS: {rps:.4f}")
    print(f"Decomposition:")
    print(f"  Uncertainty (Base Difficulty): {uncertainty:.4f}")
    print(f"  Reliability (Calibration Error, lower is better): {reliability_mc:.4f}")
    print(f"  Resolution  (Discrimination, higher is better): {resolution_mc:.4f}")
    print(f"Check sum: U - Res + Rel = {uncertainty - resolution_mc + reliability_mc:.4f} (should match Brier {brier:.4f})")

try:
    df = pd.read_csv('/root/data/predictions_log.csv')
    quick_validate_decomposition(df)
except Exception as e:
    print(f"Error: {e}")
