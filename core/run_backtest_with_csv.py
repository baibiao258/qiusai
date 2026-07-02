import pandas as pd
import numpy as np

def rps_score(y_true, y_proba):
    cdf_true = np.cumsum(y_true, axis=1)
    cdf_pred = np.cumsum(y_proba, axis=1)
    n_cat = y_proba.shape[1] - 1
    return np.mean(np.sum((cdf_true - cdf_pred) ** 2, axis=1) / n_cat)

def eval_backtest(df):
    if 'actual_hda' not in df.columns or df['actual_hda'].isna().all() or (df['actual_hda'] == '').all():
        print("No actual results available for evaluation in the log.")
        return

    df = df.dropna(subset=['actual_hda'])
    df = df[df['actual_hda'].isin(['H', 'D', 'A'])].copy()
    
    if len(df) == 0:
        print("No valid matches to evaluate.")
        return

    y_true = np.zeros((len(df), 3))
    y_proba = np.zeros((len(df), 3))
    label_to_idx = {'A': 0, 'D': 1, 'H': 2}
    
    for i, (_, row) in enumerate(df.iterrows()):
        y_proba[i] = [row['pred_a']/100, row['pred_d']/100, row['pred_h']/100]
        y_true[i, label_to_idx[row['actual_hda']]] = 1
        
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
        reliability_mc += (nk / len(df)) * np.sum((fk - ok)**2)
        resolution_mc += (nk / len(df)) * np.sum((ok - base_rates)**2)

    print(f"\nEvaluating {len(df)} matches (from log):")
    print(f"Brier Score: {brier:.4f}")
    print(f"RPS:         {rps:.4f}")
    print(f"Decomposition:")
    print(f"  Uncertainty (Base Difficulty): {uncertainty:.4f}")
    print(f"  Reliability (Calibration Error, lower is better): {reliability_mc:.4f}")
    print(f"  Resolution  (Discrimination, higher is better): {resolution_mc:.4f}")

df = pd.read_csv('/root/data/predictions_log.csv')
print(df['actual_hda'])
