import json
import numpy as np

def rps_score(y_true, y_proba):
    cdf_true = np.cumsum(y_true, axis=1)
    cdf_pred = np.cumsum(y_proba, axis=1)
    n_cat = y_proba.shape[1] - 1
    return np.mean(np.sum((cdf_true - cdf_pred) ** 2, axis=1) / n_cat)

def eval_json():
    try:
        with open('/root/data/optuna_backtest.json', 'r') as f:
            data = json.load(f)
        
        # Let's inspect
        print("Data loaded:", type(data), len(data))
    except Exception as e:
        print("Error:", e)
eval_json()
