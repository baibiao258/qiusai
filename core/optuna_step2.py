#!/usr/bin/env python3
"""Step 2: Optuna only (load saved features)"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = '/root/data'
X33 = np.load(os.path.join(DATA_DIR, 'X33.npy'))
y33 = np.load(os.path.join(DATA_DIR, 'y33.npy'))
with open(os.path.join(DATA_DIR, 'dates.json')) as f:
    dates = np.array(json.load(f))

tm = dates < '2021-06-01'
vm = (dates >= '2021-06-01') & (dates < '2022-11-20')
Xt, Xv = X33[tm], X33[vm]
yt, yv = y33[tm], y33[vm]
print(f'Train: {len(Xt)} Val: {len(Xv)}', flush=True)

cw = compute_class_weight('balanced', classes=np.unique(yt), y=yt)
sw = np.array([cw[list(np.unique(yt)).index(c)] for c in yt])
yv_oh = np.zeros((len(yv), 3))
yv_oh[np.arange(len(yv)), yv] = 1

def objective(trial):
    p = {
        'max_depth': trial.suggest_int('max_depth', 2, 4),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 150, 450),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 12.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 25.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.35, 0.60),
        'subsample': trial.suggest_float('subsample', 0.60, 0.85),
        'min_child_weight': trial.suggest_float('min_child_weight', 3.0, 15.0),
        'random_state': 42, 'eval_metric': 'mlogloss', 'verbosity': 0
    }
    m = XGBClassifier(**p, early_stopping_rounds=25)
    m.fit(Xt, yt, eval_set=[(Xv, yv)], sample_weight=sw, verbose=False)
    yp = m.predict_proba(Xv)
    return float(np.mean(np.sum((yp - yv_oh)**2, axis=1)))

print('Optuna 60 trials...', flush=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=60, show_progress_bar=True)

print(f'\nBest Brier: {study.best_value:.4f}')
for k, v in study.best_params.items():
    print(f"  '{k}': {v},")

with open(os.path.join(DATA_DIR, 'optuna_best.json'), 'w') as f:
    json.dump({'best_params': study.best_params, 'best_brier': study.best_value}, f, indent=2)
print(f'Saved: optuna_best.json')
print('DONE', flush=True)
