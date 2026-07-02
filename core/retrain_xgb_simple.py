#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrain_xgb_simple.py
从 training_data.csv 训练 XGBoost，特征:
- market_odds (市场赔率)
- form_home_win/gf/ga
- form_away_win/gf/ga
共7维
"""
import json, pickle, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

DATA_DIR = Path("/root/data")
TRAIN_CSV = DATA_DIR / "training_data.csv"
OUT_MODEL = DATA_DIR / "xgb_model_simple.pkl"
OUT_CAL = DATA_DIR / "calibrators_simple.pkl"

FEATURE_COLS = [
    "market_odds",
    "form_home_win", "form_home_gf", "form_home_ga",
    "form_away_win", "form_away_gf", "form_away_ga",
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def main():
    df = pd.read_csv(TRAIN_CSV)
    log(f"训练集: {len(df)} 行")

    # 过滤 market_odds > 1
    df = df[df["market_odds"] > 1].copy()
    log(f"有效样本 (market_odds>1): {len(df)}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values

    # label 映射: 3→0(H), 1→1(D), 0→2(A)
    label_map = {3: 0, 1: 1, 0: 2}
    y_mapped = np.array([label_map.get(v, v) for v in y])

    log(f"特征维度: {X.shape[1]}")
    log(f"标签分布: H={sum(y_mapped==0)} D={sum(y_mapped==1)} A={sum(y_mapped==2)}")

    # 5-fold CV
    params = dict(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
        use_label_encoder=False,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_losses, cv_accs = [], []
    for fold, (tr, va) in enumerate(skf.split(X, y_mapped)):
        m = XGBClassifier(**params)
        m.fit(X[tr], y_mapped[tr], eval_set=[(X[va], y_mapped[va])], verbose=False)
        p = m.predict_proba(X[va])
        cv_losses.append(log_loss(y_mapped[va], p))
        cv_accs.append(accuracy_score(y_mapped[va], p.argmax(axis=1)))
        log(f"  Fold {fold+1}: log_loss={cv_losses[-1]:.4f} acc={cv_accs[-1]:.4f}")

    log(f"CV: log_loss={np.mean(cv_losses):.4f}±{np.std(cv_losses):.4f}  acc={np.mean(cv_accs):.4f}")

    # 全量训练
    xgb = XGBClassifier(**params)
    xgb.fit(X, y_mapped, verbose=False)
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(xgb, f)
    log(f"模型: {OUT_MODEL}  ({xgb.n_features_in_}维, {xgb.n_estimators}棵)")

    # 校准器
    proba = xgb.predict_proba(X)
    calibrators = {}
    for i, name in enumerate(["home", "draw", "away"]):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(proba[:, i], (y_mapped == i).astype(int))
        calibrators[name] = iso
    with open(OUT_CAL, "wb") as f:
        pickle.dump(calibrators, f)
    log(f"校准器: {OUT_CAL}")

    # 校准后准确率
    proba_cal = np.column_stack([calibrators[n].predict(proba[:, i])
                                 for i, n in enumerate(["home", "draw", "away"])])
    proba_cal /= proba_cal.sum(axis=1, keepdims=True)
    acc_cal = (proba_cal.argmax(axis=1) == y_mapped).mean()
    log(f"校准后 acc: {acc_cal:.4f}")

    # 特征重要性
    for col, imp in sorted(zip(FEATURE_COLS, xgb.feature_importances_), key=lambda x: -x[1]):
        log(f"  {col}: {imp:.4f}")

if __name__ == "__main__":
    main()
