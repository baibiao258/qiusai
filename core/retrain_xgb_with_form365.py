#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrain_xgb_with_form365.py
===========================
复用 train_xgb_club.py 的 29 维特征结构，
form 数据源切换到 365scores form_state.json
输出: /root/data/xgb_model_365.pkl + calibrators_365.pkl
"""
from __future__ import annotations
import json, os, sys, pickle, argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

DATA_DIR        = Path("/root/data")
FORM_STATE_PATH = DATA_DIR / "form_state.json"
FORM_CLUB_PATH  = DATA_DIR / "form_club.json"
TRAIN_CSV       = DATA_DIR / "training_data.csv"
OUT_MODEL       = DATA_DIR / "xgb_model_365.pkl"
OUT_CALIBRATORS = DATA_DIR / "calibrators_365.pkl"
LOG_PATH        = DATA_DIR / "retrain_365.log"


class FormBook:
    def __init__(self):
        self._data: dict = {}
        if FORM_STATE_PATH.exists():
            with open(FORM_STATE_PATH, encoding="utf-8") as f:
                self._data = json.load(f)
            print(f"[FormBook] form_state.json: {len(self._data)} 支球队")
        if FORM_CLUB_PATH.exists():
            with open(FORM_CLUB_PATH, encoding="utf-8") as f:
                club = json.load(f)
            added = 0
            for t, v in club.items():
                if t not in self._data:
                    self._data[t] = v
                    added += 1
            print(f"[FormBook] fallback form_club.json 补充: {added} 支")

    def recent_form(self, team: str, n: int = 5,
                    before_date: Optional[str] = None) -> List[float]:
        entries = self._data.get(team, [])
        if before_date:
            entries = [e for e in entries if len(e) < 3 or str(e[2]) < before_date]
        if not entries:
            return [0.0, 0.0, 0.0, 0.0]
        recent = entries[-n:]
        wins, gf_list, ga_list = 0, [], []
        for e in recent:
            h, a = int(e[0]), int(e[1])
            gf_list.append(h)
            ga_list.append(a)
            if h > a:
                wins += 1
        m = len(recent)
        avg_gf = sum(gf_list) / m
        avg_ga = sum(ga_list) / m
        return [wins / m, avg_gf, avg_ga, avg_gf - avg_ga]


def build_features(row: pd.Series, fb: FormBook) -> Optional[np.ndarray]:
    try:
        home = str(row["home_team"])
        away = str(row["away_team"])
        date = str(row.get("date", ""))[:10] or None

        f_h5  = fb.recent_form(home,  5, before_date=date)
        f_a5  = fb.recent_form(away,  5, before_date=date)
        f_h12 = fb.recent_form(home, 12, before_date=date)
        f_a12 = fb.recent_form(away, 12, before_date=date)

        form_feat = [
            f_h5[0]  - f_a5[0],
            f_h5[3]  - f_a5[3],
            f_h12[0] - f_a12[0],
            f_h12[3] - f_a12[3],
            f_h5[1]  - f_a5[1],
            f_h5[2]  - f_a5[2],
        ]

        b15       = [float(row.get(f"b15_{i}",  0.0)) for i in range(15)]
        gold      = [float(row.get(f"gold_{i}", 0.0)) for i in range(5)]
        odds_feat = [float(row.get("odds_home", 0.0)),
                     float(row.get("odds_draw", 0.0)),
                     float(row.get("odds_away", 0.0))]
        xg_feat   = [float(row.get(f"xg_{i}",  0.0)) for i in range(8)]

        feat = b15 + gold + odds_feat + form_feat + xg_feat
        assert len(feat) == 29, f"特征维度错误: {len(feat)}"
        return np.array(feat, dtype=np.float32)
    except Exception:
        return None


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def retrain(min_samples: int = 50):
    log("=" * 60)
    log("retrain_xgb_with_form365 开始")
    fb = FormBook()

    if not TRAIN_CSV.exists():
        log(f"ERROR: 找不到 {TRAIN_CSV}")
        sys.exit(1)

    df = pd.read_csv(TRAIN_CSV)
    log(f"训练集: {len(df)} 行, 列: {list(df.columns)[:8]}...")

    rows_feat, rows_label, skipped = [], [], 0
    for _, row in df.iterrows():
        feat = build_features(row, fb)
        if feat is None:
            skipped += 1
            continue
        rows_feat.append(feat)
        rows_label.append(int(row["label"]))

    log(f"有效样本: {len(rows_feat)}, 跳过: {skipped}")
    if len(rows_feat) < min_samples:
        log(f"ERROR: 样本不足 {min_samples}")
        sys.exit(1)

    X = np.vstack(rows_feat)
    y = np.array(rows_label)
    log(f"标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    xgb_params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_losses = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = XGBClassifier(**xgb_params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        loss = log_loss(y[va], m.predict_proba(X[va]))
        cv_losses.append(loss)
        log(f"  Fold {fold+1}/5  log_loss={loss:.4f}")

    log(f"CV log_loss: {np.mean(cv_losses):.4f} ± {np.std(cv_losses):.4f}")

    # 全量训练
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X, y, verbose=False)
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(xgb, f)
    log(f"模型保存: {OUT_MODEL}")

    # Isotonic 校准器
    proba = xgb.predict_proba(X)
    calibrators = {}
    for i, name in enumerate(["away", "draw", "home"]):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(proba[:, i], (y == i).astype(int))
        calibrators[name] = iso
    with open(OUT_CALIBRATORS, "wb") as f:
        pickle.dump(calibrators, f)
    log(f"校准器保存: {OUT_CALIBRATORS}")

    # 训练集简验
    proba_cal = np.column_stack([
        calibrators[n].predict(proba[:, i])
        for i, n in enumerate(["away", "draw", "home"])
    ])
    proba_cal /= proba_cal.sum(axis=1, keepdims=True)
    acc = (proba_cal.argmax(axis=1) == y).mean()
    log(f"训练集准确率: {acc:.4f}  校准后log_loss: {log_loss(y, proba_cal):.4f}")
    log("重训完成")

    return {
        "cv_log_loss_mean": float(np.mean(cv_losses)),
        "cv_log_loss_std":  float(np.std(cv_losses)),
        "train_acc":        float(acc),
        "n_samples":        len(rows_feat),
        "model_path":       str(OUT_MODEL),
        "calibrators_path": str(OUT_CALIBRATORS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-samples", type=int, default=50)
    args = ap.parse_args()
    print(json.dumps(retrain(args.min_samples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
