#!/usr/bin/env python3
"""
retrain_xgb_with_odds.py — 用市场赔率特征重训XGBoost
====================================================

基于 prepare_training_data.py 的输出,
构建包含市场赔率的29+1维特征,
重训XGBoost模型。

输出:
  - /root/data/xgb_model_30.pkl (新模型)
  - /root/data/calibrators_v2.pkl (新校准器)
"""

import json
import os
import sys
import math
from datetime import datetime

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
import joblib

sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')

DATA_DIR = '/root/data'


def load_training_data():
    """加载带市场赔率的训练数据"""
    path = os.path.join(DATA_DIR, 'training_data_with_odds.json')
    with open(path) as f:
        return json.load(f)


def load_shared_models():
    """加载共享模型"""
    import joblib
    dc = joblib.load(os.path.join(DATA_DIR, 'dc_model.pkl'))
    elo = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))
    # 加载俱乐部 DC 模型 (可选)
    dc_club = None
    dc_club_path = os.path.join(DATA_DIR, 'dc_club.pkl')
    if os.path.exists(dc_club_path):
        dc_club = joblib.load(dc_club_path)
        print(f"  俱乐部DC模型: ✅ ({len(dc_club.teams_)} 队)")
    return dc, elo, dc_club


def compute_dc_probs(dc_model, home, away, dc_club=None):
    """DC概率计算, 国家队模型优先, 俱乐部模型回退"""
    try:
        lam_h, lam_a = dc_model.predict_lambda(home, away, neutral=True)
        if lam_h is not None:
            pass  # 国家队模型成功
        elif dc_club is not None:
            # 回退到俱乐部模型
            try:
                lam_h, lam_a = dc_club.predict_lambda(home, away, neutral=False)
            except Exception:
                return None, None, None
        else:
            return None, None, None
    except Exception:
        if dc_club is not None:
            try:
                lam_h, lam_a = dc_club.predict_lambda(home, away, neutral=False)
            except Exception:
                return None, None, None
        else:
            return None, None, None
    
    if lam_h is None or lam_a is None:
        return None, None, None
    
    from scipy.stats import poisson
    max_g = 8
    ph = [poisson.pmf(i, lam_h) for i in range(max_g)]
    pa = [poisson.pmf(i, lam_a) for i in range(max_g)]
    p_h, p_d, p_a = 0, 0, 0
    for i in range(max_g):
        for j in range(max_g):
            p = ph[i] * pa[j]
            if i > j: p_h += p
            elif i == j: p_d += p
            else: p_a += p
    return [p_a, p_d, p_h], lam_h, lam_a


def build_features(match, dc_model, elo, dc_club=None):
    """构建特征 (29基础 + 1市场赔率 = 30维)
    
    DC概率: 国家队模型优先, 俱乐部模型回退, 都失败则均匀概率.
    """
    home, away = match['home_en'], match['away_en']

    elo_h = elo.get(home, 1500)
    elo_a = elo.get(away, 1500)

    # DC概率 (国家队→俱乐部→均匀值)
    dc_p, lam_h, lam_a = compute_dc_probs(dc_model, home, away, dc_club)
    if dc_p is None or lam_h is None:
        # DC不可用 (俱乐部比赛等): 回退为均匀值
        dc_p = [1/3, 1/3, 1/3]
        lam_h = lam_a = 1.5  # 全局平均λ

    # Winsorize截断: DC概率限制在[0.01, 0.99], 防止极端值
    dc_p = [max(0.01, min(0.99, p)) for p in dc_p]
    lam_h = max(0.1, min(5.0, lam_h))
    lam_a = max(0.1, min(5.0, lam_a))

    # 近5场form (用占位值, 实际应该从form_state读取)
    fh5 = [0.5, 1.5, 1.2, 0.3]  # win_rate, avg_gf, avg_ga, avg_gd
    fa5 = [0.5, 1.5, 1.2, 0.3]

    # Gold特征 (用占位值)
    gold = [0.0, 0, 0, 0.0, 0.0]

    # 概率特征 (Elo隐含概率)
    op_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
    op_a = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))

    # 原始29维特征
    b15 = [
        (elo_h - elo_a) / 400, lam_h, lam_a, lam_h - lam_a,
        math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),
        dc_p[0], dc_p[1], dc_p[2],
        fh5[0], fa5[0],
        fh5[1] - fa5[2], fa5[1] - fh5[2],
        fh5[1] - fa5[1], fh5[0] - fa5[0],
        1,
    ]
    odds_feat = [op_h, op_a, 0.0]
    form_feat = [fh5[1], fh5[2], fa5[1], fa5[2], fh5[0] * 3, fa5[0] * 3]

    # 新增: 市场赔率特征 (第30维)
    market_implied = match.get('market_implied_prob', 0.0)

    # 29维 + 1维市场赔率 = 30维 (赛事阶段特征尚未填充, 留待后续)
    feat = b15 + gold + odds_feat + form_feat + [market_implied]

    return np.array(feat)


def main():
    print("📡 加载数据...")
    data = load_training_data()
    print(f"  训练数据: {len(data)} 场")

    print("\n📡 加载模型...")
    dc_model, elo, dc_club = load_shared_models()
    print(f"  DC模型: ✅ | Elo: {len(elo)} 队 | 俱乐部DC: {'✅' if dc_club else '❌'}")

    # 构建特征和标签
    print("\n🔧 构建特征...")
    X = []
    y = []
    valid_count = 0
    skip_count = 0

    for m in data:
        feat = build_features(m, dc_model, elo, dc_club)
        if feat is None:
            skip_count += 1
            continue

        # 标签: spf_result ('3'=主胜, '1'=平局, '0'=客胜)
        result = str(m['spf_result'])
        if result == '3':
            label = 2  # H
        elif result == '1':
            label = 1  # D
        else:
            label = 0  # A

        X.append(feat)
        y.append(label)
        valid_count += 1

    X = np.array(X)
    y = np.array(y)

    print(f"  有效样本: {valid_count}")
    print(f"  跳过样本: {skip_count}")
    print(f"  特征维度: {X.shape[1]}")

    # 训练XGBoost
    print("\n🏋️ 训练XGBoost...")
    from xgboost import XGBClassifier

    # 时间序列分割
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # 预测
        y_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)

        # 评估 (multiclass Brier)
        from sklearn.metrics import log_loss
        ll = log_loss(y_val, y_proba)
        acc = accuracy_score(y_val, y_pred)
        brier = ll  # Use log_loss as proxy for multiclass calibration

        scores.append({'fold': fold, 'brier': brier, 'acc': acc})
        print(f"  Fold {fold}: Brier={brier:.4f}, Acc={acc:.4f}")

    # 用全量数据训练最终模型
    print("\n🏋️ 训练最终模型 (全量数据)...")
    final_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
    )
    final_model.fit(X, y, verbose=False)

    # 特征重要性
    print("\n📊 特征重要性:")
    importance = final_model.feature_importances_
    feature_names = [
        'elo_diff', 'lam_h', 'lam_a', 'lam_diff', 'lam_ratio',
        'dc_a', 'dc_d', 'dc_h', 'fh5_wr', 'fa5_wr',
        'fh5_gf_fa5_ga', 'fa5_gf_fh5_wr', 'fh5_gf_fa5_gf', 'fh5_wr_fa5_wr',
        'bias',
        'h2h_gd', 'tier_major', 'tier_friendly', 'fh12_gf_fa12_ga', 'fa12_gf_fh12_wr',
        'op_h', 'op_a', 'op_0',
        'fh5_gf', 'fh5_ga', 'fa5_gf', 'fa5_ga', 'fh5_wr3', 'fa5_wr3',
        'market_implied',
        'points_diff', 'rank_diff', 'is_knockout', 'round_num',  # 赛事阶段特征
    ]

    top_idx = np.argsort(importance)[-10:]
    for i in reversed(top_idx):
        print(f"  {feature_names[i]}: {importance[i]:.4f}")

    # 保存模型
    output_path = os.path.join(DATA_DIR, 'xgb_model_30.pkl')  # 30维模型 (29基础+1市场赔率)
    joblib.dump(final_model, output_path)
    print(f"\n✅ 模型保存到: {output_path}")

    # 训练Isotonic校准器
    print("\n🔧 训练Isotonic校准器...")
    # 用最后一折的数据做校准
    X_cal = X[val_idx]
    y_cal = y[val_idx]
    y_proba_cal = final_model.predict_proba(X_cal)

    calibrators = {}
    for j, key in enumerate(['away', 'draw', 'home']):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(y_proba_cal[:, j], (y_cal == j).astype(float))
        calibrators[key] = ir

    cal_path = os.path.join(DATA_DIR, 'calibrators_v2.pkl')
    joblib.dump(calibrators, cal_path)
    print(f"✅ 校准器保存到: {cal_path}")

    # 保存训练报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': valid_count,
        'n_features': X.shape[1],
        'cv_scores': scores,
        'mean_log_loss': np.mean([s['brier'] for s in scores]),
        'mean_acc': np.mean([s['acc'] for s in scores]),
        'feature_importance': dict(zip(feature_names, importance.tolist())),
    }

    report_path = os.path.join(DATA_DIR, 'retrain_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ 报告保存到: {report_path}")

    print(f"\n{'='*60}")
    print(f"  📊 训练完成")
    print(f"  平均LogLoss: {report['mean_log_loss']:.4f}")
    print(f"  平均准确率: {report['mean_acc']*100:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
