#!/usr/bin/env python3
"""
train_clean_xgb.py — 精简版XGBoost重训
=========================================
去掉所有死特征 (form/gold/h2h/tier占位符),
只保留11个活特征 + 时间序列交叉验证修复.
"""
import json, os, sys, math
from datetime import datetime
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

sys.path.insert(0, '/root')
DATA_DIR = '/root/data'

def load_training_data():
    path = os.path.join(DATA_DIR, 'training_data_with_odds.json')
    with open(path) as f:
        return json.load(f)

def compute_dc_probs(dc_model, home, away, dc_club=None):
    """DC概率, 国家队优先, 俱乐部回退"""
    try:
        lam_h, lam_a = dc_model.predict_lambda(home, away, neutral=True)
        if lam_h is not None:
            pass  # 国家队模型成功
        elif dc_club is not None:
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

def build_clean_features(match, dc_model, elo, dc_club=None):
    """构建精简特征 (11维) — 支持俱乐部DC回退"""
    home, away = match['home_en'], match['away_en']
    
    elo_h = elo.get(home, 1500)
    elo_a = elo.get(away, 1500)
    
    # DC概率 (国家队→俱乐部→均匀值)
    dc_p, lam_h, lam_a = compute_dc_probs(dc_model, home, away, dc_club)
    if dc_p is None or lam_h is None:
        dc_p = [1/3, 1/3, 1/3]
        lam_h = lam_a = 1.5
    
    # Winsorize截断: DC概率限制在[0.01, 0.99], 防止极端值
    dc_p = [max(0.01, min(0.99, p)) for p in dc_p]
    lam_h = max(0.1, min(5.0, lam_h))
    lam_a = max(0.1, min(5.0, lam_a))
    
    # Elo隐含概率
    op_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
    op_a = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))
    
    # 11维干净特征
    feat = [
        (elo_h - elo_a) / 400,  # elo_diff
        lam_h,                   # 主队预期进球
        lam_a,                   # 客队预期进球
        lam_h - lam_a,           # 预期进球差
        math.log(max(lam_h, 0.01) / max(lam_a, 0.01)),  # 预期进球比
        dc_p[0],                 # DC客胜概率
        dc_p[1],                 # DC平局概率
        dc_p[2],                 # DC主胜概率
        op_h,                    # Elo主胜概率
        op_a,                    # Elo客胜概率
        match.get('market_implied_prob', 0.0),  # 市场赔率概率
    ]
    
    return np.array(feat)

FEATURE_NAMES = [
    'elo_diff', 'lam_h', 'lam_a', 'lam_diff', 'lam_ratio',
    'dc_a', 'dc_d', 'dc_h',
    'op_h', 'op_a',
    'market_implied',
]

def main():
    print("📡 加载数据...")
    data = load_training_data()
    print(f"  训练数据: {len(data)} 场")
    
    print("\n📡 加载模型...")
    dc_model = joblib.load(os.path.join(DATA_DIR, 'dc_model.pkl'))
    elo = joblib.load(os.path.join(DATA_DIR, 'elo_ratings.pkl'))
    dc_club = None
    dc_club_path = os.path.join(DATA_DIR, 'dc_club.pkl')
    if os.path.exists(dc_club_path):
        dc_club = joblib.load(dc_club_path)
    print(f"  DC模型: ✅ ({len(dc_model.teams_)}队) | Elo: {len(elo)} 队 | 俱乐部DC: {'✅' if dc_club else '❌'} ({len(dc_club.teams_) if dc_club else 0}队)")
    
    print("\n🔧 构建特征...")
    X = []
    y = []
    for m in data:
        feat = build_clean_features(m, dc_model, elo, dc_club)
        result = str(m['spf_result'])
        if result == '3':
            label = 2  # H
        elif result == '1':
            label = 1  # D
        else:
            label = 0  # A
        X.append(feat)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    print(f"  总样本: {len(X)}, 特征维度: {X.shape[1]}")
    
    # ── 按日期排序确保时序分割公平 ──
    dates = [m['date'] for m in data]
    sorted_idx = np.argsort(dates)
    X = X[sorted_idx]
    y = y[sorted_idx]
    dates_sorted = [dates[i] for i in sorted_idx]
    print(f"  日期范围: {dates_sorted[0]} → {dates_sorted[-1]}")
    
    # ── 时间序列交叉验证 (按时间顺序, 不用随机打乱) ──
    # 手动分割: 按时间分3折
    n = len(X)
    fold_sizes = [n // 3, n // 3, n - 2 * (n // 3)]
    splits = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        splits.append((list(range(start)), list(range(start, end))))
        start = end
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        if len(train_idx) < 10 or len(val_idx) < 10:
            continue
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        date_range = f"{dates_sorted[train_idx[0]]}~{dates_sorted[train_idx[-1]]} | val: {dates_sorted[val_idx[0]]}~{dates_sorted[val_idx[-1]]}"
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        
        ll = log_loss(y_val, y_proba)
        acc = accuracy_score(y_val, y_pred)
        brier_mc = ll  # multiclass Brier proxy
        
        scores.append({'fold': fold, 'brier': brier_mc, 'acc': acc, 'date_range': date_range})
        print(f"  Fold {fold}: Brier={brier_mc:.4f}, Acc={acc*100:.1f}% | {date_range}")
    
    # ── 训练最终模型 ──
    print("\n🏋️ 训练最终模型 (全量数据)...")
    final_model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
    )
    final_model.fit(X, y, verbose=False)
    
    # 特征重要性
    print("\n📊 特征重要性 (11维):")
    importance = final_model.feature_importances_
    for idx in np.argsort(importance)[::-1]:
        print(f"  {FEATURE_NAMES[idx]}: {importance[idx]:.4f}")
    
    # 保存模型 - v28 (clean 11-dim)
    output_path = os.path.join(DATA_DIR, 'xgb_model_28.pkl')
    joblib.dump(final_model, output_path)
    print(f"\n✅ 模型保存到: {output_path}")
    
    # ── 校准器 (用全部数据, 自校准) ──
    print("\n🔧 训练Isotonic校准器...")
    y_proba_full = final_model.predict_proba(X)
    
    calibrators = {}
    for j, key in enumerate(['away', 'draw', 'home']):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(y_proba_full[:, j], (y == j).astype(float))
        calibrators[key] = ir
    
    cal_path = os.path.join(DATA_DIR, 'calibrators_v3.pkl')
    joblib.dump(calibrators, cal_path)
    print(f"✅ 校准器保存到: {cal_path}")
    
    # 报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': 'xgb_model_28',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': FEATURE_NAMES,
        'cv_scores': scores,
        'mean_log_loss': np.mean([s['brier'] for s in scores]),
        'mean_acc': np.mean([s['acc'] for s in scores]),
        'feature_importance': dict(zip(FEATURE_NAMES, importance.tolist())),
    }
    report_path = os.path.join(DATA_DIR, 'train_report_v28.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ 报告保存到: {report_path}")
    
    print(f"\n{'='*60}")
    print(f"  📊 v28 训练完成")
    print(f"  特征: {X.shape[1]}维 (去掉14个死特征)")
    print(f"  样本: {len(X)}")
    print(f"  平均LogLoss: {report['mean_log_loss']:.4f}")
    print(f"  平均准确率: {report['mean_acc']*100:.1f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
