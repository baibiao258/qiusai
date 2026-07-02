#!/usr/bin/env python3
"""
train_national_xgb.py — 国家队专用 v28 模型
=============================================
只使用英文名数据（dc_model 可覆盖）, 去掉俱乐部噪音.
"""
import json, os, sys, math, numpy as np, joblib
from datetime import datetime
from sklearn.metrics import log_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
sys.path.insert(0, '/root')
DATA_DIR = '/root/data'

# 加载模型
print("📡 加载模型...")
dc = joblib.load(f'{DATA_DIR}/dc_model.pkl')
elo = joblib.load(f'{DATA_DIR}/elo_ratings.pkl')
print(f"  DC国家队: {len(dc.teams_)} 队 | Elo: {len(elo)} 队")

# 加载数据, 只保留英文名
data = json.load(open(f'{DATA_DIR}/training_data_with_odds.json'))
nat = [m for m in data if not any('\u4e00' <= c <= '\u9fff' for c in m['home_en'] + m['away_en'])]
print(f"  训练数据: {len(nat)} 场 (过滤掉 {len(data)-len(nat)} 场中文名)")

# 构建特征
def get_dc(home, away):
    try:
        lam_h, lam_a = dc.predict_lambda(home, away, neutral=True)
        if lam_h is None: return None, None, None
        p = dc.predict_proba(home, away, neutral=True)
        return p, lam_h, lam_a
    except: return None, None, None

X, y, dates_arr = [], [], []
for m in nat:
    h, a = m['home_en'], m['away_en']
    mi = m.get('market_implied_prob', 0.0)
    eh = elo.get(h, 1500); ea = elo.get(a, 1500)
    
    pr, lam_h, lam_a = get_dc(h, a)
    if pr is None:
        continue  # 跳过dc_model覆盖不了的
    
    # dc_probs = [A, D, H]
    dc_probs = np.clip([pr[2], pr[1], pr[0]], 0.01, 0.99)
    lam_h = max(0.1, min(5.0, lam_h)); lam_a = max(0.1, min(5.0, lam_a))
    op_h = 1/(1+10**((ea-eh)/400)); op_a = 1/(1+10**((eh-ea)/400))
    
    feat = [(eh-ea)/400, lam_h, lam_a, lam_h-lam_a,
            math.log(max(lam_h,0.01)/max(lam_a,0.01)),
            dc_probs[0], dc_probs[1], dc_probs[2], op_h, op_a, mi]
    
    result = str(m['spf_result'])
    label = 2 if result == '3' else (1 if result == '1' else 0)
    X.append(feat); y.append(label); dates_arr.append(m['date'])

X = np.array(X); y = np.array(y)
print(f"  有效样本: {len(X)} (dc_model覆盖)")

# 时间序列分割
sort_idx = np.argsort(dates_arr)
X, y = X[sort_idx], y[sort_idx]
dates_sorted = [dates_arr[i] for i in sort_idx]
n = len(X)
split = int(n * 0.7)

X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
print(f"\n训练集: {split} 场 ({dates_sorted[0]}→{dates_sorted[split-1]})")
print(f"验证集: {n-split} 场 ({dates_sorted[split]}→{dates_sorted[-1]})")

# 训练
print("\n🏋️ 训练 XGBoost...")
model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                      use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 验证
pred = model.predict_proba(X_val)
pred_class = model.predict(X_val)
acc = accuracy_score(y_val, pred_class)
ll = log_loss(y_val, pred)
print(f"  验证准确率: {acc*100:.1f}%")
print(f"  LogLoss: {ll:.4f}")

# 特征重要性
FEATURE_NAMES = ['elo_diff', 'lam_h', 'lam_a', 'lam_diff', 'lam_ratio',
                 'dc_a', 'dc_d', 'dc_h', 'op_h', 'op_a', 'market_implied']
importance = model.feature_importances_
print(f"\n📊 特征重要性:")
for idx in np.argsort(importance)[::-1]:
    print(f"  {FEATURE_NAMES[idx]}: {importance[idx]:.4f}")

# 保存
model_path = f'{DATA_DIR}/xgb_model_nat.pkl'
joblib.dump(model, model_path)
print(f"\n✅ 国家队模型保存: {model_path}")

# 校准器
cal = {}
y_proba = model.predict_proba(X)
for j, key in enumerate(['away', 'draw', 'home']):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(y_proba[:, j], (y == j).astype(float))
    cal[key] = ir
cal_path = f'{DATA_DIR}/calibrators_nat.pkl'
joblib.dump(cal, cal_path)
print(f"✅ 校准器保存: {cal_path}")

# 报告
report = {
    'timestamp': datetime.now().isoformat(),
    'model': 'xgb_model_nat',
    'n_train': split, 'n_val': n-split,
    'val_acc': acc, 'val_logloss': ll,
    'feature_importance': dict(zip(FEATURE_NAMES, importance.tolist())),
}
json.dump(report, open(f'{DATA_DIR}/train_report_nat.json', 'w'), indent=2)
print(f"✅ 报告保存")

print(f"\n{'='*50}")
print(f"  国家队模型完成: {acc*100:.1f}% 验证准确率")
print(f"{'='*50}")
