"""Fix training data format + retrain nat model"""
import json, os, sys, math, numpy as np, joblib
from datetime import datetime
from collections import Counter
from sklearn.metrics import log_loss, accuracy_score

# Make numpy types JSON serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)
from xgboost import XGBClassifier

sys.path.insert(0, '/root')
sys.path.insert(0, '/root/wc_2026_upgrade')
DATA_DIR = "/root/data"
PATH = f"{DATA_DIR}/training_data_with_odds.json"
BACKUP = f"{DATA_DIR}/training_data_with_odds.json.pre_retrain"

# Backup
os.system(f"cp {PATH} {BACKUP}")
print(f"Backup: {BACKUP}")

with open(PATH) as f:
    data = json.load(f)
print(f"Total: {len(data)}")

# === Step 1: Normalize SPF ===
# Old format: '3'=H, '1'=D, '0'=A
# New format: 'H'=H, 'D'=D, 'A'=A
# Target: '3'/'1'/'0'
SPF_MAP = {'H':'3', 'D':'1', 'A':'0', '3':'3', '1':'1', '0':'0'}
changes = 0
for m in data:
    old = str(m.get('spf_result', ''))
    new = SPF_MAP.get(old, old)
    if old != new:
        m['spf_result'] = new
        changes += 1

print(f"SPF normalized: {changes} changed")
spf_check = Counter(str(m.get('spf_result','')) for m in data)
print(f"SPF distribution: {dict(spf_check)}")

# === Step 2: Normalize market_implied fields ===
# Old format: single 'market_implied_prob' (scalar)
# New format: 'market_implied_h', 'market_implied_d', 'market_implied_a'
# Training script uses 'market_implied_prob' at line 38
for m in data:
    if 'market_implied_prob' not in m or not m['market_implied_prob']:
        # Check if we have the three-part format
        mip = None
        for key in ['market_implied_prob', 'mi']:
            val = m.get(key)
            if val and val != 0:
                mip = val
                break
        if mip is None:
            # Try to compute from h/d/a
            h = m.get('market_implied_h')
            d = m.get('market_implied_d')
            a = m.get('market_implied_a')
            if d and d != 0:
                m['market_implied_prob'] = d  # use draw probability as before
            elif h and a:
                # Use 1/(1+h/a) as a proxy
                m['market_implied_prob'] = d or h or a or 0.0
        else:
            m['market_implied_prob'] = mip

# Count coverage
mi_count = sum(1 for m in data if m.get('market_implied_prob'))
print(f"market_implied_prob coverage: {mi_count}/{len(data)}")

# Save fixed data
with open(PATH, 'w') as f:
    json.dump(data, f, indent=2)
print(f"Saved normalized data to {PATH}")

# === Step 3: Train nat model ===
print("\n" + "="*60)
print("RETRAINING NAT MODEL")
print("="*60)

# Load models
print("\nLoading models...")
dc = joblib.load(f"{DATA_DIR}/dc_model.pkl")
elo = joblib.load(f"{DATA_DIR}/elo_ratings.pkl")
print(f"  DC: {len(dc.teams_)} teams | Elo: {len(elo)} teams")

# Filter English names only
nat = [m for m in data if not any('\u4e00' <= c <= '\u9fff' for c in m['home_en'] + m['away_en'])]
print(f"  National team matches: {len(nat)}/{len(data)}")

def get_dc(home, away):
    try:
        lam_h, lam_a = dc.predict_lambda(home, away, neutral=True)
        if lam_h is None: return None, None, None
        p = dc.predict_proba(home, away, neutral=True)
        return p, lam_h, lam_a
    except:
        return None, None, None

# Build feature matrix
X, y, dates_arr = [], [], []
skipped_no_dc = 0
for m in nat:
    h, a = m['home_en'], m['away_en']
    mi = m.get('market_implied_prob', 0.0) or 0.0
    eh = elo.get(h, 1500)
    ea = elo.get(a, 1500)
    
    pr, lam_h, lam_a = get_dc(h, a)
    if pr is None:
        skipped_no_dc += 1
        continue
    
    dc_probs = np.clip([pr[2], pr[1], pr[0]], 0.01, 0.99)
    lam_h = max(0.1, min(5.0, lam_h))
    lam_a = max(0.1, min(5.0, lam_a))
    op_h = 1/(1+10**((ea-eh)/400))
    op_a = 1/(1+10**((eh-ea)/400))
    
    feat = [(eh-ea)/400, lam_h, lam_a, lam_h-lam_a,
            math.log(max(lam_h,0.01)/max(lam_a,0.01)),
            dc_probs[0], dc_probs[1], dc_probs[2], op_h, op_a, mi]
    
    result = str(m['spf_result'])
    label = 2 if result == '3' else (1 if result == '1' else 0)
    X.append(feat)
    y.append(label)
    dates_arr.append(m['date'])

print(f"  Skipped (no DC coverage): {skipped_no_dc}")
print(f"  Effective samples: {len(X)}")

X = np.array(X)
y = np.array(y)

# Time-series split (70/30)
sort_idx = np.argsort(dates_arr)
X, y = X[sort_idx], y[sort_idx]
dates_sorted = [dates_arr[i] for i in sort_idx]
n = len(X)
split = int(n * 0.7)

X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

FEATURE_NAMES = ['elo_diff', 'lam_h', 'lam_a', 'lam_diff', 'lam_ratio',
                 'dc_a', 'dc_d', 'dc_h', 'op_h', 'op_a', 'market_implied']

print(f"\nTraining set:   {split} ({dates_sorted[0]} → {dates_sorted[split-1]})")
print(f"Validation set: {n-split} ({dates_sorted[split]} → {dates_sorted[-1]})")

# Check class balance
train_dist = Counter(y_train)
val_dist = Counter(y_val)
print(f"Train label dist: {dict(sorted(train_dist.items()))}")
print(f"Val label dist:   {dict(sorted(val_dist.items()))}")

# Train
print("\nTraining XGBoost...")
model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                      use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Validate
pred = model.predict_proba(X_val)
pred_class = model.predict(X_val)
acc = accuracy_score(y_val, pred_class)
ll = log_loss(y_val, pred)

# Brier score
brier = np.mean((pred[np.arange(len(y_val)), y_val] - 1)**2)

print(f"\n{'='*50}")
print(f"  Validation accuracy: {acc*100:.1f}%")
print(f"  LogLoss:            {ll:.4f}")
print(f"  Brier:              {brier:.4f}")
print(f"{'='*50}")

# Feature importance
importance = model.feature_importances_
print(f"\nFeature importance:")
for idx in np.argsort(importance)[::-1]:
    print(f"  {FEATURE_NAMES[idx]:20s}: {importance[idx]:.4f}")

# Compare with old model
old_model_path = f"{DATA_DIR}/xgb_model_nat.pkl"
if os.path.exists(old_model_path):
    old_model = joblib.load(old_model_path)
    old_pred = old_model.predict_proba(X_val)
    old_acc = accuracy_score(y_val, old_model.predict(X_val))
    old_ll = log_loss(y_val, old_pred)
    old_brier = np.mean((old_pred[np.arange(len(y_val)), y_val] - 1)**2)
    print(f"\n{'='*50}")
    print(f"OLD model on same validation set:")
    print(f"  Accuracy: {old_acc*100:.1f}% (was {acc*100:.1f}%)")
    print(f"  LogLoss:  {old_ll:.4f} (was {ll:.4f})")
    print(f"  Brier:    {old_brier:.4f} (was {brier:.4f})")
    delta_acc = (acc - old_acc) * 100
    print(f"  ΔAccuracy: {delta_acc:+.1f}pp")
    print(f"{'='*50}")

# Save model
model_path = f"{DATA_DIR}/xgb_model_nat.pkl"
joblib.dump(model, model_path)
print(f"\nSaved model: {model_path}")

# Save calibrator (optional, was disabled before)
from sklearn.isotonic import IsotonicRegression
cal = {}
y_proba = model.predict_proba(X)
for j, key in enumerate(['away', 'draw', 'home']):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(y_proba[:, j], (y == j).astype(float))
    cal[key] = ir
cal_path = f"{DATA_DIR}/calibrators_nat.pkl"
joblib.dump(cal, cal_path)
print(f"Saved calibrator: {cal_path}")

# Save report
report = {
    'timestamp': datetime.now().isoformat(),
    'model': 'xgb_model_nat',
    'n_total': n,
    'n_train': split,
    'n_val': n-split,
    'train_date_range': f"{dates_sorted[0]} → {dates_sorted[split-1]}",
    'val_date_range': f"{dates_sorted[split]} → {dates_sorted[-1]}",
    'val_acc': acc,
    'val_logloss': ll,
    'val_brier': brier,
    'feature_importance': dict(zip(FEATURE_NAMES, importance.tolist())),
    'data_source': 'thestatsapi_2.5k',
}
json.dump(report, open(f"{DATA_DIR}/train_report_nat.json", 'w'), indent=2, cls=NumpyEncoder)
print(f"Saved report: {DATA_DIR}/train_report_nat.json")

print(f"\n{'='*60}")
print(f"  DONE: nat model retrained on {n} matches")
print(f"  Accuracy: {acc*100:.1f}% | LogLoss: {ll:.4f}")
print(f"{'='*60}")
