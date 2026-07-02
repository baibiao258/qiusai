#!/usr/bin/env python3
"""校准曲线分析: 从 predictions_log.csv 提取真实赛果，绘制校准曲线"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

rows = list(csv.DictReader(open('/root/data/predictions_log.csv')))
labeled = [r for r in rows if r.get('actual_hda','').strip() in ('H','D','A')]
print(f"有赛果记录: {len(labeled)}")

y_true_hda = []
probs_H, probs_D, probs_A = [], [], []

for r in labeled:
    hda = r['actual_hda']
    try:
        pH = float(r.get('pred_h', 0)) / 100
        pD = float(r.get('pred_d', 0)) / 100
        pA = float(r.get('pred_a', 0)) / 100
    except:
        continue
    y_true_hda.append(hda)
    probs_H.append(pH)
    probs_D.append(pD)
    probs_A.append(pA)

y_true = np.array(y_true_hda)
pH = np.array(probs_H)
pD = np.array(probs_D)
pA = np.array(probs_A)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'Football Prediction Calibration Curve (n={len(y_true)})', fontsize=16, fontweight='bold')

# 1. Home win
ax = axes[0, 0]
y_bin_H = (y_true == 'H').astype(int)
try:
    frac, mean_pred = calibration_curve(y_bin_H, pH, n_bins=5, strategy='quantile')
    ax.plot(mean_pred, frac, 's-', color='#e74c3c', linewidth=2, markersize=8, label='Home Win')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.fill_between(mean_pred, frac, mean_pred, alpha=0.15, color='#e74c3c')
except Exception as e:
    ax.text(0.5, 0.5, str(e), ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Actual Hit Rate')
ax.set_title(f'Home Win (H) - actual {y_bin_H.mean()*100:.1f}%')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# 2. Draw
ax = axes[0, 1]
y_bin_D = (y_true == 'D').astype(int)
try:
    frac, mean_pred = calibration_curve(y_bin_D, pD, n_bins=4, strategy='quantile')
    ax.plot(mean_pred, frac, 's-', color='#3498db', linewidth=2, markersize=8, label='Draw')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.fill_between(mean_pred, frac, mean_pred, alpha=0.15, color='#3498db')
except Exception as e:
    ax.text(0.5, 0.5, str(e), ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Actual Hit Rate')
ax.set_title(f'Draw (D) - actual {y_bin_D.mean()*100:.1f}%')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# 3. Away win
ax = axes[1, 0]
y_bin_A = (y_true == 'A').astype(int)
try:
    frac, mean_pred = calibration_curve(y_bin_A, pA, n_bins=5, strategy='quantile')
    ax.plot(mean_pred, frac, 's-', color='#2ecc71', linewidth=2, markersize=8, label='Away Win')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.fill_between(mean_pred, frac, mean_pred, alpha=0.15, color='#2ecc71')
except Exception as e:
    ax.text(0.5, 0.5, str(e), ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Actual Hit Rate')
ax.set_title(f'Away Win (A) - actual {y_bin_A.mean()*100:.1f}%')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# 4. Overall accuracy by confidence bin
ax = axes[1, 1]
max_probs = np.maximum(np.maximum(pH, pD), pA)
max_labels = np.where(pH >= np.maximum(pD, pA), 'H',
               np.where(pD >= pA, 'D', 'A'))
hits = (max_labels == y_true).astype(int)

bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
bin_centers, bin_hit_rates, bin_counts = [], [], []
for i in range(len(bins)-1):
    mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
    if mask.sum() > 0:
        bin_centers.append((bins[i] + bins[i+1]) / 2)
        bin_hit_rates.append(hits[mask].mean())
        bin_counts.append(mask.sum())

ax2 = ax.twinx()
if bin_centers:
    ax.bar(bin_centers, bin_counts, width=0.06, alpha=0.3, color='#9b59b6', label='Count')
    ax2.plot(bin_centers, [h*100 for h in bin_hit_rates], 'D-', color='#e67e22', linewidth=2, markersize=10, label='Hit Rate')
    for c, n in zip(bin_centers, bin_counts):
        ax.text(c, max(bin_counts)*0.05, str(n), ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Max Predicted Probability')
ax.set_ylabel('Count')
ax2.set_ylabel('Hit Rate (%)', color='#e67e22')
ax2.set_ylim(0, 110)
ax.set_title(f'Overall Accuracy: {hits.mean()*100:.1f}%')
ax.grid(True, alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('/root/data/calibration_curve.png', dpi=150, bbox_inches='tight')
print(f"Saved: /root/data/calibration_curve.png")

# Numeric diagnostics
print(f"\n{'='*60}")
print(f"  Calibration Diagnostics")
print(f"{'='*60}")

for name, p_arr, y_bin in [('Home(H)', pH, y_bin_H), ('Draw(D)', pD, y_bin_D), ('Away(A)', pA, y_bin_A)]:
    avg_pred = p_arr.mean() * 100
    actual_rate = y_bin.mean() * 100
    gap = actual_rate - avg_pred
    brier = np.mean((y_bin - p_arr)**2)
    flag = 'OVERCONF' if gap < -10 else 'CONSERV' if gap > 10 else 'OK'
    print(f"  {name}: pred={avg_pred:.1f}% actual={actual_rate:.1f}% gap={gap:+.1f}pp [{flag}] brier={brier:.4f}")

# Per-group diagnostics
def infer_action(row):
    league = row.get('league', '')
    if league == 'UEFA Nations League': return 'SKIP'
    try:
        ph = float(row.get('pred_h', 0)) / 100
        pd_ = float(row.get('pred_d', 0)) / 100
        pa = float(row.get('pred_a', 0)) / 100
    except: return 'UNKNOWN'
    if abs(ph - 1/3) < 0.05 and abs(pd_ - 1/3) < 0.05 and abs(pa - 1/3) < 0.05:
        return 'WATCH_UNIFORM'
    try:
        oh = float(row.get('odds_h', 0) or 0)
        od = float(row.get('odds_d', 0) or 0)
        oa = float(row.get('odds_a', 0) or 0)
    except: oh = od = oa = 0
    margin = 0
    for p, o in [(ph, oh), (pd_, od), (pa, oa)]:
        if o > 1: margin = max(margin, (p - 1/o) * 100)
    if league == 'Friendship' or league == '友谊赛':
        if margin < 20: return 'WATCH_LOW'
    if margin < 10: return 'WATCH_LOW'
    return 'RECOMMEND'

print(f"\n{'='*60}")
print(f"  Per bet_action (inferred) Diagnostics")
print(f"{'='*60}")

action_stats = {}
for r in labeled:
    a = infer_action(r)
    hda = r['actual_hda']
    try:
        ph = float(r.get('pred_h', 0)) / 100
        pd_ = float(r.get('pred_d', 0)) / 100
        pa = float(r.get('pred_a', 0)) / 100
    except: continue
    max_p = max(ph, pd_, pa)
    pred_cls = 'H' if ph >= max(pd_, pa) else ('D' if pd_ >= pa else 'A')
    hit = 1 if pred_cls == hda else 0
    if a not in action_stats:
        action_stats[a] = {'preds': [], 'hits': [], 'briers': []}
    action_stats[a]['preds'].append(max_p)
    action_stats[a]['hits'].append(hit)
    iH = 1 if hda == 'H' else 0
    iD = 1 if hda == 'D' else 0
    iA = 1 if hda == 'A' else 0
    action_stats[a]['briers'].append(((iH-ph)**2 + (iD-pd_)**2 + (iA-pa)**2) / 3.0)

for action in ['WATCH_UNIFORM', 'WATCH_LOW', 'RECOMMEND']:
    if action not in action_stats: continue
    s = action_stats[action]
    n = len(s['hits'])
    avg_p = np.mean(s['preds']) * 100
    hit_rate = np.mean(s['hits']) * 100
    gap = hit_rate - avg_p
    avg_brier = np.mean(s['briers'])
    print(f"  {action} (n={n}): max_prob={avg_p:.1f}% hit={hit_rate:.1f}% gap={gap:+.1f}pp brier={avg_brier:.4f}")
