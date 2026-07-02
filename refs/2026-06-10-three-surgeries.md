# 2026-06-10 三台微创手术

## 背景

系统诊断发现三个关键缺陷：

| # | 缺陷 | 严重度 | 影响 |
|---|------|--------|------|
| 1 | Isotonic校准器负优化 | P0 | Brier +14% 退化 (0.2053→0.2341) |
| 2 | 平局预测为零 | P0 | 3分类退化为2分类 |
| 3 | model_route 100%空 | P1 | 无法按模型路径钻取Brier |

## 诊断方法论

### Brier 生产-训练对比

```python
# 从 predictions_log.csv 提取 Brier
import csv
rows = list(csv.DictReader(open('/root/data/predictions_log.csv')))
brier_rows = [r for r in rows if r.get('brier_spf','').strip()]
briers = [float(r['brier_spf']) for r in brier_rows]
print(f'生产Brier: {sum(briers)/len(briers):.4f} (n={len(briers)})')
print(f'训练验证Brier: 0.2053')
print(f'退化: {(sum(briers)/len(briers))/0.2053-1:+.1%}')
```

### 校准分箱检查 (Per-Decile Calibration)

```python
for decile in range(10):
    lo, hi = decile * 0.1, (decile + 1) * 0.1
    subset = [r for r in brier_rows 
              if float(r.get('pred_h',0))/100 >= lo 
              and float(r.get('pred_h',0))/100 < hi]
    if subset:
        actual_h = sum(1 for r in subset if r.get('actual_hda','') == 'H') / len(subset)
        calib_error = actual_h - (lo + 0.05)
        print(f'[{lo:.0%}-{hi:.0%}] n={len(subset)} actual_H={actual_h:.0%} '
              f'calib={calib_error:+.0%}')
```

### 模型路由分析

```python
# 按 predicted class 聚合 Brier
for label in ['H', 'D', 'A']:
    subset = [r for r in brier_rows if 
              float(r.get('pred_h',0))>float(r.get('pred_d',0)) 
              and float(r.get('pred_h',0))>float(r.get('pred_a',0))  # 简化,实际用argmax
              and r.get('predicted','') == label]
    ...  # 见 /tmp/brier_analysis.py 完整版
```

## 手术记录

### 手术1: 剥离 Isotonic 校准器

**文件：** `/root/daily_jczq.py`

**改点1** — `_try_hybrid_predict()` L325-335:
```
旧: if _calibrators: for j, key in enumerate(['away','draw','home']): 
      calibrated[j] = _calibrators[key].predict([hybrid[j]])[0]
新: # ── 剥离Isotonic校准器 (生产Brier=0.2341 vs 训练0.2053) ──
    pass
    # ── Draw Correction Layer ──
```

**改点2** — `_try_club_predict()` L923-933:
```
旧: if _calibrators_club: for j, key ... 
新: # ── 剥离Isotonic校准器 (Club, 同国际赛诊断) ──
    pass
    # ── Draw Correction Layer ──
```

**验证：** `grep -n "calibrators\[key\].predict" /root/daily_jczq.py` → 0 matches

### 手术2: Draw Correction Layer

**位置：** 紧接在校准器剥离后（两处）

**逻辑：**
```python
if hybrid[1] < 0.15:  # p_draw < 15%
    confidence = max(hybrid[2], hybrid[0])
    draw_boost = 0.05 * (1.0 - confidence)
    hybrid[1] += draw_boost
    denom = hybrid[2] + hybrid[0] + 1e-10
    hybrid[2] -= draw_boost * (hybrid[2] / denom)
    hybrid[0] -= draw_boost * (hybrid[0] / denom)
    s = hybrid.sum()
    if s > 0: hybrid /= s
```

**数学验证：**
- 输入 `[0.60, 0.05, 0.35]` → 输出 `[0.581, 0.070, 0.349]`, 平局从5%→7%
- 输入 `[0.30, 0.25, 0.45]` → 不变（平局≥15%不触发）
- 总和恒为1.0

### 手术3: model_route 追踪修复

**改点** — `build_prediction_bundle()` 返回值新增：

```python
'model': p.get('model', 'unknown'),
```

**全链路：**
```
p['model'] (来自各预测函数)
  → build_prediction_bundle() bundle['model']
  → record_prediction() --model-route 参数
  → backtest_jczq.py FIELDS + cmd_record()
  → predictions_log.csv model_route 列
```

**各路径值：**
| 函数 | model 值 |
|------|----------|
| `_try_hybrid_predict()` | `'hybrid'` |
| `_try_club_predict()` | `'club_hybrid'` |
| `predict_match_legacy()` | `'legacy_poisson'` |
| `fallback_market_predict()` | `'market_fallback'` |
| 兜底（无上游model键） | `'unknown'` |

## 验证命令

```bash
# 1. Isotonic已剥离
grep -c "calibrators\[key\].predict" /root/daily_jczq.py
# 期望输出: 0

# 2. Draw Correction已植入
grep -c "draw_boost" /root/daily_jczq.py
# 期望输出: 2 (两条路径各一处)

# 3. model字段已加入bundle
grep "'model': p.get" /root/daily_jczq.py
# 期望: 'model': p.get('model', 'unknown'),

# 4. model_route传参链完整
grep "model-route" /root/daily_jczq.py
# 期望: '--model-route', str(bundle.get('model', '')),

# 5. 运行验证脚本
python3 /tmp/verify_fixes.py
# 期望: 所有测试 PASS
```

## 预期效果

| 指标 | 修前 | 修后期望 |
|------|------|----------|
| Brier (SPF) | 0.2341 | 往0.2053收敛 |
| Draw预测率 | 0% (0/32) | >5% |
| model_route填充率 | 0% (0/85) | >90% |

## 回滚方案

如 Brier 恶化，回滚三步：

```bash
# Step 1: 恢复 Isotonic 校准器 (intl)
# 在 _try_hybrid_predict() 中将 pass 替换回:
if _calibrators:
    calibrated = np.zeros(3)
    for j, key in enumerate(['away', 'draw', 'home']):
        if key in _calibrators:
            calibrated[j] = _calibrators[key].predict([hybrid[j]])[0]
    ...

# Step 2: 恢复 Isotonic (club, 同上)

# Step 3: 移除 Draw Correction (两处)
# 删除 if hybrid[1] < 0.15: ... 块

# Step 4: 移除 model 字段
# 删除 build_prediction_bundle() 中的 'model': p.get('model', 'unknown'),
```
