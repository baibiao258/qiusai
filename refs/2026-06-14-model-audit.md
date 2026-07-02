# 2026-06-14 系统性模型审计报告

## 触发背景

用户要求全面分析预测模型和流程。从数据、特征、模型训练、推理链路四个维度逐层审计。

## 审计方法 (可直接复用)

### 步骤1: 模型文件清单

```bash
ls -la /root/data/xgb_model*.pkl
```

检查项: 文件数、时间戳(确认最近是否重训过)、文件大小(大=多估计器或高维)

### 步骤2: 特征健康度扫描

对每个 `xgb_model_*.pkl`:

```python
import joblib, numpy as np
model = joblib.load('xgb_model_29.pkl')
imp = model.feature_importances_
zero_mask = imp == 0.0
print(f'{model.n_features_in_}维, 死特征: {zero_mask.sum()}/{model.n_features_in_}')
```

### 步骤3: 训练数据四层审计

```bash
# 1. 总览
python3 -c "
import json
d = json.load(open('/root/data/training_data_with_odds.json'))
print(f'总数: {len(d)}')
print(f'日期范围: {min(m[\"date\"] for m in d)} ~ {max(m[\"date\"] for m in d)}')
from collections import Counter
for t, c in Counter(m['tournament'] for m in d).most_common():
    print(f'  {t}: {c}')
# 中文名检测
cn = sum(1 for m in d if any(ord(c)>127 for c in m['home_en']))
print(f'中文队名: {cn}/{len(d)}')
"
# 2. DC覆盖率
python3 -c "
import json, joblib, sys; sys.path.insert(0, '/root')
d = json.load(open('/root/data/training_data_with_odds.json'))
dc = joblib.load('/root/data/dc_model.pkl')
skipped = sum(1 for m in d if dc.predict_lambda(m['home_en'], m['away_en'], neutral=True)[0] is None)
print(f'DC可预测: {len(d)-skipped}/{len(d)}')
"

# 3. 市场赔率覆盖率
python3 -c "
d = json.load(open('/root/data/training_data_with_odds.json'))
nonz = sum(1 for m in d if m.get('market_implied_prob', 0) > 0.01)
print(f'market_implied覆盖率: {nonz}/{len(d)}')
"
```

### 步骤4: 时间序列交叉验证

用 `train_clean_xgb.py` 中的手动时间序列分割方式，输出每折日期范围+指标。

```python
# 按日期排序
dates = [m['date'] for m in data]
sorted_idx = np.argsort(dates)
# 3折
n = len(X)
fold_sizes = [n // 3, n // 3, n - 2 * (n // 3)]
# 每折: train=[0..start], val=[start..end]
```

### 步骤5: A/B 测试

对比多版模型在同一数据上的概率分布:

```python
# 每场比赛, 对每对模型计算 max(|p_A - p_B|)
diffs = []
for m in test_matches:
    p_a = model_a.predict_proba(feat_a)[0]
    p_b = model_b.predict_proba(feat_b)[0]
    diffs.append(max(abs(p_a - p_b)))
mean_max_diff = np.mean(diffs)
```

## 审计结果 (2026-06-14)

### 数据层

| 项 | 值 | 
|----|-----|
| 训练数据 `training_data_with_odds.json` | 510 条 |
| 时间范围 | 2024-01-12 ~ 2026-06-13 |
| 国家队:俱乐部 | 360:150 |
| 中文队名数 | 150 (全部俱乐部) |
| DC可预测 | 360/510 (70.6%) — 中文名全部跳过 |
| market_implied覆盖率 | 451/510 (88.4%) |

### 特征层 (v29=29维, v30=30维)

**死特征清单 (importance=0 in v29)**:
```
fh5_wr, fa5_wr, fh5_gf_fa5_ga, fa5_gf_fh5_wr, fh5_gf_fa5_gf, fh5_wr_fa5_wr,
bias, h2h_gd, tier_major, tier_friendly, fh12_gf_fa12_ga, fa12_gf_fh12_wr,
op_0, fh5_gf, fh5_ga, fa5_gf, fa5_ga, fh5_wr3, fa5_wr3
= 19/29 死特征 (65.5%)
```

原因: form 特征是占位值 `[0.5, 1.5, 1.2, 0.3]`，gold/h2h/tier 全为 0.0

### 模型层 (时间序列CV)

| Fold | 训练期 | 验证期 | LogLoss | Acc |
|------|--------|--------|---------|-----|
| 0 | 2024-01 ~ 2024-06 | 2024-06 ~ 2026-05 | 0.6785 | 78.8% |
| 1 | 2024-01 ~ 2026-05 | 2026-05 ~ 2026-06 | 1.0039 | 42.9% |

最新折 Acc(42.9%) ≈ baseline(猜胜42.5%) — 模型在最新数据上无效。

### A/B 测试

| 模型A | 模型B | mean_max_diff | 结论 |
|-------|-------|--------------|------|
| v28(11) | v29(29) | 61.3% | v29 受死特征干扰严重 |
| v28(11) | v30(30) | 1.4% | 13个死特征外额外+1维市场赔率影响极小 |
| v29(29) | v30(30) | 61.8% | 同上 |

### 根因追溯链

```
590条 500.com数据 (中文队名)
  → prepare_training_data.py 无 _resolve_name() 映射
    → training_data_with_odds.json 混入中文队名
      → DC模型返回None (仅226支国家队)
        → 150条俱乐部匹配数据特征退化
          → 训练时死特征无信号
            → 模型预测不稳定
              → 最新数据Fold Acc=42.9%
```

### 修复动作

| # | 动作 | 状态 | 影响 |
|---|------|------|------|
| 1 | retrain_xgb_with_odds.py 增加 DC 回退(均匀值) | ✅ | 510条全量使用 |
| 2 | 创建 train_clean_xgb.py (11维干净特征) | ✅ | 消除死特征噪声 |
| 3 | 时间序列CV改手动分割+日期显示 | ✅ | 暴露分布漂移 |
| 4 | prepare_training_data.py 加 _resolve_name() | ⏳ | 从源头解决中文名问题 |
