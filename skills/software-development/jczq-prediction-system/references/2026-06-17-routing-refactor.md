# 2026-06-17: 国际赛路由重构 + 模型优先级 + 状态码升级

## 背景

原 `_try_hybrid_predict()` 双轨隔离: 国际赛走 DC+Pinnacle, 跳过 XGBoost。世界杯期间国际赛占 100%, 等于主动废弃了 24维 XGBoost 特征(Form/H2H/赛事分类)。

## 改动概要

修改文件: `/root/daily_jczq.py` (line 779→880 段)

### 1. 统一混合路由

**旧**: `if is_intl: DC + Pinnacle; else: build_feat→XGB→DC+XGB`

**新**: 共享 46维特征构建 + XGBoost推理 → 动态 DC+XGB融合 → Pinnacle微调(仅国际赛)

### 2. 模型优先级 (关键发现)

| 模型 | 维度 | 优先级 | 20场WC Acc | 结论 |
|------|------|--------|-----------|------|
| nat_11d | 11 | **1** | **90%** | 6月15日重训, 32K数据, 纯强度特征最稳健 |
| v33_shadow | 34 | 2 | 60% | 含stage_feat但过拟合, 比纯DC还差 |
| v30_shadow | 30 | 3 | — | 最旧 |

**教训**: V33(34维) 高维特征在 WC 样本外场景引入过拟合噪声。不要因某个 XGB 版本表现差就切断整个 XGB 管线——应逐版本测试。

### 3. Pinnacle 市场校正层

```
DC+XGB融合 → 检查 divergence > 15% → 15%权重Pinnacle微调
```

权重从 30% 降至 15% (验证发现 30% 过度扭曲)。

### 4. 维度注册表 Bug 修复

`xgb_model_33.pkl` 实际 `n_features_in_=34` 但注册表写 33。修复: line 547 `dims: 33→34`。

### 5. 状态码升级

新增:
- `DATA_INSUFFICIENT` — form_state 缺失任一队数据
- `PREDICTION_STALE` — 模型文件最新 mtime >7天
- 10个状态码全部加上中文标注 (BET_ACTION_LABELS 字典)

改:
- `SKIP_DATA` → `DATA_INSUFFICIENT` (line 930)
- `compute_bet_action` 新增 Rule 6 (模型过时检查)

终端输出:
```
旧: bet_action: WATCH_NO_ODDS
新: 📋 bet_action: 有概率无赔率[500.com熔断]
```

## 验证

```bash
# 语法
python3 -c "import ast; ast.parse(open('daily_jczq.py').read()); print('OK')"

# 路由验证
python3 -c "
from daily_jczq import _try_hybrid_predict
p = _try_hybrid_predict('Portugal', 'DR Congo', 'World Cup', is_intl=True)
print(p['model'])  # xgb_dc_nat_11d
"

# 状态码验证
python3 -c "
from daily_jczq import compute_bet_action, _check_model_staleness
stale, f, age = _check_model_staleness()
print(f'stale={stale}, age={age}d')
print(compute_bet_action('World Cup', 'xgb_dc_nat_11d', None, [], 0, {}))  # RECOMMEND
"

# CSV 验证
python3 -c "
import csv
with open('/root/data/predictions_log.csv') as f:
    rows = list(csv.DictReader(f))
from collections import Counter
print(Counter(r.get('bet_action','') for r in rows))
"
```
