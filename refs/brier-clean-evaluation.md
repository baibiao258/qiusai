# Brier 评估清洗方法 (CLEAN)

## 为什么需要 CLEAN

predictions_log.csv 存在两类系统性污染：
1. **假预测 (0%)**: market_fallback 在无赔率时返回 `[0,0,0]`，Brier=0.3333 的无意义惩罚
2. **重复行**: 同一场比赛在不同日期被预测，但 match_date 不同时是合法不同场次（如背靠背赛程）

## CLEAN 清洗流水线

`evaluate_brier.py` 的 `clean()` 函数执行两层清洗：

### 去伪 (0% 假预测)

```python
# pred_h/pred_d/pred_a 全部为 0 → 跳过
if pred_h == 0.0 and pred_d == 0.0 and pred_a == 0.0:
    continue
```

### 去重 (保留最新预测)

```python
key = (home_cn, away_cn, match_date)  # 有 match_date 时
key = (home_cn, away_cn)               # 空 match_date 时兜底
# 保留 date 最大的记录
```

**关键**: 去重键包含 match_date。同一对球队在不同日期的比赛是独立样本（如小组赛背靠背），不应去重。

## 三版本对比

evaluate_brier.py 输出三个版本:

| 版本 | 过滤 | 用途 |
|------|------|------|
| RAW | 无 | 历史兼容, 与旧数据可比 |
| NO-ZERO | 去伪 | 看假预测的影响量 |
| CLEAN | 去伪+去重 | **真实系统基准** |

## 校准曲线解读

六档置信区间，重点关注 `[80-100%)` 档偏差：

| 偏差幅度 | 判断 | 行动 |
|----------|------|------|
| < ±5pp | 良好 | 无需干预 |
| ±5~15pp | 轻微 | 监控 |
| ±15~30pp | 中度 | 需要校准 |
| > ±30pp | **严重** | 立即校准 (如 xgb 80-100% 档 -49.3%) |

## Brier 四版本对比脚本

```bash
python3 -c "
import pandas as pd, numpy as np
df = pd.read_csv('/root/data/predictions_log.csv')
df['brier_spf'] = pd.to_numeric(df['brier_spf'], errors='coerce')
br = df[df['brier_spf'].notna()]

# RAW
print(f'RAW:       Brier={br[\"brier_spf\"].mean():.4f} n={len(br)}')

# NO-ZERO (filter 0% pred)
for c in ['pred_h','pred_d','pred_a']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
valid = ~((br['pred_h']==0)&(br['pred_d']==0)&(br['pred_a']==0))
print(f'NO-ZERO:   Brier={br[valid][\"brier_spf\"].mean():.4f} n={len(br[valid])} ✂{(~valid).sum()}')

# DEDUP by (home, away, match_date)
br_cl = br[valid].drop_duplicates(subset=['home_cn','away_cn','match_date'], keep='last')
print(f'CLEAN:     Brier={br_cl[\"brier_spf\"].mean():.4f} n={len(br_cl)}')
"
```

## xgb 校准效果 what-if 回测

```python
from daily_jczq import _calibrate_xgb_probs
# 遍历 CSV 的 xgb 行, 用校准后概率重新计算 Brier
# 结果: xgb Brier 0.2541 → 0.2393 (-5.8%), Acc 50% 保持不变
```

## 报警阈值

| 指标 | 值 | 意义 |
|------|-----|------|
| Brier > 0.25 | 过自信惩罚 | 检查校准器 |
| Acc < 43% | 方向性偏差 | 检查模型 |
| `[80-100%)` 偏差 > -30pp | 严重过自信 | 启用盖帽 |
| 平局预测 < 20% of 实际平局 | 平局盲区 | 启用反哺 |
| N < 30 | 样本量不足 | 标记 provisional |
