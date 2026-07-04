# 概率校准偏误诊断与修复方法论

> 本文档记录 2026-07-04 三轮修复中沉淀的方法论，用于指导未来类似偏误的诊断与修正。
> 相关 commit: `7ed2ba4`, `4b1a831`, `13182e5`

---

## 1. 发现阶段：定位偏误区间

### 置信度分桶

不要只看整体准确率（如 HDA 57.7%），整体数字会被正常区间稀释，掩盖局部区间的高偏误。

```python
# 按 pred_h 分桶，对比准确率和实际平局率
buckets = [(30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(90,100)]
for lo, hi in buckets:
    bucket = [r for r in verified if lo <= max(pred_h,pred_d,pred_a)*100 < hi]
    acc = sum(1 for r in bucket if pred_correct(r)) / len(bucket)
    print(f"{lo}-{hi}%: n={len(bucket)}  acc={acc:.1%}")
```

关键信号：**高置信度桶的准确率低于低置信度桶**（如 80-90% 桶 42.1% < 整体 57.7%）。

### 二项检验确认显著性

对可疑区间用 scipy.stats.binomtest 验证：

```python
from scipy.stats import binomtest
# 80-90% 桶期望至少 80% 准确，实际 8/19
p_value = binomtest(8, 19, 0.80, alternative='less').pvalue  # p ≈ 0.0003
```

### 滑动窗口可视化

对于连续型预测概率，滑动窗口比固定分桶更能揭示平滑趋势：

```python
# w=15 的滑动窗口扫描全部 0-100% 范围
pts.sort(key=lambda p: p['pred_h'])
for i in range(len(pts)):
    chunk = pts[max(0,i-7):min(len(pts),i+8)]
    actual_draw = sum(p['is_draw'] for p in chunk) / len(chunk)
    avg_pred_d = sum(p['pred_d'] for p in chunk) / len(chunk)
    gap = actual_draw - avg_pred_d
```

滑动窗口曾揭示：pred_h 低端 gap 为负（模型多估平局），高端 gap 急剧转正（模型低估平局），两个方向的偏误在中间抵消，使得整体 gap 看似正常。

---

## 2. 归因阶段：区分三类根因

### 类型 A：标签/评估层错配

**特征**：评估指标系统性偏低，但分布是纯随机噪声，没有任何置信度方向性。

**案例**：`actual_rq_result` 列被写成了 H/D/A 字母而非 让胜/让平/让负，导致 RQ 准确率被钉在 8%（接近随机匹配的 11.1% = 1/9 命中）。修复后回到 37.9%。

**鉴别方法**：随机抽 5 行，人工心算 actual vs pred 的标签是否语义一致。

### 类型 B：参数方向错误

**特征**：数学公式正确但参数符号或作用域错了，影响幅度通常在 1-5 个百分点。

**案例**：DC rho 修正对 handicap≠0 反向生效（让平概率被额外压低 1.34%），因为 tau 的四格修正坐标与让平判定区间不重合。

**鉴别方法**：对照测试——`rho=0` vs `rho=-0.13` 的 A/B 对比。如果 rho 对让平概率的贡献是负的，就是符号用反了。

**数学上限验证**：将参数推到理论极限（rho→-0.25），看最大可解释缺口。如果理论极限只能解释 2pp 而实际缺口 42pp → 工具用错了，见类型 C。

### 类型 C：行为学/结构性现象

**特征**：缺口巨大（10pp+），且模型结构本质上无法捕捉。

**案例**：pred_h≥75% 时平局被低估 +21.8pp，80-85% 区间甚至 +42pp。根因是弱队主动摆大巴的战术选择，Poisson × DC 模型只能描述"随机波动"，无法建模"战略性放弃控球求平"。

**鉴别方法**：（类型 B 的数学上限验证）当参数推到理论极限仍无法解释大部分缺口时，说明不是参数问题，是模型架构问题。

---

## 3. 修正阶段：后验校正 vs 改底层模型

### 原则

**不改底层 Poisson/DC 模型**。原因：
- 底层模型是所有玩法（SPF/RQ/HTFT/Goals/Score）的共享基础，改一个地方可能破坏其他玩法的校准
- 行为学现象（如摆大巴）无法通过调整统计分布的参数来建模
- 后验层可以独立开启/关闭/版本管理，不影响其他玩法

### 校正层接口模式

```python
# bundle_builder.py 中的模式
# 1. 在 0-1 尺度操作（先于 *100 转换）
# 2. 只调整 Δ，不重算模型
# 3. 从 pred_h 和 pred_a 按比例扣减，保持总和=1.0
# 4. 用 model_note 标记版本

if pred_h >= 0.75 and pred_h < 0.87:
    delta = _draw_correction_delta(pred_h)
    if delta > 0:
        pd_adj = pred_d + delta
        scale = (1.0 - pd_adj) / (1.0 - pred_d)
        pred_h *= scale
        pred_a *= scale
        pred_d = pd_adj
        model_note_postfix = '+draw_postcal_v1'
```

### 分段拟合方法

数据量较小时（N<100），用分段常数 + 线性插值而非多项式拟合，避免过拟合：

```python
def _draw_correction_delta(pred_h):
    # 4 个 knot 的折线，(pred_h, Δ)
    knots = [(0.75, 0.356), (0.78, 0.0), (0.81, 0.422), (0.87, 0.0)]
    if pred_h < knots[0][0] or pred_h >= knots[-1][0]:
        return 0.0
    for i in range(len(knots)-1):
        x1, y1 = knots[i]
        x2, y2 = knots[i+1]
        if x1 <= pred_h < x2:
            t = (pred_h - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
```

### knot 调整纪律

1. 初始拟合使用全量历史数据
2. **绝不基于同一批数据微调**（会产生 +42pp 看到就改到 +35pp 的过拟合）
3. 待积累至少 30 场新的 out-of-sample 数据后再评估是否需要更新 knots

---

## 4. 追溯阶段：版本标记

每次校正必须有版本标识，写入 `model_note` 字段，确保回测可以区分修正前后的预测输出。

### 规范

| 版本 tag | 含义 | 触发条件 |
|:--|:--|:--|
| `+draw_postcal_v1` | 平局概率后验校正 v1 | pred_h ∈ [0.75, 0.87) |
| `+疲劳度调整` | 疲劳度校正 | 已有逻辑（独立于本方法论） |

### 在回测中区分

```python
# predictions_log.csv 的 model_note 列
# 未校正: "xgb_dc_nat_11d"
# 已校正: "xgb_dc_nat_11d+draw_postcal_v1"
```

按 `model_note` 过滤即可比较校正前后的校准曲线。

---

## 案例参考：2026-07-04 draw_postcal_v1

### 发现

| 指标 | 数值 |
|:--|:--|
| 嫌疑区间 | pred_h 80-85% |
| 样本量 | 48 场（pred_h≥75%） |
| 实际平局率 | 35.4% |
| 模型平均 pred_d | 13.6% |
| 缺口 | +21.8pp（80-85% 桶 +42pp） |
| rho 理论极限 | +2pp（不够用） |

### 修正效果（回代）

| 指标 | 修正前 | 修正后 |
|:--|:--:|:--:|
| avg_pred_d | 13.6% | 31.0% |
| 缺口 | +21.8pp | +4.4pp |
| 缩减幅度 | — | 80% |

### 待验证

- 4b1a831 之后的新比赛 out-of-sample 表现
- 三个桶（75-80%, 80-85%, 85-90%）的独立准确率
- 累计至少 30 场新样本后评估是否需要重拟合 knots

---

*文档版本: 2026-07-04 | 对应 commit: 13182e5*
