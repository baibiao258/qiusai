# 回测方法论 — 合成赔率陷阱与正确做法 (2026-06-08)

## 核心原则

**回测必须使用真实历史赔率，永远不要用自己的模型概率生成合成赔率。**

## 合成赔率陷阱 (Synthetic Odds Trap)

### 错误路径
```
DC模型概率 → +overround(1.10) → 合成市场赔率 → EV计算 → Kelly下注 → 虚假ROI
```

### 为什么是循环论证
1. DC模型输出P(home), P(draw), P(away)
2. 合成赔率 = 1 / (P × overround)
3. 用合成赔率计算EV = P × (odds-1) - (1-P)
4. 由于 odds = 1/(P×overround)，代入后 EV ≈ P × (1/(P×overround) - 1) - (1-P)
5. 当 overround > 1 时，EV 始终 > 0（对所有选项！）
6. 结果：看似"有价值"的下注遍地都是，ROI 曲线完美上升

### DC均匀分布退化加剧问题
- DC对未知队伍（国际友谊赛）默认输出 33.3%/33.3%/33.3%
- 合成赔率 = 1/(0.333 × 1.10) ≈ 2.70/2.70/2.70（所有选项等价）
- 任何真实市场赔率偏离2.70都被误判为"EV正"
- 实际上模型对这些比赛零预测能力

### 检测方法
```python
# 检查DC是否在均匀分布模式
dc_uniform = (abs(dc_h - 1/3) < 0.02 and abs(dc_d - 1/3) < 0.02 and abs(dc_a - 1/3) < 0.02)
if dc_uniform:
    print("⚠️ DC退化为均匀分布, EV计算无参考价值")
```

## 正确的回测方法

### 1. 使用真实历史赔率
- 500.com历史数据（如果可获取）
- The Odds API历史端点（需付费）
- Bet365历史赔率数据集（如football-data.co.uk）

### 2. DC-Only实时验证
用真实市场赔率验证DC模型：
```python
# dc_real_odds_test.py 已实现
# 对比: DC概率 vs 500.com真实赔率 → 计算真实EV
# 只在DC有训练数据的俱乐部赛事上有意义
```

### 3. Kelly策略回测
```python
# backtest_kelly.py 已实现
# 三种方案对比:
# A: Quarter-Kelly, EV≥5%, 单场上限5%, 日上限15%
# B: Half-Kelly, EV≥3%, 单场上限8%, 日上限20%
# C: Quarter-Kelly, EV≥8%, 单场上限3%, 日上限10%
```

## 文件位置
- `/root/backtest_kelly.py` — Kelly策略回测管线（三方案对比）
- `/root/dc_real_odds_test.py` — DC vs 真实赔率诊断工具
- `/root/daily_alert.py` — 每日预警脚本（提取价值投注汇总）

## 历史赔率数据
- `data/odds_data.json` — 2195场Bet365赔率（英格兰第五级别联赛，与俱乐部数据不重叠）
- `data/today_500_odds.json` — 今日500.com真实赔率（SPF/让球/比分/总进球/半全场）
