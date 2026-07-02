# 系统全景分析 (2026-06-14)

## 数据源状态

| 数据源 | 状态 | 问题 |
|--------|------|------|
| **500.com** | ✅ 每日稳定 | 单点依赖, odds_history.json (17场) 熔断兜底 |
| **football-data.org** | ✅ 24h 缓存 | Tier One 10次/min, 无备选 |
| **365scores 投票/FIFA/趋势** | 🔴 闲置 | 10维特征在 scores365_adjuster.py 做固定 ±5pp, 模型学不到非线性 |
| **365scores CSV积累** | ✅ 已修好 | SID=1 过滤, football_games.csv 纯足球 |
| **historical_kaijiang.csv** | ✅ 3248场 | 只覆盖已开奖 |
| **predictions_log.csv** | ❌ 无回填 | 158行无 actual_result → 无法做 train-serve skew 检测 |

## 核心模型栈

| 组件 | 位置 | 评价 |
|------|------|------|
| XGBoost v29 | 主模型 | Brier=0.2053 |
| XGBoost v30/33 | 影子 | 多版本共存, 管理成本高 |
| Elo Rating | daily_jczq.py:1176 | 24行, K=32固定, decay=180天固定 |
| DC Model | 独立 | 后验patch |
| HTFT Model | half_full_model.py | 球队级 r_ht |
| Calibrators | 多个.pkl | Isotonic→Temperature 已切换 |

## 6 大结构性缺陷

### 1. 365scores 特征闲置 (🔴高, 🟢易修)
10维特征 `[vote_h/d/a, vote_count_log, pop_rank_diff, trend_win_rate_diff, fifa_rank_diff]` 跑在 ±5pp 固定规则里, 没进 XGB 特征向量。修复收益: 预期 Brier 降 2-5pp。

### 2. 无结果反馈闭环 (🔴高, 🟢易修)
`predictions_log.csv` 158行, actual_* 全空 → Brier 不可追踪 → 无法判断模型 drift → 无法评估优化效果。

### 3. Elo 过于原始 (🟡中, 🟡中)
- K=32 不分联赛
- 180天半衰期统一
- 无比分权重 (净胜球)
- 无主场优势独立

### 4. 4 层后处理叠加 (🟡中, 🔴难修)
`XGB → Temperature Scaling → Draw Correction → Friendly Discount → scores365_adjuster`
每层修上层的偏差。理想: 1个校准就够了。

### 5. 无交叉验证 (🟡中, 🟢易修)
模型从 6/12 起没重训过。train() 函数仅 24 行, 只算泊松+Elo。

### 6. 特征版本碎片化 (🟢低, 🟢易修)
v29/v30/v33/club_v37 四个版本用 _FEATURE_REGISTRY 管理, 复杂度有但收益不明确。

## 建议优先级

| 优先级 | 任务 | 收益 | 难度 |
|--------|------|------|------|
| P0 | 365scores 10维入模 (准备已完成) | Brier -2~5pp | 低 |
| P0 | backfill_results.py 结果闭环 | 迭代基础 | 低 |
| P1 | XGB 定期重训管线 | 持续优化 | 中 |
| P1 | Elo 升级 (比分加权+K自适应+主场) | 联赛精度 | 中 |
| P2 | 后处理层简化 | 维护成本 | 高 |
| P2 | 多数据源冗余 (备选赔率) | 运维安全 | 高 |
