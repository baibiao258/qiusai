# 2026-06-17 系统健康审计: 模型碎片化与准确率裂谷

## 触发场景

用户要求全面分析当前预测模型的架构、数据流、算法协同配合，以及不足和优化方向。

## 审计方法

4 层审计 (继承自 `references/2026-06-15-holistic-audit-method.md`):

1. **模型文件清单** — 列出所有 `.pkl` 文件，检查时间戳、大小、死特征比例
2. **predictions_log 统计** — 总记录、bet_action 分布、model_route 分布、回填率、Brier 覆盖
3. **训练数据概览** — 总样本数、赛事分布、spf_result 类型混检
4. **关键数据文件检查** — form_state、Poisson-Elo 先验、DC 模型属性完整性

## 定量发现

### 1. 模型碎片化 (9 XGB + 4 DC + 4 校准器)

| 模型文件 | 日期 | 维度 | 死特征率 | 状态 |
|----------|------|------|----------|------|
| xgb_model_nat.pkl | 06-15 | 11 | **0%** ✅ | 生产主推 |
| xgb_model_29.pkl | 06-17 | 29 | **0%** ✅ | active |
| xgb_model_33.pkl | 06-11 | 34 | **68%** ❌ | shadow |
| xgb_model_30.pkl | 06-14 | 30 | **63%** ❌ | shadow |
| xgb_model_28.pkl | 06-14 | 11 | — | 废弃 |
| xgb_model_17d.pkl | 06-15 | 17 | 0% | 冗余 |
| xgb_model_club.pkl | 06-08 | 37 | 11% | 未被路由触发 |
| xgb_model_simple.pkl | 06-09 | — | — | 废弃 |
| xgb_model_20_3.pkl | 06-02 | — | — | 废弃 |

DC 模型:
- `dc_model.pkl`: DixonColes, 712 队, 6月17日, 生产 ✅
- `dc_model_club.pkl`: 属性不完整(无 attack/defense/teams 标准属性), 6月8日, 未被使用

4 套校准器文件:
- `calibrators.pkl`, `calibrators_v2.pkl`, `calibrators_v3.pkl`, `calibrators_club.pkl`, `calibrators_nat.pkl`, `calibrators_simple.pkl`
- 代码已剥离 (2026-06-14), 但文件仍在磁盘上

### 2. 20pp 准确率裂谷 (生产 vs 回测)

| 指标 | 历史回测(600场) | 每日验证(34场) | 差距 |
|------|----------|----------|------|
| Acc | 64.5% | **44.1%** | **-20.4pp** |
| Brier | 0.4613 | **0.7373** | +0.276 |
| LogLoss | 0.7925 | **1.867** | +1.074 |

根因: 回测用 polished 特征管线含 market_implied + full form, 生产走 nat-11d 简化版。不是同一个模型。回测数据止于 2026-03-31, 生产数据是 2026-06-17 的世界杯。

### 3. 训练数据分布偏斜

| 数据源 | 总样本 | 国际赛 | 俱乐部 |
|--------|--------|--------|--------|
| training_data_with_odds.json | 2,436 | **2,436 (100%)** | 0 |
| thestats_training_data.json | 32,001 | ~416 (1.3%) | **31,585 (98.7%)** |

nat_11d 模型在纯国际赛数据上训练 → 世界杯表现尚可。
V29/V33 用 32K 俱乐部为主的数据训练 → 国际赛预测被稀释。

### 4. predictions_log 现状

| 指标 | 值 |
|------|-----|
| 总记录 | 18 |
| 已回填 | **0** (0%) |
| 有 Brier | **0** |
| RECOMMEND | 12 |
| WATCH | 6 |
| model_route 多样性 | 仅 `xgb_dc_nat_11d` + `market_fallback` 2种 |
| 365scores 覆盖 | **3/18** (16.7%) |

### 5. 数据流断点

- Lineup 缓存不存在 (`thestats_lineups_cache.json` 空)
- 疲劳度特征已编写 (`fatigue_features.py`) 但未入任何生产模型
- 高级特征缓存(13维 TheStats)状态未知
- Poisson-Elo 先验 32K 场仅用于 `_lookup_prior_elo/lambda` 回退, prior_poisson 路由从未触发

## 建议动作

### P0
1. 回填当前 18 场赛果 → 获取实际校准指标
2. 删除 V33/V30 死特征模型 → 从 `_FEATURE_REGISTRY` 移除 + 清除磁盘文件
3. 清理废弃 XGB 模型 → 只保留 nat_11d + v29 + club 共 3 个
4. 修 365scores 名称匹配 → 世界杯球队不匹配的根因

### P1
5. nat vs V29 A/B 对比 → 用已有 159 场 Brier 数据分出胜负
6. 修复俱乐部路由 → 芬兰联赛等入 form_state 白名单
7. 删除全部校准器文件 → 防误加载

### P2
8. 训练集按赛事类型分层 → 国际赛/俱乐部各训各的
9. 启用疲劳度特征
10. 扩展回测至 2026-06

## 参考

- 完整分析输出见本次对话的用户终稿
- 审计脚本: `_model_audit.py`, `_data_audit.py`, `_dc_inspect.py` (一次性脚本, 不保留)
