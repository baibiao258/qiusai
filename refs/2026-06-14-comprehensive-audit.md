# 2026-06-14 全面系统审计

## 审计范围

对预测系统进行全面4层诊断，覆盖模型架构、数据流、算法配合。

## 发现的问题 (按优先级)

### P0: 训练数据标签污染 (已修复)
- **问题**: `training_data_with_odds.json` 中 131/491 条 `spf_result` 是 int 类型（非 str）
- **影响**: `result == '3'` 不匹配 int，29 条标签错误（7.3%）
- **修复**: 所有训练脚本统一 `str(m['spf_result'])`
- **效果**: nat 模型验证准确率 64.4% → 75.4% (+11pp)

### P0: 平局概率灭绝 (已修复)
- **问题**: `calibrated_predictor.py` 的 `_blend_with_market()` 中 `elo_arr = [elo_h, 0, 1-elo_h]` 平局硬编码为 0
- **影响**: 所有比赛平局概率被系统性压低到 0%
- **修复**: 用 Elo 差估算平局概率 `elo_draw = 0.25 * (1 - abs(2*elo_h - 1))`
- **效果**: 荷兰vs日本平局 5.6% → 13.2%

### P0: 双管线模型不一致 (已修复)
- **问题**: daily_jczq.py 用 xgb_model_29 (29维)，calibrated_predictor.py 用 xgb_model_nat (11维)
- **修复**: 两管线统一到 nat 模型 (11维)

### P1: 500.com 连续 6 天熔断 (待排查)
- **日期**: 6/9~6/14 每天至少 1 次抓取失败
- **日志**: `/root/data/500breaker.log`
- **兜底**: `_load_fallback_odds()` 用 `odds_history.json` 兜底

### P1: 俱乐部路径路由追踪缺失
- **问题**: `model_route` 字段 170 行为空
- **原因**: `build_prediction_bundle()` 返回值无 `'model'` 键

### P2: 训练数据不足
- **现状**: 491 条，有效 330 条（英文明 + DC 覆盖）
- **目标**: 扩充到 1000+ 条
- **方法**: The Odds API 赛后自动拉取已完成赛果

### P2: bet_action 区分度不足
- **问题**: RECOMMEND/WATCH/WATCH_FRIENDLY 命中率相同 (~40%)
- **建议**: 基于 Brier Score 设差异化阈值

## 模型架构评估

### nat 模型 (11维) 特征构成
```
[elo_diff, lam_h, lam_a, lam_diff, lam_ratio, dc_a, dc_d, dc_h, op_h, op_a, market_implied]
```

### 优势
- 纯结构特征，无 train-serve skew
- 验证准确率 75.4%，Brier 0.1339
- LogLoss 0.819

### 劣势
- 训练数据仅 395 条（纯国际比赛英文名）
- 无 form/gold/h2h 特征
- 俱乐部比赛覆盖不足

### 建议
1. 短期: 统一用 nat 模型为主，退役 29/30 维模型
2. 中期: 扩充训练数据到 1000+ 条
3. 长期: 重新评估 form/gold 特征的加入

## 算法配合分析

### Dixon-Coles (DC)
- 作用: 提供基础概率分布
- 优势: 数据稀缺时仍有效
- 劣势: 独立 Poisson 假设低估平局
- 改进: Draw Correction Layer 已植入

### XGBoost (XGB)
- 作用: 用特征工程捕捉非线性关系
- 优势: 市场赔率特征重要性最高 (#1, 15.32%)
- 劣势: 18 个死特征导致模型臃肿
- 改进: 统一到 11 维干净特征

### Elo 评分
- 作用: 提供两队实力对比
- 优势: 简单有效，可解释性强
- 劣势: 不考虑近期状态
- 改进: 已加入 form 特征补偿

### 市场赔率
- 作用: 提供市场共识
- 优势: 单一最强预测因子
- 劣势: 需要实时数据
- 改进: 动态市场权重融合 (10%-42%)

### 融合方式
- 熵基动态权重 (α=0.30, β=0.50)
- Draw Correction Layer (平局<15%时补偿)
- Isotonic 校准器已剥离

## 下一步行动

1. 验证 Germany vs Curaçao SPF 修复效果
2. 排查 500.com 熔断原因
3. 扩充训练数据
4. 优化 bet_action 区分度
