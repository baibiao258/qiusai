# 预测系统全景分析 (2026-06-15)

## 核心结论

系统架构成熟稳健，主要瓶颈在**数据时效性**而非算法。最大问题: training_data_with_odds.json（含竞彩赔率的特征数据）仅 263 场，全部来自 2024-11，导致 v30/v33 含市场赔率的模型无法用 2025-2026 年数据训练。

## 模型架构速览

```
路由: Club(37d) → Intl(29/33d) → Poisson+Elo → Market Fallback
融合: DC(0.4) + XGBoost(0.6) — 熵权动态调节 ∈ [0.1, 0.9]
特征: 29d = b15 + gold5 + odds3 + form6
后处理: Draw Correction → Motivation Patch → Temperature Scaling → Friendly Discount → 365scores Adjuster
```

## 6 维度不足

### P0 (严重影响精度)

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 1 | training_data_with_odds.json 止于 2024-11 (263场) | v30/v33 模型缺近期有赔率训练 | 500 wanchang 批量回填 + retrain |
| 2 | 365scores 特征入模数据不足 | 39d 模型不可用 | 等积累到 200+ 样本 (约 6/28) |
| 3 | wanchang 被误判为不可抓 | 之前无法补历史比分 | ✅ 已修复 (curl+GBK) |

### P1 (显著优化)

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 4 | Temperature Scaling T=1.2 硬编码未重调 | 校准偏差大 | 网格搜索最优 T |
| 5 | 回填闭环断裂 (174预测0回填) | 无实时 drift 监控 | 检查 cron |
| 6 | 俱乐部赛 whitelist 仅 16 联赛 | 日职/韩职/北欧用国际赛模型 | 动态扩充 |
| 7 | v33 stage 特征 points_diff/rank_diff 是占位符 | 2/4 维是噪声 | 需计算真实值 |

### P2 (长期改进)

| # | 问题 | 建议 |
|---|------|------|
| 8 | _validate_feature_dims 只覆盖 hybrid 路径 | 扩展到 club/shadow/simple |
| 9 | HTFT 模型双路径不稳定 | 统一到 half_full_model |
| 10 | 友谊赛折扣在 _try_hybrid_predict 中缺失 | 从 predict_match.py 同步 |
| 11 | Motivation Patch 硬编码 0.85 乘数 | 数据驱动 |

## 当前数据状态

| 数据源 | 记录数 | 日期范围 | 状态 |
|--------|--------|---------|------|
| international_results.json | 49,409 | 1872~2026-06-12 | ✅ |
| training_data_with_odds.json | 263 | 2024 止 | ❌ 断档 |
| predictions_log.csv | 174 | 2026-06-08~14 | ✅ 需回填 |
| 500_history_backfill.csv | 0 | — | 待执行 |
| xgb_model_29.pkl (Brier) | — | val=0.1453 | ✅ 重训后 |
| football_games.csv | 566 | 6/5~6/7 | 积累中 |

## 优先级路线图

```
Week 1: 500 wanchang 回填 2025-01~06 → merge intl → retrain 全套模型
Week 2: Temperature 网格搜索 → backfill cron 修复 → 俱乐部 whitelist 扩充
Week 3: 365scores 入模 (6/28 前后达 200+) → 重新优化 Draw Correction
Ongoing: 每日 02:00 365scores 积累, 09:30+13:30 双次回填
```
