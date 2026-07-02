# 预测系统架构全面审查 (2026-06-08)

## 系统架构总览

两套管线并行：WC 2026 冠军模拟 + Daily JCZQ 每日竞彩

```
wc_2026_phase1.py  (Dixon-Coles, Elo, 数据加载, TEAMS_2026)
       │
       ├──→ wc_2026_final.py        (WC 2026 冠军模拟, 200K MC, Isotonic校准)
       │
       ├──→ daily_jczq.py          (每日竞彩, 500.com + 365scores, 市场校准器)
       │
       └──→ wc_10edition_backtest.py (10届回测, 1986-2022, 604场)
```

共享层: Dixon-Coles → 29维特征 → XGBoost → Hybrid融合
差异化层: WC=Isotonic校准+MC模拟+市场凸组合, JCZQ=市场校准器+5玩法赔率

### 新增支撑模块 (2026-06-08)

```
feature_helper.py      → H2H缓存(7481对) + 12场form缓存(336队), 修复gold特征train-serve skew
scores365_adjuster.py  → 365scores后验调整器: 投票/趋势/人气3信号融合, ±5pp上限
backtest_pipeline.py   → 回测管线: --verify 核验已有预测, --backtest N 历史回测
update_form_state.py   → form_state每日自动更新 + H2H/form_12缓存重建
```

## 数据流

| 数据源 | 用途 | 大小 | 状态 |
|--------|------|------|------|
| international_results.json | 国际比赛结果 (1872+) | 7.1MB | ✅ |
| odds_data.json | 500.com赔率 | 430KB | ✅ |
| form_state.json | 球队近期状态 | 61KB | ✅ |
| h2h_cache.json | 预计算H2H特征 | — | ✅ 新增 |
| form_12.json | 预计算12场form | — | ✅ 新增 |
| predictions_log.csv | 预测记录 | 7.7KB (27条) | ⚠️ actual全空 |
| theodds_api_data.json | 冠军市场赔率 | — | ❌ 文件不存在 |

## Baseline 回测指标 (2026-06-08)

| 指标 | 值 | 随机基线 | 完美值 |
|------|-----|---------|--------|
| Brier Score | 0.4613 | 0.667 | 0.0 |
| RPS | 0.1475 | 0.333 | 0.0 |
| Log Loss | 0.7925 | 1.099 | 0.0 |
| 准确率 | 64.5% | 33.3% | 100% |
| 主胜准确率 | 80.5% | — | — |
| 平局准确率 | 14.4% | — | — |
| 客胜准确率 | 78.2% | — | — |

测试集: 600场 (2025-09-09 ~ 2026-03-31), 训练集1400场

## 关键改进记录 (2026-06-08)

### ✅ 已修复

1. **Gold特征train-serve skew** — predict_match.py和daily_jczq.py的gold特征从占位符`[0.0, 1, 0, 0.0, 0.0]`改为feature_helper.py真实值(H2H净胜球+12场form差异)。平局准确率从4.7%提升到14.4%。

2. **365scores后验调整器** — scores365_adjuster.py集成到daily_jczq.py的predict_match_wrapper。投票/趋势/人气3信号加权融合，±5pp上限。数据到位后自动生效。

3. **form_state自动更新** — update_form_state.py + cron每日06:00运行，从football-data.org和international_results.json拉取昨日结果追加到form_state.json，并重建H2H和form_12缓存。

4. **回测管线** — backtest_pipeline.py: --verify 核验predictions_log中已结束比赛，--backtest N 历史滚动回测。输出Brier/RPS/LogLoss/准确率/校准度。cron每日08:00自动运行验证。

### 🔴 待修复 (仍存在)

1. **缺失3个导入模块** — mc_uncertainty_helper, mc_market_weight_helper, half_full_model 已移至wc_2026_upgrade/但wc_2026_final.py导入路径仍可能找不到。
2. **预测日志无反馈闭环** — actual_*列全部为空，回测管线已就绪但无数据可核验。
3. **每日管线缺少Isotonic校准** — 已有calibrated_xgb.pkl但daily_jczq.py未加载使用。

### 🟡 待优化

4. **联赛模型回退质量差** — domestic联赛走predict_match_legacy()纯泊松+Elo，hybrid DC+XGB仅覆盖国际赛球队。
5. **市场权重固定40%** — mc_market_weight_helper.py已有动态逻辑但未在daily_jczq.py中使用。
6. **半全场模型过于简单** — half_full_model.py用r_ht=0.45固定比率推算，无独立训练。
7. **友谊赛折扣粗糙** — 30%拉向均匀分布是硬编码，未区分主力/替补出战。
8. **Elo未覆盖弱队默认1500** — 佛得角、约旦等非主流队起点不合理。
9. **进球截断MAX_GOALS=6** — 极端比分(7-0)被截断。
10. **无伤停/阵容数据集成** — lineup_risk.py存在但未接入预测流程。

## 改进优先级 (更新)

### P0 — 已完成 ✅
1. ~~补全缺失模块~~ (部分完成, 3模块已移至wc_2026_upgrade/)
2. ~~修复MC缓存form特征~~ ✅ (wc_2026_final.py已修复)
3. ~~Gold特征补全~~ ✅ (feature_helper.py)

### P1 — 校准精度
4. 每日管线引入Isotonic校准 (calibrated_xgb.pkl已存在, 需接入daily_jczq.py)
5. ~~独立校准集（60/20/20分割）~~ ✅ (wc_2026_final.py已实现)
6. Platt Scaling退路

### P2 — 验证与监控
7. ~~自动回填actual结果~~ (backtest_pipeline.py --verify已就绪, 等数据)
8. ~~集成Brier分解到每日评估~~ ✅ (daily_jczq.py已有quick_validate)
9. Bootstrap置信区间

### P3 — 架构加固
10. 统一Config（/root/config/prediction.yaml）
11. 动态融合权重调优（Optuna搜索阈值）
12. 熔断器（500.com失败3次后降级到纯模型）✅ (已实现)
13. 第三名分配约束满足算法
14. 模型版本Git跟踪
15. 联赛模型升级 (DC+XGB覆盖英超/德甲等)
16. 365scores特征入模 (需积累历史数据后重训练XGB)

## Cron Jobs

| 名称 | 时间 | 功能 | 状态 |
|------|------|------|------|
| form_state_daily_update | 每日06:00 | 更新form + 重建H2H/form_12缓存 | ✅ 新增 |
| daily_backtest_verify | 每日08:00 | 核验predictions_log, 计算Brier/RPS | ✅ 新增 |
| 365scores收集 | 每日02:00 | 抓取365scores投票/趋势数据 | ✅ 已有 |
| WC冠军概率更新 | 每日06:00 UTC | wc_2026_final.py全管线 | ✅ 已有 |
