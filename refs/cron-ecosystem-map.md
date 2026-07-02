# 竞彩预测系统 Cron 生态系统 (15 jobs)

## 分类总览

### 一、数据采集层（原料）

| # | Job | ID | Schedule(UTC) | 脚本 | 数据源 | 模型作用 |
|---|-----|-----|--------------|------|--------|---------|
| 1 | 365scores每日数据收集 | 3fee9087ae2c | 02:00 | collect_365scores_daily.py | 365scores API | 特征工程输入(投票/趋势/FIFA排名) |
| 2 | 世界杯每日赔率拉取 | f22f1d2494f3 | 00:00 | fetch_worldcup_odds.py | The Odds API | 市场赔率40%权重混合 |
| 3 | form_state每日更新 | 7c1895c6d655 | 06:00 | update_form_state.py | football-data.org API | H2H交锋+近况form特征 |

### 二、预测输出层（成品）

| # | Job | ID | Schedule(UTC) | 脚本 | 数据源 | 模型作用 |
|---|-----|-----|--------------|------|--------|---------|
| 4 | 双色球分析报告 | a06bbb000385 | 03:00 | lottery-analysis skill | data.17500.cn | 独立彩票分析(与足球无关) |
| 5 | 竞彩足球分析报告 | 7eb1bcff2779 | 03:00 | daily_jczq.py | 500.com+football-data+365scores | 主输出:5玩法预测 |
| 6 | daily-jczq-alert | 3b404abedaf4 | 08:00 | daily_alert.sh→daily_alert.py | 当日预测+赔率 | 价值投注预警(EV/Kelly) |

### 三、回填层（校准燃料）

| # | Job | ID | Schedule(UTC) | 脚本 | 数据源 | 模型作用 |
|---|-----|-----|--------------|------|--------|---------|
| 7 | backfill-am | 6d912cb676ec | 02:00 | backfill_am.sh→backfill_results.py | results JSON+kaijiang+football-data | Brier Score计算 |
| 8 | backfill-pm | 571c46a2a622 | 05:30 | backfill_pm.sh→backfill_results.py | 同上 | 二次补充赛果 |

### 四、回测验证层（体检报告）

| # | Job | ID | Schedule(UTC) | 脚本 | 数据源 | 模型作用 |
|---|-----|-----|--------------|------|--------|---------|
| 9 | 竞彩自动回测 | 7163975dc922 | 07:00 | backtest_runner.sh→backtest_jczq.py | 500.com开奖页(curl) | 5玩法准确率监控 |
| 10 | daily_backtest_verify | 35ea8dbf7337 | 08:00 | backtest_pipeline.py --verify | predictions_log.csv | Brier/RPS/ROI校准验证 |
| 11 | 365scores定期回测 | e4139f534e3e | 周一08:00 | periodic_backtest_365scores.py | 365scores历史JSON | 验证365scores特征提升力 |

### 五、世界杯专项

| # | Job | ID | Schedule(UTC) | 脚本 | 数据源 | 模型作用 |
|---|-----|-----|--------------|------|--------|---------|
| 12 | 世界杯模型每日更新 | b2148e127b3a | 06:00 | wc_2026_final.py | 多源(DC+XGB+赔率+回测) | 冠军概率+MC模拟 |

### 六、监控与可视化

| # | Job | ID | Schedule(UTC) | 脚本 | 数据源 | 模型作用 |
|---|-----|-----|--------------|------|--------|---------|
| 13 | 365scores数据积累监控 | 796bb77aeee1 | 09:00 | monitor_365scores_data.py | 365scores目录JSON | 数据质量门禁 |
| 14 | daily_dashboard | b7dcf59ba96c | 08:30 | dashboard_generator.py | backtest_results+predictions_log | 可视化监控面板 |
| 15 | 友谊赛前向验证周报 | 863e77536f71 | 周一10:00 | forward_valid*.py | 回测数据 | 友谊赛场景诊断 |

## 关键诊断知识

### Agent-driven vs Pure-script 区分

**这是诊断cron失败的第一步。**

```
cron jobs.json 中 no_agent 字段:
  no_agent=False (🤖) → 需要LLM模型API → 模型配额耗尽时全部失败
  no_agent=True  (📜) → 纯脚本执行 → 不依赖LLM → 只受数据源/API影响
```

**诊断流程:**
1. `cat /root/.hermes/cron/jobs.json` → 检查 no_agent 标志
2. 手动运行底层脚本验证(排除模型依赖)
3. 如果手动运行成功但cron失败 → 模型API配额问题(429)
4. 如果手动也失败 → 数据源问题

### HTTP 429 Quota Exhausted 故障模式

**症状:** 所有🤖类cron同批次失败，📜类可能也报错(session创建消耗少量配额)
**根因:** Hermes模型API(mimo-v2.5/xiaomi provider)配额耗尽
**验证:** 手动运行任意脚本确认数据源正常
**恢复:** 等待配额重置，或切换provider

### Model Deprecation 故障模式 (2026-06-12)

**症状:** `RuntimeError: Error code: 400 - Not supported model <old_name>`
**根因:** Provider下线旧模型名(如mimo-v2.5-pro-ultraspeed→mimo-v2.5)
**修复:** (1) 更新config.yaml的model.default (2) 重启gateway (3) 重跑失败的cron
**注意:** config.yaml改动需gateway重启才生效；cron job中pinned model需单独update

### no_agent Script Path Resolution (2026-06-12)

**陷阱:** `no_agent=True` cron job的`script`字段是**文件名**(相对路径解析到`HERMES_HOME/scripts/`)，不是完整命令。写`python3 /root/backfill_results.py 2>&1`会报`Script not found`。
**修复:** 创建wrapper shell脚本:
```bash
#!/bin/bash
# /root/.hermes/scripts/backfill_am.sh
python3 /root/backfill_results.py 2>&1
```
然后设置`script: backfill_am.sh`。调度器自动识别.sh后缀用bash执行。
**已应用:** backfill_am.sh, backfill_pm.sh, daily_alert.sh
**详细文档:** hermes-agent skill → references/cron-script-path-resolution.md

### 手动验证命令

```bash
# 回测脚本
cd /root && bash /root/.hermes/scripts/backtest_runner.sh

# 赛果回填
cd /root && python3 /root/backfill_results.py

# 核心预测(需500.com数据)
cd /root && python3 /root/daily_jczq.py

# 赔率拉取
cd /root && python3 /root/.hermes/scripts/fetch_worldcup_odds.py
```
