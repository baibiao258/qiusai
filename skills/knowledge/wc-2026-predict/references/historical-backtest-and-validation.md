# 历史回测与校验说明

## 数据源角色分离

| 文件 | 用途 | 来源 |
|------|------|------|
| `/root/data/predictions_log.csv` | 每日预测日志 + 赛后回填 | daily_jczq.py 写入, backfill_results.py 补充 actual_* |
| `/root/data/results/YYYY-MM-DD.json` | 500.com 开奖原始数据 | historical_kaijiang.py 每日 cron 写入 |
| `/root/data/backtest_results.json` | 回测指标累计归档 | backtest_pipeline.py 追加 |
| `/root/data/wc_completed_results.json` | WC 完赛结果累积库 | accumulate_results.py 写入 |

## 每日校验 (`--verify`) 工作流

### 命令
```bash
python3 /root/backtest_pipeline.py --verify
```

### 校验逻辑
1. 遍历 `predictions_log.csv` 所有行
2. 跳过 `checked=1` (已核验)
3. 跳过 `date >= today` (未来比赛)
4. 跳过 `actual_score` 为空 (无赛果无法验证)
5. 对满足条件的行: 解析比分 → 计算 Brier/RPS/LogLoss/Acc → 写入 `checked=1`
6. 若有新核验比赛 → 追加一条 `type="daily_verify"` 到 `backtest_results.json`

### 常见结果解读

**场景 A: "今日新增核验: 0 场"**
这是最常见的 cron 执行结果。原因:
- `predictions_log.csv` 中 157/168 行已核验 (历史累积)
- 剩余未核验行缺少 `actual_score` (赛果数据尚未回填)
- 回填滞后 1-3 天是正常现象 — backfill 依赖多源数据 (kaijiang → football-data.org → 365scores)，非实时
- 不表示 pipeline 出错

**排查步骤** (当认为应该有新核验但显示 0 场时):
```bash
# 1. 检查文件完整性
wc -l /root/data/predictions_log.csv
python3 -c "import csv; rows=list(csv.DictReader(open('/root/data/predictions_log.csv'))); print(f'total={len(rows)} checked={sum(1 for r in rows if r.get(\"checked\")==\"1\")} unchecked_past={sum(1 for r in rows if r.get(\"checked\")!=\"1\" and r.get(\"date\",\"\") < \"2026-06-29\")} have_score={sum(1 for r in rows if r.get(\"checked\")!=\"1\" and r.get(\"date\",\"\") < \"2026-06-29\" and r.get(\"actual_score\",\"\").strip())}')"

# 2. 确认 backtest_results.json 最近一条记录
python3 -c "import json; r=json.load(open('/root/data/backtest_results.json')); print(json.dumps(r[-1], indent=2))"

# 3. 检查脚本退出码
python3 /root/backtest_pipeline.py --verify; echo "EXIT=$?"
```

**场景 B: "今日新增核验: N 场 (N>0)"**
- 新结果被成功回填到 CSV 后自动校验
- 追加记录到 `backtest_results.json`
- 在报告中列出: Brier/RPS/LogLoss/Acc 以及逐场明细

**场景 C: 脚本输出被截断**
backtest_pipeline.py 会对每个缺少实际比分的比赛输出一行警告。当未回填比赛较多时 (10-30 行), 输出可达 76K 字符, 超过 terminal 截断阈值。
```bash
# 解决: 重定向到文件再读末尾摘要
python3 /root/backtest_pipeline.py --verify > /tmp/verify_out.txt 2>&1
# 检查是否有新增核验: 看文件尾部是否有 "核验结果" 标题
grep -E "核验|Brier|准确率|已保存" /tmp/verify_out.txt
# 检查 backtest_results.json 是否新增
wc -l /root/data/backtest_results.json
```

## backtest_results.json 格式

```json
[
  {
    "timestamp": "2026-06-27T08:05:51",
    "type": "daily_verify",
    "n_matches": 1,
    "brier": 0.1916,
    "rps": 0.071,
    "log_loss": 0.436,
    "accuracy": 1.0,
    "details": [
      {
        "code": "周四026", "home": "瑞士", "away": "波黑",
        "actual_hda": "H",
        "pred_h": 0.646, "pred_d": 0.221, "pred_a": 0.132,
        "brier": 0.1916, "rps": 0.071, "log_loss": 0.436, "accuracy": 1.0
      }
    ]
  },
  {
    "timestamp": "2026-06-08T13:25:32",
    "type": "historical_backtest",
    "n_matches": 150,
    "train_size": 350,
    "test_size": 150,
    "date_range": "2026-01-18 ~ 2026-03-31",
    "brier": 0.4919,
    "rps": 0.1503,
    "log_loss": 0.8418,
    "accuracy": 0.6333
  }
]
```

- `type=daily_verify`: 对已回填赛果的预测做事后核验 (小样本, 1-50 场)
- `type=historical_backtest`: 对历史国际赛做滚动回测 (大样本, 150-600 场)
- `accuracy` 基线: 33.3% (随机 3 分类), 当前 hybrid 模型 ~44-65%

## 历史滚动回测 (`--backtest`)

```bash
# 默认 500 场 (70% 训练, 30% 测试)
python3 /root/backtest_pipeline.py --backtest

# 指定场数
python3 /root/backtest_pipeline.py --backtest --n 600
```

回测使用已训练的 DC + XGB + Elo 模型, 按时间顺序滚动预测。详细结果写入 `/root/data/backtest_details.json`。

## 赛果回填管线

赛果回填由 `/root/backfill_results.py` 处理, 每天 cron 两次 (UTC 01:30 + 05:30):

```bash
# 查看回填统计 + Brier 分析
python3 /root/backfill_results.py --stats

# 查看每日趋势 (Brier drift + 联赛分级)
python3 /root/backfill_results.py --report

# 强制回填指定日期范围
python3 /root/backfill_results.py --from-date 2026-06-27 --to-date 2026-06-28
```

回填流程: kaijiang JSON → kaijiang CSV → football-data.org → 检查冲突 → 标记 filled -> 校验

### WC 完赛结果累积

独立于 predictions_log 的 WC 赛果累积管线:

```bash
python3 /root/wc_2026_upgrade/accumulate_results.py
```
结果存入 `/root/data/wc_completed_results.json`。不自动进入 training_data_with_odds.json — 需额外运行 `scripts/check_training_gap.py`。

## 故障排查指南

| 现象 | 可能原因 | 操作 |
|------|---------|------|
| verify 输出大量"缺少实际比分" | 赛果尚未回填 | 正常, 检查 backfill cron 是否运行 |
| backtest_results.json 未更新 | 无新核验比赛 | 运行排查脚本 stats 命令 |
| 脚本输出被截断 | 警告行过多 | 重定向到文件后读取 |
| 某场比赛已结束但 actual_score 为空 | backfill 未覆盖 | 运行 `backfill_results.py --stats` |
