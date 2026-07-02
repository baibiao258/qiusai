# 训练数据审计 (2026-06-09 检查)

## 模型文件时间戳

| 文件 | 最后修改 | 说明 |
|------|---------|------|
| `/root/data/calibrators.pkl` | 2026-06-10 06:08 | 每天预测时加载/更新 |
| `/root/data/xgb_model_30.pkl` | 2026-06-09 10:24 | 最近一次重训 |
| `/root/data/dc_model.pkl` | 2026-06-10 06:08 | 每天预测时更新 |
| `/root/data/xgb_model_29.pkl` | — | daily_jczq.py 加载的预测模型 |

## 训练数据范围

`training_data_with_odds.json` (XGB+calibrators 的训练数据):
- **263 场**
- **日期范围**: 2024-01-13 → 2024-11-15
- **数据来源**: historical_kaijiang.py (500.com开奖页) 抓取的3248场中筛选出的带市场赔率且有特征匹配的场次

## 风险

训练数据不包含 2025/2026 年的比赛。2026-06 的友谊赛预测时，模型只能用2024年的规律外推。

## 更新流程

重训练命令:
```bash
python3 /root/wc_2026_upgrade/prepare_training_data.py    # 数据准备
python3 /root/wc_2026_upgrade/retrain_xgb_with_odds.py    # XGB+校准器重训
```

回测验证:
```bash
python3 /root/backtest_pipeline.py                        # 回测600场
```
