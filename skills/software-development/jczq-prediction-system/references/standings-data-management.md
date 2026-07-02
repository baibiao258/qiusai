# 积分榜数据管理

## 数据来源
- **API**: football-data.org v4 `/competitions/{code}/standings`
- **脚本**: `/root/update_tournament_state.py`
- **缓存**: `/root/data/tournament_api_cache.json` (24小时有效)

## 存储文件

### `/root/data/tournament_state.json` (世界杯专用)
```json
{
  "球队名": {
    "home_group_points": 3,    // 小组赛积分
    "home_group_rank": 1,      // 排名
    "away_group_points": 0,    // 客场积分 (杯赛用)
    "away_group_rank": 1,      // 客场排名
    "is_knockout": false,      // 是否淘汰赛阶段
    "round_num": 1             // 当前轮次 (1-7)
  }
}
```

### `/root/data/league_features.json` (联赛统计特征)
包含各联赛的平均进球、主胜率等统计特征，用于模型输入。

## 关联脚本

| 脚本 | 功能 | 调用频率 |
|------|------|----------|
| `update_tournament_state.py` | 从API拉取世界杯积分榜 | 每日/手动 |
| `predict_group_stage.py` | 小组赛预测 | 每日 |
| `simulate_knockout.py` | 淘汰赛模拟 | 每日 |
| `ml_football.py:235-247` | 动态计算联赛积分榜(特征) | 预测时 |

## 积分榜特征工程 (build_tournament_stage_features)

输入: `(home, away, league)` → 输出: 4维特征向量
1. 积分差 (归一化到[-1,1])
2. 排名差 (归一化)
3. 是否淘汰赛 (0/1)
4. 轮次编码 (归一化到[0,1])

## API 调用限制
- 免费层: 10次/分钟
- 缓存策略: 24小时TTL
- 限流处理: 429时使用缓存数据

## 注意事项
- 只有世界杯(WC)有独立积分榜API
- 五大联赛积分榜需从 `ml_football.py` 动态计算
- 队名需通过 `team_name_mapping.json` 做中英映射
