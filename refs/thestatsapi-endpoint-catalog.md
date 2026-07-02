# TheStatsAPI 端点目录 (2026-06-15)

## 概述

REST API (`https://api.thestatsapi.com/api`)，认证 `Authorization: Bearer {key}`。完整文档: `https://api.thestatsapi.com/llms.txt`

## 已使用的端点

| 端点 | 用途 | 覆盖率 |
|------|------|--------|
| `GET /football/competitions` | 翻页扫出149赛事，筛选17国家队 | — |
| `GET /football/matches?competition_id=CID&date_from=...&status=finished` | 拉历史基础数据 | 100% (2,289场) |
| `GET /football/matches/{id}/stats` | xG/控球/射正 | ~12% |
| `GET /football/matches/{id}/odds` | Betfair标准盘 | ~8% |

## 关键参数陷阱

`competition_id` 必须用单数。`competition_ids=CID1,CID2` 被静默忽略。

## 高价值未用端点

### Team Stats
```
GET /football/teams/{team_id}/stats?season_id={season_id}
```
返回: `form` (WWDLW), 积分, 排名, 进球/失球。季级滚动形式特征。

### Lineups
```
GET /football/matches/{match_id}/lineups
```
返回: 阵型(4-3-3), 首发XI。开赛前~1h确认。当前系统无此维度。

### Live Odds
```
GET /football/matches/{match_id}/odds/live
```
对比 opening vs live 发现资金流向。

### Shotmap
```
GET /football/matches/{match_id}/shotmap
```
每次射门的 xG/身体部位/坐标。赛后分析用，np_xG 比 xG 更稳定。

## 回填管线历史

Phase1(并行拉基础) → Phase2(并行拉stats+odds) → merge_training_data.py → retrain_nat.py
总耗时 ~5min (2289场, 5M/mo配额)
