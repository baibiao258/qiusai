# TheStatsAPI 端点探测日志 (2026-06-15)

## 核心发现: `/statistics` 不存在, 正确端点是 `/stats`

| 目标 | 正确路径 | 错误路径(404) | 
|------|---------|---------------|
| 技术统计(射正/控球/xG/黄牌/犯规) | `/football/matches/{id}/stats` | `/statistics`, `/events`, `/xg`, `/expected_goals` |
| 开盘赔率(4家bookmaker) | `/football/matches/{id}/odds` | — |
| 裁判信息 | `/football/matches/{id}` → `referee.name` | 无独立裁判端点 |
| 首发阵容 | `/football/matches/{id}/lineups` | — |

## `/stats` 响应结构

```json
{
  "match_id": "mt_...",
  "overview": {
    "ball_possession": {"all": {"home": 49, "away": 51}, "first_half": {...}, "second_half": {...}},
    "expected_goals": {"all": {"home": 1.33, "away": 0.28}, ...},
    "big_chances": {"all": {"home": 4, "away": 0}, ...},
    "total_shots": {"all": {"home": 13, "away": 6}, ...},
    "shots_on_target": {"all": {"home": 7, "away": 2}, ...},
    "fouls": {...}, "yellow_cards": {...}, "red_cards": {...},
    "corner_kicks": {...}, "passes": {...}, "accurate_passes": {...}
  },
  "shots": {"shots_inside_box": {...}, "shots_outside_box": {...}, ...},
  "attack": {"touches_in_penalty_area": {...}, ...},
  "passes": {"accurate_crosses": {...}, ...},
  "defending": {"tackles": {...}, "clearances": {...}, ...},
  "goalkeeping": {"saves": {...}, ...},
  "np_expected_goals": {"all": {"home": 1.36, "away": 0.28}, ...}
}
```

## `/odds` 响应结构

```json
{
  "match_id": "mt_...",
  "bookmakers": [
    {"bookmaker": "Pinnacle", "markets": {"match_odds": {
      "home": {"opening": "1.930", "last_seen": "1.850"},
      "draw": {"opening": "3.400", "last_seen": "3.460"},
      "away": {"opening": "4.150", "last_seen": "5.110"}
    }}},
    {"bookmaker": "Bet365", ...},
    {"bookmaker": "Betfair Exchange", ...},
    {"bookmaker": "Kambi", ...}
  ]
}
```

注意: 赔率值以字符串返回, 需要 `float()` 转换。同时包含 `btts` (both teams to score) 和 `total_goals` over/under 市场。

## `/match/{id}` 裁判结构

```json
{
  "referee": {"id": "ref_57285416", "name": "Falcon Perez, Yael"},
  "odds_available": true,
  "xg_available": true,
  "score": {"home": 5, "away": 1, "half_time_home": 2, "half_time_away": 1},
  "venue": {"name": "Estadio BBVA", "city": "Monterrey"}
}
```

## 性能统计

- 单个 API 请求耗时: 约 0.8-3s (视端点而定)
- 并行请求 (ThreadPoolExecutor max_workers=4): 约 1.5-3s 完成 2 个端点
- 历史数据 (2024年比赛) 的 `/stats` 和 `/odds` 依然可用, 不限于最新比赛
- Pinnacle 开盘赔率在所有比赛上均可用 (`odds_available: True`)
