# 365scores API 调查记录

## Endpoint 探测结果 (2026-06-10)

| Endpoint | 状态 | 数据 |
|----------|------|------|
| `/data/games/` | ✅ 200 | 主数据源，含投票/趋势/FIFA排名/人气 |
| `/web/game/` | ✅ 200 | 单场详情，含 recentMatches ID 列表 |
| `/web/standings/` | ✅ 200 | 友谊赛返回空（无积分榜） |
| `/web/trends/` | ❌ 500 | 无论比赛是否已结束都返回 500 |
| `/web/game/stats/` | ❌ 500 | 同上 |
| 其他变体 (`/v2/`, `/web/insights/`, `/web/h2h/`) | ❌ 404 | 不存在 |

## `/data/games/` 返回结构

```json
{
  "Games": [{
    "ID": 4712160,
    "Comp": 570,
    "Comps": [
      {
        "Name": "Portugal",
        "Trend": [1, 1, 2, 1, 0],        // [W, D, L, ?, ?]
        "Rankings": [{"Name": "FIFA", "Position": 5}],
        "PopularityRank": 25905
      },
      {
        "Name": "Nigeria",
        "Trend": [2, 1, 1, 2, 1],
        "Rankings": [{"Name": "FIFA", "Position": 26}],
        "PopularityRank": 4583
      }
    ],
    "WhoWillWinReults": {"Vote1": 50440, "VoteX": 4686, "Vote2": 4782},
    "HasTrends": true
  }]
}
```

## Trend 字段含义

- 数组长度: 多数为 5，少数为 0-4
- 前 3 位: [胜, 平, 负] 场数（近5场）
- 后 2 位: 含义不明（值域 0-6，不符合胜平负逻辑）
- 总和分布: 2-5 最常见，暗示近5场统计

## `/web/trends/` 返回 500 的可能原因

1. 需要登录态/认证 token
2. 仅对特定赛事类型开放（如联赛而非友谊赛）
3. 服务端暂时不可用（多次测试均失败）

## 结论

盘路走势、xG统计、红牌犯规等高级数据目前无法通过公开API获取。
可用数据: 投票 + Trend[0:2](W/D/L) + FIFA排名 + 人气指数。
