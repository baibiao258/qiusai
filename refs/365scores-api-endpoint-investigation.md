# 365scores API 端点探测记录 (2026-06-14)

## 动机
探明 365scores 各 API endpoint 的实际可用性，找出我们未使用但有用的数据字段，确认是否需要 Playwright 或额外工具。

## 环境
- 请求库: `requests`
- 头部: `User-Agent: Mozilla/5.0`, `Accept: application/json`
- 无 cookie/无 session/无 token

## 端点逐项结果

### ✅ 可用: `webws.365scores.com/data/games/`
**URL**: `https://webws.365scores.com/data/games/`
**参数**: `lang=1, app-type=1, cid=2, sport-type=1`
**认证**: 无（连 User-Agent 都不需要）
**返回体**: JSON，含 `Games`, `Competitions`, `Countries`, `Bookmakers`(空), `CurrentDate` 等

**可用过滤器参数**（2026-06-14 发现）:
- `teamId=N` — 只返回指定球队的比赛
- `compId=N` — 只返回指定联赛的比赛
- `sport-type=1` — 只返回足球  （注意: SID=1 是字段名, sport-type=1 是参数名, 两者不同）

**已验证**: 连裸 `Accept: application/json` 都不带返回 200, Game 数=425+。

### ⚠️ 部分可用: 不同 `app-type` 变体
```
app-type=0: 返回 Notifications
app-type=2: 返回 Notifications
app-type=3: 返回 Notifications
```

### ❌ 不可用: `webws.365scores.com/web/trends/`
所有参数变体返回 500:
```
?gameId=N&lang=1  → 500
?gameId=N         → 500
/gameId           → 404
```

### ❌ 不可用: `momentumsr.365scores.com`
**URL**: `https://momentumsr.365scores.com/api/SportRadarMomentum/GetMomentum?partnerId=46927441`
- 返回 HTML Widget (SportRadar 动量图表)
- Content-Type: `text/html`
- 无内嵌 JSON 数据
- 仅用于 iframe embed 可视化，不可编程提取

### ❌ 不可用: `365scores.com/match/...` (纯 requests)
- 纯 React SPA，无 SSR
- 无 `__NEXT_DATA__` script
- 页面 shell < 65KB，数据在运行时通过 JS 调用 `webws.365scores.com` 加载
- 需要用 Playwright/headless browser 才能提取页面内数据

### ❌ 不可用: 阵容/球员 endpoint
`data/members/`, `data/players/`, `data/squad/`, `data/teams/` — 全部 404。
尽管 API 返回 `HasSquad=True` (258 队) 和 competition 层 `HasSquads=True` (31 联赛)，但无公开 API。

## SID 字段 — 体育项目识别

`SID` 是 Game 和 Competition 级别的字段，比 `competition` 名字更可靠：

| SID | 体育项目 | 已验证 |
|-----|---------|--------|
| 1 | 足球 (Soccer) | ✅ |
| 2 | 篮球 (Basketball) | ✅ |
| 3 | 网球 (Tennis) | ✅ |
| 4 | 冰球 (Ice Hockey) | ✅ |
| 5 | 未知 (疑似 e-sports 或手球) | ❓ |
| 7 | 棒球 (Baseball) | ✅ |
| 8 | 排球 (Volleyball) | ✅ |
| 9 | 橄榄球 (Rugby) | ✅ |

**重要**: `filter_sid=1` 是 100%精准的足球过滤方式，比维护关键词列表更可靠。
已在 `fetch_365scores.py` 的 `extract_games()` 中实现。

## 已读 vs 未读字段 (data/games/ API)

### 已提取字段
- `WhoWillWinReults` → `Vote1, VoteX, Vote2` (投票分布)
- `Trend` → 主客队近 5 场趋势
- `PopularityRank` → 主客队人气排名
- `Rankings` (from Comps) → FIFA 排名
- `HasLineups`, `HasStatistics`, `HasNews`, `HasBuzz`, `HasDoubtful`, `HasMissingPlayers`
- `SocialStats.Comments`
- `Scrs` [0:2] → 终场比分
- `Events` → 进球、红黄牌事件
- `Venue`, `Attendance`
- `LineupsStatusText`

### 未提取但存在的字段
| 字段 | 类型 | 说明 | 价值 |
|------|------|------|------|
| `Winner` | int | -1=平局, 1=主胜, 2=客胜 | 中: 赛后回填 backup |
| `Scrs[2:4]` | float[2] | 半场比分 (H:A) | 中: HTFT 验证 |
| `HasBets` | bool | 是否有赔率数据 (全线=False) | 低: 无数据 |
| `HasTable` (Competition) | bool | 联赛是否有积分榜 | 低: 需另外 endpoint |
| `HasTopPerformers` | bool | 是否有最佳球员数据 | 中: 标记可用性 |
| `Round`, `Group`, `Stage` | int | 比赛轮次/分组/阶段信息 | 低: tournament_state 更全 |
| `HasSquad` (Comps) | bool | 球队是否有阵容数据 | 低: 无 API 可拉 |

## GitHub 社区发现的关联项目

- **leoronchini/bet-science-ai**: 提到 365scores 是 SPA，需要 Playwright。**结论**: 针对的是 365scores 网站前端，不是我们用的 REST API。我们的 API 端点用 requests 能正常返回 425+ 场比赛的完整 JSON 数据。
- **federicorabanos/futbol-data-visualizacion**: 在 Jupyter Notebook 中使用 `momentumsr.365scores.com/api/SportRadarMomentum/GetMomentum`。**结论**: 该 endpoint 返回 HTML Widget, 推测是在 notebook 中以 iframe embed 做可视化，非数据提取。

## 结论

1. **当前 API 足够**: 无需 Playwright, 无需 token, 最核心的投票/FIFA排名/人气/趋势数据都已正确提取
2. **5/157 不是 API 问题**: 是 365scores 数据只累积了 3 天, 与 historical_kaijiang 的 892 天重叠很小
3. **SID=1 是正确过滤方式**: 已上线, 每日约 200 场纯足球数据
4. **未来发展**: 若需要阵容/评分/技术统计等数据, 需 Playwright 或 365scores 商业 API Key
