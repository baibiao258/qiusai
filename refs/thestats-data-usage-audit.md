# TheStatsAPI 数据使用审计 (2026-07-01)

## 已用 vs 未用的数据全景

基于 [llms.txt](https://api.thestatsapi.com/llms.txt) 全量排查。

### ✅ 已使用的数据和字段

| 端点 | 提取字段 | 用途模块 |
|------|---------|---------|
| `GET /football/matches` | `home_team.name`, `away_team.name`, `score.home`, `score.away`, `score.half_time_home`, `score.half_time_away`, `utc_date`, `competition_id`, `status`, `id` | `pull_training_data.py`, `retrain_poisson_elo.py`, `backfill_results.py` |
| `GET /football/matches/{id}` | `referee.name` | `thestats_advanced_features.py` — 裁判特征 5 维 |
| `GET /football/matches/{id}/odds` | `bookmakers[Pinnacle/Bet365].markets.match_odds.home/draw/away.opening` | `thestats_advanced_features.py` — 市场隐含概率 3 维 |
| `GET /football/matches/{id}/stats` | `overview.{shots_on_target, expected_goals, ball_possession, total_shots}.all` | `thestats_advanced_features.py` — 压制力 5 维 |

约 API 总能力的 **15~20%**。`daily_jczq.py` 预测时并不直接调 TheStatsAPI，而是加载从该数据训练的 `.pkl` 模型文件。

### ❌ 高价值未使用数据 (按优先级)

#### 1. 积分榜 standings (最高优先级)
**端点**: `GET /football/competitions/{id}/seasons/{sid}/standings`
**字段完全可用**: `position`, `points`, `goal_difference`, `goals_for`, `goals_against`, `matches_played`, `form("WWDLW")`, `wins`, `draws`, `losses`
**集成方案**:
- Phase 1 (当前): `pull_standings_cache.py` → `/root/data/standings_cache.json` + `standings_lookup.lookup_both()` → 返回 dict 供输出展示 + bet_analysis 使用
- Phase 2 (下次重训): 追加 3 维特征到 `_try_club_predict` 的 gold features (rank_diff/max_teams, pt_diff/max_pts, gd_diff/50)，重训 `xgb_model_club.pkl` (17→20 维)
- **当前状态**: Phase 1 已完成，7 联赛 standings 已缓存 (136 队, 43KB)

#### 2. BTTS/总进球 Over-Under 市场赔率
**端点**: 已有 `/matches/{id}/odds`，但只解析了 `match_odds`
**未提取字段**: `markets.btts.yes/no`; `markets.total_goals.{line}.over/under`
**价值**: 总进球预测校准信号 — 目前 `compute_goals_distribution()` 纯靠泊松 λ，无市场校准层

#### 3. 半场/下半场 stats 拆分
**端点**: 已有 `/matches/{id}/stats`，但只取了 `.all` 全场聚合
**未提取字段**: `ball_possession.first_half`, `expected_goals.first_half`, `expected_goals.second_half` 等
**价值**: 替换 `daily_jczq.py` 中硬编码的 `HALF_FULL_R_HT = 0.45` — 用真实半场 xG 比例提升半全场预测

#### 4. 其他可选

| 数据 | 端点 | 应用方向 |
|------|------|---------|
| 球队赛季统计 | `/teams/{id}/stats?season_id=` | 赛季级稳定均值 vs 近5场小样本 |
| 阵容+身价 | `/teams/{id}/players` | market_value 作为实力代理变量 |
| 裁判生涯数据 | `/matches/{id}/referee` → `career` | 比本地 JSON 缓存更权威 |
| Shotmap | `/matches/{id}/shotmap` | 每次射门坐标 + xG → 更高阶进攻质量特征 |
| 比赛阵容 | `/matches/{id}/lineups` | 世界杯考勤: 主力缺阵检测 |
| 事件时间线 | `/matches/{id}/timeline` | 进球/红牌/换人 → 比分走势特征 |

## Standings 集成管线

### 缓存脚本

```bash
python3 /root/pull_standings_cache.py           # 全量拉取 (7 联赛)
python3 /root/pull_standings_cache.py --dry-run  # 预览
python3 /root/pull_standings_cache.py --stats    # 查看缓存
```

输出: `/root/data/standings_cache.json` — 按 comp_id 索引的 136 队数据

### 查询模块

```python
from standings_lookup import lookup_both, lookup_team, load_standings_cache

hi, ai, features = lookup_both("Arsenal FC", "Liverpool FC")
# features = [rank_diff/38, pt_diff/85, gd_diff/50]
# hi = {"comp_id":"comp_3039", "position":1, "points":85, ...}
```

队名匹配策略: 精确 → +FC后缀 → AFC前缀 → 归一化子串

### 当前 7 联赛 season_id 映射

| 联赛 | comp_id | 当前 season_id | 队数 |
|------|---------|---------------|------|
| Premier League | comp_3039 | sn_6125938 | 20 |
| Bundesliga | comp_4643 | sn_5789634 | 18 |
| LaLiga | comp_8814 | sn_7246390 | 20 |
| Ligue 1 | comp_0256 | sn_6120181 | 18 |
| Liga Portugal Betclic | comp_8385 | sn_6120591 | 18 |
| Eredivisie | comp_3809 | sn_9674249 | 18 |
| Championship | comp_8321 | sn_3064530 | 24 |

> 新赛季开始时需更新 season_id。可以通过 `GET /football/competitions/{comp_id}` 的 `current_season_id` 字段获取最新 ID。

## Pre-run 脚本审查模式

运行关键脚本前 (backfill_results.py / evaluate_brier.py / daily_jczq.py):

1. **读代码** — 阅读完整脚本而非仅头部注释
2. **查依赖** — 验证 import 和 subprocess 依赖 resolve
3. **验运行时** — 确认数据文件存在、API Key 有效、模型 .pkl 存在
4. **找 bug** — 特别关注: `\\n` 转义、路径硬编码、subprocess 参数拼装、幂等性设计
5. **顺数据流** — 确认执行顺序正确 (backfill → evaluate → predict)
6. **报告** — 结论先行，分严重影响/低影响/注意事项三级

## 命名约定

| 脚本 | 职责 |
|------|------|
| `pull_standings_cache.py` | 从 TheStatsAPI 拉取 standings 并缓存到本地 JSON |
| `standings_lookup.py` | 查询模块: 队名模糊匹配、双队特征提取 |
| `backfill_results.py` | 多源赛果回填 (5层数据源优先级) |
| `evaluate_brier.py` | Brier Score A/B 评估 + 数据清洗 |
| `daily_jczq.py` | 每日预测主管线 |
