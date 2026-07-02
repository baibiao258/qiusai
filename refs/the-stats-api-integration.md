# TheStatsAPI 集成现状与数据差距 (2026-07-02 更新)

> Base URL: `https://api.thestatsapi.com/api`
> Auth: `Authorization: Bearer {THE_STATS_KEY}` (5M req/month)
> API Key 来源: `THE_STATS_KEY` 环境变量
> 文档: `https://api.thestatsapi.com/llms.txt`

## 当前使用状态

**API 已全面上线，非试用阶段。** Key 为付费 Tier（500万次/月），通过 `THE_STATS_KEY` 环境变量注入。所有训练数据拉取、模型训练、赛果回填均依赖此 API。

## 已使用的端点与字段

### `/football/matches` — 训练数据主体

- **字段提取**: `home_team.name`, `away_team.name`, `score.home/away`, `score.half_time_home/away`, `utc_date`, `competition_id`, `id`
- **参数**: `competition_id=`, `status=finished`, `date_from=`, `date_to=`, `per_page=100`
- **用途模块**:
  - `pull_training_data.py` → `thestats_training_data.json` (2,528+ 场，22 赛事)
  - `retrain_poisson_elo.py` → `poisson_elo_prior.json` (Elo + λ)
  - `retrain_xgb_model.py` / `retrain_xgb_v3.py` → XGBoost `.pkl` (11-17维)
  - `retrain_dc_model.py` → `dc_model.pkl`
  - `backfill_results.py` → 赛果兜底 (`match_from_thestats`, data source #5)

### `/football/matches/{id}/odds` — 高阶特征 (3维)

- **字段提取**: `bookmakers[Pinnacle/Bet365].markets.match_odds.home/draw/away.opening` → 去抽水隐含概率 [prob_H, prob_D, prob_A]
- **用途**: `thestats_advanced_features.py` 的 `_get_odds()`

### `/football/matches/{id}/stats` — 高阶特征 (5维)

- **字段提取**: `overview.shots_on_target.expected_goals.ball_possession.total_shots.all` (全场聚合)
- **用途**: `thestats_advanced_features.py` 的 `_build_pressure_features()`
- **已知局限**: 只用了 `.all`，未取 `.first_half` / `.second_half` 半场拆分

### `/football/matches/{id}` — 裁判特征 (5维)

- **字段提取**: `referee.name` → 关联本地 `referee_strictness.json`
- **用途**: `thestats_advanced_features.py` 的 `get_referee_data()`

### 回填过滤

- `statusGroup == 4` 或 `status == "finished"` → 筛选完赛比赛
- `score.final_score.home/away` 或 `score.home/away` → 实际比分
- 已回填 2,289 场 (2024-2026)，训练数据从 491 场扩至 ~2,528 场

### 已知参数坑

- 参数名 `competition_id`（单数）不是 `competition_ids`（复数）
- `_fetch_all_thestats_matches()` 用翻页 Pattern: 每页 100 场，最多 50 页，页间 0.3s 间隔

## 可调用但尚未集成的数据

基于 `llms.txt` 完整审计，以下数据已可用但未被系统使用，按 ROI 排序：

### 1. 联赛积分榜 (standings) — 🥇

**端点**: `GET /football/competitions/{id}/seasons/{sid}/standings`

**为什么该拿**：
- `_try_club_predict()` 的 gold 特征 `[h2h_gd, 0, 0, fh12_gf_diff, fa12_gf_diff]` 当前无排名/净胜球差
- 排名差 + 净胜球差是足球预测 Top-5 经典特征
- 1 次调用/赛季即可缓存整赛季

**响应格式** (已验证):
```json
{
  "data": [
    {"team": {"name": "Arsenal"}, "position": 1, "points": 85,
     "goal_difference": 44, "wins": 26, "draws": 7, "losses": 5,
     "goals_for": 80, "goals_against": 36, "matches_played": 38, "form": "WWDLW"}
  ]
}
```

**7 联赛 season_id** (已验证):
| 联赛 | comp_id | season_id |
|------|---------|-----------|
| 英超 | comp_3039 | sn_6125938 |
| 德甲 | comp_4643 | sn_5789634 |
| 西甲 | comp_8814 | sn_7246390 |
| 法甲 | comp_0256 | sn_6120181 |
| 葡超 | comp_8385 | sn_6120591 |
| 荷甲 | comp_3809 | sn_9674249 |
| 英冠 | comp_8321 | sn_3064530 |

### 2. BTTS / O-U 市场赔率 — 🥈

**端点**: `/matches/{id}/odds` (已调用，只解析了 `match_odds`)

未提取字段:
```
markets.btts.yes/no              → 双方进球概率
markets.total_goals.2.5.over/under → 大2.5/小2.5
```

**用途**: 对 `compute_goals_distribution()`（当前仅泊松 λ）做市场校准

### 3. 半场/下半场 stats 拆分 — 🥉

**端点**: `/matches/{id}/stats` (已调用，只取了 `.all`)

每个统计项都有 `.first_half` / `.second_half`:
```json
{
  "expected_goals": {
    "all": {"home": 1.8, "away": 0.6},
    "first_half": {"home": 0.7, "away": 0.2},
    "second_half": {"home": 1.1, "away": 0.4}
  }
}
```

**用途**: 替代 `daily_jczq.py` 硬编码 `HALF_FULL_R_HT = 0.45`，直接提升半全场预测

### 4. 球队赛季统计

**端点**: `GET /football/teams/{id}/stats?season_id=`

返回: `position, points, goals_for/against, form ("WWDLW"), matches_played`
用途: 替代当前近5场滑动窗口，提供赛季级稳定均值

### 5. 阵容名单 + 身价

**端点**: `GET /football/teams/{id}/players`

返回: 全队球员位置、年龄、身价 (`market_value`)
用途: 杯赛主力缺阵检测，当前疲劳度特征 weak

### 6. 裁判职业生涯数据

**端点**: `GET /football/matches/{id}/referee` (独立端点，非 match detail)

返回: `career.games, yellow_cards, red_cards, yellow_red_cards`
用途: 比当前本地 JSON 缓存更权威

### 7. 射门地图 & 比赛阵容

| 数据 | 端点 | 用途 |
|------|------|------|
| Shotmap | `/matches/{id}/shotmap` | 每次射门坐标 + xG，构建进攻质量特征 |
| Lineups | `/matches/{id}/lineups` | 赛前首发确认（世界杯决赛阶段关键） |
| Timeline | `/matches/{id}/timeline` | 进球/换人/红牌事件流 |

## 当前架构限制

**`thestats_advanced_features.py` 未接入预测管线**: 13 维高阶特征（市场隐含概率 3 + 压制力 5 + 裁判 5）虽已实现提取和缓存，但 `daily_jczq.py` 的 `_try_hybrid_predict()` 未调用 `get_all_advanced_features()`。该模块是独立工具，仅在手动 `preload` 模式下运行。

集成分叉: (a) 预测时实时调 API → 慢、有断联风险；(b) 预加载缓存 → 需额外 cron

## 推荐下一步

**standings 集成** (工程最小、收益最直接):
1. 写 `pull_standings_cache.py` → 7 次调用 → `/root/data/standings_cache.json` (~5KB)
2. 在 `_try_club_predict()` 的 `gold` 特征追加 `rank_diff/38` 和 `gd_diff/50` 两维
3. 对 9 大联赛俱乐部比赛生效

## 参数坑历史

- `competition_id`(单数) 不是 `competition_ids`(复数) — 422 错误
- TheStatsAPI 队名质量: 中英混杂，中文名需经过 `team_name_mapping.json` (2,275 条) 映射
- 参数 `per_page` 默认 20，最大 100 — 回填必须设 `per_page=100` 否则页数爆炸

## API 数据质量陷阱：嵌套字段可能为 null

**关键事实**: TheStatsAPI 在 stats 不可用（比赛未进行、数据缺失、或该队无统计记录）时，不会省略字段，而是返回 `null`（JSON null）。

### 示例问题响应

```json
GET /football/matches/{id}/stats
→ {
    "overview": null,                    // ← 整个 overview 为 null
    "shots": {"shots_inside_box": null}  // ← 嵌套字段也为 null
  }
```

### 为什么标准 `.get()` 防护不够

```python
# Python 陷阱: .get(key, default) 在 key 存在但值为 None 时返回 None, 而非 default
stats.get("overview", {})  # → None (因为 "overview" key 存在, 值是 None)
# 而不是预期的 {} 
```

### 正确的防护模式

直接在数据读取代码中嵌入此函数（不要单独 import，保持自包含）：

```python
def _safe_get(d, *keys, default=0):
    """Safe nested dict access, handling None at any level."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d if d is not None else default

# 使用示例:
xg = _safe_get(overview, "expected_goals", "all", "home")
sot = _safe_get(overview, "shots_on_target", "all", "away")
shots_ibox = _safe_get(shots, "shots_inside_box", "all", "home")
```

### 已受影响/修复的代码

- `/root/thestats_advanced_features.py` `_get_team_recent_stats()` — 2026-07-02 修复。原始 `.get("overview", {})` 在第 2 场比赛（América Mineiro）因 overview=null 崩溃。替换为 `_safe_get()` 后 17 场全部成功。

### 预防性检查

新增使用该 API 任何嵌套字段的代码时，先用 `_safe_get()` 模式而非链式 `.get()`。下列端点已知可能返回 null 嵌套字段：
- `/football/matches/{id}/stats` — overview, expected_goals, shots_on_target, ball_possession, big_chances, shots_inside_box
- `/football/matches/{id}` — referee 字段 (可能 null)
- 所有带 `.all.home/.all.away` 嵌套结构的统计端点