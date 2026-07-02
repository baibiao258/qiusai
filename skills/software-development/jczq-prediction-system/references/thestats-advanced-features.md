# TheStatsAPI 高阶特征工程 + 5年 Elo/Poisson 重训

## 1. thestats_advanced_features.py — 13 维高阶特征向量

**文件**: `/root/thestats_advanced_features.py`
**Cron**: `thestats_daily_cache_build` (02:30 UTC, job_id=3dd572d3b983)

### 三大维度

| 维度 | 维数 | 端点 | 说明 |
|------|------|------|------|
| 过程压制力 (Process Dominance) | 5 | `/football/matches/{id}/stats` (注意: 是 `/stats`, 不是 `/statistics`!) | home_sot_dominance, home_da_dominance, xG_diff, danger_ratio |
| 国际盘口隐含概率 | 3 | `/football/matches/{id}/odds` | Pinnacle → Bet365 → Betfair 开盘 1X2 去抽水后概率 |
| 裁判/得牌预期 | 5 | `/football/matches/{id}` (referee字段) + 本地裁判数据库 | referee_strictness, avg_yc, avg_rc, card_tendency ×2 |

### 🗺️ 端点发现 (2026-06-15)

| 目标 | 正确端点 | 错误端点 (404) |
|------|---------|----------------|
| 技术统计 (射正/控球/xG/犯规/黄牌) | `/football/matches/{id}/stats` | `/statistics`, `/events`, `/xg` |
| 开盘赔率 (Pinnacle/Bet365 等4家) | `/football/matches/{id}/odds` | — |
| 裁判信息 | `/football/matches/{id}` → `referee.name` | 无独立裁判端点 |
| 首发阵容 | `/football/matches/{id}/lineups` | — |
| xG数据 | `/stats` → `np_expected_goals` 节 | `/xg`, `/expected_goals` |

### 实战细节

1. **开盘赔率 (opening)**: 返回 bookmaker 数组。优先 Pinnacle, 其次 Bet365, 最后 Betfair。需要去抽水: `prob_i = (1/odds_i) / sum(1/odds_all)`。给各bookmaker的权重: Pinnacle > Bet365 > Betfair > Kambi。
2. **裁判数据库**: 从 match detail 的 `referee.name` 提取, 结合 `/stats` 的 yellow_cards/red_cards 构建。构建脚本 `/root/build_referee_fast.py` (200场样本, 34名裁判, 默认黄牌3.5张/场)。
3. **过程压制力计算**: 提取两队近5场比赛的统计均值, 计算: SoT_diff, xG_diff, poss_diff, danger_ratio, defend_ratio。注意: 旧比赛依然有 `/stats` 数据 (2024年比赛可用)。
4. **Pinnacle市场校正层**: 在 daily_jczq.py 中, 当 Pinnacle 赔率可用且与模型分歧 >15% 时, 以 15% 权重折衷。实现:
   ```python
   divergence = np.max(np.abs(pinn_probs - hybrid))
   if divergence > 0.15:
       market_weight = 0.15
       hybrid = (1 - market_weight) * hybrid + market_weight * pinn_probs
   ```
   权重从 30% 调低至 15% (验证发现分歧过大时30%扭曲太多)。

### 双轨集成方式 (2026-06-15 架构变更)

由于训练数据 98.7% 为俱乐部比赛, XGBoost 在国际赛上拖累 DC 模型。实现**双轨隔离路由**:

| 路线 | is_intl | 模型 | 高阶特征用途 |
|------|---------|------|-------------|
| A: 国际赛 | True | DC + Pinnacle市场校正 | 仅用前3维 (Pinnacle赔率) 做市场校正层, 不入XGB特征 |
| B: 俱乐部/联赛 | False | DC + XGBoost 全特征 | 13维全部经 `feat_46` 入 XGBoost 特征向量 |

**路线 A (国际赛)** — 13 维特征仅用于市场校正:
```python
if is_intl:
    hybrid = dc_ado.copy()  # [A, D, H]
    if thestats_adv_feat[0] > 0:  # Pinnacle 赔率可用
        pinn_probs = np.array(thestats_adv_feat[:3])
        divergence = max(abs(pinn_probs - hybrid))
        if divergence > 0.15:
            hybrid = 0.85 * hybrid + 0.15 * pinn_probs
```

**路线 B (俱乐部)** — 13 维特征拼接到 46 维特征向量末尾:
```python
else:
    feat_46 = np.array([b15 + gold + odds_feat + form_feat + stage_feat + thestats_adv_feat])
    # ... XGBoost predict ...
```

详细路由见 `references/dual-route-isolation.md`。

### 数据流

```
03:00 UTC thestats-adv-features-preload (cron, job_id=064a8a625719)
  └── python3 thestats_advanced_features.py preload
        ├── /matches?date=today → 获取当天所有比赛
        ├── /stats → 压制力特征 (仅路线B)
        ├── /odds → Pinnacle赔率 (路线A+B)
        └── referee_strictness.json → 裁判特征 (仅路线B)

03:05 UTC daily_jczq.py
  └── _try_hybrid_predict(match_id, is_intl)
        ├── is_intl=True  → 读缓存前3维 → 市场校正
        └── is_intl=False → 读缓存13维 → 入特征向量
```

## 2. retrain_poisson_elo.py — 5年全史 Elo + 时间衰减 Poisson λ

**文件**: `/root/retrain_poisson_elo.py`
**产出**: `/root/data/poisson_elo_prior.json`

### 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| START_DATE | 2021-01-01 | 5 年窗口，覆盖上届世界杯周期 |
| 赛事覆盖 | 22 个 | World Cup + Premier League + Bundesliga + LaLiga + Ligue 1 + 葡超+荷甲+英冠+巴甲+MLS+J1+K1+EURO+Copa America 等 |
| 总场次 | 32,001 场 | |
| HALF_LIFE_DAYS | 1.5 × 365.25 | Poisson λ 时间衰减半衰期 |
| MAX_RECENT_MATCHES | 30 | λ 最多用最近 30 场 |
| ELO_INIT | 1500 | Elo 初始值 |
| ELO_K | 20 | Elo 敏感系数 |
| 主场优势修正 | +100 Elo | 相当于~0.64 胜率 |

### 输出 JSON 结构

```json
{
  "meta": { "generated_at": "...", "total_matches": 32001, ... },
  "elo": { "Arsenal": 1804.1, "Barcelona": 1791.1, ... },
  "lambda_prior": { "Arsenal": {"lambda_home": 1.856, "lambda_away": 1.464, "n_matches": 30, "total_n": 312}, ... },
  "home_advantage": { "Premier League": 0.4612, ... }
}
```

712 支球队 Elo, 609 支球队 Poisson λ。

### 集成到 daily_jczq.py

`predict_match_legacy()` 优先查询先验：

```
_lookup_prior_elo(team) → 命中则取先验 Elo, 否则回退 1500
_lookup_prior_lambda(team, is_home) → 命中则取衰减后 λ, 否则回退 train() 产出的 ts/ga
model 标记为 prior_poisson (区分 legacy_poisson)
```

### 增量更新

```bash
python3 retrain_poisson_elo.py incremental
```

集成在 `backfill_results.py` 的 `backfill()` 末尾：回填赛果后自动触发增量拉取昨天完赛数据，更新 Elo 和 home_advantage。

### 性能验证

**Elo Top 10 收敛 (32,001 场后)**:
| 排名 | 球队 | Elo |
|------|------|-----|
| 1 | Sporting | 1833.7 |
| 2 | Benfica | 1821.2 |
| 3 | PSV Eindhoven | 1819.4 |
| 4 | FC Bayern München | 1813.9 |
| 5 | Arsenal | 1804.1 |
| 6 | FC Porto | 1802.8 |
| 7 | Barcelona | 1791.1 |
| 8 | Manchester City | 1782.0 |
| 9 | GNK Dinamo Zagreb | 1773.6 |
| 10 | Al Ahly FC | 1773.1 |
