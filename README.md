# 竞彩足球/世界杯预测系统 (qiusai)

> 全栈足球预测系统 — 覆盖数据采集、模型训练、实时预测、赛果回填、Brier 评估的完整闭环。

## 系统架构总览

```
500.com (竞彩数据+实时赔率) ──┐
365scores (投票/趋势/阵容) ──┼──→ daily_jczq.py (每日管线) → predictions_log.csv
The Odds API (世界杯赔率) ───┤      │
football-data.org (联赛结果) ─┘      ├→ build_prediction_bundle() → 5玩法输出
TheStatsAPI (xG/阵容) ──────────┘      ├→ telegram_bot.py → TG推送
                                        └→ backfill_results.py → Brier评估
                                                  ↓
Model Training ← training_data_with_odds.json
    ├─ Dixon-Coles (泊松攻防模型)
    ├─ XGBoost (29维/11维/33维)
    ├─ Elo Rating System
    ├─ Stacking (LR Meta-learner)
    └─ Isotonic/Platt Calibration
```

## 目录结构

| 目录 | 内容 |
|------|------|
| `core/` | 核心预测管线入口脚本 |
| `wc_upgrade/` | 世界杯专用模型特征工程+训练 |
| `scripts/` | 辅助脚本(赔率抓取/回测/数据准备) |
| `data/` | 数据文件(模型/training data/缓存) |
| `skills/` | Hermes Agent 技能文件(完整工作流文档) |
| `refs/` | 设计文档和参考 |

---

## 数据采集 (Data Collection)

### 1. 500.com (竞彩官网) — 实时赔率+开售清单

**核心脚本**:
- `core/daily_jczq.py` → `scrape_500_odds_today()` — 主入口, 异步抓取4玩法(SPF/RQSPF/JQS/BQC)
- `wc_upgrade/async_500_scraper.py` — aiohttp+BeautifulSoup 并发爬虫 (~2秒完成4页面)
- `scripts/scraper_500_analysis.py` — 分析页爬虫(FIFA排名/近10场/亚盘/澳门心水)
- `wc_upgrade/historical_kaijiang.py` — 从 zx.500.com 抓取历史收盘SP赔率(3248场+)

**API端点**:
- `https://trade.500.com/static/jc/odds/{playid}/?date={YYYY-MM-DD}` — 实时赔率
- `https://odds.500.com/fenxi/shuju-{sid}.shtml` — 分析页
- `https://live.500.com/` — 平均欧赔兜底源

**爬取逻辑**: DOM 解析 `data-type/data-value/data-sp` 三属性, 按 `data-fixtureid` 合并4页面

### 2. 365scores — 公众投票/趋势/阵容数据

**核心脚本**:
- `core/daily_jczq.py` → `fetch_365scores_data()` — API调用+队名映射
- `scripts/collect_365scores_daily.py` — 每日02:00 cron自动抓取, 缓存到 CSV
- `scripts/fetch_365scores.py` — 单次抓取工具

**API端点**:
- `https://webws.365scores.com/web/games/current/?sports=1&date=YYYY-MM-DD&games=1&lang=1`

**数据字段**: 公众投票%, 近期趋势, 阵容信息, 伤病, FPI评分

### 3. The Odds API — 世界杯赔率

**核心脚本**:
- `wc_upgrade/daily_wc_pipeline.py` → `fetch_odds()`/`fetch_scores()`
- `scripts/fetch_worldcup_odds.py` — 夺冠赔率刷新

**API端点**:
- `https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup_winner/odds/` — 夺冠赔率
- `https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/` — 单场1X2
- `https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/scores/?daysFrom=3` — 完赛结果

**API Key**: 环境变量 `THE_ODDS_API_KEY` (500免费额度/月)

### 4. TheStatsAPI — xG/阵容/裁判特征

**核心脚本**:
- `core/thestats_advanced_features.py` — 高级特征提取
- `wc_upgrade/fetch_thestats_features.py` — 特征批量拉取
- `wc_upgrade/thestats_backfill.py` — 历史数据回填(2289场)

**API端点**: `https://api.thestatsapi.com/` (Key: `THE_STATS_KEY`)

**数据**: 17国家队赛事ID, 含xG/odds, 覆盖2024-2026

### 5. football-data.org — 联赛赛果+积分榜

**核心脚本**:
- `scripts/fetch_league_data.py` — 9联赛×3赛季历史数据
- `scripts/update_tournament_state.py` — 世界杯积分榜更新
- `scripts/pull_standings_cache.py` — 7联赛积分榜缓存

**API端点**: `https://api.football-data.org/v4/competitions/{code}/matches`
**Key**: `FOOTBALL_API_KEY` (Tier One, 10次/分钟)

### 6. football-data.co.uk — CSV离线数据

**URL模式**: `https://www.football-data.co.uk/mmz4281/{season}/{league}.csv`
**覆盖**: 3赛季×9大联赛 ≈ 9300场 (含Bet365赔率+统计)

---

## 预测模型 (Prediction Models)

### Dixon-Coles (DC) 泊松攻防模型

**定义**: `core/dc_model_definition.py` (DixonColes类)
**训练**: `scripts/retrain_dc_model.py`, `scripts/retrain_poisson_elo.py`

**参数**:
- `dc_model.pkl` (国家队, 226队, γ=0.25, half_life=540天)
- `dc_club.pkl` (俱乐部, 2174队, γ=0, 从63K场训练)
- `dc_club_en.pkl` (俱乐部英文, 152队, 2,743场)

**输出**: `predict_proba(home, away, neutral) → [Home, Draw, Away]`
**警示**: 对未知队伍默认均匀分布(1/3, 1/3, 1/3)

### XGBoost 特征驱动模型

**多版本并存**:

| 模型 | 维度 | 验证准确率 | 状态 |
|------|------|-----------|------|
| `xgb_model_nat.pkl` | 11维 | **75.4%** | ✅ 推荐 |
| `xgb_model_29.pkl` | 29维 | 64.3% | ⚠️ 死特征问题 |
| `xgb_model_30.pkl` | 30维 | 64.3% | ⚠️ |
| `xgb_model_33.pkl` | 33维 | — | 🔵 影子模式 |
| `xgb_model_club.pkl` | 37维 | 53.5% | ✅ 俱乐部专用 |

**11维nat模型特征**: elo_diff, lam_h, lam_a, lam_diff, lam_ratio, dc_h/d/a, op_h, op_a, market_implied

**29维完整特征**: 15维基线 + 5黄金(H2H/大赛/友谊赛/form) + 3赔率 + 6滚动形式

**训练脚本**: `scripts/retrain_xgb_v3.py`, `wc_upgrade/train_national_xgb.py`

### Elo Rating

**存储**: `data/elo_ratings.pkl` (国家队, 以英文名为key)
**俱乐部**: `data/elo_club.pkl` (150天半衰期)
**Elo赔率**: `op_h = 1/(10^(-diff/400)+1)`

### 混合融合策略

**熵基动态权重** (2026-06-08):
```
W_xgb = clamp(α + β*C, 0.10, 0.90)
C = 1 - H/log₂(3)  # H = Shannon Entropy
α=0.30, β=0.50
```
- XGB预测尖锐(80/12/08) → W_xgb≈0.56
- XGB预测均匀(34/33/33) → W_xgb≈0.30

**凸组合市场校准**: `blended = (1-mw)*model + mw*market`

### 蒙特卡洛模拟 (世界杯冠军)

- **200K次并行** (2进程) — `core/wc_2026_final.py`
- **12×4正确赛制** — 72场小组赛 → R32 → R16 → QF → SF → Final
- **淘汰赛签表**: `ranked` (Elo配对) / `official` (FIFA路书)
- **东道主加成**: per-team 值 (USA=0.1445, Mexico=0.10, Canada=0.07)
- **加时/点球**: Poisson 30min + Elo-based penalty shootout

---

## 每日预测管线 (Daily Pipeline)

### 执行流程 (`daily_jczq.py`)

```
1. async_500_scraper → 4玩法赔率并发抓取 (2秒)
2. 365scores数据加载 (CSV缓存/API)
3. 双重模型路由:
   ├─ _try_club_predict() → elo+dc+xgb_club (Brier 0.21)
   └─ _try_hybrid_predict() → elo+dc+xgb_nat (Brier ~0.20)
4. 后处理:
   ├─ Draw Correction (平局补偿 → 0%→~7%)
   ├─ Empirical Smoothing (cap=0.75裁剪)
   ├─ 赛事阶段特征 (积分/排名/轮次)
   └─ 战意不足调整 (小组赛第3轮强队降权)
5. bet_action计算 (RECOMMEND/WATCH/SKIP)
6. EV/Kelly分析 → output + CSV
```

### 5玩法输出

1. **胜平负(SPF)**: H/D/A全概率 + 推荐
2. **让球胜平负(RQSPF)**: 让球值 + 让胜/让平/让负
3. **半全场(HTFT)**: 9向概率Top6
4. **比分(Score)**: 泊松50K MC, Top15分布
5. **总进球(Goals)**: 0~12球完整13档分布

### 赛果回填 + Brier评估

- `scripts/backfill_results.py`: 多源回填(results JSON→kaijiang CSV→football-data.org)
- 自动计算 Brier Score (多分类, r=3)
- `scripts/backfill_evaluate_brier.py`: 清洗+分模型版本对比

---

## 关键数据文件

| 文件 | 大小 | 内容 |
|------|------|------|
| `data/international_results.json` | 16MB | 1872-2026国际比赛结果 |
| `data/training_data_with_odds.json` | 1.5MB | 2364场带赔率训练数据 |
| `data/predictions_log.csv` | 168KB | 每日预测记录(已回填) |
| `data/team_name_mapping.json` | — | 2275条中文→英文队名映射 |
| `data/elo_ratings.pkl` | 284KB | 国家队Elo评分 |
| `data/dc_model.pkl` | 8KB | Dixon-Coles模型(226队) |
| `data/xgb_model_nat.pkl` | 656KB | XGBoost国家队模型 |
| `data/standings_cache.json` | — | 7联赛积分榜缓存 |

---

## 世界杯专用模型 (WC 2026)

### 双DC管线架构
```
calibrated_predictor.py
├─ Pipeline A: dc_model.pkl (国家队226队, 英文) + xgb_model_nat.pkl
├─ Pipeline B: dc_club.pkl (俱乐部2174队, 中文) + Elo
└─ Pipeline C: Elo + Market (兜底)
```

### 冠军概率模拟
- `core/wc_2026_final.py` — 每日06:00 cron自动运行
- 200K MC模拟, 输出Top15冠军概率 + 每轮晋级概率
- 48队淘汰过滤已集成

### 单场比赛预测
- `core/predict_match.py` — `python3 predict_match.py "Home" "Away"`
- 10秒出结果, 含DC/XGB/Hybrid/Market四列概率

---

## 策略与风控

### EV/Kelly模块 (`scripts/bet_math.py`)
- **Kelly Criterion**: `f* = p - (1-p)/(odds-1)`
- **半-Kelly**: 单注≤5%总资金
- **is_sane_bet() 五道保险**: odds>30跳过, prob<15%跳过, market_fallback过滤, odds>5+prob<25%跳过

### 赛事过滤
- 友谊赛 → WATCH_FRIENDLY (ROI -58.1%)
- UEFA Nations League → SKIP_LEAGUE (ROI -72.5%)
- 世界杯 → RECOMMEND

### bet_action规则 (`compute_bet_action()`)
基于历史ROI的赛事分级过滤, 结合margin_pp门槛

---

## 系统监控

### 模型健康检查
```bash
python3 scripts/backfill_results.py --stats  # Brier + 准确率
python3 scripts/backfill_results.py --report # 每日趋势
```

### 每日cron任务

| cron_id | 时间(UTC) | 任务 | 脚本 |
|---------|-----------|------|------|
| `3fee9087ae2c` | 02:00 | 365scores采集 | `collect_365scores_daily.py` |
| `6d912cb676ec` | 01:30 | 赛果回填(AM) | `backfill_results.py` |
| `571c46a2a622` | 05:30 | 赛果回填(PM) | `backfill_results.py` |
| `f22f1d2494f3` | 00:00 | 夺冠赔率刷新 | `fetch_worldcup_odds.py` |
| `b2148e127b3a` | 06:00 | 全量重训+MC | `wc_2026_final.py --bracket=official` |
| `c6532ca9a1eb` | 每30min | 赛前赔率监控 | `pre_match_odds_refresh.py` |

### 关键API Keys
- `FOOTBALL_API_KEY` — football-data.org (10次/分钟)
- `THE_ODDS_API_KEY` — The Odds API (500次/月免费)
- `THE_STATS_KEY` — TheStatsAPI (500万次/月)

---

## 历史修复摘要 (2026-06-14 重大P0修复)

| 问题 | 影响 | 修复 |
|------|------|------|
| spf_result类型混淆 | 29条标签错, 准确率64.4%→75.4% | 统一`str(m['spf_result'])` |
| _blend_with_market draw=0硬编码 | 平局系统性低估 | Elo差估算真实平局概率 |
| 双管线模型不一致 | daily vs calibrated不同模型 | 统一到nat模型 |
| market_h未定义 | SPF显示0.0% | `locals().get('market_h', 0)` |
| form数据硬检查return None | 无form队伍无法预测 | 改为软检查+警告 |
| Isotonic校准器负优化 | Brier 0.2053→0.2378 | 全量剥离, 回退raw概率 |
| Draw Correction | 0/32场平局预测 | 注入draw_boost补偿 |

---

## 快速入门

```bash
# 每日预测管线
python3 core/daily_jczq.py

# 世界杯单场预测
python3 core/predict_match.py "Brazil" "Argentina"

# 世界杯冠军模拟
python3 core/wc_2026_final.py --bracket=official

# 赛果回填
python3 scripts/backfill_results.py --stats

# 模型重训练
python3 scripts/retrain_xgb_v3.py
```

---

## 技术栈

- **Python 3.11**: 核心语言
- **XGBoost 2.x**: 特征驱动模型
- **scikit-learn 1.5+**: 校准器/评估/stacking
- **NumPy/SciPy**: 泊松/Skellam分布/Elo计算
- **aiohttp/BeautifulSoup**: 异步爬虫
- **joblib**: 模型序列化
- **pandas**: 数据处理
- **joblib**: 模型文件IO
