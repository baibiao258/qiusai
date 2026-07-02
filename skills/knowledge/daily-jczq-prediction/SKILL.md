---
name: daily-jczq-prediction
description: Use when the user asks to run next-day football lottery / JCZQ prediction workflow from live 500.com buyable matches, enrich with 365scores data, and output full 5-play predictions for 90-minute settlement. Only predicts matches that play on the NEXT calendar day. Covers dual-track model (club 37-dim XGB + international 29-dim XGB with 34-dim shadow) with xG-proxy luck-factor features and tournament stage features (dynamic from tournament_state.json). Includes form data update from 365scores API, XGB retraining pipeline, and pre-match odds monitoring.
version: 1.6.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [jczq, football, 500.com, 365scores, prediction, daily, spf, rqspf, htft, goals, score]
    related_skills: [jczq-analysis, hermes-agent]
---

# Daily JCZQ Prediction

## Overview

This skill runs the current production workflow for **today's buyable JCZQ matches**. It uses `500.com` to determine which matches are actually on sale today, enriches them with `365scores` public vote / recent trend / popularity signals, and then runs the current prediction chain to produce a full report covering:

- `胜平负`
- `竞彩让球胜平负`
- `半全场`
- `比分`
- `总进球数`

## Model Architecture (2026-06-14 Updated)

**Primary model: nat (11-dim)** — unified across daily and WC pipelines.

Features: `[elo_diff, lam_h, lam_a, lam_diff, lam_ratio, dc_a, dc_d, dc_h, op_h, op_a, market_implied]`

- Training: 395 English-name national team matches (time-series 70/30 split)
- Validation accuracy: **75.4%** (LogLoss 0.819)
- Clean features only — no form/gold/h2h (avoids train-serve skew)

**Deprecated models**: xgb_model_29 (18 dead features, label bug), xgb_model_30 (64.3% with same skew).

**Fusion**: Entropy-based dynamic weighting (α=0.30, β=0.50) → DC + XGB hybrid → Draw Correction Layer.

Settlement scope is always **90 minutes regular time including stoppage time**, excluding extra time and penalty shootouts.

This skill is for **execution**, not theory. When triggered, run the pipeline and give the user the actual current-day predictions.

## When to Use

Use when the user says things like:

- `执行今天的竞彩预测`
- `今日竞彩预测`
- `执行竞彩预测`
- `去500看今天能买哪几场，然后预测`
- `把今天所有可买比赛预测一下`

Do not use for:
- Historical backtest requests only
- Single-match deep dive where the user specifies one exact match only
- Requests that are purely about model design or refactor discussion

**预测回顾**: 用户问"昨天的预测怎么样"/"回顾一下"时，参见 `references/prediction-review-workflow.md`。分析 settled_at 匹配目标日期的记录，逐场计算 SPF/RQ/HTFT/Goals 命中率，识别模式（强队局vs冷门局）。注意 actual_hda 格式不统一（H/D/A 或 胜/平/负）需映射。

## Ground Truth Workflow

### Data roles

- `500.com` decides **which matches are buyable today** and provides play odds context
- `365scores` provides **public vote / trend / popularity / lineup-risk** enhancement signals
- `daily_jczq.py` is the current **orchestrator / production entrypoint**
- `predictions_log.csv` is the structured output ledger

### Model Fusion

The prediction engine fuses two models:

### Dixon-Coles (DC) — structural model
- Poisson-based attack/defense rating model
- Handles data-scarce matchups well (uses historical λ estimates)
- Output: `[Home, Draw, Away]` → converted to `[Away, Draw, Home]` for alignment

### XGBoost — feature-driven model
- **International**: 29-dimensional features (Elo, DC λ, form, H2H, odds-based, golden features)
- **Club**: 37-dimensional features (29 基线 + 8 xG-proxy: 运气因子 5场/12场均值 + streak + volatility，主客各4)
- Trained with Optuna-optimized defensive parameters
- Output: `[P(away), P(draw), P(home)]`

### Fusion: Entropy-based Dynamic Weighting (2026-06-08)
Replaces the old `min_games` hard threshold logic:

- **Function**: `compute_dynamic_xgb_weight(xgb_probs, alpha=0.30, beta=0.50)` in `daily_jczq.py`
- **Formula**: Shannon Entropy `E = -Σ(p·log₂(p))` → Confidence `C = 1 - E/log₂(3)` → `W_xgb = clamp(α+β·C, 0.10, 0.90)`
- **Behavior**:
  - Sharp XGB prediction (80/12/08) → `W_xgb ≈ 0.56` (high confidence)
  - Flat XGB prediction (34/33/33) → `W_xgb ≈ 0.30` (min confidence, DC dominates)
  - Strong away favorite (70/15/15) → `W_xgb ≈ 0.43` (moderate)
- **Key improvement**: Weight depends on per-match prediction quality, not training data volume
- **Homologous function** in `wc_2026_final.py` (uses numpy) — must stay in sync

### Dual-Track Model Routing (2026-06-08, 2026-06-14 增强)
`predict_match_wrapper()` routes predictions through two model tracks, **now with explicit routing logging**:

```
predict_match_wrapper(home, away)
  ├─ 1. _try_club_predict()   → elo_club + dc_club + xgb_club (Brier 0.21)
  ├─ catch: _try_hybrid_predict() → elo_intl + dc_intl + xgb_intl (Brier 0.46)
  └─ catch: return None (顶层 caller 决定是否 fallback 到 legacy_poisson)
```

**Routing logging (2026-06-14)**: Each attempt logs `{'model': ..., 'success': True/False}` to `r['routing']['tried']`. The selected model is stored in `r['routing']['selected']`. This replaces silent try/except cascade with audit-trail routing:

```python
r['routing'] = {
    'tried': [
        {'model': 'club', 'success': True/False},
        {'model': 'intl', 'success': True/False},
    ],
    'selected': 'club|intl'
}
```

**Pitfall**: If both tracks return None, the function returns None without logging a legacy attempt. The caller must log that separately. The legacy track (`predict_match_legacy()`) is NOT called inside `predict_match_wrapper()` — it's the caller's responsibility. This is by design: legacy uses Poisson+Elo which needs different data (league history), not available at the wrapper level.

### Isotonic 校准器已剥离 (2026-06-10 诊断, 2026-06-14 代码确认)

**代码修改已完成**: 
- `predict_match.py` 的 `_load_calibrators()` 直接返回 `(None, None)` → 强制走 Temperature Scaling (T=1.2)
- `daily_jczq.py` 的 `_load_club_models()` 不再加载 `calibrators_club.pkl` → `_calibrators_club` 保持 None
- 校准器文件 (`calibrated_xgb.pkl` 6.4MB, `calibrators.pkl` 2.5KB, `calibrators_club.pkl` 2.7KB) 仍存在于磁盘但不再被读取

**恢复条件**: 积累 200+ 场同质数据 (纯 XGB 输出概率 + 真实赛果) 后, 用 Sigmoid/Platt Scaling 训练专属校准器. 注意训练数据必须来自同一模型, 不能混合不同概率来源 (2024 市场赔率 + 2026 XGB 输出混训已验证失败, Brier 0.2053→0.2378).

- **Club track**: 217 teams, DC ρ=0.25, half_life=150d, XGB 37-dim (含xG-proxy)
- **International track**: 336 teams, DC ρ=0.25, half_life=540d, XGB 29-dim
- Club data: `/root/data/elo_club.pkl`, `form_club.json`, `dc_model_club.pkl`, `xgb_model_club.pkl`, `xg_proxy_club.json`
- Routing: if club form data exists → club track; else → international track

### Dynamic Market Weight (2026-06-08)
`build_prediction_bundle()` blends model probabilities with market implied probabilities:

- Uses `mc_market_weight_helper.market_weight_for_match()` for weight calculation
- Weight range: 10%-42% based on Elo gap + neutral flag + market strength
- Wider Elo gap → more market reliance (market is efficient for mismatches)
- Neutral matches → less market reliance
- Strong market confidence → modestly more weight
- Only applied when 500.com odds are available

### Fatigue Features (2026-06-08)
`fatigue_features.py` computes rotation risk from 500.com future fixtures:

- **Input**: future_fixtures (from 500.com) + match date + competition type
- **Output**: `{home/away}_rotation_risk` (0-1), `{home/away}_fatigue` (0-1), `rotation_diff`
- **Logic**: World Cup in 6 days → 70% rotation risk; in 9 days → 40%
- **Integration**: When `rotation_diff ≥ 0.1`, auto-adjusts H/D/A probabilities via `fatigue_adjustment()`
- **Pitfall**: 500.com team names have FIFA rank prefix (e.g., `[7]荷兰`), must strip `[\\d+\\]` before matching fixtures

### Tournament Stage Features (2026-06-11 Phase 1, 动态化 Phase 2)
`build_tournament_stage_features()` adds 4 dimensions for cup competition context:

- **Features**: `[points_diff, rank_diff, is_knockout, round_encoded]`
- **Integration**: Added to `_try_hybrid_predict()` after gold+odds+form features, producing 33-dim vector
- **Config source**: Dynamically loaded from `/root/data/tournament_state.json` via `_load_tournament_state()` (带缓存, 24h TTL)
- **Updated by**: `/root/update_tournament_state.py` — daily cron (02:00 UTC), fetches from football-data.org `/competitions/WC/standings` endpoint
- **Fallback**: If JSON missing or team not found, uses `DEFAULT_TOURNAMENT_STATE` (points=0, rank=2, round=1)
- **Non-cup fallback**: Returns `[0,0,0,0]` for league matches (no effect on predictions)
- **League detection**: Checks `'世界杯' in league or '杯' in league` to activate features
- **Pitfall**: football-data.org API returns different English names than expected (e.g., "South Korea" not "Korea Republic", "Ivory Coast" not "Côte d'Ivoire"). TEAM_NAME_MAP in update_tournament_state.py must match actual API output. Verify with: `python3 update_tournament_state.py --dry-run`

**Model Dimension Compatibility Pattern**:
```python
# Auto-detect model dimensions and slice features accordingly
feat_dim = _xgb_model.n_features_in_ if hasattr(_xgb_model, 'n_features_in_') else 29
if feat_dim == 29:
    feat = feat_33[:, :29]  # Use only first 29 dims for old model
else:
    feat = feat_33  # Use all 33 dims for new model
```
This allows seamless upgrade: current 29-dim model ignores new features; future 33-dim model uses them.

**Remaining feature gaps** (see `references/world-cup-feature-gaps.md`):
- Phase 2: Squad training duration & key player absence (5 dims)
- Phase 3: Referee tendencies & VAR usage (4 dims)

### Dynamic Lambda by Tournament Stage (2026-06-11)

`fallback_market_predict()` uses `STAGE_LAM` to select average goals (λ) based on tournament stage:

```python
STAGE_LAM = {
    'group':   2.55,  # 小组赛
    'last_16': 2.30,  # 16强
    'quarter': 2.10,  # 8强
    'semi':    2.00,  # 半决赛
    'final':   1.90,  # 决赛
    'third':   2.20,  # 三四名决赛
}
```

`_detect_stage(league)` extracts stage from league name via keyword matching.

**Pitfall**: Detection order matters — "1/8决赛" contains "决赛" but means round-of-16, not final. Code must check more-specific patterns first (`last_16` before `final`). Add fallback log for unrecognized World Cup stage names to catch naming inconsistencies.

### Post-Processing: Motivation Drop Rule (2026-06-11)

In `_try_hybrid_predict()`, after Draw Correction Layer, a post-processing rule handles strong teams with low motivation in group stage round 3:

**Trigger conditions** (all must be true):
1. `round_num_normalized ≈ 0.428` (round 3 of 7, range 0.33-0.53)
2. `|points_diff| >= 0.5` (≥3 points gap, clear favorite)
3. `'世界杯' in league` AND `is_knockout == False` (World Cup group stage only)

**Adjustment**: Strong team (higher points) probability reduced by 15%, split equally between draw and weak team win.

```python
# In _try_hybrid_predict(), after Draw Correction Layer:
stage_feat = build_tournament_stage_features(home, away, league)
round_num_normalized = stage_feat[3]
points_diff = stage_feat[0]

is_third_round = abs(round_num_normalized - 0.428) < 0.1
is_big_gap = abs(points_diff) >= 0.5
is_world_cup_group = league and '世界杯' in league and not stage_feat[2]

if is_third_round and is_big_gap and is_world_cup_group:
    strong_idx, weak_idx = (2, 0) if points_diff > 0 else (0, 2)
    cut_amount = hybrid[strong_idx] * 0.15
    boost_each = cut_amount / 2.0
    hybrid[strong_idx] -= cut_amount
    hybrid[1] += boost_each   # draw
    hybrid[weak_idx] += boost_each  # weak team
    # re-normalize
```

**Rationale**: World Cup round 3 frequently features qualified teams rotating squads / coasting. Historical data shows upsets increase 2-3x in this scenario.

### Pre-Match Odds Monitoring (2026-06-11)

Cron-based odds refresh system to detect stale odds risk before match kickoff:

- **Script**: `/root/scripts/pre_match_odds_refresh.py`
- **Cron**: Every 30 minutes (`*/30 * * * *`)
- **Detection**: Compares current odds vs previous snapshot, alerts on >10% change
- **Logs**: `/root/data/odds_alerts.log` (abnormal changes), `/root/data/odds_refresh.log` (run log)
- **History**: `/root/data/odds_history.json` (snapshots for comparison)

**Usage**: Before placing bets, manually run `python3 /root/scripts/pre_match_odds_refresh.py` to verify odds freshness. If alerts exist, re-run `daily_jczq.py` for updated predictions.

### Isotonic Calibration (REMOVED 2026-06-10)

**已从生产管线剥离。** Isotonic 校准器在2026年国际友谊赛数据上表现负优化（Brier从原始0.2053退化到0.2341）。详见上方 [Calibration Strategy](#calibration-strategy-2026-06-10-更新)。

保留以下历史参考信息：

- **Club XGB**: Per-class Isotonic on cal set (20%), evaluated on val set (20%)
  - Raw Brier 0.2020 → Calibrated 0.1937
  - Raw Acc 47.7% → Calibrated 53.5%
- **International XGB**: CalibratedClassifierCV (sklearn)
  - 60/20/20 train/cal/val split
- **HT/FT XGB**: Per-class Isotonic on 9-class outputs
  - Raw Acc 16.4% → Calibrated 30.3% (Isotonic is critical here)

### 500.com spf/nspf 映射修复

`scrape_500_odds_today()` 中的映射逻辑：

```python
# playid=269 数据含义:
# handicap != 0 时: spf=让球胜平负, nspf=标准1X2
# handicap == 0 时: spf=标准1X2
#
# 修复映射:
if handicap != 0 and nspf_raw and nspf_raw.get('3'):
    std_h = nspf_raw['3']  # 标准1X2
    rq_h_val = spf_raw['3']  # 让球
else:
    std_h = spf_raw['3']  # fallback: spf就是标准1X2
```

### 500.com nspf为空时的欧赔兜底 (2026-06-09 修复)

**问题**: 当 handicap≠0 且 nspf 为空时（多见于让2+球的强弱悬殊对阵），`spf` 字段是让球赔率而非标准赔率。直接赋值给 `odds_h/d/a` 会导致：
- SPF市场展示错误的赔率
- 隐含概率算错（基于让球赔率）
- EV/Kelly 完全不可信

**修复方案**: `_fetch_live_odds_map()` — 从 `live.500.com` 获取平均欧赔作为标准赔率兜底。

**技术路径**:
1. 请求 `https://live.500.com/`（GBK编码）
2. 提取 JS 变量 `liveOddsList = {...}`，其结构为 `fid -> { '0': [h, d, a], '3': [...], ... }`，其中 key `'0'` 是多家博彩公司的平均欧赔
3. 从 HTML 构建 code→fid 映射：`<input ... value="{fid}" />周二201</td>`
4. 在环中当 handicap≠0 且 nspf 为空时，用平均欧赔替换 `std_h/d/a`

**验证**（2026-06-09 实测通过）：
```
周二203 阿根廷vs冰岛: 2.24/3.45/2.55 → 1.16/6.94/14.71
周六005 卡塔尔vs瑞士: 1.98/3.85/2.74 → 12.32/6.12/1.23
周日009 德国vs库拉索: 1.94/4.60/2.52 → 1.03/16.91/44.44
```

**`apply_euro_fallback()` 已废弃**：旧方案通过 `scraper_500_analysis.py` 的分析缓存欧赔兜底，但因分析页缓存冷启动滞后于赔率数据，实际运行中几乎从不生效。已被 `_fetch_live_odds_map()` 取代。

**⚠️ 已知陷阱：live.500.com 的数据源特点**
- fid 与 trade.500.com 的 data-id 不同 — live.500.com 有自己独立的 fid
- 部分欧洲低级别联赛或无竞彩对照的比赛可能不在 liveOddsList 中
- 平均欧赔是多家博彩公司简单算术平均，未去重/未剔除 outlier
- live.500.com 页面结构可能变更 — 当前依赖的 tr id / input value 正则需保持维护

### 亚盘价值集成 (2026-06-08)

`build_prediction_bundle()` 中调用 `ah_probs()` 扫描以竞彩让球为中心的 ±2 范围盘口，输出 `ah_fair_odds` 字典。

输出示例:
```
亚盘公平赔率: AH 0.00=1.50 | AH 0.25=1.65 | AH 0.50=1.84
```

`asian_handicap.py` 函数:
- `ah_probs(λ_h, λ_a, h)` → win/push/lose + fair_odds
- `find_ah_odds(λ_h, λ_a, h, market_odds)` → EV + Kelly
- `scan_ah_value(λ_h, λ_a, [(h1,odd1), ...])` → 批量扫描正EV

### CLV 回测与赛果回填

```bash
# 抓取历史收盘赔率后运行回测
python3 /root/clv_backtest.py --fetch
python3 /root/clv_backtest.py --report

# 比赛结束后回填赛果
python3 /root/backfill_results.py
python3 /root/backfill_results.py --dry-run  # 预览
```

- **CLV回测**: `/root/clv_backtest.py` — 用500.com真实历史收盘赔率 vs 我们的早期赔率计算Closing Line Value。解决"合成赔率"回测陷阱。`--fetch`抓取历史赔率，`--report`看报告
- **赛果回填**: `/root/backfill_results.py` — 多源回填(results JSON→kaijiang CSV→football-data.org) + checkpoint + Brier Score。幂等设计，cron每天09:30+13:30北京时间。详见 [赛果回填管线](#赛果回填管线-2026-06-10-重写)
- **回填checkpoint**: `/root/data/backfill_checkpoint.json` — 记录最后成功回填日期，重启后跳过已处理范围
- **关键**: CLV 必须在同源赔率间计算 (竞彩 vs 竞彩 或 欧赔 vs 欧赔)。跨体系比较 (竞彩让球 vs 欧赔1X2) 会得到虚假负值

## 赛果回填管线 (2026-06-10 重写)

`/root/backfill_results.py` — 多源赛果回填 + Brier Score 计算

### 数据源优先级
1. `/root/data/results/YYYY-MM-DD.json` (500.com kaijiang, 每日cron生成)
2. `/root/data/historical_kaijiang.csv` (历史开奖CSV, 3248+场)
3. football-data.org API (9大联赛, 需API Key)

### 幂等设计
- 只更新 `result_status=missing` 的记录
- 已有 `actual_hda` 的记录永不覆盖
- 多源结果冲突时标记 `result_status=conflict`
- checkpoint `/root/data/backfill_checkpoint.json` 记录最后成功处理的日期

### Brier Score 计算
多分类 Brier: `(1/r) * Σ_j (I_j - p_j)²`, r=3 (H/D/A)
填充赛果后自动计算写入 `brier_spf` 字段。

### Cron 配置
- `backfill-am` (`6d912cb676ec`): 每天 UTC 01:30 (北京09:30)
- `backfill-pm` (`571c46a2a622`): 每天 UTC 05:30 (北京13:30)
- 幂等设计：两次运行互补，已成功的快速跳过

### CLI 用法
```bash
python3 /root/backfill_results.py                        # 回填所有缺失赛果
python3 /root/backfill_results.py --from-date 2026-06-01 # 从指定日期开始
python3 /root/backfill_results.py --to-date 2026-06-09   # 到指定日期截止
python3 /root/backfill_results.py --dry-run              # 只展示不修改
python3 /root/backfill_results.py --source results,kaijiang  # 指定数据源
python3 /root/backfill_results.py --stats                # 显示回填统计+Brier分析
python3 /root/backfill_results.py --report               # 每日趋势报告: Brier drift + 联赛分级 + 行动建议
```

### 首次回填结果 (2026-06-14 更新)
- 赛果覆盖: 107/174 (61.5%)
- Brier覆盖: 96/174 (55.2%)
- 平均 Brier (SPF): 0.2465 (n=96)
- 每日趋势 via `--report`: Brier drift 检测, 联赛分级, 行动建议
- 未填充原因: 8条未来比赛(06-16/06-18) + 8条kaijiang无匹配(周一/周三场次code不同)

### CSV 新增字段
`brier_spf`, `result_status` (missing/filled/conflict/postponed), `settled_at`, `backfill_source`, `match_key`

## Output Display Protocol (2026-06-10)

**CRITICAL: Raw terminal output must be displayed in full, not summarized.**

When the user asks to see predictions, the display protocol IS PART OF THE WORKFLOW, not optional. Do NOT output summaries like "已执行完毕" / "共26场" / "以下省略" / "内容与之前一致". The user's exact requirement:

> "不要给我概况，不要给我摘要，如果聊天窗口有限制就分块贴出来"

### Steps (non-negotiable)

1. **Run `_show_tomorrow.py` with file redirection**:
   ```bash
   python3 /root/_show_tomorrow.py $(date +%F) > /root/data/show_output.txt 2>&1
   ```

2. **Count total lines**:
   ```bash
   wc -l /root/data/show_output.txt
   ```

3. **Display in 60-line chunks using `sed -n`**, continuing until the last line:
   ```bash
   sed -n '1,60p' /root/data/show_output.txt
   sed -n '61,120p' /root/data/show_output.txt
   sed -n '121,180p' /root/data/show_output.txt
   ...
   ```

4. **For each chunk**, output ONLY:
   - `开始第X/Y块` (exact one-line header, no embellishment)
   - The raw chunk content

   No summaries, no interpretation, no "以下省略", no "已显示前X行", no "共X场匹配" before chunks, no bullet-point recap after chunks. The user's explicit words:
   > "不要给我概况，不要给我摘要，不要告诉我'已经保存到文件'"
   > "输出内容只能是两部分：简短说明'开始第X/Y块' + 对应分块的原始文本"
   > "不允许插入你自己的概括、解读、摘要"

5. **File attachment fallback**: If chat window truncation persists despite 60-line chunking, provide the file path (`/root/data/show_output.txt`) AND continue posting chunks. Do NOT replace chunks with the file path — both must be provided.

### When this protocol applies

Trigger when user says any of:
- `显示预测结果` / `输出预测`
- `我要内容` / `把完整输出返回给我`
- `把脚本的终端原始输出完整返回给我`
- Any request that follows a predict-and-show workflow

The `_show_tomorrow.py` script reads `/root/data/predictions_log.csv` (written by `daily_jczq.py`) and formats the output. It supports filtering by date, competition prefix, or match code.

## CSV Schema (2026-06-10 扩展)

predictions_log.csv 新增字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| bet_action | str | 赛事过滤标签 (RECOMMEND/WATCH/WATCH_FRIENDLY/SKIP_LEAGUE) |
| model_route | str | 模型路由 (hybrid/market_fallback/club) |
| match_key | str | 稳定主键: date\|league\|home\|away\|time |
| pred30_h/d/a | float | A/B测试: 30维模型概率(%) |
| brier_spf | float | 单场Brier Score (胜平负) |
| result_status | str | missing/filled/conflict/postponed |
| settled_at | str | 回填完成时间 (ISO) |
| backfill_source | str | 回填数据来源 (kaijiang/results:YYYY-MM-DD) |

## 赛果回填管线 (2026-06-10 上线)

- **脚本**: `/root/backfill_results.py` — 多源回填 + Brier Score 计算
- **数据源优先级**: results JSON → kaijiang CSV (可配置)
- **幂等设计**: 只填充 result_status=missing 的记录
- **Checkpoint**: `/root/data/backfill_checkpoint.json`
- **Cron**: `backfill-am` (UTC 01:30 = 北京09:30), `backfill-pm` (UTC 05:30 = 北京13:30)
- **用法**: `python3 backfill_results.py [--dry-run] [--from-date] [--to-date] [--stats]`

## bet_action 逻辑 (2026-06-11 updated)

`compute_bet_action()` 在 `daily_jczq.py` 中:
- UEFA Nations League → SKIP_LEAGUE
- market_fallback + 世界杯 → **SKIP_WORLD_CUP_FALLBACK** (EV循环论证 + 高波动风险)
- market_fallback + 世界杯 → **SKIP_WORLD_CUP_FALLBACK** (EV循环论证 + 高波动风险)
- market_fallback (其他) → WATCH
- 友谊赛 → **WATCH_FRIENDLY** (硬编码, 无margin门槛)
- 其他 (包括世界杯/亚洲杯/欧国联等正赛) → RECOMMEND

**重要**: 联赛名称从500.com动态提取（`m5.get('league', '')`），不再硬编码。世界杯、亚洲杯等大赛现在正确识别为正赛，不会被误判为友谊赛。

友谊赛降级原因: Isotonic校准器在友谊赛上严重过度自信 (RECOMMEND组70%置信度, 0%命中率)。恢复条件: 积累200+场2026年回填数据后重训校准器。

## A/B测试: 29维 vs 30维 (2026-06-10 上线)

- `_load_shared_models()` 自动加载 `xgb_model_30.pkl` (存在时)
- `_try_hybrid_predict()` 并行推理29维和30维, 30维结果写入 `pred30_h/d/a`
- 30维推理完全隔离: 失败时不影响主路由
- 30维结果不参与bet_action逻辑或终端展示
- 目标: 2-3周后对比 Brier(xgb29) vs Brier(xgb30)

## Current production files

- Main script: `/root/daily_jczq.py`
- Ledger: `/root/data/predictions_log.csv`
- **世界杯特征盲区分析**: `references/world-cup-feature-gaps.md` — 当前29维特征构成、Top 3杯赛盲区(赛事阶段/集训磨合/裁判因素)、新增特征建议代码、实施路线图
- **赛事阶段特征函数**: `build_tournament_stage_features()` in `daily_jczq.py` — 4维特征(points_diff, rank_diff, is_knockout, round_encoded), 从 `tournament_state.json` 动态加载
- **赛事状态更新脚本**: `/root/update_tournament_state.py` — 从 football-data.org `/standings` 端点获取世界杯积分榜, 带24h缓存, cron每天02:00 UTC运行
- **赛事状态数据**: `/root/data/tournament_state.json` — 48支球队的积分/排名/轮次 (由 update_tournament_state.py 生成)
- **赛事状态缓存**: `/root/data/tournament_api_cache.json` — API响应缓存 (24h有效期, 避免限流)
- Form updater (365scores): `/root/update_form_from_365.py`
- Training data builder: `/root/build_training_data.py`
- XGB retrain (simple): `/root/retrain_xgb_simple.py`
- 500.com breaker log: `/root/data/500breaker.log`
- Backtest script: `/root/.hermes/scripts/backtest_jczq.py`
- **Kelly策略回测**: `/root/backtest_kelly.py` — 三方案对比(Q-Kelly 5%/H-Kelly 3%/Q-Kelly 8%)
- **DC真实赔率诊断**: `/root/dc_real_odds_test.py` — DC vs 500.com真实赔率验证
- 每日预警脚本: `/root/daily_alert.py` — 运行daily_jczq.py提取价值投注汇总
- **展示工具**: `/root/_show_tomorrow.py` — 从 predictions_log.csv 读取并格式化输出 5 玩法预测，支持按日期/竞彩前缀/场次编码过滤。用法: `python3 _show_tomorrow.py [周四|2026-06-14|周四002]`
- **每日预警cron**: job `3b404abedaf4` — 每天08:00 UTC运行daily_alert.py
- 365scores fetcher: `/root/fetch_365scores.py` (含 `filter_sid=1` 参数, 见 `references/365scores-api-endpoint-investigation.md`)
- 365scores cron writer: `/root/collect_365scores_daily.py` (cron `3fee9087ae2c` 02:00 UTC, 通过 SID=1 过滤纯足球)
- half_full_model module: `/root/wc_2026_upgrade/half_full_model.py`
- **Gold特征补全**: `/root/feature_helper.py` — H2H+12场form缓存, 修复gold特征train-serve skew
- 365scores后验调整器: `/root/scores365_adjuster.py` — 投票/趋势/人气3信号融合, ±5pp上限
- 365scores综合数据(阵容/伤病/H2H/xG/公众投票): 见 `references/365scores-enrichment-data.md`
- **回测管线**: `/root/backtest_pipeline.py`
- **赛果回填**: `/root/backfill_results.py` — 多源回填+checkpoint+brier计算, 每天cron两次(backfill-am 01:30UTC, backfill-pm 05:30UTC)
- **校准分析**: `/root/calibration_analysis.py` — 校准曲线PNG+数值诊断
- **sigmoid校准器训练**: `/root/train_sigmoid_calibrator.py` — Platt Scaling (暂不适用, 异质数据问题)
- **form_state更新**: `/root/update_form_state.py` — 每日06:00 cron自动更新form+缓存
- **俱乐部数据管线**: `/root/club_data_pipeline.py` — 构建俱乐部Elo+form+DC
- **俱乐部XGB训练**: `/root/train_xgb_club.py` — 37维俱乐部专用XGB (Brier=0.1937 cal)
- **xG-proxy特征**: `/root/xg_proxy.py` — luck_factor = actual_goals - DC_lambda, 8维(主客各4: proxy_5/12, streak, volatility)
- **xG-proxy数据**: `/root/data/xg_proxy_club.json` — 217队运气因子状态快照
- **EV/Kelly模块**: `/root/bet_math.py` — 单场赔率EV计算+Kelly Criterion仓位推荐，已集成到daily_jczq.py输出端。**安全审计(2026-06-11)**: Kelly钳位5%、is_sane_bet五道保险、同场相关性折扣。详见 `references/bet-math-safety-audit-20260611.md`
- **500.com HTML结构逆向分析**: `references/500-html-structure-analysis.md` — 500.com竞彩页面DOM完整逆向：`tr.bet-tb-tr`的data-*属性(让球值fixture_id等)、赔率data-type/data-value/data-sp三属性、spf/nspf语义区分(让球≠0时spf是让球赔率非标准1X2)、抓取代码模板
- **500.com分析爬虫v2**: `/root/scraper_500_analysis.py` — 抓取FIFA排名/近期战绩/赢盘率/大球率/澳门心水/亚盘/首发阵容/世界杯赛程 + **v2新增: 历史10场逐场欧赔+亚盘(含盘路/大小)、当前欧赔/亚盘、matchid主键、未来赛事(世界杯/欧国联/亚洲杯)完整赛程**。1小时缓存，已集成到daily_jczq.py展示层。详见 `references/500-analysis-scraping.md`
- **疲劳度特征**: `/root/fatigue_features.py` — 从500.com未来赛事计算轮换风险(0-1)/疲劳度/主客差异。世界杯前6天友谊赛→轮换70%。已集成到daily_jczq.py，rotation_diff≥0.1时自动调整概率。详见 `references/500-analysis-scraping.md`
- **赛事阶段动态Lambda**: `STAGE_LAM` in `daily_jczq.py` — 按小组赛/淘汰赛阶段选择Poisson平均进球数(2.55→1.90)，`_detect_stage()` 从联赛名提取阶段关键词。详见上方 [Dynamic Lambda by Tournament Stage](#dynamic-lambda-by-tournament-stage-2026-06-11)
- **亚盘价值计算**: `/root/asian_handicap.py` — Skellam分布计算任意盘口的赢盘/走水/输盘概率。`find_ah_odds()`单场EV+Kelly, `scan_ah_value()`批量扫描。详见 `references/asian-handicap-skellam.md`
- **CLV回测**: `/root/clv_backtest.py` — 用500.com真实历史收盘赔率 vs 我们的早期赔率计算Closing Line Value。解决"合成赔率"回测陷阱。`--fetch`抓取历史赔率，`--report`看报告
- **赛果回填**: `/root/backfill_results.py` — 多源回填(results JSON→kaijiang CSV→football-data.org) + checkpoint + Brier Score。幂等设计，cron每天09:30+13:30北京时间。详见 [赛果回填管线](#赛果回填管线-2026-06-10-重写)
- **回填checkpoint**: `/root/data/backfill_checkpoint.json` — 记录最后成功回填日期，重启后跳过已处理范围
- **半全场XGB训练**: `/root/train_htft_club.py` — 9分类半全场模型 (acc=30.3%)
- **半全场预测器**: `/root/htft_predictor.py` — 替代r_ht=0.45的纯数学推导
- **联赛数据拉取**: `/root/fetch_league_data.py` — football-data.org 9联赛历史数据
- **500.com异步爬虫**: `/root/wc_2026_upgrade/async_500_scraper.py` — aiohttp+BeautifulSoup, 4玩法并发~2秒, 按data-fixtureid合并
- **历史开奖爬虫**: `/root/wc_2026_upgrade/historical_kaijiang.py` — 从zx.500.com抓取收盘SP赔率, 3248场CSV, 支持断点续传
- **真实赔率回测**: `/root/wc_2026_upgrade/real_odds_backtest.py` — 30维特征+市场赔率+Isotonic校准+赛事过滤, 真实ROI=+69.86%
- **训练数据准备**: `/root/wc_2026_upgrade/prepare_training_data.py` — 合并kaijiang+国际赛, 生成带市场赔率的训练集
- **XGB重训练**: `/root/wc_2026_upgrade/retrain_xgb_with_odds.py` — 用市场赔率+赛事阶段特征重训XGBoost, 输出xgb_model_33.pkl (34维). 详见 `references/xgb-retraining-33dim.md` 和 `references/33dim-training-and-postprocessing.md`
- `/root/data/team_name_mapping.json` — 101条中文→英文队名映射
- **赛前赔率监控**: `/root/scripts/pre_match_odds_refresh.py` — 每30分钟cron检查盘口变化, >10%波动报警. 日志: `/root/data/odds_alerts.log`, 快照: `/root/data/odds_history.json`
- 赛事过滤标签: `bet_action` in bundle dict (RECOMMEND/WATCH/SKIP_LEAGUE)。详见 `references/bet-action-filtering.md`
- EV/Kelly风控: `bet_math.is_sane_bet()` 过滤 odds>30/prob<15%/fallback比分&半全场
- **P0热修复记录 (2026-06-14)**: `references/2026-06-14-p0-fixes.md` — 校准器全量剥离、路由日志、500.com熔断兜底
- **500.com wanchang 完场比分抓取**: `references/500-wanchang-scraping.md` — 2026-06-15 确认: 静态 GBK HTML 非 SPA, curl 替代 requests 防超时, 已验证 63K+ 场回填
- **系统全景分析 (2026-06-15)**: 见 references/500-wanchang-scraping.md#训练数据断档 — 最大瓶颈: training_data_with_odds.json 263场止于2024-11
- **全面系统审计 (2026-06-14)**: `references/2026-06-14-comprehensive-audit.md` — 4层诊断结果、8个关键问题(P0-3/P1-2/P2-3)、模型架构评估、算法配合分析
- **P0热修复记录 (2026-06-14)**: `references/2026-06-14-p0-fixes.md` — 校准器全量剥离、路由日志、500.com熔断兜底
## Scope Rule (2026-06-09; 2026-06-10 override note)

**Default mode: Only predict matches that play TOMORROW (the next calendar day).**

If a match is available for betting today but won't play until 3 days later, **do not predict it**. The user's exact instruction:

> "预测只做隔天的，比如明天将要比赛的才做预测，明天不比，即使可以买了也不预测"

Filtering logic: check match `endtime` field — if it does NOT start with `MM-DD` of tomorrow's date, skip it. Tomorrow = `date.today() + timedelta(days=1)`.

This applies to the final output AND to value-bet summaries. Do not include matches outside tomorrow's scope in any output table.

**Override: "今天可购买" mode.** When the user explicitly asks for "今天可购买比赛" / "today's buyable matches" / "今天买的所有场次" / "重新完整执行今天可购买竞彩足球比赛", the tomorrow-only filter does NOT apply. Show ALL matches currently for sale on 500.com regardless of their kickoff date. The `_show_tomorrow.py $(date +%F)` command inherently returns the full buyable set; do not add an additional tomorrow filter on top of it.

Detect the override: user says "今天可购买" or "today's buyable" or "全部" or "所有可买" or "完整执行" — these all signal ALL-mode. User says "明天的" or "隔天" — these signal tomorrow-only mode.

## Exact Execution Steps

### Step 0. Determine tomorrow's date

```python
import datetime
tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
# e.g. if today is 2026-06-09, tomorrow_str matches '06-10'
tomorrow_mmdd = tomorrow[5:]  # '06-10'
```

### Step 1. Run the production script

Use:

```bash
python3 /root/daily_jczq.py
```

This script already does the operational chain:

1. find today's buyable matches from `500.com`
2. fetch `365scores` enhancement signals (prefers daily CSV cache, falls back to live API)
3. run current model stack
4. print the predictions
5. write/update `/root/data/predictions_log.csv`

### Step 1b. Verify odds mapping before trusting EV/Kelly

Before presenting output, verify that today's odds mapping is correct:

```bash
python3 /root/.hermes/skills/knowledge/daily-jczq-prediction/scripts/verify_odds_mapping.py
```

If affected matches > 0, those matches' SPF display odds, implied probabilities, and EV values are **unreliable**. For affected matches where model has hybrid prediction (DC+XGB), the model probabilities themselves are OK but EV/Kelly against the wrong market odds is not.

### Step 1c. nspf-empty matches: automated euro odds fix

In `daily_jczq.py`, `scrape_500_odds_today()` now calls `_fetch_live_odds_map()` at startup to fetch average euro odds from `live.500.com` as a fallback. When `handicap != 0` and `nspf` is empty, it replaces the wrongly-mapped spf (handicap) odds with the live euro average.

This is the **preferred** fix path — it covers all affected matches in one shot without per-match manual scraping.

If the automated fix fails or produces suspicious values, fall back to manual per-match scraping:

```bash
# 1. Find shuju_id from trade page HTML
# Search for: data-homesxname="TeamA" data-awaysxname="TeamB" + nearby href="/fenxi/shuju-XXXXXX.shtml"
# 2. Fetch analysis page
# https://odds.500.com/fenxi/shuju-{sid}.shtml
# 3. Extract euro odds
# Regex: <p class="pub_table_pl"><span>([\d.]+)</span><span>([\d.]+)</span><span>([\d.]+)</span></p>
# 4. Use euro odds as std_h/std_d/std_a in prediction bundle
```

Verified working example (2026-06-09):
- Argentina vs Iceland (shuju-1405672) → euro odds 1.16/6.96/14.74

### Step 2. Filter output for tomorrow only

After the script finishes, **discard predictions for matches not playing tomorrow**. Apply the `tomorrow_mmdd` filter to match endtime:

```python
for match in all_predictions:
    if match['endtime'].startswith(tomorrow_mmdd):
        output.append(match)
```

Report only the filtered set to the user.

### Step 2. Treat script output as the primary source of truth

Do **not** manually rewrite picks from memory.

Use the actual script output for:

- `胜平负` probabilities and pick
- `让球` probabilities and pick
- `半全场` top outcomes and main pick
- `比分` top outcomes and main pick
- `总进球数` distribution and main pick

### Step 3. Preserve the display order

For each match, output in this order:

1. **500.com分析**: FIFA排名 → 近10场 → 赢盘率/大球率 → 近3场历史赔率 → 交战历史 → 澳门推介 → 亚盘 → 世界杯赛程 → 疲劳度特征
2. `胜平负`
3. `竞彩让球`
4. `半全场`
5. `比分`
6. `总进球数`

This is the user-facing contract.

### Step 4. Keep settlement scope explicit

Every reply should clearly state:

- `90分钟常规时间（含伤停补时）`
- `不含加时赛和点球大战`

## Output Rules

### Bet Action Filtering (2026-06-10)

Each bundle now carries a `bet_action` label: `'RECOMMEND'` | `'WATCH'` | `'SKIP_LEAGUE'`.

Rules applied in `build_prediction_bundle()` → `compute_bet_action()`:

1. **UEFA Nations League** → `SKIP_LEAGUE` (historical ROI -72.5%)
2. **友谊赛 + max margin < 20pp** → `WATCH` (historical ROI -58.1%)
3. All others → `RECOMMEND`

`margin_pp` = max edge across all scenarios: `max(s.prob - 1/s.odds) × 100` for all s.odds > 1.

**Display**: per-match shows `👀 bet_action: WATCH（友谊赛 margin<20pp，仅观察不推荐）` or `🚫 bet_action: SKIP_LEAGUE（...）`.

**Global summary filtering**: `format_value_summary()` receives only RECOMMEND bundles. SKIP_LEAGUE and WATCH bundles are excluded from the value-bet table. The summary line reports count: `ℹ️ 已过滤 N 场赛事类型不推荐场次 (SKIP_LEAGUE/WATCH)`.

**Per-match display**: WATCH/SKIP_LEAGUE matches still show full 5-play predictions with all details (model stays transparent). Only the value-bet summary table is filtered.

### HT/FT 半全场胜胜 低概率外推标记 (2026-06-10)

When `s.play == '半全场'` and `s.pick == '胜胜'` and `s.prob < 0.20` and `model_type == 'hybrid'`:
- Bundle sets `htft_warning = True`
- Per-match display shows: `⚠️ 半全场胜胜: 低概率外推(模型概率<20%)，仅供参考`

This does NOT suppress the value bet from the table — it only adds a warning label to the per-match output. The `is_sane_bet()` longshot filter handles suppression separately.

### Required match-level fields

For each buyable match, include all of the following:

- Match code + teams + kickoff time
1. **500.com分析**: FIFA排名 → 近10场 → 赢盘率/大球率 → 近3场历史赔率(欧赔+亚盘) → 交战历史 → 澳门推介 → 亚盘 → 世界杯赛程 → 疲劳度(轮换风险/距下一场天数)
- `胜平负`: full `主/平/客` probabilities and main pick
- `竞彩让球`: handicap direction, full `让胜/让平/让负` probabilities and main pick
- `半全场`: main pick plus top outcomes
- `比分`: main score plus top 6-8 outcomes
- `总进球数`: main pick plus distribution summary
- `365scores` support info if available: vote split, sample size, recent trend

### Pick integrity rules

Main picks must come from each play's own argmax, not from narrative override:

- `胜平负` main pick = max of `pred_h / pred_d / pred_a`
- `让球` main pick = max of `pred_rq_win / pred_rq_draw / pred_rq_loss`
- `半全场` main pick = max of `htft_probs`
- `比分` main pick = first item of the score ranking
- `总进球数` main pick = max of the goals distribution

Do not replace the main pick with a market-lean label.

### Market information placement

Market information may appear only as:

- `市场分歧`
- `市场倾向`
- `市场价值提示`
- `价值投注` (EV/Kelly analysis from bet_math.py)

It must not overwrite the model main pick.

### EV/Kelly Analysis Output

After all match predictions, `daily_jczq.py` outputs:

1. **Per-match value line**: When EV > 2% AND `is_sane_bet()` passes, shows top 3 value bets with EV and Half-Kelly
2. **Global summary table**: All value bets across all matches (EV > 5% AND `is_sane_bet()` passes), sorted by EV, with filter count
3. **Quarter-Kelly total position**: Sum of Quarter-Kelly across all filtered value bets, **with correlation discount** (same-match bets grouped, max per group taken), capped at 15% daily

**Longshot Bias 过滤** (`is_sane_bet()` 五道保险, 统一应用):
- 保险1: odds > 30 → 跳过 (数字海市蜃楼)
- 保险2: prob < 15% → 跳过 (低信心不上榜)
- 保险3: market_fallback + 比分/半全场 → 跳过 (泊松外推不可信)
- 保险4: odds > 5 + prob < 25% → 跳过 (世界杯爆冷高发区)
- 保险5: market_fallback + 胜平负/让球 → 跳过 (EV循环论证)

**Kelly 安全**: `compute_kelly()` 输出钳位到 `MAX_SINGLE_BET = 0.05` (单注≤5%总资金)，防止极端 edge 导致仓位过大。

**同场相关性警告**: 当同一场比赛有 >1 个 value bet 时，输出 `⚠️ 同场 N 注高度正相关` 警告。

详见 `references/bet-math-safety-audit-20260611.md`。

The EV/Kelly analysis uses `bet_math.py` module. See `references/bet-math-ev-kelly.md` for formulas, data structures, and `is_sane_bet()` filter details.

### Backtesting Pitfalls

See `references/backtest-methodology.md` for the critical synthetic odds trap and correct backtesting methodology.

## Calibration Strategy (2026-06-10 更新)

### Isotonic 校准器已于 2026-06-10 剥离

**现状：不再使用任何全局校准器。** Isotonic 校准器已在 `_try_hybrid_predict()` 和 `_try_club_predict()` 中被完全注释掉。

**剥离原因（2026-06-10 生产诊断）：**
- 生产 Brier=0.2341（32场已回填） vs 训练验证 Brier=0.2053 — 退化+14%
- 校准分箱检查：几乎所有分箱都存在严重系统性偏倚（calib_error +20~75pp）
- 平局校准：概率0-20%区间实际平局率60%（校准器错误地将A结果拉到中间区）
- 根因：校准器训练数据（263场2024年数据，features=market_implied）与生产数据（2026年XGB输出）特征分布不同，Isotonic 在小样本上产生过拟合补偿

**恢复条件：** 积累200+场同质数据后，用纯XGB输出概率训练专属校准器（优先选 Sigmoid/Platt Scaling 而非 Isotonic）。

### Empirical Smoothing Overlay (2026-06-20, 叠加在 Draw Correction 之上)

`_calibrate_xgb_probs()` 在 `build_prediction_bundle()` 的市场融合后、×100 前调用。只对 `xgb` 路径生效。

- **Cap 0.75**: 主胜/客胜 >75% 触发降温
- **pass_frac 0.3**: 超出部分的 30% 保留
- **draw_frac 0.5**: 超出部分的 50% 注入平局
- **效果**: xgb Brier 0.2541 → 0.2393 (-5.8%), Acc 50% 不变
- **相关 skill**: `xgb-calibration`

详见 `references/brier-clean-evaluation.md` 的 what-if 回测方法。

### Draw Correction Layer (2026-06-10 植入)

**问题：** 诊断发现0/32场预测平局，3分类退化为2分类。根因：
- DC 独立 Poisson 假设天然低估平局
- Elo 二元胜负偏向进一步压缩平局空间
- 训练数据平局比例~25%但模型学到"宁可错也不猜平"

**解决方案：** 在两个推理路径（`_try_hybrid_predict`, `_try_club_predict`）的最终概率输出前植入：

```python
if hybrid[1] < 0.15:  # p_draw < 15%
    confidence = max(hybrid[2], hybrid[0])  # 主胜或客胜的最高概率
    draw_boost = 0.05 * (1.0 - confidence)  # 保守补偿系数
    hybrid[1] += draw_boost
    denom = hybrid[2] + hybrid[0] + 1e-10
    hybrid[2] -= draw_boost * (hybrid[2] / denom)
    hybrid[0] -= draw_boost * (hybrid[0] / denom)
    # 重新归一化
```

**效果：** 平局从~5%提升到~7%（低置信场景），高置信场景不受影响。总和恒为1.0。

### model_route 追踪修复 (2026-06-10)

**问题：** `predictions_log.csv` 的 `model_route` 字段100%为空。根因链：`build_prediction_bundle()` 返回值中没有 `'model'` 键 → `record_prediction()` 的 `--model-route` 传空 → `backtest_jczq.py` 写空。

**修复：** `build_prediction_bundle()` 返回值新增 `'model': p.get('model', 'unknown')`，其中 `p['model']` 来自四种生产路径：
- `_try_hybrid_predict()` → `'hybrid'`
- `_try_club_predict()` → `'club_hybrid'`
- `predict_match_legacy()` → `'legacy_poisson'`
- `fallback_market_predict()` → `'market_fallback'`

**兜底：** 当上游未设 `model` 键时，写入 `'unknown'` 以区分"空值"和"未知路由"。

**教训：** 新增 bundle 字段时必须同步三处：(a) `build_prediction_bundle()` 返回值 (b) `backtest_jczq.py` FIELDS + cmd_record 解析 (c) `daily_jczq.py` `record_prediction()` 传参。

## Defense-in-Depth: 数据质量防御体系 (2026-06-20)

防止假预测(0%概率)写入 predictions_log.csv 的多层防御:

```
L1 源头: fallback_market_predict() 空赔率时 return None
L2 入库: record_prediction() pred_h/d/a 全0时 skip
L3 兜底: implied_probs_from_odds() 全0赔率→均匀分布
L4 评估: evaluate_brier.py clean() 后过滤0%假样本
```

### match_date 空值源头修复 (2026-06-20)
`_thestats_list_todays_matches()` 在 TheStatsAPI 兜底路径中未设置 `match_date` 字段。修复: result.append 加入 `'match_date': today_s`。

## Brier 评估清洗方法

新的 `evaluate_brier.py` 支持三版本对比：RAW(原始) / NO-ZERO(去伪) / CLEAN(去伪+去重)。
详见 `references/brier-clean-evaluation.md`。

## A/B Shadow Deployment (2026-06-10 起, 2026-06-11 升级)

`xgb_model_33.pkl`(34维: 29基线 + 1市场赔率 + 4赛事阶段)作为影子模型并行推理，结果写入`pred30_h/d/a`字段。主路由仍用`xgb_model_29.pkl`。500.com赔率缺失时34维自动跳过(except pass隔离)。影子模型加载优先级: xgb_model_33 > xgb_model_30 > 不加载。

## Backfill Pipeline (2026-06-10 上线)

- 脚本: `/root/backfill_results.py` — 多源(results JSON→kaijiang CSV), 幂等, checkpoint, Brier自动计算
- Cron: `backfill-am`(01:30 UTC) + `backfill-pm`(05:30 UTC), 每天两次互补
- Checkpoint: `/root/data/backfill_checkpoint.json`
- CSV扩展字段: `brier_spf`, `result_status`, `settled_at`, `backfill_source`, `match_key`, `pred30_h/d/a`

## Recommended Response Pattern

Use this structure:

1. One-line conclusion first: how many buyable matches today and what competition types they are
2. Then per match, output the full 5-play prediction block
3. Then optional short ranking: strongest 1-3 signals vs skip/caution matches

### 应对用户直觉反驳：模型≠直觉时

用户可能会质疑预测（如"巴西打日本不应该稳赢吗？"）。遵循数据驱动回复框架：

1. **承认合理性** — "你的直觉是正常的"（不否定用户）
2. **亮出核心数据** — 模型概率（主58.6% / 平24.2% / 客17.2%）还是看好巴西，但不是碾压
3. **拆解原因** — 近况胜率（25%）、对手小组赛韧性、淘汰赛保守倾向、Pinnacle聪明钱态度
4. **让球选项的覆盖逻辑** — 让负（巴西平或输）覆盖结果面最宽，不是"巴西会输"

不要只说"模型预测它是..."，要给数据根因：近况趋势、赔率变化、对手小组赛表现。

### 让球推荐必须用竞彩标准术语

**永远用主场视角的竞彩标准术语表述让球推荐：让胜/让平/让负。**
不要用"日本+1"、"巴西-1"这类国际盘口表述，用户会混淆。

正确：
- "推荐：让负（巴西-1负=巴西赢不了盘）"
- "让负覆盖：巴西平局 或 日本胜"

错误：
- "买日本+1" ❌
- "让负（巴西-1负=日本+1）" ❌（加了括号解释反而更乱）

只需要说"让负"并附带**主队视角的一行说明**即可。

### Step 5. Display results via _show_tomorrow.py (mandatory)

Do NOT verbally summarize or reformat the output. Use the [Output Display Protocol](#output-display-protocol-2026-06-10) below — save to file, chunk in 60-line blocks, display raw.

```bash
python3 /root/_show_tomorrow.py $(date +%F) > /root/data/show_output.txt 2>&1
wc -l /root/data/show_output.txt
# Then display in 60-line chunks
```

## 用户决策原则 (2026-06-10)

**评估闭环 > 模型复杂度**: 用户明确优先级——先打通赛果回填+Brier监控，再切换模型版本(29→30维)。没有真实标签，模型版本比较、校准器重训、bet_action效果评估都是空谈。任何涉及"要不要用新模型/新特征/新校准器"的决策，都必须先有回填数据支撑。

## Quality Checks

After running the script, verify at least these:

- [ ] output contains today's match count
- [ ] each match has all 5 plays
- [ ] script wrote or updated `/root/data/predictions_log.csv`
- [ ] no manual probability rewriting in the final answer

### Quick Sanity Checks (pre-retraining)

Before retraining any XGB model, run these checks to catch silent data corruption:

```bash
# 1. 检查训练数据标签类型 (P0-1: spf_result int/str 混型检测)
python3 -c "
import json
d=json.load(open('/root/data/training_data_with_odds.json'))
int_ct=sum(1 for m in d if isinstance(m.get('spf_result'),int))
str_ct=sum(1 for m in d if isinstance(m.get('spf_result'),str))
print(f'spf_result: int={int_ct} str={str_ct}')
assert int_ct==0 or str_ct > int_ct, 'int类型过多, 训练脚本会映射错误标签!'
"

# 2. 检查管线中是否有 draw=0 硬编码 (P0-2: 平局灭绝检测)
grep -rn 'arr.*\[.*0.*1-' /root/*.py /root/wc_2026_upgrade/*.py 2>/dev/null || echo 'no draw=0 arrays found'
# 检查 _blend_with_market 函数是否有 draw=0
grep -A5 'elo_arr.*=.*np.array\|mkt_arr.*=.*np.array' /root/wc_2026_upgrade/calibrated_predictor.py

# 3. 检查校准器是否仍在使用 (P1-1: 校准器残留检测)
grep -rn 'calibrat.*predict\|_cal.*predict' /root/*.py /root/wc_2026_upgrade/*.py 2>/dev/null || echo 'calibrator clean'

# 4. 检查双管线模型是否一致 (P0-3: 模型不一致检测)
echo "daily_jczq model:"
grep -n 'xgb_model.*=.*joblib.load' /root/daily_jczq.py | head -3
echo "calibrated_predictor model:"
grep -n 'xgb_model.*=.*joblib.load' /root/wc_2026_upgrade/calibrated_predictor.py
```

## Historical Backtest Baseline (2026-06-14 Updated)

**nat model (11-dim, clean labels):**
- Validation accuracy: 75.4% (n=118)
- Brier Score: 0.1339 (random=0.667)
- LogLoss: 0.819
- Draw recall: 64.7%, Home recall: 84.3%, Away recall: 72.7%

Run `python3 /root/backtest_pipeline.py --backtest --n 600` for legacy 29-dim model baseline:
- Brier Score: 0.4613 (random=0.667)
- Accuracy: 64.5% (random=33.3%)

## Real-Time Brier Monitoring (2026-06-10)

赛果回填闭环打通后，可实时监控模型校准质量:
```bash
python3 /root/backfill_results.py --stats
```
输出包含:
- 总体 Brier Score (SPF): 平均/最小/最大
- 按 model_route 切分: hybrid vs market_fallback vs club
- 按 bet_action 切分: RECOMMEND vs WATCH 命中率

首次基线 (2026-06-10, n=32): **Brier = 0.2341**
- 随机基线: 0.667 (3分类均匀分布)
- 当前模型: 0.2341 (比随机好 65%)
- 校准器训练数据仅263场(2024年)，2026年分布偏移可能导致Brier上升

## XGB Retraining with Market Odds (2026-06-09)

### Problem
Original XGB model (xgb_model_29.pkl) was trained without market odds features, despite market odds being the strongest single predictor of match outcomes.

### Solution
1. **Data preparation**: `prepare_training_data.py` merges kaijiang closing SP with international_results.json
2. **Feature engineering**: Added 30th feature `market_implied = 1/sp`
3. **Training**: `retrain_xgb_with_odds.py` trains new XGB model
4. **Calibration**: New Isotonic calibrators trained on validation set

### Key Findings
- **Market odds importance**: 15.32% (rank #1 among 30 features)
- **Top 5 features**: market_implied > op_h > elo_diff > dc_h > lam_ratio
- **Model performance**: Mean LogLoss 0.6410, Mean Accuracy 77.9%

### Files
- Training data: `/root/data/training_data_with_odds.json`
- New model: `/root/data/xgb_model_30.pkl`
- New calibrators: `/root/data/calibrators_v2.pkl`
- Training report: `/root/data/retrain_report.json`

### Retraining Command
```bash
# Step 1: Prepare training data (with tournament stage features)
python3 /root/wc_2026_upgrade/prepare_training_data.py

# Step 2: Retrain XGB (34-dim: 29 base + 1 market_odds + 4 tournament_stage)
python3 /root/wc_2026_upgrade/retrain_xgb_with_odds.py
# Output: /root/data/xgb_model_33.pkl (shadow model)
# See references/xgb-retraining-33dim.md for details
```
# See references/xgb-retraining-33dim.md for details
```

**Real Odds Backtest (2026-06-09, with Isotonic calibration + competition tier filtering + market odds retraining)**:
- Dataset: 263 merged matches (from 3248 kaijiang)
- Valid bets: 234 (spf_sp > 0)
- Triggered bets: 80 (EV > 5%, tier > 0.3)
- Hit rate: 70.0% (56/80)
- **Real ROI: +69.86%** (improved from -3.94% baseline)
- Monthly: 2024-01 (+155.1%), 2024-02 (+431.7%), 2024-03 (+61.4%), 2024-06 (+24.1%), 2024-07 (+44.8%), 2024-09 (+39.2%), 2024-10 (+85.3%), 2024-11 (+168.3%)

**Competition Tier Filtering (COMPETITION_TIER)**:
Based on actual ROI by tournament, dynamically adjust EV threshold:
```python
COMPETITION_TIER = {
    'AFC Asian Cup': 1.2,           # +194.7% ROI
    'FIFA World Cup qualification': 1.0,  # +15.0% ROI
    'UEFA Euro': 0.7,               # -2.4% ROI
    'Copa América': 0.6,            # -12.7% ROI
    'Friendly': 0.2,                # -58.1% ROI (filtered)
    'UEFA Nations League': 0.2,     # -72.5% ROI (filtered)
}
# Dynamic EV threshold = base_ev / tier_weight
# tier_weight < 0.3 → skip
```

**Optimization Path**:
1. ✅ Isotonic calibration: ROI -3.94% → +3.24% (+7.18pp)
2. ✅ Competition tier filtering: ROI +3.24% → +37.64% (+34.4pp)
3. ✅ XGB retraining with market odds: ROI +37.64% → +69.86% (+32.22pp)

**XGB Retraining Results (2026-06-09)**:
- New model: `xgb_model_30.pkl` (30 features, +1 market_implied)
- Market odds feature importance: #1 (15.32%)
- Hit rate: 70.0% (up from 28.3%)
- All months profitable
- Training data: 263 merged matches with closing SP odds

**Club model baseline** (separate):
- Brier Score: 0.1937 (calibrated) — much better than international due to more data
- Accuracy: 53.5% (calibrated)

**HT/FT model baseline**:
- Accuracy: 30.3% (calibrated) vs 25.5% baseline (r_ht=0.45 math derivation)
- Top-3 accuracy: 60.8%
- Brier: 0.8168 (calibrated)

## Form State Compatibility (2026-06-09)

### Problem
`form_state.json` had old format `[h, a]` entries that needed compatibility with new format `[h, a, date]`.

### Solution
The `update_form_from_365.py` script handles both formats:
- Old: `[home_goals, away_goals]` (no date)
- New: `[home_goals, away_goals, "YYYY-MM-DD"]` (with date)

### Deduplication Logic
```python
if not any(x[0]==gh and x[1]==ga and (len(x)<3 or x[2]==date_str)
           for x in form_state[team]):
    form_state[team].append([gh, ga, date_str])
```

### Sorting Logic
```python
def _sort_key(x):
    if isinstance(x, (list, tuple)) and len(x) >= 3:
        return str(x[2])
    return '0000-00-00'
form_state[team] = sorted(form_state[team], key=_sort_key)[-25:]
```

## Training Data Generation (2026-06-09)

### Problem
No training data CSV existed for XGB retraining.

### Solution
Created `build_training_data.py` to generate training features from existing data sources.

### Data Sources
1. `/root/data/training_data_with_odds.json` (263 matches)
2. `/root/data/form_state.json` (336 teams)

### Output
`/root/data/training_data.csv` (263 rows, 7 features + label)

### Features
- `market_odds` (from training_data_with_odds.json)
- `form_home_win/gf/ga` (computed from form_state.json)
- `form_away_win/gf/ga` (computed from form_state.json)

### Usage
```bash
python3 /root/build_training_data.py
```

## 365scores Form Data Source (2026-06-09)

### Problem
football-data.org API doesn't cover international friendlies, so `update_form_state.py` was returning 0 matches daily.

### Solution
Use 365scores API (`webws.365scores.com`) as primary form data source.

**API Endpoint:**
```
https://webws.365scores.com/web/games/current/?sports=1&date=YYYY-MM-DD&games=1&startIndex=0&count=200&withTop=true
```

**Key field mappings:**
- Team name: `game.homeCompetitor.name`
- Score: `game.homeCompetitor.score`
- Status: `game.statusGroup == 4` (finished)

**Script:** `/root/update_form_from_365.py`
- Reads 365scores API, updates `/root/data/form_state.json`
- Compatible with old `[h,a]` format and new `[h,a,date]` format
- Deduplication: `x[0]==gh and x[1]==ga and (len(x)<3 or x[2]==date_str)`
- Cron: `0 6 * * * cd /root && python3 /root/update_form_from_365.py --days 2`

**Verification:**
```bash
python3 /root/update_form_from_365.py --days 7
ls -la /root/data/form_state.json  # check mtime
python3 -c "import json; fs=json.load(open('/root/data/form_state.json')); print(len(fs))"
```

## Parallel Model Integration (2026-06-09)

### Pattern
Run a second simpler XGB model alongside the main model, log both predictions, compare for consensus/disagreement signals.

### Implementation
1. Load `xgb_model_simple.pkl` + `calibrators_simple.pkl` in `_load_shared_models()`
2. In `_try_hybrid_predict()`, compute simple_pred from market_odds + form features
3. Add `simple_pred` and `simple_conf` to prediction result dict
4. Pass through `build_prediction_bundle()` → `record_prediction()` → CSV

### Feature mapping (simple model)
- Input: `[market_odds, form_home_win, form_home_gf, form_home_ga, form_away_win, form_away_gf, form_away_ga]` (7-dim)
- market_odds derived from `1/op_h` (Elo-implied odds)
- form features from `recent_form()` (same as main model)

### CSV fields
- `simple_pred`: H/D/A or empty
- `simple_conf`: 0.0-1.0

### Consensus analysis
```python
main = 'H' if pred_h > pred_d and pred_h > pred_a else ('D' if pred_d > pred_h else 'A')
simple = row['simple_pred']
agree = main == simple
```

### Typical consensus rate: ~68% (from 2026-06-09 test with 19 matches)

## XGB Retraining with Form Features (2026-06-09)

### Training Data Generation
**Script:** `/root/build_training_data.py`
- Source: `/root/data/training_data_with_odds.json` (263 matches)
- Adds form features from `form_state.json`
- Output: `/root/data/training_data.csv` (7 features + label)

**Features:**
- `market_odds` (market odds, strongest predictor)
- `form_home_win`, `form_home_gf`, `form_home_ga` (home form)
- `form_away_win`, `form_away_gf`, `form_away_ga` (away form)

### Model Training
**Script:** `/root/retrain_xgb_simple.py`
- Input: `/root/data/training_data.csv` (263 rows)
- Model: XGBoost, 200 trees, depth 3, lr 0.05
- Output: `/root/data/xgb_model_simple.pkl` + `/root/data/calibrators_simple.pkl`

**Performance (time-series holdout 80/20):**
- CV accuracy: 59.4%
- Test accuracy: 57.5%
- Test EV>5% bets: 44, Hit rate: 54.5%, ROI: +34.1%

**Feature importance:**
- `market_odds`: 0.289 (strongest)
- `form_home_gf`: 0.137
- `form_away_ga`: 0.133
- Form features total: 0.711

## football-data.org API Quirks

- **Seasons endpoint returns 404** — do NOT call `/competitions/{code}/seasons`
- **Use matches endpoint**: `/competitions/{code}/matches?season=YYYY` works
- **Use standings endpoint**: `/competitions/{code}/standings` — returns structured group tables with position, played, points, goalDifference. More accurate than computing from match data. Returns `standings[].table[]` with `team.name`, `team.shortName`, `position`, `playedGames`, `points`, etc.
- **Free tier**: only ~3 seasons of historical data (2023-2025). 2022+ returns 403.
- **Rate limit**: 10 requests/minute. Use 6.5s interval between requests. **Cache aggressively** — implement 24h local cache (`tournament_api_cache.json`) to stay under limits.
- **Half-time scores available**: `score.halfTime.home/away` field exists.
- **Team name mapping pitfall**: API returns different English names than expected (e.g., "South Korea" not "Korea Republic", "Ivory Coast" not "Côte d'Ivoire", "Cape Verde Islands" not "Cabo Verde", "Congo DR" not "Congo", "Bosnia-Herzegovina" not "Bosnia and Herzegovina"). Always verify actual API output with `--dry-run` before committing TEAM_NAME_MAP changes. Reference: `daily_mjm_analysis` repo (`/root/daily_mjm_analysis/collectors/football_data_collector.py`) has a working example of standings collection.

## DC Model Convergence Handling

`predict_match()` returns a **tuple** `(None, error_msg)` when DC doesn't converge, not just `None`. Always check:
```python
result = predict_match(home, away, match_type='friendly')
if isinstance(result, tuple):
    print(f'Skipped: {result[1]}')
    continue
if result is None:
    print('Model failed')
    continue
```

## Club vs National Team Isolation

Club and national team data MUST use separate models:
- `/root/data/elo_club.pkl` + `/root/data/form_club.json` + `/root/data/dc_model_club.pkl` + `/root/data/xg_proxy_club.json`
- `/root/data/elo_ratings.pkl` + `/root/data/form_state.json` + `/root/data/dc_model.pkl`

The `club_data_pipeline.py` builds club-specific Elo (half_life=150d), DC (ρ=0.25), and xG-proxy (luck factors from DC lambda residuals). xG-proxy is auto-generated after DC training in `save_all()`.

## Incremental Data Saving

Long-running fetch scripts (like `fetch_league_data.py`) MUST save data incrementally after each league/batch, not at the end. If the process is killed, only unsaved data is lost. Pattern:
```python
# After each league completes:
with open(output_path, 'w') as f:
    json.dump(all_matches, f, ensure_ascii=False)
```

## Backfill Pipeline (2026-06-10)

赛果自动回填 + Brier Score 计算系统，每天 cron 两次。

### 架构
- 脚本: `/root/backfill_results.py`
- Checkpoint: `/root/data/backfill_checkpoint.json`
- Cron: `backfill-am` (UTC 01:30 = 北京09:30), `backfill-pm` (UTC 05:30 = 北京13:30)
- 数据源优先级: results JSON → kaijiang CSV → football-data.org

### 设计原则
- **幂等**: 只填充 `result_status=missing` 的记录，已填充的永不覆盖
- **checkpoint**: 记录最后处理日期，重启后从断点-1天继续
- **冲突检测**: 多源结果不一致时标记 `result_status=conflict`
- **Brier Score**: 填充后自动计算 `(1/3)*Σ(I_j - p_j)²`
- **cron频率**: 每天两次互补，第一次失败第二次补上

### 新增 CSV 字段
- `match_key`: 稳定主键 `date|league|home|away|time`
- `brier_spf`: 单场多分类 Brier Score (4位小数)
- `result_status`: missing/filled/conflict/postponed
- `settled_at`: 回填完成时间 (ISO)
- `backfill_source`: 数据来源 (如 `kaijiang`, `results:2026-06-09`)

### backtest_jczq.py FIELDS 扩展注意事项
扩展 FIELDS 时，同时更新 `cmd_record()` 的解析逻辑 AND `record_prediction()` 的 CLI 传参。
回填字段（actual_*, brier_spf, result_status, settled_at, backfill_source）必须加入保护列表，
防止 `record_prediction()` 覆盖已回填的数据。

## System Health Check (2026-06-10)

Run these diagnostics when asked to review model quality or suspect degradation:

```bash
# 1. Brier overview
python3 /root/backfill_results.py --stats

# 2. Full Brier + accuracy + per-class breakdown
python3 -c '
import csv; rows = list(csv.DictReader(open("/root/data/predictions_log.csv")))
m = []
for r in rows:
    b = r.get("brier_spf","").strip()
    if not b: continue
    s = r.get("actual_score","").strip()
    if not s: continue
    p = s.split(":")
    if len(p) != 2: continue
    try: hg, ag = int(p[0]), int(p[1])
    except: continue
    a = "H" if hg > ag else ("D" if hg == ag else "A")
    ph, pd_, pa = float(r.get("pred_h",0)), float(r.get("pred_d",0)), float(r.get("pred_a",0))
    pr = "H" if ph > pd_ and ph > pa else ("D" if pd_ > pa else "A")
    m.append((float(b), a, pr, ph, pd_, pa))
print(f"n={len(m)} mean_brier={sum(x[0] for x in m)/len(m):.4f}")
corr = sum(1 for x in m if x[1]==x[2])
print(f"acc={corr}/{len(m)}={corr/len(m)*100:.1f}%")
print(f"draw_pred={sum(1 for x in m if x[2]==\"D\")} actual_draw={sum(1 for x in m if x[1]==\"D\")}")
for l in ["H","D","A"]:
    s = [x for x in m if x[2]==l]
    if s: print(f"  pred={l}: n={len(s)} brier={sum(x[0] for x in s)/len(s):.4f} acc={sum(1 for x in s if x[1]==x[2])}/{len(s)}")
'

# 3. Model file freshness
ls -la /root/data/dc_model.pkl /root/data/xgb_model_29.pkl /root/data/xgb_model_30.pkl /root/data/elo_ratings.pkl /root/data/calibrators.pkl

# 4. Club pathway check
python3 -c "
import os
for f in ['dc_model_club.pkl','xgb_model_club.pkl','elo_club.pkl','calibrators_club.pkl','form_club.json','xg_proxy_club.json','h2h_cache_club.json']:
    p = os.path.join('/root/data', f); e = os.path.exists(p); s = os.path.getsize(p) if e else 0
    print(f'{f}: exists={e} size={s}')
"

# 5. Draw prediction rate
python3 -c "
import csv
rows = list(csv.DictReader(open('/root/data/predictions_log.csv')))
total = len(rows)
draw = sum(1 for r in rows if float(r.get('pred_d',0)) > float(r.get('pred_h',0)) and float(r.get('pred_d',0)) > float(r.get('pred_a',0)))
print(f'Draw predictions: {draw}/{total} = {draw/max(total,1)*100:.1f}%')
print(f'WARNING: target ~25% for calibrated model')
"

# 6. Predictions by league
python3 -c "
import csv
from collections import Counter
rows = list(csv.DictReader(open('/root/data/predictions_log.csv')))
for l, c in Counter(r.get('league','') for r in rows).most_common():
    print(f'{l}: {c}')
"
```

### Thresholds for concern
| Metric | Warning | Critical |
|--------|---------|----------|
| Brier vs training (0.2053) | >0.22 | >0.25 |
| Accuracy | <45% | <40% |
| Draw prediction rate | <5% | <1% (FIXED via Draw Correction Layer 2026-06-10) |
| Club pathway working | No | No |
| model_route filled | <50% | <10% (FIXED 2026-06-10, monitor next run) |

## System Audit Findings (2026-06-14)

### Model Architecture Recommendation: Unify on nat model (11-dim)

**Rationale**: 29/30-dim models have 18 dead features (importance=0) causing train-serve skew, scoring 64.3% vs nat's 75.4% after label fix. Structural features (Elo/DC/market) are stable; soft features (form/gold/h2h) create drift.

**Current production models**:
- `xgb_model_nat.pkl` (11-dim): 75.4% acc, LogLoss 0.819 — CLEAN, RECOMMENDED
- `xgb_model_29.pkl` (29-dim): 18 dead features, trained with label bug — DEPRECATED
- `xgb_model_30.pkl` (30-dim): retrained 2026-06-14, 64.3% — still has dead features

**Action required**: Switch `daily_jczq.py` `_load_shared_models()` to load nat model and add 11-dim feature branch in `_try_hybrid_predict()`.

### Isotonic Calibration Status (2026-06-14 更新)

- **daily_jczq.py**: Isotonic calibrator loaded but NOT applied (stripped 2026-06-10)
- **calibrated_predictor.py**: Isotonic calibrator NOW stripped (same day fix)
- Both pipelines use raw model probabilities + Draw Correction Layer

---

## Responding to User "想买..." Queries (2026-07-01)

When the user asks about **buying a specific match** (e.g., "想买今天的世界杯比赛英格兰vs刚果金"), follow this workflow:

### Step 1: Verify match stage freshness
**Do NOT trust tournament_state.json blindly.** It may be stale if the cron hasn't run:
```bash
ls -la /root/data/tournament_state.json  # check mtime
python3 -c "
import json, datetime
d=json.load(open('/root/data/tournament_state.json'))
# If mtime > 3 days ago, state may be unreliable
# Cross-check with: is_knockout field for teams
"
```

If tournament_state.json mtime > 3 days old, **flag it explicitly** — do not make claims about group/knockout stage based on stale data. The user may have real-time tournament knowledge that overrides stale cache.

### Step 2: Check if match is buyable on 500.com
Run the scraper and grep for the match:
```bash
python3 /root/wc_2026_upgrade/async_500_scraper.py | python3 -c "
import sys,json
d=json.load(sys.stdin)
for m in d.get('result',[]):
    if '英格兰' in m['home'] and '刚果' in m['away']:
        print(m)
"
```

### Step 3: Read the prediction from predictions_log.csv
```bash
cd /root && awk -F',' '\$5==\"英格兰\" && \$6==\"刚果(金)\"' data/predictions_log.csv | tail -1
```

### Step 4: Give direct, structured recommendation
Include all 5 plays with probabilities, market odds, EV analysis, and a **clear recommendation hierarchy** (best value first, safest pick first). Address follow-up questions directly — do not re-explain the model architecture. The user wants:

- Which result most likely (SPF: 主74.4%)
- What's the best value (让负@3.90, EV+41%)
- What's the goal/score pattern (1:0, 1球, 胜胜)
- Risk assessment (e.g., "这是小组赛末轮，可能轮换" or "淘汰赛全主力，但刚果金死守")

### Step 5: Be honest about uncertainty
When the user asks "确定吗？", present both the model's probabilistic view AND real-world counter-evidence (e.g., "英格兰上一场4-2，但刚果金0-1输哥伦比亚能守"). The model says ≤3 goals 89.4% but let the user decide with full info.

### Stage-dependent adjustment rules
- **Group stage**, especially round 3: Possible motivation adjustment (强队轮换). The 战意不足调整 only applies when ALL of: (a) 世界杯, (b) round 3, (c) is_knockout=False, (d) points_diff ≥ 3.
- **Knockout stage**: NO motivation adjustment. Full strength expected. Prediction probabilities should be taken at face value.
- **Detection**: tournament_state.json `is_knockout` field. If state is stale, ask the user or check external sources.

## Common Pitfalls

0. **CRITICAL: tournament_state.json can be stale (2026-07-01 found)** — The cron `update_tournament_state.py` may not be running or football-data.org API may return no data. Always check `/root/data/tournament_state.json` mtime before making claims about group/knockout stage. If >72h old, explicitly flag as stale. The `is_knockout` field and `round_num` may be wrong. User's real-time tournament knowledge trumps stale cache. Do NOT apply 战意不足 motivation adjustment without verifying the round is actually group stage round 3.

0. **CRITICAL: _show_tomorrow.py 显示旧数据 (2026-06-15 发现)** — pipeline 刚写入新预测后，`_show_tomorrow.py $(date +%F)` 可能仍显示旧 CSV 行（如 SPF 0.0%/0.0%/0.0%），因为 CSV 同场次有多行，show 脚本按完整度评分选行而非按时间。**检测**: 对比 show 输出与 pipeline 终端输出的 SPF 概率，若偏差 >5pp 说明读到旧数据。**修复**: 直接用 pipeline 终端输出（`daily_jczq.py > /root/data/today_output.txt 2>&1`），不依赖 _show_tomorrow.py。**根因**: `predictions_log.csv` 同一 match_key 多行，show 脚本选 `goals_full` 键数最多的行，但旧行可能 goals_full 更完整。**永久修复方向**: show 脚本应优先选 `settled_at` 最新或 `date` 最大的行。

0. **CRITICAL: spf_result 训练数据类型安全 (2026-06-14 发现, 已修复)** — `training_data_with_odds.json` 中 `spf_result` 字段含 131/491 条 int 类型（非 str）。训练脚本 `result == '3'` 不匹配 int，导致 29 条标签错误（7.3%）。修复: 所有训练脚本统一 `str(m['spf_result'])`。影响: nat 模型验证准确率从 64.4% → 75.4%。**检查方法**: `python3 -c "import json; d=json.load(open('/root/data/training_data_with_odds.json')); print('int:', sum(1 for m in d if isinstance(m.get('spf_result'),int)), 'str:', sum(1 for m in d if isinstance(m.get('spf_result'),str)))"`

0. **CRITICAL: _blend_with_market 平局硬编码0 (2026-06-14 发现, 已修复)** — `calibrated_predictor.py` 的 `_blend_with_market()` 函数中 `elo_arr = [elo_h, 0, 1-elo_h]` 将平局概率硬编码为 0，系统性压低平局预测。修复: 用 Elo 差估算真实平局概率 `elo_draw = 0.25 * (1 - abs(2*elo_h - 1))`。效果: 荷兰vs日本平局 5.6% → 13.2%。**检查方法**: `grep -n '0, 1-elo_h\|0, 1-elo_a' /root/wc_2026_upgrade/calibrated_predictor.py`

0. **CRITICAL: 双管线模型不一致 (2026-06-14 发现, 已修复)** — daily_jczq.py 用 xgb_model_29 (29维+剥离校准器)，calibrated_predictor.py 用 xgb_model_nat (11维+活跃校准器+无Draw Correction)。修复: 两管线统一到 nat 模型。详见 [references/2026-06-14-audit.md](references/2026-06-14-audit.md)。**检查方法**: `grep -n 'xgb_model.*=.*joblib.load' /root/daily_jczq.py /root/wc_2026_upgrade/calibrated_predictor.py`

0. **CRITICAL: 500.com league field not propagated (2026-06-11 DISCOVERED, FIXED)** — `async_500_scraper.py` correctly extracts `simpleleague` attribute (e.g., "世界杯", "英超") and includes it in the scraper output dict. However, `scrape_500_odds_today()` was NOT including the `'league'` field in its return dict. Downstream code hard-coded `'友谊赛'` for ALL international matches:
   ```python
   # BUG (fixed): league field missing from return dict
   bundle = build_prediction_bundle(..., '友谊赛', ...)  # ALL international = friendly
   
   # FIX: use league from 500.com, fallback to '友谊赛'
   league_name = m5.get('league', '') or '友谊赛'
   bundle = build_prediction_bundle(..., league_name, ...)
   ```
   **Root cause**: Data field existed in scraper output but was dropped in the intermediate function. `compute_bet_action()` checks `if '友谊赛' in league` → `WATCH_FRIENDLY`, so World Cup was classified as friendlies and not recommended.
   **Fix locations** (all in `/root/daily_jczq.py`):
   1. `scrape_500_odds_today()` return dict: added `'league': row.get('league', '')`
   2. `build_prediction_bundle()` call: changed hard-coded `'友谊赛'` to `league_name`
   3. `compute_fatigue_features()` call: same league_name fix
   4. Header print: changed "国际友谊赛" to "国际赛事（来自500.com）"
   **Verification**: `python3 async_500_scraper.py 2026-06-11 269` → `league: "世界杯"`. Output now shows `[世界杯]` instead of `[友谊赛]`.
   **Lesson**: When adding new fields to scraper output, trace the ENTIRE data path: scraper → `scrape_500_odds_today()` return dict → `build_prediction_bundle()` → `compute_bet_action()`. Missing any link means the field is silently lost.

0. **football-data.org API rate limiting — cache aggressively (2026-06-11 established)** — Free tier allows only 10 requests/minute. For daily tournament state updates, implement 24h local cache (`tournament_api_cache.json`). `update_tournament_state.py` checks cache freshness before calling API; `--force` bypasses cache. Pattern: `load_cache()` → check TTL → `fetch_standings_from_api()` → `save_cache()`. On API failure (429/5xx), fall back to stale cache rather than failing silently. Reference repo: `/root/daily_mjm_analysis/collectors/football_data_collector.py` uses same pattern with `retries=2` and `time.sleep(0.5 * (attempt + 1))` backoff.

0. **Model dimension compatibility when extending features (2026-06-11 established)** — When adding new features to XGB models, use auto-detection to maintain backward compatibility:
   ```python
   feat_dim = _xgb_model.n_features_in_ if hasattr(_xgb_model, 'n_features_in_') else 29
   if feat_dim == 29:
       feat = feat_33[:, :29]  # Slice to match old model
   else:
       feat = feat_33  # Full features for new model
   ```
   This pattern allows deploying new features (e.g., tournament stage 4-dim) before the new model is trained. The old model silently ignores extra features via slicing; the new model uses them when ready. Check `n_features_in_` attribute (sklearn ≥1.0) rather than hardcoding.

0. **Zero draw predictions (2026-06-10 discovered, SAME-DAY FIX)** — The 3-class model degenerated to 2-class. Out of 32 backfilled matches, 0 predicted draws. Root cause: DC independent Poisson + Elo binary bias + training data imbalance. **Fix**: Draw Correction Layer implanted in both `_try_hybrid_predict()` and `_try_club_predict()` — when p_draw < 15%, applies `draw_boost = 0.05 × (1.0 - confidence)` with proportional home/away reduction.

0. **Isotonic calibrator removed from production (2026-06-10 fixed)** — Previously, despite the diagnosis that "raw prob Brier=0.2053 is better than any calibrator", `calibrators.pkl` was STILL applied in `_try_hybrid_predict()` after fusion. Fixed: both `_try_hybrid_predict()` and `_try_club_predict()` now skip all calibrator predict() calls. Production Brier should converge back toward 0.2053 baseline.

0. **Club DC+XGB pathway dead (2026-06-10)** — `_try_club_predict()` always returns None. Model files exist but form_club.json or xg_proxy_club.json may be missing/empty. Run club pathway diagnostic to confirm.

0. **model_route field was empty for all CSV records — FIXED 2026-06-10** — `build_prediction_bundle()` now carries `'model': p.get('model', 'unknown')` through to CSV. After next daily_jczq.py run + backfill, model_route will distinguish hybrid / club_hybrid / legacy_poisson / market_fallback.

0. **CSV source_tag ≠ model routing (2026-06-10)**  \\\n   `predictions_log.csv` 的 `source_tag` 列始终为 `"500+365"`，不反映模型路由。\\\n   实际路由（`hybrid` / `market_fallback`）在 pipeline 打印输出中通过 `模型: xxx` 标识，\\\n   来自 bundle dict 的 `model` 字段，CSV 中无对应列。\\\n   分析代码不应通过 `source_tag` 判断模型类型。如需批量分析，用新增的 `model_route` 列。

0. **Isotonic校准器在友谊赛上严重过度自信 (2026-06-10 诊断)**  \\\n   32场友谊赛校准曲线: RECOMMEND组70%置信度, 0%命中率, 校准差-70.2pp。\\\n   根因: 263场训练数据(2024)分布与2026友谊赛截然不同, Isotonic在小样本上过拟合。\\\n   治本: 积累200+场2026年回填数据后, 用纯XGB输出概率训练sigmoid校准器。\\\n   详见 `references/calibration-pitfalls.md`。

0. **全局校准器在特征漂移时适得其反 (2026-06-10 发现)**  \\\n   2024训练数据用 market_implied=1/spf_sp 作为特征, 2026推理用 XGB 输出概率。\\\n   混合训练的 Sigmoid 校准器 (LogisticRegression Platt Scaling) 让 Overall Brier 从 0.2053 恶化到 0.2378。\\\n   结论: 无校准 > 错误校准。当原始 Brier < 0.22 时, 不确定的校准器不如不校准。

0. **365scores lineup/news 数据可用时间窗 (2026-06-10)**  \\\n   365scores 的 `has_lineups` / `has_news` / `has_missing_players` 标志在比赛 >24h 前均为 `false`。\\\n   当日开售清单跨越 8 天（如 6/10 开售 6/11-6/18），仅最近 1-2 场可能释放阵容信息。\\\n   阵容调整对友谊赛的影响无法提前预判，模型不确定性高于联赛。

0. **极端让球比赛模型退化 (2026-06-10)**  \\\n   让球 ≥ ±2 且弱队不在 9 大联赛训练集时（德国vs库拉索-3、西班牙vs佛得角-2），\\\n   DC 输出均匀分布(1/3,1/3,1/3)，XGB 因 form/elo 全零退化为 0/0/0%。\\\n   SPF=0/0/0%，RQ=0/0/100%（让球方净胜 3 球）。检测: SPF 总和 < 1%。\\\n   应对: 用市场赔率判断，模型概率不可信，Poisson 外推的比分/半全场失真。

0. **NEXT-DAY-ONLY scope (2026-06-09 用户确认)**  \\\n   用户明确要求仅预测**明天比赛**。即使500.com上可以看到并购买之后日期的比赛，也不应输出。必须用 `endtime` 过滤。如果明天只有1-2场，就只输出1-2场，不扩范围。

0. **合成赔率回测陷阱 (CRITICAL — 2026-06-08 发现)**  \\\n   **永远不要用自己的模型概率生成合成赔率来做回测。** 这是循环论证。\\\n   - 错误路径: DC概率 → +overround → 合成赔率 → EV计算 → Kelly下注 → "518% ROI"\\\n   - 为什么是假的: DC对未知队伍默认输出1/3均匀分布，任何偏离1/3的赔率都被误判为"价值"\\\n   - 正确做法: 用真实市场赔率(500.com/The Odds API)计算EV\\\n   这些都只是合成赔率回测的虚假信号\n   详见 references/backtest-methodology.md
   2024训练数据用 market_implied=1/spf_sp 作为特征, 2026推理用 XGB 输出概率。\
   混合训练的 Sigmoid 校准器 (LogisticRegression Platt Scaling) 让 Overall Brier 从 0.2053 恶化到 0.2378。\
   结论: 无校准 > 错误校准。当原始 Brier < 0.22 时, 不确定的校准器不如不校准。

0. **365scores lineup/news 数据可用时间窗 (2026-06-10)**  \
   365scores 的 `has_lineups` / `has_news` / `has_missing_players` 标志在比赛 >24h 前均为 `false`。\
   当日开售清单跨越 8 天（如 6/10 开售 6/11-6/18），仅最近 1-2 场可能释放阵容信息。\
   阵容调整对友谊赛的影响无法提前预判，模型不确定性高于联赛。

0. **极端让球比赛模型退化 (2026-06-10)**  \
   让球 ≥ ±2 且弱队不在 9 大联赛训练集时（德国vs库拉索-3、西班牙vs佛得角-2），\
   DC 输出均匀分布(1/3,1/3,1/3)，XGB 因 form/elo 全零退化为 0/0/0%。\
   SPF=0/0/0%，RQ=0/0/100%（让球方净胜 3 球）。检测: SPF 总和 < 1%。\
   应对: 用市场赔率判断，模型概率不可信，Poisson 外推的比分/半全场失真。

0. **NEXT-DAY-ONLY scope (2026-06-09 用户确认)**  \
   用户明确要求仅预测**明天比赛**。即使500.com上可以看到并购买之后日期的比赛，也不应输出。必须用 `endtime` 过滤。如果明天只有1-2场，就只输出1-2场，不扩范围。

0. **合成赔率回测陷阱 (CRITICAL — 2026-06-08 发现)**  \
   **永远不要用自己的模型概率生成合成赔率来做回测。** 这是循环论证。\
   - 错误路径: DC概率 → +overround → 合成赔率 → EV计算 → Kelly下注 → "518% ROI"\
   - 为什么是假的: DC对未知队伍默认输出1/3均匀分布，任何偏离1/3的赔率都被误判为"价值"\
   - 正确做法: 用真实市场赔率(500.com/The Odds API)计算EV\
   这些都只是合成赔率回测的虚假信号
   详见 references/backtest-methodology.md

   4. **CSV 同场次多行问题 (2026-06-10)**：每次 daily_jczq.py 运行追加新行，同一 match 有多个记录。展示脚本必须按完整度评分选行（goals_full 键数 > htft_full > score_full > time > odds）。

   5. **goals_full 截断已修复 (2026-06-10)**：record_prediction() 原用 goals_top5(5条) 已改为 goals_all(13条 0~12球)。详见 references/csv-full-distribution-serialization.md。

   6. **CSV 概率是百分比 (2026-06-10)**：predictions_log.csv 的 pred_* 字段存百分比值(66.0=66%)，不是小数。

33. **bet_action + model_route 未写入CSV (2026-06-10 发现并修复)**: \\
    `record_prediction()` 传了 40+ CLI 参数给 backtest_jczq.py 但漏传 `--bet-action` 和 `--model-route`。`backtest_jczq.py` 的 FIELDS 列表也缺少这两个字段。结果：CSV 中 bet_action 全为 N/A，无法追踪赛事过滤效果。\\
    **修复**: backtest_jczq.py FIELDS 加 `bet_action`/`model_route`，cmd_record() 加解析，daily_jczq.py record_prediction() 加传参。\\
    **教训**: 新增 bundle 字段时必须同步三处: (1) build_prediction_bundle 返回值 (2) backtest_jczq.py FIELDS + cmd_record 解析 (3) daily_jczq.py record_prediction() 传参。

34. **daily_jczq.py 加载 xgb_model_29 而非 xgb_model_30 (2026-06-10 发现)**: \\
    `_load_shared_models()` 第208行加载 `xgb_model_29.pkl`(29维)，而 `xgb_model_30.pkl`(含 market_implied 特征，重要性 #1=15.32%) 已于 06-09 训练完成但从未被加载。\\
    **切换风险**: 30维模型需推理时有市场赔率构造 market_implied 特征。若 500.com 赔率抓取失败，30维模型会因缺特征崩溃。\\
    **建议**: 优先加载 30维 + try/except 降级 29维。

33. **nspf为空时rq赔率被吞 (2026-06-10 发现并修复)**:

34. **record_prediction()漏传新字段 (2026-06-10 发现并修复)**: \\
    新增bundle字段(如bet_action, model_route, match_key)时，必须同步三处: \\
    (a) `backtest_jczq.py` 的 `FIELDS` 列表加字段名 \\
    (b) `backtest_jczq.py` 的 `cmd_record()` 加 `elif k == "field-name": row["field_name"] = v` 解析 \\
    (c) `daily_jczq.py` 的 `record_prediction()` cmd列表加 `'--field-name', str(bundle.get('field_name', ''))` \\
    漏掉任何一处都会导致CSV中该字段为空。\\
    同时要在 `cmd_record()` 的保护列表中添加回填字段(actual_*, brier_spf, result_status, settled_at, backfill_source)，防止record_prediction覆盖已回填的数据。

39. **backtest_pipeline.py --verify cron 输出截断 (2026-06-30 发现, 2026-07-01 增加工作流)** — predictions_log.csv 含异常行（字段值嵌入列名，date 为空）时，`--verify` 对每条异常行输出巨量警告，远超终端 50KB 截断限制，实际 summary 被吞。

**诊断步骤**:
   (1) 检查 `backtest_results.json` 最后一条是否为今天 → 若不是则说明无新核验
   (2) `python3 -c "import csv; rows=list(csv.DictReader(open('/root/data/predictions_log.csv'))); print(sum(1 for r in rows if r.get('checked')!='1' and r.get('actual_score','').strip()), 'ready')"`
   (3) 若 ready=0 且 backtest_results.json 无新条目，则脚本正常（只是没有新数据可核验）。

**执行策略** (2026-07-01): 当 pipeline 输出被截断时，不要依赖其输出。改用 `execute_code` 直接执行验证逻辑 — 读取 CSV、解析比分、逐行计算 Brier/RPS/准确率、写回已更新行、追加结果到 `backtest_results.json`。这完全绕过 pipeline 的打印噪音，且能在单次调用内完成所有工作。参考工作流：
   ```python
   from hermes_tools import read_file
   import csv, json, math, os, numpy as np
   # 直接读 CSV, 筛选可核验行, 算指标, 写回
   ```
   
**已知陷阱**: 部分行 code 与 date 不匹配（如 周三 codes 出现在周二日期下），backfill 所有数据源均无赛果，导致这些行永久滞留为未核验状态。这些行不影响其他行的核验，但会持续出现在 pipeline 的 warning 中。检测方法：`duplicate_codes` 结合日期检查。

**修复方案**: 清理 predictions_log.csv 中 date 为空的异常行；或在 `daily_verify()` 中将警告改为 stderr、stdout 只输出 summary。

    当handicap≠0且nspf为空时，`else`分支的 `rq_h_val = to_float(nspf_raw.get('3')) if nspf_raw else 0.0` 返回0.0。因为rq赔率应从spf_raw取，不是nspf_raw。修复：在 `handicap!=0 and not nspf_raw` 块中显式从spf_raw重新赋值 rq_h/d/a。\
    详见 `references/500-api-spf-nspf-quirk.md`。

34. **赛事过滤规则未生效 (2026-06-10 发现)**：\
    COMPETITION_TIER 过滤（针对友谊赛×0.20、Nations League×0.20）只在 `real_odds_backtest.py` 中实现，`daily_jczq.py` 中没有。回测发现友谊赛 ROI -58.1%、Nations League ROI -72.5%，但当前每日预测对所有赛事类型一视同仁。\
    **建议**: 在 `daily_jczq.py` 的 EV/Kelly 输出端添加赛事类型加权或跳过逻辑。\
    详见 `references/data-layer-audit-qa.md`。

35. **predictions_log.csv 缺 Brier 字段 (2026-06-10 发现)**：\
    CSV 记录了 pred_h/d/a 但未记录校准后的 Brier Score。无法按路由（国际/俱乐部）在线监控校准质量。需要在 `record_prediction()` 中写入 `brier` 列。

36. **校准器训练数据仅 263 场 (2026-06-10 发现)**：\
    `training_data_with_odds.json` 只有 2024-01-13 → 2024-11-15 的 263 场比赛。模型文件虽每天更新，但训练数据滞后。2026 年的分布偏移可能让校准效果退化。\
    详见 `references/data-layer-audit-qa.md`。

37. **simple vs main 模型分歧解读缺失 (2026-06-10 发现)**：\
    今天有 8 场分歧（不是用户猜测的 6 场），且未自动标记或解读。分歧可能标志着模型过拟合或数据异常，建议加入自动分歧分析环节。

34. **Isotonic 校准器过拟合诊断 (2026-06-10 确认)**: \\
    263场训练数据 + Isotonic回归 = 严重过度自信。校准曲线诊断: \\
    - RECOMMEND组: 70%置信度, 0%命中率, 校准差 -70.2pp \\
    - 主胜: 40%预测概率, 62.5%实际命中率, 校准差 +22.5pp \\
    - 客胜: 32.7%预测概率, 18.8%实际命中率, 校准差 -14.0pp \\
    **根因**: Isotonic是非参数方法, 在小样本(263场)上对局部波动过度拟合。 \\
    **处置**: 友谊赛全部降级为 WATCH_FRIENDLY。详见 `references/calibration-diagnostics.md`。

35. **Platt Scaling (sigmoid) 在异质数据上同样失败 (2026-06-10 验证)**: \\
    用2024年数据(market_implied=1/spf_sp) + 2026年数据(XGB输出) 混合训练sigmoid校准器, \\
    Overall Brier 从 0.2053 恶化到 0.2378。 \\
    **根因**: 两批数据的概率来源不同 (市场赔率倒数 vs 模型输出), LR学到的sigmoid变换不通用。 \\
    **教训**: 校准器训练数据必须来自同一模型的同质输出, 不能混合不同概率来源。 \\
    **正确做法**: 等2026年回填数据达到200+场后, 用纯(XGB输出, 赛果)对训练校准器。

36. **原始概率(无校准)反而校准最好 (2026-06-10 发现)**: \\
    265场合并数据诊断: Overall Brier=0.2053, 各类别gap在±5pp以内。 \\
    Isotonic和sigmoid校准器都让结果变差。 \\
    **结论**: 当训练数据不足或分布异质时, 不校准 > 强行校准。

37. **友谊赛必须全部 WATCH (2026-06-10 止血规则)**: \\
    `compute_bet_action()` 中 `if '友谊赛' in league: return 'WATCH_FRIENDLY'` \\
    不再用 margin_pp 门槛 (旧逻辑: margin<20pp才WATCH, 导致高margin友谊赛仍RECOMMEND) \\
    **恢复条件**: 等友谊赛回填数据达到50+场, 训练友谊赛专属校准器后再恢复margin门槛。

38. **校准曲线分析方法 (2026-06-10 建立)**: \\
    分析脚本: `/root/calibration_analysis.py` \\
    诊断维度: 按class(H/D/A) × 按bet_action(UNIFORM/LOW/RECOMMEND) \\
    关键指标: 平均预测概率 vs 实际命中率 → gap(pp) + Brier \\
    **何时触发**: 每次重训校准器后, 每次回填数据超过50场时 \\
    详见 `references/calibration-diagnostics.md`。

1. **Confusing script output with manual interpretation**
   The prediction text must come from `daily_jczq.py`, not from memory or ad-hoc estimates.

2. **Forgetting that 500.com defines buyable scope**
   The user asked for `今天可以买哪几场比赛`; if a match is not in today's 500.com buyable set, do not present it as today's JCZQ match.

3. **Mixing market value with model main pick**
   A positive EV note does not justify changing the main recommendation label.

4. **Dropping plays from the output**
   The user expects all five: `胜平负 / 让球 / 半全场 / 比分 / 总进球数`.

5. **Using the wrong settlement scope**
   Always speak in `90分钟` terms unless the user explicitly asks otherwise.

6. **Treating weak friendly signals as strong bets**
   For international friendlies, margin quality matters. If the model output is close or noisy, call that out instead of overclaiming.

7. **赛果回填不要只押注单一数据源**
   `predictions_log.csv` 的真实回测闭环依赖 `actual_*` 回填。实践中，`500.com kaijiang` 可能对友谊赛覆盖不足，而 `football-data.org` 也可能在特定日期窗口返回 0 个 `FINISHED` 结果，即使 API 请求成功、队名标准化正确也无法回填。因此回填链路必须设计成多源顺序，而不是把问题误判为解析器或队名映射故障。

8. **先验证"赛果是否存在"，再写回填逻辑**
   当 `predictions_log.csv` 中 `actual_hda/actual_score` 全为空时，先检查数据源是否真的提供该日期窗口的完赛赛果：
   - `500.com kaijiang` 页面是否有对应 `周XNNN` 行
   - `football-data.org v4/matches` 是否返回 `FINISHED` 且 `fullTime.home/away` 非空
   - 本地缓存（如 `/root/data/results/*.json`、`international_friendlies_*.csv`）是否覆盖目标日期
   若这些源都没有赛果，结论应明确为"数据完备性缺失"，不要继续把时间浪费在队名匹配或 HTML 解析上。

9. **中文队名可直接复用 `team_name_normalizer.py` 做 football-data 匹配**
   `predictions_log.csv` 中球队名存的是中文，回填时应先走 `team_name_normalizer.normalize_team_name()` 转成模型标准英文名，再与外部源（football-data、365scores 等）匹配。若发现常用国家队别名缺失（如 `南非`、`佛得角`、`埃及`、`塞内加尔`、`库拉索`、`科特迪瓦`），优先补到别名字典，而不是临时手写散落映射。

10. **为未来回填预埋多源顺序框架**
    推荐顺序：
    1. `/root/data/results/*.json`
    2. `/root/data/international_friendlies_*.csv`
    3. `football-data.org`
    4. `500.com kaijiang`
    5. `365scores` 或其他比分源
    6. **ESPN API (WC 2026)**: `https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=YYYYMMDD` — 免费无认证，仅 FIFA World Cup。详见 `jczq-prediction-system` skill 的 `references/espn-api-score-backfill.md`
    这样即使单一源缺失，也不会阻塞整个 daily_jczq 的历史滚动回测。
 **WARNING**: 不要随意添加新字段到固化流程版——用户明确要求保持 500.com+365scores+模型直出的简洁架构。新功能（market_calibrator、lineup_risk、disc_spf 等）已被用户否定并回退。详见 `jczq-analysis/references/fixed-flow-architecture-and-rollback.md`。

11. **half_full_model 模块不存在**：`daily_jczq.py` 第 60 行 `from half_full_model import predict_half_full_prods` 会尝试从 `/usr/local/lib/hermes-agent/{models,strategy}` 导入，但这些目录下也无此模块。try/catch 包裹后静默降级，半全场/总进球功能从 DC λ + Poisson 网格推导（而非专用模型）。影响：半全场准确率受限于 DC 的独立性假设。

12. **Isotonic 校准已集成到每日管线 (2026-06-09 修复)**: `calibrators.pkl` (home/draw/away 各一个 IsotonicRegression) 已加载到 `_load_shared_models()`，在 `_try_hybrid_predict()` 的 hybrid 融合后执行校准。校准流程：raw XGB 概率 → Isotonic 预测 → 归一化。真实回测 ROI 从 -3.94% 提升到 +3.24% (+7.18pp)。校准器训练数据为 2024-2026 年竞彩完赛比赛。

13. **predictions_log.csv 反馈闭环缺失**：所有 `actual_hda`、`actual_score`、`actual_htft`、`actual_rq_result`、`actual_goals` 列全部为空。系统无法自动评估历史预测质量。回填应走多源顺序（500.com kaijiang → football-data.org → 365scores），详见 `jczq-analysis/references/backtest-data-fallback-chain.md`。

15. **`compute_dynamic_xgb_weight()` 与 wc_2026_final.py 的副本必须同步**：两个同名函数逻辑一致但实现不同（daily_jczq.py 用纯 math，wc_2026_final.py 用 np）。修改 α/β 或钳位范围时必须同时改两处。验证检查：`python3 -c "from wc_2026_final import compute_dynamic_xgb_weight as f1; from daily_jczq import compute_dynamic_xgb_weight as f2; assert f1([0.85,0.10,0.05])==f2([0.85,0.10,0.05])"`
### 500.com 抓取熔断与降级日志（2026-06-08 上线, 2026-06-14 增强）
`scrape_500_odds_today()` 现已集成 3 次重试 + 指数回退 (3s/6s/9s) 和结构化降级日志。每个 playid 独立重试，失败后隔离（其他 playid 不受影响）。

**熔断兜底 (2026-06-14)**: 当异步抓取全量熔断时, 不再直接返回 `[]` 跳过市场校准。改为调用 `_load_fallback_odds()`:

1. 加载 `/root/data/odds_history.json` (17 场历史快照)
2. 过滤 `start >= today` (当前/未来比赛)
3. 转换格式与 `scrape_500_odds_today()` 一致 (含 spf/nspf/jqs/bf/bqc 映射)
4. 标记 `std_odds_source='fallback_stale'`
5. 返回兜底数据 (无数据时才返回 [])

**熔断日志**: 写入 `/root/data/500breaker.log`, 终端输出 `🟡 500.com 熔断, 使用 odds_history.json 兜底 (N 场, 标记 stale)`.

**预期行为**：
- 偶发单次失败 → 重试成功，静默修复
- 持续失败 → 终端黄字告警 + 历史赔率兜底 (含 stale 标记)
- 全量熔断且无兜底数据 → 红字告警 + 返回 []

17. **Gold特征train-serve skew（2026-06-08 修复）**：predict_match.py和daily_jczq.py的gold特征原先用占位符`[0.0, 1, 0, 0.0, 0.0]`，但训练时gold=[h2h_gd, tier_major, tier_friendly, fh12_attack_def, fa12_attack_wr]有真实值。现已通过feature_helper.py修复：预计算H2H缓存(7481对)+12场form缓存(336队)，推理时调用`build_gold_features()`获取真实值。平局准确率从4.7%提升到14.4%。

18. **xG-proxy train-serve skew风险（2026-06-08）**：xG-proxy特征(运气因子=实际进球-DC_lambda)使用全局快照`xg_proxy_club.json`而非增量构建。训练时所有历史样本使用最终状态，引入轻微前视偏差。已知tradeoff，暂可接受。若需修复，需在train_xgb_club.py中逐场增量重建xg_proxy_state（类似form_state的buffer模式）。

18. **365scores后验调整器（2026-06-08 集成）**：scores365_adjuster.py已集成到daily_jczq.py的predict_match_wrapper。当365scores数据可用时，自动用投票/趋势/人气3信号加权融合微调模型概率(±5pp上限)。调整器在predict_match_wrapper中以try/except包裹，失败不影响主预测。365scores数据每日02:00 cron抓取，当天预测时自动生效。

19. **半全场XGB模型（2026-06-08 训练）**：htft_predictor.py替代了r_ht=0.45的纯数学推导。模型用10,077场俱乐部比赛的HT/FT数据训练9分类XGB，Isotonic校准后acc=30.3%（基线25.5%）。compute_htft_topn()自动选择XGB或数学回退。注意：club_matches.json中ht_h/ht_a字段来自football-data.org的score.halfTime，部分旧数据可能缺失。

20. **动态市场权重（2026-06-08 集成）**：build_prediction_bundle()中新增市场隐含概率与模型概率的加权融合，使用mc_market_weight_helper的Elo gap+neutral+market_strength逻辑。权重范围10%-42%。仅当500.com赔率可用时生效，否则跳过融合。

21. **俱乐部模型训练数据源**：football-data.org免费tier仅支持~3赛季历史（2023-2025），2022及更早返回403。seasons端点返回404，必须用matches?season=YYYY直接拉取。fetch_league_data.py已适配此限制。

22. **DC模型均匀分布退化 (2026-06-08)**：DC对国际友谊赛/未知队伍默认输出均匀分布(1/3, 1/3, 1/3)，λ=1.00-1.00。这意味着模型对这些比赛**零预测能力**。检测方法：`abs(dc_h - 1/3) < 0.02 and abs(dc_d - 1/3) < 0.02`。遇到此情况时：
    - 不要基于DC概率计算EV（会得到虚假信号）
    - 不要在回测中使用合成赔率（会得到虚假ROI）
    - 应使用纯市场赔率分析或DC仅限有训练数据的俱乐部赛事

23. **每日预警cron (2026-06-08)**：已设置cron job `3b404abedaf4`，每天08:00 UTC (16:00北京时间) 运行 `/root/daily_alert.py`，提取价值投注汇总并推送。脚本调用daily_jczq.py，提取"💎 价值投注汇总"区块，输出邮件格式。deliver=all 推送到所有已连接渠道。

24. **500.com队名FIFA排名前缀 (2026-06-08)**：`m5['home_cn']` 格式为 `"[7]荷兰"` (带方括号排名)，而 `future_fixtures` 中是纯队名 `"荷兰"`。`"[7]荷兰" in "荷兰"` 返回 False (子串方向反了)。匹配前必须 `re.sub(r'\[\d+\]', '', name).strip()`。

24. **500.com分析缓存冷启动滞后于赔率数据**：`scrape_500_odds_today()` 和 `scrape_500_analysis()` 在 pipeline 中是先后调用的，但 analysis 依赖于从 odds 数据中提取 shuju_id 或 match_codes。当 odds 数据中有 27 场而 analysis cache 只有 3 场（前一天遗留），说明新赛事的分析页尚未被爬取缓存。原因：
   - `scrape_500_analysis(match_codes)` 需要在 odds 数据中预先提供 `shuju_id`，而 `scrape_500_odds_today()` 可能未返回 shuju_id（需要额外从页面解析）
   - 如果 match_codes 中 id 为空，`scrape_500_analysis()` 会从第一个已知比赛页提取同期列表 — 但依赖的硬编码 matchid（如 `1411007`）可能是旧比赛
   - **症状**：预测输出中大部分比赛缺少 `FIFA排名/近10场/赢盘率/澳门心水/亚盘/首发` 等分析数据，只有 market 基础概率
   - **缓解**：用已知的有效 shuju_id 触发刷新（如先手动打开 `https://odds.500.com/fenxi/shuju-{有效id}.shtml` 确认同期列表），或通过 `FETCH_FRESH=1` 环境变量强制跳过缓存
   - **注意**：analysis 数据是增强展示层（FIFA排名/交战历史/澳门推介），不影响模型核心概率计算，缺失时预测仍可正常输出

25. **365scores 多体育混杂与队名跨语言匹配**：`fetch_365scores_data()` 返回 297 场数据，但其中大部分不是足球（篮球/棒球/网球/排球/综合体育）。`build_365_map()` 通过 `normalize_match_pair()` 做英语→中文队名映射来过滤，最终能匹配到竞彩场次的只有少数。这不是 bug——365scores API 按 lang=1 返回所有体育项目，过滤由队名标准化层自动完成。
   - 验证方式：直接打印 365scores 比赛名称列表，观察有多少能看懂队名
   - 当 mapped 数量远低于 500.com 场次数时，说明该日 365scores 覆盖不足，后验调整器的调整量较小（±1-2pp 而非 ±5pp）
   - **不要**尝试提高匹配召回率（如模糊匹配 3 字母缩写），会引入虚假匹配风险

25. **CRITICAL: 500.com API spf/nspf 数据含义颠倒 (2026-06-08 发现, 2026-06-09 修复)**：\n    playid=269 的 `spf` 字段 **不是标准胜平负(1X2)**。实际含义：rangqiu≠0 时 spf=让球胜平负、nspf=标准1X2。修复：`odds_h/d/a`←nspf, `rq_h/d/a`←spf。nspf为空时(阿根廷vs冰岛/西班牙vs佛得角/德国vs库拉索等)用 `_fetch_live_odds_map()` 从 live.500.com 获取平均欧赔兜底。\n\n27. **predictions_log.csv 只写 top1 不写完整分布 (2026-06-09 修复)**:
    **现象**: 读取 CSV 时，比分/半全场/总进球只有一个"模型主推荐"，没有完整概率分布。
    **根因**: 写入端（不是读取端）只保存了 `pred_top_score`/`pred_top_htft`/`pred_top_goals`，
    没有把 `score_top8`/`htft_top6`/`goals_top5` 的完整分布序列化存入 CSV。
    **修复**: `daily_jczq.py` 的 `record_prediction()` 在写入时序列化完整分布为 JSON 字符串，
    追加 `--score-full`/`--htft-full`/`--goals-full` 到 backtest 脚本的 CLI 参数。
    backtest_jczq.py 新增 `score_full`/`htft_full`/`goals_full` 三个 JSON 字段。
    **读取时**: `json.loads(row['score_full'])` 还原为完整概率字典。
    **注意**: 新字段对历史行为空，只对修复后的新行有值。

26. **live.500.com 编码陷阱 (2026-06-09)**：\n    页面为 GBK 编码，用 `resp.read().decode('gbk', errors='replace')` 解码。`re` 模块对 CJK 字符的匹配需要注意——`周` 字后面的数字是 `二`（汉字）而非直接数字，code 格式为 `周二201`（周X + 数字编号）。提取 code→fid 映射时正确的正则：`r'value=\"(\\d+)\"\\s*/>\\s*(周[一二三四五六七日]\\d+)'`。

26. **疲劳度日期格式 (2026-06-08)**：500.com时间格式 `"06-09 02:45"` (MM-DD HH:MM, 无年份)。`[:10]` 截取得到 `"06-09 02:4"` 不是有效日期。`_parse_date()` 需要检测此格式并补年份 `f"2026-{date_str[:5]}"`。

28. **CRITICAL: subprocess脚本stdout必须纯JSON (2026-06-09 生产事故)**:
    async_500_scraper.py 的重试警告 `print(...)` 输出到了 stdout，
    daily_jczq.py 用 json.loads(proc.stdout) 解析时崩溃:
    `Expecting value: line 1 column 5` → 整个500.com数据熔断 → 27场预测全部丢失。
    **修复**: 所有非数据输出改为 `print(..., file=sys.stderr)`，stdout只留给最终JSON。
    **验证**: `python3 async_500_scraper.py 2026-06-09 269,270 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok'))"` → True
    **教训**: subprocess通信中，stdout=数据通道，stderr=日志通道，永远不要混用。

29. **长尾偏差/天价EV幻觉 (2026-06-09)**:
    market_fallback 场次的 Poisson 外推会产生荒谬的高 EV (如 +15800%)。
    **根因**: 主模型失明 → 欧赔反推 → Poisson 概率未惩罚弱队进攻 → 赔率×概率乘数效应。
    **三道保险过滤** (`bet_math.is_sane_bet()`):
    - 保险1: odds > 30 → 跳过 (数字海市蜃楼)
    - 保险2: prob < 0.15 → 跳过 (低信心不上榜)
    - 保险3: model_type == 'market_fallback' 且 play in ('比分','半全场') → 跳过 (泊松外推不可信)
    **应用**: per-match 显示 + 全局汇总表统一过滤。过滤计数显示在汇总行 `(过滤N个)`。
    **效果**: 79→36个推荐, 最高EV从+15800%降至+280%。
    **model_type 传递**: BetScenario 新增 model_type 字段, 从 p.get('model','') → analyze_match → analyze_scenario → is_sane_bet。
    - `references/system-architecture-audit.md` — 2026-06-10 full system audit: architecture overview, critical findings (zero draw predictions, calibration degradation, club pathway dead), improvement roadmap, diagnostic commands.
- `references/2026-06-10-three-surgeries.md` — 2026-06-10 system diagnosis and three surgical fixes: Isotonic calibrator removal, Draw Correction Layer, model_route tracking fix. Includes full diagnostic commands, patch details, and rollback plan.
    - `references/bet-math-ev-kelly.md`

30. **输出必须包含完整概率分布，不是top1摘要 (2026-06-09 用户纠正)**:
    用户要求看到每个玩法的**全部概率**，不要只输出"模型主推荐"。
    正确格式: 胜平负: 主51.2% | 平27.0% | 客21.9%
    比分: 1:0(12.9%) | 1:1(11.6%) | 2:0(10.6%) | ...
    CSV已存完整分布(score_full/htft_full/goals_full JSON字段)。

30. **Isotonic校准器是ROI主要驱动力 (2026-06-09 验证)**:
    ROI -3.94% → +69.86% 主要来自校准器修正系统性概率偏差，不是XGB更准确。
    意味着换真实form输入后校准器需重训，ROI+69.86%不可外推到未来。

31. **form_state.json是真实数据非占位值 (2026-06-09 澄清)**:
    recent_form()从form_state.json读取336队真实比赛记录。
    唯一问题: 最后更新6月2日(7天前)，football-data.org对友谊赛返回空数据。
    已设cron每天6点尝试更新。详见 references/real-odds-backtest-insights.md。

32. **CRITICAL: nspf为空时的错误赔率转换 (2026-06-09 发现并修复)**:
    当handicap≠0且nspf为空时（如阿根廷-2冰岛），竞彩只开了让球玩法。
    旧逻辑的bug: 让球赔率(2.34/3.23/2.55)被错误转换为1.16/7.03/15.46。
    **根因**: `apply_euro_fallback()`用平均欧赔覆盖了std_h/d/a，但平均欧赔基于让球赔率反推。
    **修复(两处)**:
    1. `scrape_500_odds_today()`: nspf为空时直接设 std_h/d/a = 0（不转换）
    2. `apply_euro_fallback()`: nspf为空时不覆盖odds_h/d/a，改为记录euro_odds_ref
    **效果**: SPF显示"未开售"，EV计算自动跳过SPF，只用rqspf做价值分析
    **输出格式**: 胜平负: 未开售 | 让球(让2): 让胜46.8% / 让平22.7% / 让负30.5% | 市场SP(让球): 2.34-3.23-2.55

## Verification Checklist

- [ ] Ran `python3 /root/daily_jczq.py`
- [ ] Used script output as source of truth
- [ ] Confirmed today's buyable match count from 500.com pipeline
- [ ] **Filtered output to TOMORROW's matches only** (check endtime starts with tomorrow's MM-DD)
- [ ] Included all 5 play categories for every match
- [ ] Stated 90-minute settlement scope explicitly
- [ ] Preserved model-main-pick vs market-info separation
- [ ] Confirmed ledger path `/root/data/predictions_log.csv`
- [ ] **Verified backfill status** — `python3 backfill_results.py --stats`, check 赛果覆盖%, Brier平均值
- [ ] **Verified odds mapping** — ran `verify_odds_mapping.py`, noted any affected matches
- [ ] For nspf-empty matches, automated `_fetch_live_odds_map()` fix in daily_jczq.py covers them; no manual scraping needed
- [ ] **Data-layer audit** (user may ask Q1-Q7):
  - form_state.json freshness (mtime + cron schedule)
  - fallback ratio (if >20% = training data gap, not mapping issue)
  - simple vs main model divergence count
  - calibrators.pkl training data date range (currently 2024 only)
  - bet_action filtering report (how many WATCH/SKIP)
  - CSV league field present, Brier field missing (known gap)

### Odds mapping verification (run per-session)

Before trusting EV/Kelly outputs, verify today's odds mapping:

```bash
cd /root && python3 /root/.hermes/skills/knowledge/daily-jczq-prediction/scripts/verify_odds_mapping.py
```

If affected matches > 0, those matches' SPF display odds, implied probabilities, and EV values are **unreliable** if the automated `_fetch_live_odds_map()` fix also failed or wasn't applied. Check `daily_jczq.py` output for the message: `🌐 live.500.com 平均欧赔兜底加载: N 场`.

## 训练数据扩展: football-data.co.uk (2026-06-15)

**现状瓶颈**: `training_data_with_odds.json` 仅 491 场，含 2025 全年空白，2026 年仅 152 场。nat 模型 75.4% 准确率在 152 场上验证，样本量不足。

**football-data.co.uk 提供**: 3 赛季 × 9 大联赛 = **~9,300 场**（含 Bet365 赔率 + 射门/角球/犯规/红黄牌统计）

### 数据源格式

```python
# URL 模式: https://www.football-data.co.uk/mmz4281/{season}/{league}.csv
# season: 2324, 2425, 2526 (对应 2023-24, 2024-25, 2025-26)
# league: E0=英超, SP1=西甲, D1=德甲, I1=意甲, F1=法甲, etc.
# 返回 106-132 列: 含 B365H/B365D/B365A (Bet365 主/平/客赔率)
# 另含多家博彩公司赔率(BWH/BWD/BWA, PSH/PSD/PSA, WHH/WHD/WHA 等)
# 统计列: HS/AS(射门), HST/AST(射正), HF/AF(犯规), HC/AC(角球), HY/AY(黄牌)
```

### 直接读取

```python
import pandas as pd
url = 'https://www.football-data.co.uk/mmz4281/2425/E0.csv'
df = pd.read_csv(url)  # 380 场英超, 120 列
```

### 通过 soccerdata 库读取

```python
from soccerdata import MatchHistory
mh = MatchHistory(leagues=['ENG-Premier League'], seasons=['2024-2025'])
df = mh.read_games()  # 返回 Pandas DataFrame, 含标准化队名
```

### 与现有训练数据格式的映射

| training_data_with_odds.json | football-data.co.uk 列 |
|---|---|
| home_en | HomeTeam |
| away_en | AwayTeam |
| ft_h / ft_a | FTHG / FTAG |
| spf_result | FTR (H/D/A) |
| market_odds / market_implied_prob | 1/B365H (或多家博彩平均) |
| points_diff / rank_diff | 需从 league table 计算 |
| tournament / date | Div / Date |

### 快速拉取脚本模板

```python
import pandas as pd, json
from datetime import datetime

SEASONS = {'2324':'2024','2425':'2025','2526':'2026'}
LEAGUES = {'E0':'ENG-Premier League', 'SP1':'ESP-La Liga', 'D1':'GER-Bundesliga',
           'I1':'ITA-Serie A', 'F1':'FRA-Ligue 1'}

all_matches = []
for s, syear in SEASONS.items():
    for lcode, lname in LEAGUES.items():
        url = f'https://www.football-data.co.uk/mmz4281/{s}/{lcode}.csv'
        try:
            df = pd.read_csv(url)
            for _, row in df.iterrows():
                all_matches.append({
                    'date': f'{syear}-{row["Date"][:2]}-{row["Date"][3:]}',
                    'home_en': row['HomeTeam'], 'away_en': row['AwayTeam'],
                    'tournament': lname,
                    'spf_result': {'H':'3','D':'1','A':'0'}.get(row['FTR'], ''),
                    'ft_h': int(row['FTHG']), 'ft_a': int(row['FTAG']),
                    'market_odds': max(row['B365H'], 1.01) if 'B365H' in df.columns else 2.0,
                    'market_implied_prob': 1.0 / max(row['B365H'], 1.01) if 'B365H' in df.columns else 0.5,
                })
        except Exception as e:
            print(f'  ⚠ {s} {lcode}: {e}')
print(f'Total: {len(all_matches)} matches')
```

### 已知限制
- football-data.co.uk 为 CSV 离线数据，无实时接口
- 2025-26 赛季数据在赛季结束后才完整归档
- 队名标准化：football-data.co.uk 与 form_state.json 可能用不同英文名，需映射
- 含赔率时注意剔除未来比赛（赔率是 pre-match 而非历史）
- **当前训练数据断档根因**: 原 `training_data_with_odds.json` 仅 263 场 (2024年) + 228 场 (2026年)，2025 全年空白。将 football-data.co.uk 数据转换为同一格式后，可直接用于重训 nat/XGB 模型。

## 500.com Analysis Scraping (v2)

See `references/500-analysis-scraping.md` for the complete v2 integration:
- URL pattern, encoding (gbk fallback chain — NOT pure gb2312), HTML parsing
- Match list extraction from 竞足 section
- **v2 核心新增**: 历史10场逐场欧赔+亚盘(解决"合成赔率"问题的真实数据源), 当前欧赔/亚盘, matchid主键, 完整未来赛程
- Cache strategy (1-hour TTL)
- Enrichment pattern (inject display fields without altering model logic)
- Integration into daily_jczq.py pipeline
- 疲劳度特征 (fatigue_features.py) 集成
- 亚盘价值计算 (asian_handicap.py, Skellam分布)
CLV回测 (clv_backtest.py) 用法
- 赛果回填 (backfill_results.py) 用法

See `references/competition-tier-filtering.md` for the competition tier filtering technique that improved ROI from +3.24% to +37.64%.

### 500.com 赔率爬取 (v2 — 2026-06-09 升级)

See `references/500-api-spf-nspf-quirk.md` for the critical spf vs nspf distinction.
See `references/async-scraper-architecture.md` for the new async scraper.
See `references/csv-full-distribution-serialization.md` for how full probability distributions are saved to CSV.

**架构升级**: 旧 Node.js+jsdom 方案 (`fetch_500_market.py` + `scrape_500_market.js`) 已废弃。新方案使用 Python `aiohttp` + `BeautifulSoup`，位于 `/root/wc_2026_upgrade/async_500_scraper.py`。

**核心突破**: 竞彩所有页面 DOM 结构统一，无需为每个玩法写独立解析规则。所有赔率 SP 值绑定在 data-type/data-value/data-sp 三个标准属性上：

```python
for node in container.find_all(attrs={'data-sp': True}):
    play_type = node.get('data-type')   # nspf/spf/bf/jqs/bqc
    play_value = node.get('data-value') # 3/1:0/3-3/0 等
    sp_val = float(node.get('data-sp')) # 赔率数值
```

**并发**: 4个玩法页面(playid=269/270/271/272)使用 asyncio.Semaphore 并发请求，约2秒完成（旧顺序调用8-12秒）。URL 追加 `_t={timestamp}` 穿透缓存。

**合并**: 按 `data-fixtureid` 对 4 个页面的赔率进行深度合并，每场比赛 5 种玩法在同一个 dict 中。不再需要在 Python 端跨 markets 字典拼接。

**bet-more-wrap 处理**: 比分(bf)等复杂玩法数据在紧邻主行的 `<tr class="bet-more-wrap">` 中。`_parse_html()` 通过 `row.find_next_sibling('tr', class_='bet-more-wrap')` 定位并一并搜索 data-sp 节点。

## Critical Bugs Found & Fixed (2026-06-14 Session)

### Bug: market_h undefined in _try_hybrid_predict (SPF=0.0% for WC matches)

**Symptom**: Matches like Germany vs Curaçao show `SPF: 主0.0% / 平0.0% / 客0.0%` even though nat model can predict them.

**Root cause**: `_try_hybrid_predict()` line 592 uses `market_h` variable but it's never defined in the function scope. The `except Exception: return None` at line 772 silently catches the `NameError`.

**Fix** (daily_jczq.py:591-593):
```python
# Before (BUGGY):
market_implied = market_h if market_h > 0 else op_h

# After (FIXED):
_mi = locals().get('market_h', 0)
market_implied = _mi if _mi > 0 else op_h
```

**Detection**: If SPF shows 0.0% for all three outcomes AND the match is in DC model, this bug is likely the cause.

### Bug: form data hard check causing return None

**Symptom**: `_try_hybrid_predict()` returns None for matches where one team has no form data (e.g., Curaçao in form_state.json).

**Root cause**: Line 546 `if h_key not in fs or a_key not in fs or len(fs.get(h_key, [])) < 1: return None` — hard requirement for form data.

**Fix** (daily_jczq.py:543-550): Changed to soft check:
```python
has_form = h_key in fs and a_key in fs and len(fs.get(h_key, [])) >= 1 and len(fs.get(a_key, [])) >= 1
if not has_form:
    print(f'    ⚠ {home} vs {away}: form_state无数据, 使用DC+XGB直推')
# Continue instead of return None
```

**Why form data is optional**: `recent_form()` returns default values `[0.4, 1.8, 1.6, 0.2]` for unknown teams. The nat model (11-dim) doesn't use form features at all — only DC/Elo/market.

## One-Shot Trigger Mapping

If the user says any of the following, load this skill and execute immediately:

- `执行今天的竞彩预测`
- `执行今日竞彩预测`
- `今日竞彩预测`
- `今天的竞彩预测`
- `去500看今天能买哪几场然后预测`
- `预测明天的竞彩`
- `明天有什么比赛可以买`
- `明天竞彩预测`

The correct default action is **run the pipeline now, filter for tomorrow**, not discuss how it could be done.

## 500.com Analysis Scraping (v2)