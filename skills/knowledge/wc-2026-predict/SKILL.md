---
name: wc-2026-predict
title: 2026世界杯比赛预测管线
description: 2026世界杯冠军概率日更管线 + 单场预测。每日cron运行DC+XGB+市场赔率混合模型，输出冠军概率、对比昨日变化、API信用额度。用户问"预测XX vs XX"时按步骤执行单场预测。
---

# 2026世界杯比赛预测管线

## 触发条件

用户说"预测XX比赛"、"XX vs XX比分"、"这场怎么看"等涉及2026世界杯具体比赛预测时，**必须**按此流程执行。
**日更cron任务**也使用此技能（每日更新冠军概率），参考"每日更新管线"节。

## 项目文件清单（按使用顺序）

| 文件 | 路径 | 用途 |
|------|------|------|
| 历史数据缓存 | `/root/data/international_results.json` | 国际比赛结果（1872-至今），`filter_matches()` 默认使用 `datetime.now()` 动态截止 |
| DC+XGB基础模型 | `/root/wc_2026_phase1.py` | Dixon-Coles拟合、Elo计算、特征工程 |
| **最终版管线** | **`/root/wc_2026_final.py`** | **统一冠军+亚军管线 (v2.4修复版) — 29维特征+Optuna+并行MC 200K+双签表模式(`--bracket={ranked,official}`)+东道主灵敏度+凸组合市场校准+每轮晋级概率(R16/QF/SF/Final/Champion/Runner)** |
| **XGBoost模型** | **`/root/data/xgb_model_29.pkl`** (优先, 29维形式特征) → 回退链: `xgb_model_26.pkl` → `xgb_model_20_3.pkl` | **已训练XGB (29维: 23基线+6滚动形式, Optuna参数) ~1.0MB** |
| **DC模型** | **`/root/data/dc_model.pkl`** | **已训练Dixon-Coles（全量数据2021-2026，非historical cutoff）** |
| **Elo评分** | **`/root/data/elo_ratings.pkl`** | **已计算Elo评分字典** |
| **2026分组** | **`/root/data/2026_groups.json`** | **12组×4队正式分组（摘自 WRooney108/World-Cup-Betting seed.ts，含最新FIFA排名）**；备选 `/root/data/2026_groups_official.json`；详见 `references/2026-official-groups-source.md`；可用 `scripts/extract_official_groups.py` 从 seed.ts 同步更新 |
| 实时赔率缓存 | `/root/data/theodds_api_data.json` | DraftKings/FanDuel/Betfair H2H赔率 + 夺冠赔率 |
| MC冠军概率 | `/root/data/final_results.json` | 200K并行12×4 MC冠军+亚军概率+每轮晋级概率（含EV表，统一输出） |
| 旧冠亚军旁路 | `/root/wc_2026_champ_runnerup.py` | **已废弃**，功能合并至 `wc_2026_final.py`
| Odds API刷新参考 | `references/odds-api-refresh.md` | API调用格式、响应解析、合并逻辑 |
| **夺冠赔率刷新脚本** | **`/root/fetch_worldcup_odds.py`** | **独立运行: `python3 /root/fetch_worldcup_odds.py` — 拉5家博彩公司世界杯夺冠赔率。每日00:00 UTC cron（no_agent, f22f1d2494f3）自动执行。复制到 ~/.hermes/scripts/ 供 cron 使用** |
| 单场预测脚本 | **`/root/predict_match.py`** | **独立运行: `python3 /root/predict_match.py <home> <away>` — 加载已训练模型+市场赔率，输出概率+比分MC+信心度。东道主持场加 `--home` 触发 host_bonus。10秒出结果。新增可选参数 `lineup_features` (见 Sofascore 集成)** |
| **阵容数据集成** | **`/root/sofascore_integration.py`** | **EasySoccerData→predict_match lineup_features。可选参数动态调整友谊赛折扣 (0.3→0.6) 基于首发市场价值/缺主力/阵型。详见 `references/sofascore-lineup-integration.md`** |
| 完整赛程参考（72场+场馆） | `/root/data/2026_matches_ref.json` | 从 seed.ts 提取的官方赛程，含轮次/日期/场馆 |
| 赛程提取指引 | `references/official-schedule-extraction.md` | seed.ts → 赛程/分组json的完整流程 + 队名映射陷阱 |
| 20+3黄金消融研究 | `references/golden20-ablation-study.md` | 特征演进、Optuna参数、FeatureBuffer模式 |
| 可复现实验审计手册 | `references/repro-audit-runbook.md` | 一键重训→阈值验收→72场复算→证据打包（Grok交叉审计） |
| **72场小组赛预测** | **`/root/data/group_stage_predictions.json`** | 全72场得分概率+λ预期进球（2026-06-02新生成） |
| **淘汰赛模拟脚本（Elo种子）** | **`scripts/simulate_knockout.py`** | 独立运行: `python3 /root/.hermes/skills/knowledge/wc-2026-predict/scripts/simulate_knockout.py [N]` — 基于Elo种子配对做N次MC淘汰赛模拟，输出各轮晋级概率+冠军排名。默认50K次（~4分钟） |
| **淘汰赛模拟脚本（官方FIFA签表）** | **`/root/simulate_knockout_official.py`** | 独立运行: `python3 /root/simulate_knockout_official.py [N]` — 使用 openfootball/worldcup cup_finals.txt 的官方R32配对路径，含 team-constrained-first 第三名分配算法。默认50K次。比 Elo 种子准确，推荐正式报告使用 |
| **淘汰赛模拟结果（Elo种子）** | `/root/data/knockout_simulation.json` | 50K MC淘汰赛结果（Elo种子配对） |
| **淘汰赛模拟结果（FIFA签表）** | `/root/data/knockout_fifa_bracket.json` | 50K MC淘汰赛结果（官方FIFA路书，同源 openfootball） |
| **官方淘汰赛签表** | `references/2026-official-knockout-bracket.md` | 从openfootball/worldcup提取的FIFA正式R32→Final配对路径（含第三名槽位规则） |
| **72场小组赛细项** | **`/root/data/group_stage_details.json`** | 全72场每场：DC+XGB概率、λ、比分Top8、半全场9向Top6、总进球Top6、半场概率 |
| **72场细项计算** | **`/root/compute_group_details.py`** | 独立运行: `python3 /root/compute_group_details.py` — 遍历72场调用predict_match_detail()，泊松50K MC，保存全部细项 |
| **10届回测脚本 (v2)** | **`/root/wc_10edition_backtest_v2.py`** | 独立运行: `python3 /root/wc_10edition_backtest_v2.py` — 跨 1986-2022 共 10 届 (604场) leave-one-edition-out 回测 Elo+DC+XGB29+Stacking+让球-1, 总耗时 ~37 分钟。29维特征 (15基线+5黄金+3odds+6滚动形式) + LR meta-learner 9维 stacking。详见 `references/wc-10edition-backtest.md` |
| **历史滚动回测脚本** | **`/root/backtest_runner.py`** | **独立运行: `python3 /root/backtest_runner.py` — 读取已完赛 `\/root\/data\/results\/YYYY-MM-DD.json`，跑混合模型回测、Brier/RPS/EV 模拟；不要拿 `predictions_log.csv` 直接当赛果** |
| **历史回测与校验说明** | `references/historical-backtest-and-validation.md` | `predictions_log.csv` vs `results/*.json` 数据源分离、`final_results.json` 的验证字段、当前会话确认的回测入口 |
| **Gold特征补全工具** | `/root/feature_helper.py` | 预计算H2H缓存(7481对)+12场form缓存(336队), 修复predict_match.py和daily_jczq.py的gold特征train-serve skew。`python3 feature_helper.py` 重建缓存 |
| **365scores后验调整器** | `/root/scores365_adjuster.py` | 投票/趋势/人气3信号加权融合, ±5pp上限。已集成到daily_jczq.py的predict_match_wrapper |
| **回测管线** | `/root/backtest_pipeline.py` | `--verify` 核验predictions_log, `--backtest N` 历史滚动回测。输出Brier/RPS/LogLoss/准确率。cron每日08:00运行 |
| **form_state更新脚本** | `/root/update_form_state.py` | 每日从football-data.org+international_results.json更新form_state.json + 重建H2H/form_12缓存。cron每日06:00运行 |
| **H2H特征缓存** | `/root/data/h2h_cache.json` | 预计算的H2H对特征 (7481对), 由feature_helper.py生成 |
| **12场form缓存** | `/root/data/form_12.json` | 预计算的12场form (336队), 由feature_helper.py生成 |

| **10届回测结果 (v1)** | `/root/data/wc_10edition_backtest.json` | v1 极速版 (无XGB) 每届 Elo/DC/HYB Acc+Brier+LL, 让球-1 Acc, 实际 H/D/A 基线 + 宏平均汇总 |
| **10届回测结果 (v2)** | `/root/data/wc_10edition_backtest_v2.json` | v2 完整版每届 Elo/DC/XGB29/Stack Acc+Brier, 让球-1 Acc, 5模型并列对比 + 宏平均 |
| **双DC管线 (2026-06-14新增)** | **`/root/wc_2026_upgrade/calibrated_predictor.py`** | **双管线预测器: Pipeline A (国家队dc_model 226队+XGB) → Pipeline B (俱乐部dc_club 2,174队+Elo) → Pipeline C (Elo+Market) + 置信度加权+队名标准化** |
| **每日预测管线 (2026-06-15更新)** | **`/root/wc_2026_upgrade/daily_wc_pipeline.py`** | **每日拉取The Odds API赔率 → 加载双DC管线 → 预测全部场次 → 输出摘要表。`fetch_scores()` 已修复 `daysFrom=3` (原默认不返回已完成比赛)。Step 2 检查已完成比赛仍为 TODO 占位符** |
| **赛后比分累积 (2026-06-15新增)** | **`/root/wc_2026_upgrade/accumulate_results.py`** | **从 The Odds API scores 端点 (daysFrom=3) 拉取已完成比赛赛果，累积到 `/root/data/wc_completed_results.json`。幂等设计, 每天跑一次。在 pipeline 的 fetch_scores() 修复后运行** |
| **WC已完成比赛结果 (2026-06-15新增)** | **`/root/data/wc_completed_results.json`** | **已完成比赛的累积结果数据库, 格式: [{home, away, home_score, away_score, result, commence_time, date, saved_at}]。用于模型重训前的真实标签回填** |
| **训练数据间隙检查/回填 (2026-06-24更新)** | **`scripts/check_training_gap.py`** | **每日cron中紧接accumulate_results.py运行: 遍历wc_completed_results.json, 将未进入training_data_with_odds.json的完赛结果补录。| **淘汰状态更新 (2026-07-01新增)** | **`/root/update_elimination_status.py`** | **从 wc_completed_results.json + tournament_state.json 计算已淘汰球队, 写入 eliminated 字段。每日 accumulate_results.py 后运行。详见 `references/elimination-tracking.md`** |
| **国家队专用模型** | **`/root/data/xgb_model_nat.pkl`** | **11维干净特征, 只在国际数据训练, 过滤中文名噪音. val acc=75.4% (2026-06-14重训后, 修复标签污染+form软检查+market_h修复)** |
| **国家队校准器** | **`/root/data/calibrators_nat.pkl`** | **Isotonic 校准器, 与xgb_model_nat配对** |
| **俱乐部DC(中文队名)** | **`/root/data/dc_club.pkl`** | **2,174支俱乐部, 从500_history_backfill.csv(63K场)训练, γ=0 (无主场优势)** |
| **俱乐部DC(英文队名)** | **`/root/data/dc_club_en.pkl`** | **152支俱乐部, 从football-data.org(2,743场)训练, 命名格式不兼容问题** |
| **世界杯赔率缓存** | **`/root/data/wc_2026_odds_today.json`** | **64场世界杯1X2赔率, 从The Odds API拉, 含16家bookmaker** |
| **最终预测集** | **`/root/data/wc_final_predictions.json`** (JSON) + **`/root/data/wc_final_predictions.txt`** (表格) | **64场世界杯决赛阶段预测, 含DC/XGB/Hybrid/Market/Final五列** |
| **训练数据(2436场)** | **`/root/data/training_data_with_odds.json`** | **从kaijiang+500.com+thestats合并: 含market_implied概率, 含WC完赛结果(4场, 2026-06-17追加). 不再只491场 — 通过 merge_training_data.py 从 training_data_thestats.csv 追加了~1,945场** |
| **双DC架构参考** | `references/dual-dc-model-2026-06-14.md` | **架构图、三种DC模型、Bug记录(Bug1-4)、数组顺序约定、置信度加权公式** |
| **Odds数据覆盖模式 (2026-06-21新增)** | `references/odds-data-coverage-patterns.md` | The Odds API覆盖缺口(澳大利亚vs土耳其未覆盖)、赔率文件扁平结构(home/away/odds_h/odds_d/odds_a)、跨日期赔率搜索方法论、队名一致性验证 |
| **外部分析：模型改进机会** | `references/model-improvement-opportunities.md` | FifaWorldCupPreview 对比分析：滚动形式、射手特征、概率校准、API 部署等 4 个改进方向 |
| **系统架构全面审查** | `references/system-architecture-review-2026-06-08.md` | 2026-06-08: 全架构 13 类问题 + 改进优先级 P0-P3，含缺失模块/MC form退化/无反馈闭环三个致命问题 |
| **俱乐部数据管线** | `references/club-data-pipeline.md` | 2026-06-08: 9联赛×3赛季 10,077场, 119队Elo/form/DC, football-data.org API陷阱, 增量保存 |
| **HT/FT模型+动态市场权重** | `references/htft-and-market-weight-2026-06-08.md` | 2026-06-08: 半全场9分类XGB(acc=30.3% vs 基线25.5%), 动态市场权重融合(10%-42%) |

## 执行流程（8步，不可跳过）

### Step 1: 确认比赛信息

获取用户想预测的比赛：
- 主队名、客队名
- 比赛日期（2026世界杯：6月11日-7月19日）
- 确认两队都在 `TEAMS_2026` 列表中

**队名映射**（数据文件与显示名的对应 — 完整版，来自 `team_name_normalizer` 字典）：

```
'USA' → 'United States'
'Korea Republic' → 'South Korea'
'Czechia' → 'Czech Republic'
'Bosnia & Herzegovina' → 'Bosnia'
'Bosnia and Herzegovina' → 'Bosnia'
'Türkiye' → 'Turkey'
'Ivory Coast' → "Côte d'Ivoire"
'DR Congo' → 'Congo DR'
'Congo DR' → 'Congo DR'
'São Tomé and Príncipe' → 'Sao Tome and Principe'
'Cabo Verde' → 'Cape Verde'
```

以及全名→标准名映射：
```
'United States' → 'USA'
'South Korea' → 'Korea Republic'  (注意：数据中用 'Korea Republic')
'Czech Republic' → 'Czechia'
'Turkey' → 'Türkiye'
'Bosnia' → 'Bosnia and Herzegovina'  (用于 Elo 字典存取)
```

官方赛程中的名称（seed.ts）与代码内部名称的对应是插值常见坑，更新分组或赛程后必须同步此映射。

### Step 2: 验证数据可用性

检查 `/root/data/international_results.json` 存在且非空：
- 所有目标球队在数据中有至少10+场比赛记录
- 若某队数据过少（如<5场），在输出中标注"数据稀缺，预测置信度低"

### Step 3: 加载市场赔率

从 `/root/data/theodds_api_data.json` 提取夺冠赔率用于校准（当前只有冠军盘，无单场H2H赔率）：

```python
import json
with open('/root/data/theodds_api_data.json') as f:
    md = json.load(f)
winner_odds = md.get('winner_odds', {})
total_implied = sum(1.0 / p for p in winner_odds.values())
market_probs = {t: (1.0 / p) / total_implied for t, p in winner_odds.items()}
```

夺冠赔率校准应用于MC matchups：`blended = (1-mw) * model_vec + mw * market_vec`（凸组合，model=[A,D,H], market=[rel_a, 0, rel_h]）。⚠️ **不要写非归一化的 model_weight + market_weight 加法**——旧代码用 `blended_h = hybrid[2]*MODEL_WEIGHT + rel_h*mw`，模型与市场权重不对齐会压缩draw概率。

若无赔率数据，回退使用Elo校准赔率（`make_odds(eh, ea)`）。

### Step 3.5: 快速预测路径（加载已训练模型，10秒出结果）

如果只需要对单场比赛做预测（不重训管线），直接使用独立脚本 `/root/predict_match.py`：

```bash
python3 /root/predict_match.py "Switzerland" "Bosnia and Herzegovina"
```

输出格式：Elo + λ预期进球 → 4行模型对比概率表（DC / XGB / Hybrid / Final） → MC 100K比分Top5 → 零封概率 → 结论+信心度。

也可内联Python加载已保存模型（注意DC predict_proba输出顺序）：

```python
import sys, joblib, math, numpy as np
sys.path.insert(0, '/root')
from wc_2026_phase1 import MAX_GOALS

# 加载已训练模型
xgb_model = joblib.load('/root/data/xgb_model_20_3.pkl')
dc = joblib.load('/root/data/dc_model.pkl')    # DC模型对象
elo = joblib.load('/root/data/elo_ratings.pkl')

DC_WEIGHT, XGB_WEIGHT = 0.4, 0.6  # 旧静态权重；生产管线已用熵基动态(见Step 4)

# ⚠️ dc.predict_proba(home, away, neutral=True) 返回 [Home, Draw, Away]
#    dc.predict_lambda(home, away, neutral=True) 返回 (λ_home, λ_away)
dc_p = dc.predict_proba(home, away, neutral=True)  # [H, D, A]
lam_h, lam_a = dc.predict_lambda(home, away, neutral=True)

# 构建23或29维特征（取决于加载的模型维度）
eh, ea = elo.get(home, 1500), elo.get(away, 1500)
def make_odds(eh, ea):
    dh = ea - eh; da = eh - ea
    return [1/(10**(-dh/400)+1), 1/(10**(-da/400)+1), None]
op = make_odds(eh, ea)

XGB_DIM = xgb_model.n_features_in_ if hasattr(xgb_model, 'n_features_in_') else 29

b15 = [(eh-ea)/400, lam_h, lam_a, lam_h-lam_a,
       math.log(max(lam_h,0.01)/max(lam_a,0.01)),
       dc_p[0], dc_p[1], dc_p[2],  # dc_p=[H,D,A] → feature expects [H,D,A]
       0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1]
gold = [0.0, 1, 0, 0.0, 0.0]
base_feat = b15 + gold + [op[0], op[1], op[2] if op[2] else 0.0]
if XGB_DIM > 23:
    # 6维滚动形式特征 (placeholder for 无form上下文)
    base_feat += [0.0, 0.0, 0.0, 0.0, 1.5, 1.5]
feat = np.array([base_feat])

# XGB预测: xgb_p = [P(away), P(draw), P(home)]  (classes=[0,1,2])
xgb_p = xgb_model.predict_proba(feat)[0]

# 混合: DC [H,D,A] → [A,D,H] 对齐 XGB [A,D,H]
dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])  # [A, D, H]
hybrid = DC_WEIGHT * dc_ado + XGB_WEIGHT * xgb_p  # [A, D, H]
print(f'Home: {hybrid[2]*100:.1f}% Draw: {hybrid[1]*100:.1f}% Away: {hybrid[0]*100:.1f}%')
```

**何时用快速路径：** 单场比赛预测，无需更新冠军概率。
**何时重训：** 有新比赛数据（每周新赛果）、参数/特征变更、需更新50K MC冠军概率。

### Step 4: 训练模型（全量更新用）

执行以下步骤（通过Python脚本运行，不可跳过DC拟合）：
- `load_data()` 自动缓存国际赛果
- `filter_matches()` 使用 `datetime.now()` 动态截止日期（不再硬编码 `2026-05-19`），默认覆盖最近5年

```python
import sys; sys.path.insert(0, '/root')
from wc_2026_phase1 import *
from xgboost import XGBClassifier

# 1. 加载数据
cache = '/root/data/international_results.json'
all_m = load_data(cache)
matches = filter_matches(all_m)
elo = compute_elo(all_m)

# 2. DC拟合
dc = DixonColes(time_decay_hl=540)
dc.fit(pd.DataFrame(matches))

# 3. 特征工程 + XGB训练
# 见 predict_matches_full.py 中的完整pipeline
```

**关键参数**：  \n- DC time_decay_hl=540（18个月半衰期）  \n- DC训练数据: **全量2021-2026**（不是historical cutoff）。XGB特征必须用全量dc构建，与MC预测保持一致。2022回测用独立的bt_dc（仅pre-2022数据）。  \n- 特征集：**29维 (23基线 + 6滚动形式)**  \n  - 15维基线: Elo差, λ_h/λ_a/λ差/λ比, DC概率[H,D,A], 近5场胜率×2, 进攻/防守优势, 进球/胜率差, 中性场  \n  - 5黄金特征: H2H净胜球, 大赛正赛, 友谊赛, 12场进攻优势, 12场客场进攻力  \n  - 3赔率: Elo校准赔率概率[H,D,A]  \n  - **6滚动形式(新增)**: 主队近5场场均进球/失球, 客队近5场场均进球/失球, 主队近5场积分/3, 客队近5场积分/3 — 让模型感知球队近期状态趋势  \n- XGB: **Optuna防守参数** — max_depth=4, lr=0.032, n_est=369, reg_alpha=3.05, reg_lambda=2.69, colsample_bytree=0.45, subsample=0.64, min_child_weight=8.2  \n- 混合权重：**熵基动态权重** `compute_dynamic_xgb_weight(xgb_probs, alpha=0.30, beta=0.50)` — 按 XGBoost 预测的香农熵自动分配 XGB/DC 权重。XGB 预测尖锐（如 80/12/08）→ XGB 权重大；预测均匀（如 34/33/33）→ DC 兜底。取代旧版 `min_games` 硬阈值逻辑。函数在 `wc_2026_final.py`（用 np）和 `daily_jczq.py`（纯 math）各有一份。α=0.30 是基础 XGB 权重下限，β=0.50 控制置信度调节幅度，最终钳位至 [0.10, 0.90]。  \n- **MC市场校准权重（凸组合）**：`blended = (1-mw) * model_vec + mw * market_vec`，其中 `mw` 来自 `market_weight_for_match()` 的动态权重（范围 0.10-0.42，基于Elo差距+中立+市场强度）。禁止使用 `model_weight + market_weight` 的非归一化加法。  \n  - 变量名: `MARKET_WEIGHT = 0.40`（wc_2026_final.py 顶部常量，用于缩放 mw）  \n- 特征构建：**FeatureBuffer增量模式**(O(1) per match)，避免O(n)全量扫描  \n- **新增特征验证协议**：任何新特征加入前，必须跑5次XGB（不同random_seed）的A/B测试——29维基线 vs 新特征维度——比较Brier和Acc。若Brier恶化或无统计显著改善（改善<0.001），立即回退。不要保留排名靠后导致整体负收益的特征。  \n  \n**校准集独立分割（60/20/20 — 2026-06-08 引入）**：  \n  Isotonic/Platt 校准器必须在独立于训练集和验证集的数据上拟合，否则校准效果被高估且版本漂移敏感。  \n  - 数据分割：**60% 训练 / 20% 校准 / 20% 验证**（按日期严格时序，不用随机 shuffle）  \n  - 时序分割原则：训练 < 校准 < 验证（按日期升序），防止校准器看到未来信息  \n  - 校准器拟合：IsotonicRegression 仅在 **校准集** 上 fit，**不是在训练+校准上**  \n  - 指标报告：校准器在 **验证集** (val) 上评估 Brier/Acc，not 训练集 or 校准集  \n  - 校准器保存：每次训练完成后，calibrators 通过 `joblib.dump(cal, '/root/data/calibrators.pkl')` 保存。下次加载时从 pkl 读取（**不重新拟合**），确保版本一致  \n  - 版本飘移预防：calibrators.pkl 的 mtime 必须与 xgb/dc/elo 模型 mtime 检查一致性——若天数差 > 7 或 calibrators.pkl 不存在，触发重新校准  \n  - 影响验证：2022 WC Hybrid 准确率 56.2% → **60.9%**（+4.7pp），Brier 0.1993 → **0.1808**（-0.0185），部分来自校准独立分割

⚠️ 混合权重现为**熵基动态**（见上），不再使用静态 DC×0.4+XGB×0.6。快速路径示例代码保留静态写法做演示——实际生产调用 `compute_dynamic_xgb_weight()`。

### Step 6: 运行蒙特卡洛模拟（12×4正确赛制，并行加速）

2026世界杯赛制为 **12组×4队 = 72场小组赛**，每组前2名 + 8个最佳第3名晋级32强淘汰赛。

冠军模拟用200K次（并行2进程）：
- 从DC λ参数采样进球（Poisson分布）
- 用混合概率校准结果分布（确保结果分布匹配混合概率）
- **12组×4队**正确赛制：每组6场循环赛 → 积分/净胜球排序 → 前2+8最佳第3 → R32 → R16 → QF → SF → Final
- 淘汰赛签表模式（二选一，`wc_2026_final.py` CLI `--bracket={ranked,official}`）：
  - 默认 `ranked`（向后兼容）；**每日cron使用 `official`**
  - **Elo 排名配对**（`--bracket=ranked`）：按晋级的 Elo 高低 1v32, 2v31……，不是随机 shuffle
  - **官方 FIFA 签表**（`--bracket=official`）：从 openfootball/worldcup 提取的 R32→Final 写死路径，含 team-constrained-first 第三名分配算法。已直接集成进 `wc_2026_final.py`，无需再跑独立脚本。
- 淘汰赛：加时赛（若90分钟平局，再模拟30分钟）、点球（基于Elo概率）
- 分组从 `/root/data/2026_groups.json` 加载（**已确认为正式分组，源自 WRooney108/World-Cup-Betting seed.ts，含最新FIFA排名**）
- 追踪冠军+亚军双结果，输出 EV 表和 Kelly 推荐
- **每轮晋级概率（Slide's Signed Bracket Calibration）**：输出每队 R16/QF/SF/Final/Champion/Runner 概率，便于二级市场投注（如"Spain进4强@赔率X"）。保存在 `final_results.json` 的 `round_prob` 字段。
- **东道主灵敏度分析**：自动输出 `host_bonus=[0.0, 0.07, 0.1445]` 三组**uniform 基准对比**（所有东道主用同一值），以与旧版比较。实际 200K 主 MC 使用 **per-team** `HOST_BONUS_BY_TEAM` dict + `KO_HOST_DECAY=0.5` 淘汰赛衰减（详细机制见 `references/host-bonus-mechanism.md`）

⚠️ **赛制变更历史**：2026 WC最初提案为16组×3队，但FIFA最终确认为12组×4队。旧版simulate_from_cache使用16×3赛制（3 pots × 16队，每组前2晋级），**必须使用12×4版本**。`wc_2026_final.py` 默认使用12×4正确赛制且加载市场赔率。

对于单场比赛预测，跑100,000次MC（不跑完整锦标赛，只模拟该场比赛的比分分布）。

### Step 7: 输出结果

#### Step 7.5: 当用户要求"比分概率Top5 + 推荐比分"时（72场或单场）

在已生成的1X2概率基础上，补充比分分布：

1. 用 `dc.predict_lambda(home, away, neutral=True)` 取 `(λ_home, λ_away)`
2. **快速法（纯Poisson网格）**：用泊松网格 `0..7` 计算比分联合分布 `P(i,j)=Pois(i;λ_home)*Pois(j;λ_away)`，归一化后取 Top5
3. **精确法（Monte Carlo）**：`/root/compute_group_details.py` 对每场跑 50K 泊松采样，同时得到比分、半场比分、半全场9向、总进球分布。输出保存在 `/root/data/group_stage_details.json`，可直接复用
4. 推荐比分规则：
   - 先看1X2最大项（主胜/平/客胜）
   - 再在对应比分子集里取概率最大的比分（主胜集 `i>j`，平局集 `i==j`，客胜集 `i<j`）
5. 结果必须落盘，便于复核：
   - `/root/data/group_stage_scoreline_top5.json`
   - `/root/data/group_stage_scoreline_top5.txt`

预计算全场细项已保存至 `/root/data/group_stage_details.json`，每场包含：
- `prob_h/d/a` — DC+XGB混合概率
- `lam_h/lam_a` — 预期进球
- `scores` — 比分Top8（泊松50K MC）
- `ht_scores` — 半场比分Top8
- `htft` — 半全场9向Top6（如"平/胜","胜/胜"）
- `total_goals` — 总进球分布Top6
- `ht_home/draw/away_pct` — 半场胜平负概率

输出字段最少包含：
- `top5_score_probs: [{score, prob} x5]`
- `recommended_score`
- `recommended_score_prob`
- `result_probs`（对应1X2概率）

每场比赛输出格式：

```
比赛：🇨🇦 Canada vs Bosnia 🇧🇦

【胜平负概率】
  DC:          52.8% / 27.1% / 20.1%
  XGBoost:     45.6% / 40.1% / 14.3%
  Hybrid:      49.9% / 32.3% / 17.8%
  📊 市场赔率: 52.9% / 26.8% / 20.4%
  ✅ 最终:     55.8% / 21.5% / 22.7%

【MC 10万次模拟】
  Top 比分: 1-0 (11.9%), 2-1 (10.9%), 2-0 (8.0%)
  BTS: 66.5% / O2.5: 65.4% / U1.5: 17.4%

【🃏 购彩建议】
  方向：Canada 胜（赔率1.77）
  大球：O2.5 @ 65.4% ⭐
  稳健：Canada 不败（胜+平）
  confidence: 中高
```

### Step 8: 提炼购彩建议

基于模型结果给出**可执行的购彩推荐**：

- **方向推荐**：主胜/平/客胜（给出参考赔率）
- **大球小球**：O2.5或U1.5哪个更稳（给出概率）
- **串关建议**：多场比赛组合推荐
- **置信度标尺**：
  - 高（>55%）：可做胆
  - 中（45-55%）：谨慎
  - 低（<45%）：纯娱乐

## Step 8b: 用户问具体比赛"怎么买"时的完整工作流

当用户指名某场世界杯比赛想了解投注建议时（区别于"预测比分"或"跑今日全部预测"），需要从 daily_jczq 管线输出中提取该场比赛的完整 5 玩法数据，结合 500.com 实时赔率做价值分析。**不要只跑 predict_match.py** — 那样得不到 500.com 盘口赔率和竞彩 5 玩法完整分布。

### 工作流（参考 2026-07-01 英格兰 vs 刚果金实操）

1. **跑 daily_jczq.py 获取最新数据**
   ```bash
   python3 /root/daily_jczq.py 2>&1 | tee /tmp/last_run.txt
   ```
   这会触发异步抓取 500.com 4 个玩法页面 + 365scores 增强，并写入 predictions_log.csv。

2. **从 predictions_log.csv 提取该场比赛**
   ```bash
   awk -F',' '$5~"英格兰" && $6~"刚果" || $5~"Congo" && $6~"England"' data/predictions_log.csv
   ```
   注意 CSV 中队名是中文（如"刚果(金)"），需中文 grep。若同一场比赛有多个日期的行（如 match_date 分别为 06-29/cancelled、06-30、07-01），取最新日期且无 cancelled 标记的行。

3. **关键 CSV 字段解析**
   | 列 (约位置) | 字段 | 含义 |
   |---|---|---|
   | 8 | rq | 让球数（负值=主队让球） |
   | 9-11 | pred_h/d/a | SPF 概率百分比（66.0=66%） |
   | 12-14 | pred_rq_win/draw/loss | 让球概率 |
   | 15 | pred_top_score | 最可能比分 |
   | 16 | pred_top_goals | 最可能总进球 |
   | 17 | pred_top_htft | 最可能半全场 |
   | 23-25 | odds_h/d/a | 500.com SPF 赔率 |
   | 26-28 | ev_h/d/a | 期望值（负值=无价值） |
   | 40-42 | s365_home/away_winrate | 365scores 基本面胜率 |
   | 43-44 | s365_home/away_fifa | FIFA 排名 |
   | score_full | JSON | 比分完整分布（Top15） |
   | htft_full | JSON | 半全场完整分布（9项） |
   | goals_full | JSON | 总进球完整分布（13档） |
   | simple_pred | str | 简单模型推荐 |
   | bet_action | str | RECOMMEND/WATCH |

   **注意**: pred_* 字段存的是百分比值（66.0=66%），输出时不要再 ×100。

4. **交叉验证 500.com 实时赔率**
   ```bash
   cd /root && python3 wc_2026_upgrade/async_500_scraper.py 2>/dev/null | python3 -c "
   import sys,json
   d=json.load(sys.stdin)
   for m in d['result']:
       if '英格兰' in m['home'] and '刚果' in m['away']:
           print(json.dumps(m, indent=2, ensure_ascii=False))
   "
   ```
   确认 CSV 中的 odds_h/d/a 与实时赔率一致。如果有差异（如赔率变化 >5%），提醒用户盘口已变动。

5. **检查比赛状态**
   - 查看 predictions_log.csv 的 `actual_score` 列：非空则比赛已结束
   - 查看 `result_status` 列："missing"=未赛，"filled"=已完赛
   - 如有 "CANCELLED" 记录，说明该日期场次被取消/延期，需确认最新排期
   - 可通过 football-data.org 或 TheStatsAPI 交叉验证当日赛程

6. **组合输出格式**
   按 [daily-jczq-prediction skill 的输出显示协议](#output-display-protocol-2026-06-10) 展示全 5 玩法，外加 EV/Value 分析和推荐优先级排序。格式：
   ```
   【胜平负】主X% / 平X% / 客X%  → 推荐: XX
   【竞彩让球(让X)】让胜X% / 让平X% / 让负X%  → 推荐: XX
   【比分】1:0(X%) 0:0(X%) ...  → 推荐: XX
   【总进球】1(X%) 2(X%) ...  → 推荐: XX
   【半全场】胜胜(X%) 平胜(X%) ...  → 推荐: XX
   💰 价值投注: XX(EV=+X%)
   市场分歧: ...
   ```

7. **EV/Value 分析排列优先级**
   - 正 EV 选项按 EV 从高到低排列
   - 标注 Kelly 建议仓位（½-Kelly 2.5% / ¼-Kelly 1.25%）
   - 区分"稳妥"（模型概率高但赔率低）vs "价值"（正 EV 但概率中等）
   - 同场多个正 EV 选项标注高度相关性

### 多源数据验证暗坑

- **predictions_log.csv 同一 match_key 多行问题** — 相同比赛因管线多次运行产生多条记录，展示时必须按 `settled_at` 最新或 `date` 最大行优先。`_show_tomorrow.py` 的选行逻辑是 goals_full 键数最多，可能选到旧数据。**检测**: 对比 pipeline 终端输出与 CSV 提取的 SPF 概率，若偏差 >5pp 说明读到旧行。**修复**: 直接用 pipeline 终端输出（today_output.txt），不用 CSV 回溯。
- **football-data.org 返回 0 场比赛** — 该 API 对世界杯 2026 可能无数据（免费层限制），不表示当天无比赛。以 500.com 开售清单为准。
- **比赛延期/改日** — 同一比赛码在 CSV 中出现多个 match_date（如 06-29→CANCELLED, 07-01→正常），取最新非 cancelled 的行。

## 冠军/冠亚军投注审计纪律

当用户问“世界杯冠军/冠亚军买哪些队”时，不能只读 `champ_runnerup_strategy.json` 后直接给推荐。必须先执行一致性审计：

1. 对比 `/root/data/final_results.json` 与 `/root/data/champ_runnerup_strategy.json` 的同日冠军概率；若 Top 队概率明显冲突，旁路策略不得直接用于购买建议。
2. 核验 `/root/data/2026_groups.json` 是否为 FIFA 正式分组。正式分组未公布前，只能称为“假设分组/情景模拟”，不能输出确定赛程下的冠军概率。
3. 检查 MC 淘汰赛是否使用官方 R32 对阵规则；若代码中存在 `_rnd.shuffle(qualifiers)` 随机洗牌，冠军/亚军概率仅作研究草稿。
4. 确认 `DixonColes.predict_proba(home, away)` 的顺序是 `[Home, Draw, Away]`。任何冠军/冠亚军脚本不得把它注释或当作 `[Away, Draw, Home]` 使用；必要时重构为显式字段。
5. 市场赔率必须全市场去水：`p_i=(1/odds_i)/sum(1/odds_all)`。不要使用 `+0.01` 这类硬编码常数构造相对强度。
6. 对加拿大、墨西哥、美国等东道主长赔率票，必须区分“低概率正EV小票”和“真实看好夺冠”。如果分组/签表/host_bonus 未通过审计，长赔率正EV一律降级或删除。
7. **东道主加成审计**：加拿大/墨西哥的冠军概率受 host_bonus 参数极度敏感。输出 EV 表前必须：
   - 用 host_bonus=[0.0, 0.07, 0.1445] 三组对照
   - 若 host_bonus=0.0 时该国冠军概率 <1.5%，则 host_bonus>0 时的高 EV 不构成买入信号
   - 推荐时需明确标注：“正EV来自模型东道主加成，实际球队实力不支持”
8. **淘汰赛签表来源审计**：核实 MC 淘汰赛签表来源：
   - `ranked`: 按 Elo 排名 1v32 配对（`wc_2026_final.py` 当前版本，`simulate_knockout.py` 相同）
   - `random`: 随机洗牌（旧版，不适用于购买建议）
   - `official`: FIFA 官方对阵规则（已从 openfootball/worldcup 获取，见 `references/2026-official-knockout-bracket.md`。已通过 `--bracket=official` 参数直接集成进 `wc_2026_final.py`）
   - 推荐冠军票时必须输出签表模式。
9. **双概率文件冲突检测**：输出前检查 `/root/data/final_results.json`（主模型）和 `/root/data/champ_runnerup_strategy.json`（旁路旧版，已废弃）。若同队冠军概率差值 >3pp，立即告警并拒绝使用旁路文件。

参考：`references/champ-runnerup-betting-audit.md`。



每日cron任务分两阶段自动运行：

### 阶段1 — 赔率拉取（00:00 UTC = 08:00 北京时间）

- **cron**: `f22f1d2494f3` (no_agent=True, 0 token开销)
- **脚本**: `/root/fetch_worldcup_odds.py`
- **动作**: 调用 The Odds API 拉取 `soccer_fifa_world_cup_winner` 最新夺冠赔率
- **输出**: 保存到 `/root/data/theodds_api_data.json`
- **API开销**: 1次/天 = ~30次/月（500免费额度绰绰有余）

### 阶段2 — 模型重训+MC（06:00 UTC）

- **cron**: `b2148e127b3a` (Hermes agent, 自动交付到TG)
- **命令**: `wc_2026_final.py --bracket=official` 全管线（29维特征, 官方FIFA路书）
- **步骤**:
  1. 自动加载阶段1写入的 `/root/data/theodds_api_data.json`
  2. 拉取最新 `international_results.json`（如缓存过期）
  3. 重新拟合 DC（含时间衰减）
  4. XGBoost 重训练（Optuna 参数 + 严格时序验证集切分）
  5. 混合: **熵基动态**（`compute_dynamic_xgb_weight(α=0.30, β=0.50)`, 钳位 [0.10,0.90]） + 凸组合市场校准（详见Step 3）
  6. 严格时序回测 2022 WC（64场）
  7. MC 200K 冠军模拟（12×4 赛制, `--bracket=official` 官方FIFA路书）
  - 保存模型到 `/root/data/{xgb_model_29.pkl, dc_model.pkl, elo_ratings.pkl}`（旧 `xgb_model_20_3.pkl` 保留为回退）
- **输出**: Top 15 冠军概率 + 条形图 + 涨跌对比 + 回测指标 + **每轮晋级概率表** + API剩余 → 自动推送到TG

### 阶段3 — 每日预测管线（轻量, 无模型重训）

独立于阶段2的全量重训。使用已训练模型（不重新拟合DC/XGB），只做赔率拉取→预测→结果累积。适合作为独立cron或阶段1/2之间的插队任务。

- **脚本**: `python3 /root/wc_2026_upgrade/daily_wc_pipeline.py`
- **动作**:
  1. 调用 The Odds API `/odds/` 拉今日 H2H 赔率（56场世界杯）
  2. 加载 `calibrated_predictor`（双DC管线: 国家队→俱乐部→Elo+市场兜底）
  3. 对所有场次做预测，输出摘要表
  4. 调用 `/scores/?daysFrom=3` 拉取已完成比赛（供检查，但不储存——实际累积靠下面）
- **输出**: `/root/data/wc_odds_YYYY-MM-DD.json` + `/root/data/wc_pred_YYYY-MM-DD.json`
- **依赖**: `calibrated_predictor.py`（try/except 包裹，缺失时跳过预测但赔率仍保存）

**赛后结果累积**（紧接上面运行）:

- **脚本**: `python3 /root/wc_2026_upgrade/accumulate_results.py`
- **动作**: 从 `/scores/?daysFrom=3` 拉取已完成比赛，幂等累积到结果库
- **输出**: `/root/data/wc_completed_results.json`（格式: `[{home, away, home_score, away_score, result, commence_time, date, saved_at}]`）
- **幂等**: 按 `home|away|commence_time` 去重，每天跑一次不会重复添加
- **用途**: 累积的真实标签用于后续模型重训前的回填（`predictions_log.csv` 的反馈闭环）

**⚠️ accumulate_results.py 不更新 training_data_with_odds.json** — 该脚本仅维护 `wc_completed_results.json`，不自动将新赛果附加到训练数据。若希望完赛结果进入 XGB 重训循环，需手动将新记录添加到 `/root/data/training_data_with_odds.json`。操作步骤：
    1. 读取 `wc_completed_results.json` 的 `saved_at` 日期段记录
    2. 从对应日期的 `wc_pred_YYYY-MM-DD.json` 或 `wc_odds_YYYY-MM-DD.json` 提取每场比赛的市场赔率
    3. 构造包含 `market_h/d/a`, `market_implied_h/d/a`, `stage='group_stage'`, `source='theoddsapi'` 的完整记录
    4. 按 `(date, home_en, away_en)` 去重后追加到训练数据
    5. 时间戳类特征（`possession_*`, `shots_*`）不可用时填空字符串，`home_xg/away_xg` 填 0.0
- **未来方向**: 增加一个 `accumulate_training_data.py` 或扩展 `accumulate_results.py`，将新完赛结果同时推入训练数据

**典型cron组合**（每日一次）:

```bash
cd /root/wc_2026_upgrade
python3 daily_wc_pipeline.py       # 拉赔率+预测
python3 accumulate_results.py       # 拉完赛结果
python3 /root/.hermes/skills/knowledge/wc-2026-predict/scripts/check_training_gap.py  # 回填训练数据
```

**可选验证步骤**（每日cron结束后，手动或自动执行）:

在 `accumulate_results.py` 运行后，可交叉验证预测准确率：遍历 `wc_pred_*.json` 文件，与 `wc_completed_results.json` 中的完赛结果比对。方法见 `references/wc-predictions-accuracy-tracking.md`。该验证报告会自动捕捉模型准确率趋势和衰退信号（如近期准确率降至 50% 以下提示重训需要）。— 每日报告的最后添加 `最近3日准确率：XX%` 有助于监控模型是否偏离。不必须执行 —— 不阻止继续预测和累积 —— 但如果 cron 空间允许（上下文窗口够用），跑一遍验证再用报告替换"管线完成"默认输出，对用户更有用。

**已知问题**:
- `daily_wc_pipeline.py` 内 `fetch_scores()` 只打印已完成比赛数，**不储存**——累积必须靠 `accumulate_results.py`
- `calibrated_predictor.predict()` 返回 `(hy_array, pipe_name)`，`hy_array` 顺序为 `[A%, D%, H%]`（内部 [A,D,H] 约定），展示时转 `[H, D, A]`
- 输出表中 `赔H` 列来自 bookmaker 最低主胜赔率（不是模型隐含赔率）

### 分组文件可执行同步脚本

`scripts/extract_official_groups.py` — 从 WRooney108/World-Cup-Betting 的 prisma/seed.ts 提取官方分组并更新 `/root/data/2026_groups.json`。当 seed.ts 更新后运行该脚本即可同步分组数据。

### 旧版（已废弃，勿用）

- ~~旧 cron `0f2347d6228a` 已删除（使用 --no-odds，无市场校准）~~
- ~~`fetch_winner_odds.py` 已删除（URL硬编码 `***` 而非API key变量）~~

## 已淘汰队过滤审计守则

任何时候展示冠军概率或每轮晋级概率，必须先过滤已淘汰球队。检查顺序：
1. 运行 `python3 /root/update_elimination_status.py` 更新 `tournament_state.json` 的 `eliminated` 字段
2. 在输出循环前加载淘汰列表（`wc_2026_final.py` 第 1135 行的模式）
3. 跳过 `eliminated=true` 的球队，不列入任何概率表

**用户纠正信号**：如果用户说"XX已经被淘汰了为什么还会输出"，说明淘汰过滤未启用。立即修复并添加此 pitfall。

参考 `references/elimination-tracking.md`。

## 关键pitfalls

1. **队名映射**：`team_name_normalizer` 字典处理官方赛程名→内部名的双向映射。关键对: `'USA' → 'United States'`(Elo索引名), `'Türkiye' → 'Turkey'`, `'Korea Republic' → 'South Korea'`, `'Czechia' → 'Czech Republic'`, `'Bosnia & Herzegovina' → 'Bosnia'`, `'Bosnia and Herzegovina' → 'Bosnia'`。直接传 `'USA'` 给 predict_match.py 可能找不到 Elo 评分（需 `'United States'`），必须先归一化。
2. **2022 WC 回测只验证单场引擎，不验证锦标赛路径**：DC+XGB 单场引擎（胜平负预测）跨届有效，但 2022 与 2026 的球队池/分组/淘汰赛结构不同，锦标赛路径概率（小组出线+签表）无历史回测支撑。输出夺冠概率时必须标注："单场引擎已验证，路径概率基于模拟假设"。
3. **正式分组 vs 假设分组**：2026分组已在 WRooney108/World-Cup-Betting repo 的 prisma/seed.ts 中确认。/root/data/2026_groups.json 必须从该源同步。更新分组后必须重新运行 MC，否则路径概率完全无效。
4. **赔率时效性**：赔率数据每天UTC 6:00 cron刷新，当天首次预测前检查文件修改时间
3. **H2H赔率 vs 夺冠赔率**：两者不同——H2H赔率是单场1X2，夺冠赔率是冠军概率。必须用H2H赔率校准单场预测
4. **无抽水处理**：市场赔率必须先转无抽水概率（除以总隐含概率）
5. **过拟合风险**：不要单独用XGBoost预测，必须混合DC（正则化）
6. **数据稀缺**：部分球队（如Bosnia）2021年后数据少，DC参数噪声大，需在输出中标注
7. **bashrc 交互式守卫**：`[ -z "$PS1" ] && return` 使cron/非交互shell无法source获取API Key。必须硬编码或直接export
8. **Odds API响应结构**：夺冠赔率URL `soccer_fifa_world_cup_winner`，市场key是 `outrights`（不是 `winner`）。格式 `[{bookmakers:[{markets:[{outcomes:[{name,price}]}]}]}]`
9. **curl解析**：用 `curl -D <hdr> -o <body>` 分离写入，不要 `-D -`
10. **并行MC进程序列化**：ProcessPoolExecutor 中 tuple 键丢失, mc_cache 用 "h||a" 传递。Worker 函数必须是模块级顶层函数。分组建固定而非随机抽取——2026是12组×4队, 用 /root/data/2026_groups.json。
11. **模型重训 vs 加载**：已训练模型在 /root/data/ 下。加载比重训快30倍（10秒 vs 5分）。DC模型加载后内部状态依赖原数据顺序，同时加载 elo_ratings.pkl 保持一致性。`wc_2026_final.py` 默认加载市场赔率（无 --no-odds），自动保存最新模型。
12. **历史赔率不现实**：The Odds API 免费层无历史赔率，football-data.co.uk 无世界杯CSV。Elo校准赔率 Brier=0.6132 已足够，真实赔率预期增益仅 +1-2pp (不划算)。
13. **f-string URL 构造陷阱**：API key 必须在 f-string 中用 `{API_KEY}` 插值，不可手动替换为 `***` 占位再忘记改回。曾多次出现: `fetch_winner_odds.py` 和 `daily_wc_pipeline.py` 的 `fetch_odds()`/`fetch_scores()` 都用过 `f'...?apiKey=***...'` 写死字符串而非 `{API_KEY}` 变量的 bug，导致脚本永久 401 失败都不报错（因为字符串合法）—— 测试时需打印 URL 前 100 字符确认 key 真的被拼入。
14. **DC predict_proba 输出顺序**：`dc.predict_proba(home, away, neutral)` 返回 `[Home, Draw, Away]`（不是 `[Away, Draw, Home]`）。代码中 `dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])` 的意图是转换为 `[Away, Draw, Home]` 以对齐 XGBoost 输出（`xgb_p = [P(away), P(draw), P(home)]`）。第一次编写单场预测脚本时容易搞反索引，引用顺序和显示列名必须双检查。

15. **`fetch_scores()` 默认不返回已完成比赛**：The Odds API 的 `/scores/` 端点默认只返回最近几天的赛程，且 `completed` 字段仅在匹配已完赛后为 true。`daily_wc_pipeline.py` 的 `fetch_scores()` 必须加 `&daysFrom=3` 参数才能拉取近 3 天的完赛赛果。若不传此参数，`completed` 列表永远为空（0/60 等）。`daily_wc_pipeline.py` 已修复但其他脚本若直接调用 scores 端点需同样注意。。代码中 `dc_ado = np.array([dc_p[2], dc_p[1], dc_p[0]])` 的意图是转换为 `[Away, Draw, Home]` 以对齐 XGBoost 输出（`xgb_p = [P(away), P(draw), P(home)]`）。第一次编写单场预测脚本时容易搞反索引，引用顺序和显示列名必须双检查。
15. **交付渠道偏好（本用户）**：默认只发文字，不发送语音（TTS/语音消息关闭）。即使有语音能力也不要主动转语音；世界杯模型结果、cron状态、修复进展统一用纯文本结构化输出。
16. **加拿大/墨西哥东道主推荐红线**：不要直接根据正EV推荐加拿大/墨西哥的冠军票。DC model 的 host_bonus 值使东道主 λ 提升（USA~15.5%, Mexico~10.5%, Canada~7.3%），足以将 2% 的中游队推至 5%+。用户明确不接受"加拿大墨西哥夺冠"推荐，必须区分：
    - "模型概率因东道主加成被抬高" vs "该队真实夺冠概率"
    - 在输出 EV 表前必须运行 host_bonus=[0.0, 0.07, 0.1445] 灵敏度测试
    - **per-team 值**：USA=0.1445, Mexico=0.10, Canada=0.07（2026-06-02从统一值分拆，见 references/host-bonus-mechanism.md）
    - **淘汰赛衰减**：KO_HOST_DECAY=0.5，小组赛满值淘汰赛减半
    - 若用户质疑加拿大/墨西哥推荐，立即承认建议无效，不辩解
17. **淘汰赛签表模式**：`wc_2026_final.py` v1.x 使用 `_rnd.shuffle(qualifiers)`，冠军概率被大幅平滑。v2.0 已改为 Elo 排名配对。输出冠军概率前检查 `final_results.json` 的 `bracket_mode` 字段。若为 `random` 模式，禁止用于购买建议。
    - **官方 FIFA 签表已集成**：`wc_2026_final.py` 支持 `--bracket=official` 参数，无需再跑独立脚本。内部使用 openfootball/worldcup cup_finals.txt 的 R32 配对规则，含 team-constrained-first 第三名分配算法。
    - **Elo 种子 vs 官方签表差异**：官方签表下西班牙冠军概率大幅提升（17.8%→25.7%），阿根廷大幅下降（14.9%→5.2%），因为真实路径将西班牙分在上半区（避开巴西/英格兰/阿根廷），阿/巴/葡/英挤在下半区。
18. **openfootball/worldcup 数据来源**：该 repo 包含 1930-2026 所有世界杯比赛结果+大名单。2026 淘汰赛签表路径和第三名分配规则是**正式 FIFA 发布的数据**，不是推导值。使用方法：
    - 历史结果可用于外部验证/回测（需注意 openfootball 格式与 football-data.org 格式的队名映射差异）
    - 2026 小组赛程与 `/root/data/2026_groups.json` 一致，已可确认分组正确
    - 官方 R32 配对中包含第三名槽位组合（如 `3{A/B/C/D/F}`），模拟时需要先将 12 个第三名排序，再将前 8 名分配到对应槽位
    - **第三名分配算法（team-constrained-first）**：R32 有 8 个第三名槽位（M74/M77/M79/M80/M81/M82/M85/M87），每个槽位接受特定组别集合的第三名。分配策略：先处理约束最强的第三名队（可选槽位最少的队），确保所有 8 个槽位都有有效分配。约束计算：每队的 eligible slots = {匹配槽位 | 该队组别在槽位的 eligible groups 中}。实现见 `/root/simulate_knockout_official.py` 中的 `assign_third_place_teams()`。
    - 第三名排序：积分→净胜球→进球→Elo
    - `simulate_knockout_official.py` 在每次 MC 模拟中独立分配第三名，正确对应每轮模拟的出线情况
    - 官方队名映射（cup.txt 中注释）：South Korea→Korea Republic, Iran→IR Iran, Cape Verde→Cabo Verde, DR Congo→Congo DR, Ivory Coast→Côte d'Ivoire, Czech Republic→Czechia, Turkey→Türkiye
18. **冠亚军旁路文件冲突**：`/root/wc_2026_champ_runnerup.py` 和 `/root/data/champ_runnerup_strategy.json` 已废弃（功能合并至 `wc_2026_final.py` + `final_results.json`）。不要加载旧文件。检查时若旧文件冠军概率与 `final_results.json` 偏差 >3pp，警告用户并拒绝使用旧数据。
19. **市场校准去水必须**：市场赔率转为概率时必须全市场去水 `p_i = (1/o_i) / sum(1/o_all)`。禁止使用 `+0.01` 类硬编码平局常数做相对强度——这会人为压缩强弱差距，错误保留冷门胜率。
20. **分组文件更新后必须重跑MC**：分组从假设变为正式后，冠军概率可能完全反转。每次更新 /root/data/2026_groups.json 后必须重新运行 `python3 /root/wc_2026_final.py`（至少 50K MC 验证后再跑全 200K）。
21. ~~`predict_match.py --home` 抛弃 XGBoost~~（✅ 已修复 2026-06-02）：原 bug 中 `--home` 分支只跑 DC 丢弃 XGB，现修复为传 `host_bonus` 给 `predict_match()`，DC 和 XGBoost 都用提升后的 λ 重建特征，混合 = DC×0.4 + XGB×0.6。特征中的 `neutral` flag 随主场状态切换 `0/1`。per-team bonus 从 `HOST_BONUS_BY_TEAM` 取。验证：`Canada --home` 输出 `xgb_h=26.7%` 不再是 `'-'`。
22. ~~`predict_match.py` 未对接 per-team HOST_BONUS_BY_TEAM~~（✅ 已修复 2026-06-02）：原代码用 `getattr(_dc, 'host_bonus_', 0.15)` 取全局值，Canada/Mexico/USA 拿到同一加成。现已添加 `HOST_BONUS_BY_TEAM` 定义且 `--home` 分支改为 `hb = HOST_BONUS_BY_TEAM.get(home, 0.0)`。
23. ~~`predict_today_3matches.py` 缺乏东道主感知~~（✅ 已修复 2026-06-02）：原脚本定义 `HOST_TEAMS = set()` 和 `HOST_BONUS = 0.0`，现改为 `from wc_2026_final import HOST_TEAMS, HOST_BONUS_BY_TEAM`。`predict_match()` 新增 `host_bonus` 参数，当 `host_bonus>0` 时 DC+XGB 都用提升后的 λ。
24. **`predict_today_3matches.py` 用冠军赔率校准单场**（概念错误）：该脚本 lines 87-94 用 `winner_odds.get(home, 0)` 和 `winner_odds.get(away, 0)` 做单场市场校准。冠军赔率是"赢得整个锦标赛"的概率，与单场 1X2 结果概率不是同一概念。虽然强弱队冠军概率和单场大致正相关，但量级差异大。正确做法：单场预测不应调用冠军赔率校准，除非有专门的 H2H 赔率源。
25. ~~死代码未清理~~（✅ 已清理 2026-06-02）：`_make_elimination_bracket`（55行）和 `build_golden20_feat`（42行）已被删除。两项均只定义不调用。
26. **第三名分配（team-constrained-first）**：官方签表 `3{A/B/C/D/F}` 类槽位分配不是"先到先得"也不是"随机匹配"。必须用 team-constrained-first 算法：先计算每支第三名队可选的槽位数量（eligible slots），从约束最强的队开始依次分配。约束计算示例：Czech Republic (GA) 仅对 M74{A,B,C,D,F} 和 M82{A,E,H,I,J} 2 个槽位 eligible 所以优先分配。算法见 `/root/simulate_knockout_official.py` 的 `assign_third_place_teams()`。错误分配（如 Scotland 同时出现在 M77 和 M79）会导致每队多场参赛的荒唐结果。
27. **半全场9向顺序不要搞反**：半全场标签顺序是 `[HH, HD, HA, DH, DD, DA, AH, AD, AA]`，其中第一个字母是半场结果（H=主胜, D=平, A=客胜），第二个字母是全场结果。MC 统计时，`idx = hr * 3 + fr` 其中 `hr=0(A)/1(D)/2(H)`，`fr` 同理。输出时显示为"胜/胜"(HH)、"平/胜"(DH)、"平/平"(DD)等中文标签。
28. **72场细项计算约3分钟**：`/root/compute_group_details.py` 为每场跑 50K 泊松 MC，含半场λ×0.45采样 + 9向半全场统计 + 总进球分布。总共约耗时3分钟。结果已预计算保存，无需每次重跑。
29. **Elo 种子 vs 官方签表冠军概率差异巨大**：同样是 50K MC，但仅因签表模式不同，冠军概率分布完全不同。西班牙从 17.8%→25.7%（+7.9pp），阿根廷从 14.9%→5.2%（-9.7pp）。西班牙被分在上半区避开所有南美强队，阿根廷/巴西/英格兰/葡萄牙挤在下半区。**输出冠军概率时必须注明签表模式**，两种结果不能混用。
32. **特征维度不匹配 (23→29 迁移陷阱)**：XGBoost 模型已从 23 维升级到 29 维（新增 6 维滚动形式特征：home_form_gf/ga, away_form_gf/ga, home_form_pts, away_form_pts）。**所有构建特征向量的脚本必须同步升级**，否则加载 29 维模型时 predict_proba 会报 `ValueError: Feature shape mismatch, expected: 29, got 23`。涉及文件：`compute_group_details.py`, `predict_group_stage.py`, `run_group72_repro.py`, `wc_2026_champ_runnerup.py`, `wc_2026_final_mc200k.py`。新增的 6 维特征在后向兼容场景（如 MC 预计算缓存、无真实形式数据的锦标赛模拟）中用占位值 `[0.0, 0.0, 0.0, 0.0, 1.5, 1.5]` 填充。单场预测脚本 `predict_match.py` 和 `wc_2026_final.py` 的 `build_golden20_feat_full` 使用真实形式数据。回退到旧模型 `xgb_model_20_3.pkl` 无需修改特征向量（仍为 23 维）。
33. **半场 λ ≈ 0.45 × 全场 λ**：当用户要求半场/半全场预测时，使用 `lam_ht = lam_ft * 0.45` 作为半场预期进球（基于国际比赛经验值）。Poisson 采样时，用 `lam_ht_home/away` 分别采样半场比分，与全场比分独立统计。半全场 9 向枚举 `[HH,HD,HA,DH,DD,DA,AH,AD,AA]`，索引公式 `idx = ht_result * 3 + ft_result`（ht/ft_result: A=0, D=1, H=2）。

34. **用户输出偏好：不要压缩任何预测数据**。本用户明确要求所有预测输出展示全量数据——每场至少显示 6 个比分概率、5 个总进球分布、4 个半全场选项。不能只显示 Top1 比分。全量数据已预计算并存于 `/root/data/group_stage_details.json`，每次预测直接引用即可。
35. **EasySoccerData Linux 浏览器路径**：包默认 `browser_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe"`，在Linux下必须显式传入 Playwright 装的 chrome 路径 `/root/.cache/ms-playwright/chromium-<ver>/chrome-linux64/chrome`。`playwright install chromium` 后路径会变化，用 `ls ~/.cache/ms-playwright/` 确认当前版本。详见 `references/sofascore-lineup-integration.md`。

63. **缺失导入模块 — wc_2026_final.py 依赖的三个外部模块不存在**：`mc_uncertainty_helper`、`mc_market_weight_helper`、`half_full_model` 被 wc_2026_final.py 和 daily_jczq.py 导入但**磁盘上不存在**。`wc_2026_final.py` 会在导入时直接崩溃（无 try/catch 包裹）。`daily_jczq.py` 虽被 try/catch 包裹，`half_full_model` 缺失导致半全场/总进球功能静默失效。**任何模型训练或冠军模拟前必须验证**：
    ```python
    for mod in ['mc_uncertainty_helper', 'mc_market_weight_helper', 'half_full_model']:
        try:
            __import__(mod)
            print(f"✅ {mod}")
        except ImportError:
            print(f"❌ {mod} — MISSING")
    ```
    若缺失，wc_2026_final.py 不可直接运行。解决方法：在 `/root/` 下创建三个桩模块（stub modules）或内联所需函数到主脚本。

64. **MC matchup cache 占位符 form 特征（train-serve skew）** — **这是系统性损害冠军模拟精度最严重的问题，2026-06-08 已修复**。旧代码（第 788-802 行）在 MC cache 中用 `fh5=[0.5, 0.0, 0.0, 0.0]`、`form_feat=[0.0,0.0,0.0,0.0,1.5,1.5]` 等占位符构建 XGB 的 29 维特征，XGB 训练时见过真实 form 分布（gf 0~3，ga 0~3，win_rate 0~1），推理时收到零向量，落入异常叶节点。这不是"精度损失"，而是**训练/推理分布错位（train-serve skew）**，XGB 输出方向不可预测。
    
    **修复方法（2026-06-08）**：在 MC cache 构建循环前，用全量历史数据创建 FeatureBuffer，预计算 48 支球队的 `recent_form(team, tournament_date='2026-06-11', n=5/12)` 和 `h2h(team, opponent, tournament_date, 3)`，存入 `team_form_5`/`team_form_12`/`team_h2h_3` dict。然后每对 matchup 用真实 form 值构建完整的 29 维特征向量（含 gold 特征的 h2h 净胜球和 12 场 form diff，及 form_feat 的 6 维 gf/ga/pts）。修复后 40/48 队有真实 form 数据。代码位置：`wc_2026_final.py` 约第 770 行后的 pre-compute 段。

65. **predictions_log.csv 无反馈闭环**：所有 `actual_hda`、`actual_score`、`actual_htft`、`actual_rq_result` 列全部为空。系统无法自动评估 Brier/RPS/准确率趋势。cron 赛后回填机制未实现。参见 `references/backtest-data-fallback-chain.md`（位于 jczq-analysis 技能下）中推荐的多源顺序回填框架。
36. **predict_match.py lineup_features 集成（2026-06-04新增）**：可选参数 `lineup_features=None` 由 `SofascoreFeatureExtractor.extract()` 返回的 dict 注入。触发**阵容感知动态折扣**: 任一方缺主力≥2人 (+10%) 或双方首发市值差异>30% (+10%)，把友谊赛折扣从 0.3 拉向 0.6 上限。回归测试: `lineup_features=None` 输出与原版完全一致。阵容数据**仅赛前 ~1小时发布**，赛前定时任务才有意义。E2E验证 (6/3 16场国家队比赛) 总准确率 7/16 = 43.8%，对强队差异明显的比赛有效 (Croatia-Belgium, Luxembourg-Italy)，对轮换导致的小比赛噪声 (Netherlands 0-1 Algeria 阵容差 4 倍) 帮助有限。后续可加 XGB 训练集做端到端学习。
41. **市场权重网格搜索结论（2026-06-05）**：对 w=0.00 / 0.25 / 0.40 / 0.60 / 1.00 做 5×200K MC 网格搜索，对比冠军概率 vs 市场赔率 MAE。结论：**w=0.40 维持现状最优**（MAE=0.939pp vs 42队）。更高权重虽然让概率更贴近市场，但会抹杀模型独立信号（如 Argentina 模型 9.22% vs 市场 7.54%）。当前 MARKET_WEIGHT=0.40 已在最佳平衡点。

42. **Sofascore 队名陷阱**: Sofascore 用 'Ivory Coast' 需映射到 DC 模型 'Côte d'Ivoire' (特殊字符)。Sofascore 搜索结果混合 Team/Player/Tournament，必须 `type(r).__name__=='Team'` 过滤再严格匹配名字。Player 对象的 `substitute` 是 bool 不是 int, 不可 `> 0` 比较。

43. **`dc_pred` argmax 编码反转**：2022 WC 回测中 `dc_pred = np.argmax(dp)` 其中 `dp = dc.predict_proba()` 返回 `[p_home, p_draw, p_away]`，`argmax` 返回 {0=主胜, 1=平, 2=客胜}。但 `actual` 编码是 {2=主胜, 1=平, 0=客胜}。导致所有主胜/客胜被算错——DC accuracy 从应有 ~57% 降至荒谬的 ~17%。修复：`dc_pred = 2 - np.argmax(dp)`。检查所有涉及 `actual` vs `argmax` 比较的代码。

44. **Hybrid Brier 计算的主客反转**：hybrid 概率数组混合 `dc_ado = [dp[2], dp[1], dp[0]]`（转成了 [客,平,主] 以对齐 XGBoost），但 Brier 计算时直接 `np.array([hybrid[0], hybrid[1], hybrid[2]])` 与 `actual_onehot=[主,平,客]` 比较——主客反了。修复：`hybrid_hda = np.array([hybrid[2], hybrid[1], hybrid[0]])` 先对齐再算 Brier。此 Bug 使 Hybrid Brier 被人为从 0.1819 放大到 0.2933，造成"混合模型使校准退化"的假象。

45. **验证集时序泄漏（FeatureBuffer 污染）**：旧代码用全量数据（含val）填充 FeatureBuffer 再按日期切分 train/val，导致 val 比赛的 form 特征中包含其他 val 比赛的结果。修复：**先按日期切分，再用独立 buffer 建特征**——train buffer 仅用 train 比赛，val 特征从 train buffer 构建（不把 val 比赛加入 buffer）。影响评估：Brier 0.1532→0.1540（极小，但逻辑正确性必须修复）。

46. **市场校准的凸组合**：MC matchups 的市场校准必须用 `blended = (1-mw) * model_vec + mw * market_vec`（model_vec=[A,D,H], market_vec=[rel_a,0,rel_h]），否则模型权重的非凸组合会让 draw 概率被人为压高。检查 `wc_2026_final.py` 第 627-642 行的市场校准逻辑，确保是凸组合。

48. **联赛特征 (club→national team heuristic) 已验证为无效**（2026-06-05）：从 football-data.org（22K 俱乐部比赛）提取4维联赛强度特征，通过国家→俱乐部联赛启发式映射注入 33 维 XGBoost，A/B 测试结果 Brier 0.1744→0.1747（恶化），Acc 59.59%→59.00%（恶化）。4个联赛特征中 3 个排名后 5 位，`league_diversity_diff` 是最无用特征。根本原因：Elo 评分已覆盖球队实力；俱乐部→国家队映射无真实球员大名单支撑，噪声大于信号。**教训**：football-data.org 数据是俱乐部级的，无球员国籍信息时，不可用于增强国家队模型。如有球员大名单数据（每名球员→俱乐部→联赛），可重新评估。

49. **`DixonColes._weights` 硬编码 cutoff='2026-05-19' 陷阱**（2026-06-06 已修复）：原代码 `_weights(self, dates, cutoff='2026-05-19')` 把 cutoff 写死为 2026-05-19。后果：当训练模型用于 1986-2014 早期届次时，所有历史比赛相对 cutoff 都 > 540 天（半衰期），`0.5^(days/540) ≈ 0`，DC 拟合退化为零权重，参数无意义（泊松 NLL=0，attack_/defense_ 完全失真）。**修复**: `fit()` 接受 `cutoff` 参数并透传。**通用规则**: 任何时间衰减函数（DC/Elo/EMA）的 cutoff 必须是**数据截止日**（如 test_year-01-01 或 datetime.now()），不可硬编码。验证方法: 打印 `w.sum()`，对早期数据应在数百到数千范围，不是 0。**应用到 10 届回测**: `dc.fit(df, cutoff=f"{test_year}-01-01")` 让每届训练用对应时间锚点。

50. **FeatureBuffer 性能优化 — FastBuffer 预索引模式**（2026-06-06 验证）：原 FeatureBuffer 每次构建特征 O(N) 扫描历史（10K 比赛 × 14 特征 = 14万次线性扫描），单届 1986 训练就跑了 5+ 分钟未出结果。**FastBuffer 模式**: 构造时一次性建预索引 `team → sorted matches list` + `h2h[t1<t2] → sorted matches list`，构建特征时 O(min(n, team_games)) 反向扫描。**性能**: 10K matches × 14 features 从 ~30min → ~5s。**通用模式**: 任何在循环中反复"对某队查最近 N 场"的代码，都应预索引 team→matches dict 而不是每轮扫描全表。

51. **10 届回测 (leave-one-edition-out) 必报项**（2026-06-06 验证）：单届 2022 回测会掩盖"特殊届"问题。10 届宏平均是稳健性最低门槛。必报:
- **每届 Acc/Brier/LogLoss** (DC, XGB, HYB) — 不只是 2022
- **每届让球-1 Acc** (如 handicap=-1) — 让球结果通常比 1X2 更可预测
- **每届实际 H/D/A 基线** — 基线波动大, 不能用单一 33% 基线对比
- **宏平均** (跨 10 届算术平均) — 单一总体指标
- **最强届/最弱届对比** — 暴露模型在哪些届失准 (如 1990 防御战 + 弱队爆冷)
- **让球-1 DC 是稳定可投策略** (10 届宏平均 59.4%, 各届 50-65% 波动) — 比 HDA 1X2 (52%) 更可预测

52. **sklearn 1.5+ 移除了 `LogisticRegression(multi_class=...)` 参数**（2026-06-06 验证）：10 届回测 v2 在 sklearn 1.5+ 直接报 `TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'`。LR 现在默认 multinomial (softmax), **不要传 `multi_class` 参数**。修复: `LogisticRegression(max_iter=500, C=1.0)` (无 multi_class)。任何在新环境跑 v2 失败时, 先 `python3 -c "import sklearn; print(sklearn.__version__)"` 确认版本, 1.5+ 必踩此坑。

53. **Feature build 是回测性能瓶颈, 不是 XGBoost**（2026-06-06 实测）：v2 跑 2022 WC 训练 (35K 场, 29 维) 的耗时分布:
- DC fit: 3.5s
- **Feature build (29 维): 290s** ← **95% 时间**
- XGBoost 训练: 2-3s
- Stacking LR fit: <1s

根因: `build_train_features_29()` 对 35K 场逐场调 `dc.predict_proba()` (单场 49 循环) + `recent_form()` (反向扫描) + `h2h_full()`。FastBuffer 模式 (#50) 已加速 form/h2h dict lookup, 但 DC predict_proba 仍是单场调用 — 这是真正的瓶颈。**优化方向**: 把 35K 场的 DC predict_proba 用 numpy 一次性向量化 (用 `dc.attack_/defense_` 数组 + team 索引) 估计可省 70% 时间, 10 届总耗时 37min → 11min。

54. **Stacking 在 10 届回测中边际收益仅 +0.6pp**（2026-06-06 验证）：v2 跑出 LR meta-learner 9 维 stacking (XGB 3 + DC 3 + Elo 3) vs v1 等权 HYB: Acc 52.1% → 52.7% (+0.6pp), Brier 几乎不变 (0.2001 → 0.2006, 略差)。**根因**:
- 3 模型已高度相关 (XGB 训练时已用 DC prob 作 5/29 维特征)
- 每届 meta 训练样本仅 5-7K, LR 无明显可学
- 未调 LR 超参 (C=1.0 默认), grid search 风险 > 收益

**实践**: 10 届回测直接用等权 0.6 XGB + 0.4 DC 即可, 不必上 stacking。Stacking 值得做的场景: 5+ 模型分歧 >10pp + 样本 ≥30K + 类别不平衡。

55. **XGB 训练量与测试性能非单调正相关 (v3 消融失败, 2026-06-06)**：把 XGB 训练集从 "WC-only 604 场 + 友谊赛" 扩到 "FIFA 国际 A 级 ~10万场"，A/B 测试 10 届回测 Acc +0.3pp 但 Brier 持平，统计不显著。**根因**: XGB 是 29 维特征驱动而非数据驱动 — Elo + DC 概率已浓缩 95% 信号，训练量对最终概率影响有限。**教训**: 不要为"数据量更大"而扩大训练集，先检查信号是否已被特征吸收。

56. **XGB Bagging 在 stacking meta 阶段是反效果 (v4 消融失败, 2026-06-06)**：5 XGB (不同 seed) bagging 让 XGB 单独 Acc +0.5pp (稳定性↑)，但把 bagged 输出喂给 LR meta-learner 反而拖累 stacking 表现。**根因**: bagging 让 5 XGB 概率接近均值，LR meta 可学空间被压缩。**实践**: bagging 单独用有益，与 stacking 组合无收益。

57. **`DixonColes` 类无 `lambdas` 属性 — 使用 `predict_lambda()`**：`DixonColes` 存储攻击/防守参数为 `self.attack_` 和 `self.defense_` (numpy 数组)，无 `self.lambdas` dict。获取预期进球 λ 的唯一方法是 `dc.predict_lambda(home, away, neutral=True)` → `(λ_home, λ_away)`。写回测或特征工程脚本时，不要试图用 `dc.lambdas.get(team, 1.3)` — 这会触发 `AttributeError`。如果被 `except Exception` 静默吃掉，会导致全量 `valid=False` 和 nan% 结果。

58. **多函数回测脚本中辅助函数必须在模块级定义**：`elo_odds(eh, ea)` 若定义在 `build_features_v6()` 内部（嵌套函数），则 `backtest_year()` 的测试预测循环（另一个函数）无法访问它，会抛出 `NameError`。如果被 `except Exception` 静默吃掉，所有测试预测全部失败。规则：被多个函数调用的辅助函数一律放在模块级（class 外、function 外），不要偷懒写成嵌套函数。

59. **残差 Stacking 在严格时序跨届场景下完全失败 (v5 消融失败, 2026-06-06)**：训练 LR 预测 `y_residual = y_true - hybrid_pred` 然后 `final = hybrid + alpha * residual_pred`。验证集 Acc ≈ 0 提升，Brier 反而恶化。**根因**:
- 测试时 `y_residual = 0` (无 ground truth)，模型默认输出 0 等于"不加 LR"
- 残差 stacking 假设 base model 偏置是稳定 (scaled+shifted by constant)，但跨届 (1986→2022) 严格时序场景下 bias 会随战术/规则漂移
- **通用教训**: 残差 stacking 只在 train/test 分布严格一致时有效。涉及"对 LR 输出零值估计"的所有 stacking 变体，测试时都退化为 base model。

**v3/v4/v5 综合 ROI**: 全部放弃。"更复杂的 stacking" (boosting meta, neural meta) 在 604 场样本上限下不会突破，复杂度→过拟合。

60. **多函数回测脚本中所有常量/辅助函数必须模块级 — phase_map scope 陷阱**：`phase_map = {'group':0,...}` 若定义在 `build_features_v6()` 内部，则 `backtest_year()` 的测试循环引用时触发 `NameError`。如果被 `except Exception` 静默吃掉，所有测试预测全失败。**不只是函数 — 常量 dict 也必须模块级**。见 `references/v6-phase-dynamic-k-backtest.md`。

61. **异常处理垫片维度必须与特征维度一致**：回测脚本中 `except` 分支的 `[0]*N` padding 必须与特征向量维度完全一致。v6 脚本用 `[0]*40` 但实际维度 42，导致 XGBoost predict 崩溃。**编程习惯**：在脚本顶部定义 `FEAT_DIM = 42` 常量，padding 用 `[0]*FEAT_DIM`，避免硬编码数字失步。

62. **`compute_dynamic_xgb_weight()` 在两份文件有副本，修改必须同步**：函数在 `wc_2026_final.py`（用 np 实现）和 `daily_jczq.py`（纯 math 实现）各有一份。逻辑完全一致但实现不同。修改 α/β 参数、钳位范围或熵公式时，必须同时更新两处。验证方法: `python3 -c "from wc_2026_final import compute_dynamic_xgb_weight as f1; from daily_jczq import compute_dynamic_xgb_weight as f2; assert f1([0.85,0.10,0.05])==f2([0.85,0.10,0.05])"`。若不一致，每天竞彩预测的权重与冠军管线的权重产生分歧，回测指标混乱。

66. **`market_probs_3way_from_outright()` 输出顺序 bug（2026-06-08 修复）**：该函数原返回 `[H, D, A]`，但 MC 管线全程使用 `[A, D, H]` 约定（XGBoost 输出顺序）。MC cache 构建时 `blended = (1-mw) * model_vec + mw * market_vec` 中 `model_vec=[A,D,H]` 与 `market_vec=[H,D,A]` 顺序不一致，导致 H/A 概率交叉污染。影响范围：市场权重 mw 通常较小 (0.1-0.3)，bug 被 Isotonic 校准部分吸收。修复：函数改为返回 `[p_a, p_d, p_h]`，删除 call site 的 `model_vec` 包装。**验证**：任何改动 `market_probs_3way_from_outright` 输出格式的代码必须确认 blend 处两端顺序一致。

67. **淘汰赛 outright→90min 概率反解**：WC 淘汰赛中夺冠赔率反映的是"最终晋级"而非"90分钟获胜"。关系：`P_qualify(H) = P(H_90) + ρ·P(D_90)`，其中 ρ ≈ 0.5（约 50% 的加时/点球被热门赢下）。`market_probs_3way_from_outright(rel_h, rel_a, knockout=True, model_draw=draw_prob)` 内嵌此反解逻辑。另在 MC worker 的 `_sim_match(mc, elo, h, a, ko=True)` 中设有**去尖锐化因子** (de-sharpen factor=0.65)：`ph = ph * 0.65 + 0.35 * (1-pd_)/2`，压缩淘汰赛中的 H/A 分布以修正 outright→90min 偏差。验证：Spread 从 0.300 压缩至 0.195（热门 vs 冷门 50/20 场景）。

67. **全局数组顺序约定 [A,D,H] 内部, [H,D,A] 展示 (2026-06-14 确立)** — 所有内部数组（DC 概率、XGBoost predict_proba、hybrid 融合、Market probs）统一使用 `[Away=0, Draw=1, Home=2]` 顺序。XGBoost label encoding 就是 0=A, 1=D, 2=H, 所以对齐。**仅在输出/展示时**转为 `[Home, Draw, Away]`。违反此约定会导致两类 bug:
    - **Bug A: 显示反转** — hy=[A,D,H] 但 results['h']=hy[0], 所有主客颠倒
    - **Bug B: 特征构成反转** — DC probs 填入 feature array 的顺序与训练时不一致, 模型学到错误映射
    验证: `print(f"hy=[{hy[0]:.3f},{hy[1]:.3f},{hy[2]:.3f}] → H={hy[2]:.1f}% D={hy[1]:.1f}% A={hy[0]:.1f}%")` 看中间值是否符合直觉（强队 H 应该大）。

68. **双 DC 模型架构 (2026-06-14 上线)** — 系统有三种 DC 模型:
    - `dc_model.pkl` (国家队 226队, 英文名, 主数据源)
    - `dc_club.pkl` (俱乐部 2,174队, 中文名, 从500_history_backfill.csv训练)
    - `dc_club_en.pkl` (俱乐部 152队, 英文名, 从football-data.org训练)
    推理路径: dc_model 按英文队名校准 → 命中则 Pipeline A (XGB_nat+DC_nat) → 未命中则 Pipeline B (dc_club+Elo) → 再失败则 Pipeline C (Elo+Market)。
    注意: 中文队名场景 (500.com 数据中的俱乐部比赛) 必须走 dc_club, 走 dc_model 会返回 None。

69. **国家队 vs 俱乐部训练的隔离 (2026-06-14)** — 国家队模型 `xgb_model_nat.pkl` 只使用英文队名的训练数据 (395/491条), 过滤掉 96 条中文队名记录。俱乐部数据 (500_history_backfill.csv 63K场) 用于训练 dc_club, 但不混入 XGB 训练。**不隔离的后果**: v28 混合训练中俱乐部数据使 XGB 准确率从 90.5% (国家队) 降到 64.3% (混合), 因为俱乐部数据覆盖 2,174队但每个队样本少 (~8场/队)。

70. **DC 置信度加权 (2026-06-14)**: 当球队在国际比赛数据中出现次数较少时, DC 的 Poisson 参数估计方差大。`calibrated_predictor.py` 按出场数加权: ≥200 场→1.0, ≥100→0.9, ≥50→0.8, ≥20→0.7, ≥10→0.5, ≥5→0.3, <5→0.1。融合: `final = dc_conf * hy + (1-dc_conf) * base`, base = 0.3*Elo + 0.7*Market。效果: 新手球队 (如 Curaçao 仅 4 场) 从 DC 空转 96% H → 降权到市场赔率靠拢。

71. **Winsorize 裁剪 DC 极端值 (2026-06-14)**: 对 DC 概率和 λ 值进行 `[0.01, 0.99]` 裁剪, 避免两队实力悬殊时 Poisson 参数产生 0.9999+ 的绝对预测。影响: Germany vs Curaçao DC_raw=(0.96,0.03,0.01) → 裁剪后不变, 但对 DC λ 裁到 [0.01, 99.0]。**在概率上必须裁** 否则 Isotonic 校准器接收 0.0 或 1.0 输入时崩掉或产生 NaN (calibrators_nat.pkl 曾因此被拉偏)。

72. **11维干净模型 vs 29维全量模型 (2026-06-14 消融验证)**: `xgb_model_nat.pkl` 只用11维特征，去掉14个死特征。消融结果: 11维 vs 29维 在验证集无统计差异 (都是 ~64%)。**教训**: 检查所有特征的实际重要性, 不要保留 "看起来合理" 但全是零值的死特征。

73. **DC 俱乐部模型 γ=0 的情况 (2026-06-14)** — `train_club_dc.py` 从500_history_backfill.csv训练dc_club(2,174队, 63K场)时, Dixon-Coles 的 γ (主场优势系数) 等于 0。原因: 500.com 数据可能已经在中立场景下采集, 或赛果字段没有区分主客场。γ=0 意味着淘汰赛/中立场的预测可能没问题, 但真正的联赛主场比赛会被低估主队优势。验证: `dc_club.rho_` 应约 0.15-0.25, 若不为此范围需检查训练数据的主客场标记是否正确。`dc_club_en.pkl` 在 football-data.org 数据上训练时因队名格式不兼容 (英文 vs 中文) 训练受阻, 目前不可用。

72. **Isotonic 校准器在极端分布时产生 NaN 概率 (2026-06-14)** — 当 Isotonic 校准器接收 0.0 或 1.0 的输入时, 可能输出 NaN。这是因为 Isotonic 回归在训练集边界外的外推行为未定义。校准器文件 `calibrators_nat.pkl` 在国家队 XGB 上训练, 如果 DC 概率被 Winsorize 裁剪后仍然接近 0/1, 校准器偶尔产出 NaN。修复: 在 Isotonic.predict 调用后检查 NaN 并回退 (`np.nan_to_num(cal_probs, nan=raw_probs)`)。从未处理这个问题的旧配置文件会产生校准后概率全为 0 的无效预测。

75. **CRITICAL: spf_result 训练数据类型污染 (2026-06-14 发现, 已修复)** — `training_data_with_odds.json` 中 `spf_result` 字段含 131/491 条 int 类型（非 str）。训练脚本 `result == '3'` 不匹配 int，导致 29 条标签错误（7.3%）。修复: 所有训练脚本统一 `str(m['spf_result'])`。影响: nat 模型验证准确率从 64.4% → 75.4%。**检查方法**: `python3 -c "import json; d=json.load(open('/root/data/training_data_with_odds.json')); print('int:', sum(1 for m in d if isinstance(m.get('spf_result'),int)), 'str:', sum(1 for m in d if isinstance(m.get('spf_result'),str)))"`

76. **CRITICAL: _blend_with_market 平局硬编码0 (2026-06-14 发现, 已修复)** — `calibrated_predictor.py` 的 `_blend_with_market()` 函数中 `elo_arr = [elo_h, 0, 1-elo_h]` 将平局概率硬编码为 0，系统性压低平局预测。修复: 用 Elo 差估算真实平局概率 `elo_draw = 0.25 * (1 - abs(2*elo_h - 1))`。效果: 荷兰vs日本平局 5.6% → 13.2%。**检查方法**: `grep -n '0, 1-elo_h\|0, 1-elo_a' /root/wc_2026_upgrade/calibrated_predictor.py`

77. **CRITICAL: 双管线模型不一致 (2026-06-14 发现, 已修复)** — daily_jczq.py 用 xgb_model_29 (29维+剥离校准器)，calibrated_predictor.py 用 xgb_model_nat (11维+活跃校准器+无Draw Correction)。修复: 两管线统一到 nat 模型。详见 `references/dual-dc-model-2026-06-14.md`。**检查方法**: `grep -n 'xgb_model.*=.*joblib.load' /root/daily_jczq.py /root/wc_2026_upgrade/calibrated_predictor.py`

78. **CRITICAL: form数据硬检查导致return None (2026-06-14 发现, 已修复)** — `_try_hybrid_predict()` 中 `if h_key not in fs or a_key not in fs: return None` 对没有form数据的队伍（如库拉索）直接返回None。修复: 改为软检查，缺失form时打印警告但继续预测（DC+XGB直推）。nat模型11维特征不含form，不影响预测质量。**检查方法**: `grep -n 'return None' /root/daily_jczq.py | grep -i form`

79. **CRITICAL: market_h未定义导致SPF=0.0% (2026-06-14 发现, 已修复)** — `_try_hybrid_predict()` 中 `market_implied = market_h if market_h > 0 else op_h` 但 `market_h` 从未在函数作用域定义，NameError被`except Exception: return None`静默吞掉。修复: 用 `locals().get('market_h', 0)` 兜底。影响: 德国vs库拉索等世界杯比赛SPF显示0.0%。**检查方法**: `grep -n 'market_h' /root/daily_jczq.py`

80. **Isotonic校准器已从两管线剥离 (2026-06-14 确认)** — daily_jczq.py 和 calibrated_predictor.py 都不再使用Isotonic校准。原始Brier=0.2053比任何校准器都好。恢复条件: 积累200+场2026年同质数据后重训sigmoid校准器。

81. **CRITICAL: WC完赛结果不进 training_data_with_odds.json (2026-06-17 发现)** — `accumulate_results.py` 只将新完赛结果写入 `wc_completed_results.json`，**不自动追加到 `training_data_with_odds.json`**。这意味着 `wc_completed_results.json` 有18+条记录的训练数据文件仍只有491→2,436条（主要来自历史 kaijiang+thestats），2026 WC 真实赛果尚未进入训练特征。后果：XGB 重训时看不到任何 2026 WC 完赛信号。**手动补救步骤**:
    ```python
    # In daily cron after accumulate_results.py:
    import json
    results = json.load(open('/root/data/wc_completed_results.json'))
    training = json.load(open('/root/data/training_data_with_odds.json'))
    seen = set((r['date'], r['home'], r['away']) for r in training)
    # For each result not in training, construct rec from wc_pred_{date}.json odds
    for r in results:
        key = (r['date'], r['home'], r['away'])
        if key in seen: continue
        # Load pred file for odds
        preds = json.load(open(f"/root/data/wc_pred_{r['date']}.json"))
        p = next((x for x in preds if x['home']==r['home'] and x['away']==r['away']), {})
        rec = {
            'date': r['date'], 'home_en': r['home'], 'away_en': r['away'],
            'tournament': 'FIFA World Cup', 'spf_result': r['result'],
            'home_goals': r['home_score'], 'away_goals': r['away_score'],
            'home_xg': 0.0, 'away_xg': 0.0,
            'market_h': p.get('odds_h',0), 'market_d': p.get('odds_d',0), 'market_a': p.get('odds_a',0),
            'stage': 'group_stage', 'source': 'theoddsapi',
        }
        # Compute implied probs
        oh, od, oa = rec['market_h'], rec['market_d'], rec['market_a']
        if all(x>0 for x in [oh,od,oa]):
            margin = 1/oh + 1/od + 1/oa
            rec['market_implied_h'] = (1/oh)/margin
            rec['market_implied_d'] = (1/od)/margin
            rec['market_implied_a'] = (1/oa)/margin
        training.append(rec)
        seen.add(key)
    json.dump(training, open('/root/data/training_data_with_odds.json','w'), indent=2)
    ```
    **检查方法**: 统计 training_data 中 `'source': 'theoddsapi'` 的记录数 — 若为 0，说明 WC 完赛结果从未进入训练数据。
    **修复方向**: 在 `accumulate_results.py` 或新脚本中增加自动追加到训练数据的功能。

    ```python
    except Exception as e:
        print(f"⚠ {m.get('home','?')} vs {m.get('away','?')}: {e}", file=sys.stderr)
        valids.append(False); rows.append([0]*FEAT_DIM); ys.append(1)
    ```
    `NameError`、`AttributeError`、`ValueError` 三种最常见，print 出来能省数小时排查时间。

### 训练数据回填 pitfalls (2026-06-24 updated)

82. **accumulate_results.py 不更新训练数据** — 该脚本只写 wc_completed_results.json，不自动追加到 training_data_with_odds.json。必须额外运行 `scripts/check_training_gap.py` 或手动执行 pitfall 81 的代码段。否则完赛结果永远不进训练数据，XGB 重训时看不到 2026 WC 真实标签。

83. **pred 文件不覆盖所有完赛场次** — 并非所有已完成比赛都在当日的 wc_pred 文件中。原因：(a) 某些匹配项从未被 The Odds API 的 odds 端点返回（如 Australia vs Turkey, 2026-06-14）；(b) 匹配在赛程被 schedule 而非预测。回填时必须 fallback 到跨日期 odds 文件搜索。(c) 即使 odds 文件中有该比赛，pred 文件也可能没有（pred 文件是 odds 文件在当日运行 pipeline 时的快照，不包含所有历史）。跳过记录的补录步骤见 `references/wc-results-backfill-fallback.md`。

84. **check_training_gap.py 仅做 pred 文件回填（1阶）** — 该脚本的当前实现只从 wc_pred_{date}.json 中查找匹配并回填。不做跨日期 odds 搜索，也不做无赔率兜底。文档中的"三阶回填策略"是未来方向的描述，非当前实现。运行后必须验证是否还有漏网之鱼（见 `references/wc-results-backfill-fallback.md#Detection`），如有则手动补录。

85. **跨日期赔率搜索** — 某比赛可能在更早日期的 odds 文件中存在（如 06-14 的 odds 文件覆盖最多，64 场；后续日子逐日减少）。构建 `(home, away) → (odds_h, odds_d, odds_a)` 全量查询表时，应从最早到最晚遍历所有 wc_odds_*.json，选择最近的匹配（最接近比赛时间）。参见 `references/wc-results-backfill-fallback.md` 的完整搜索命令。

85. **赔率文件是扁平结构** — wc_odds_YYYY-MM-DD.json 的格式是 `{"home", "away", "odds_h", "odds_d", "odds_a", "market_h"}` (无 bookmaker 嵌套)。不要按 The Odds API 原始响应的嵌套结构解析——管线已预解析为扁平格式。详情见 `references/odds-data-coverage-patterns.md`。

86. **队名在 scores 端点和 odds 端点一致** — 两者均使用显示名称（`Turkey` 而非 `Türkiye`, `South Korea` 而非 `Korea Republic`）。可直接匹配。但注意内部 DC 模型 (`dc_model.pkl`) 可能使用不同命名（需 team_name_normalizer 映射）。

87. **无赔率的完赛记录仍要进训练数据** — 某些比赛（如 Australia vs Turkey）从未被 The Odds API 覆盖。应将其 market_h/d/a 设为 0，market_implied 为 0.0，仍作为带真实标签的训练样本。它们缺少市场校准特征但含结果信息，对 XGB 的非赔率特征仍有正向贡献。
