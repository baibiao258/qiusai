# TheStatsAPI 数据源集成

## 概述

TheStatsAPI (`api.thestatsapi.com`) 是标准 RESTful 足球数据API，提供比赛赛果、xG、控球率、射门统计和 Betfair/Pinnacle/Bet365 赔率。本用户有 **500万次/月** 的高配额 Key。

## 关键发现

### competition_id 参数陷阱

**参数名是单数 `competition_id` 不是复数 `competition_ids`**。使用复数参数时服务器静默忽略过滤器，返回所有比赛的混合数据。使用单数参数后正常过滤。

### 国家队赛事 ID (2024-2026)

从 149 个赛事中筛选出 17 个国家队赛事:

| 赛事 | ID |
|------|----|
| FIFA World Cup | `comp_6107` |
| WCQ UEFA | `comp_2954` |
| WCQ CONMEBOL | `comp_4682` |
| WCQ CONCACAF | `comp_0836` |
| WCQ CAF | `comp_5720` |
| WCQ AFC | `comp_8973` |
| WCQ OFC | `comp_7363` |
| EURO | `comp_2949` |
| EURO Qual. | `comp_3759` |
| Copa América | `comp_5749` |
| UEFA Nations League | `comp_574977` |
| CONCACAF Nations League | `comp_193547` |
| CONCACAF Gold Cup | `comp_1376` |
| Africa Cup of Nations | `comp_1554` |
| AFCON Qual. | `comp_83579` |
| International Friendly Games | `comp_29967` |
| FIFA Series | `comp_920080` |

### 回填数据量 (2026-06-15)

- **2,289 场**已完成赛事的全量数据（2024-01-01 ~ 2026-06-15）
- 2024: 1,146 场, 2025: 857 场, 2026: 286 场
- **填补了 2025 年空白**（之前训练集 0 场）
- xG/stats 覆盖率: ~12% (280 场)
- Odds 覆盖率: ~8% (192 场)
- 去重后净新增: 2,037 场（vs 现有 491 场训练集）

## 端点结构

### 基础匹配: GET /football/matches

```
?date_from=&date_to=&status=finished&competition_id=comp_XXXX&page=N&per_page=100
```

响应: `{data: [{id, competition_id, utc_date, home_team: {id, name}, away_team: {id, name}, score: {home, away}, group_label, matchday, odds_available, xg_available}], meta: {page, total, total_pages}}`

### 统计: GET /football/matches/{id}/stats

响应路径: `data -> overview -> {expected_goals, ball_possession, shots_on_target, total_shots, ...}`。每项有 `{all: {home, away}, first_half: {...}, second_half: {...}}`。

额外: `shots`, `attack`, `passes`, `duels`, `defending`, `goalkeeping`, `np_expected_goals`

### 赔率: GET /football/matches/{id}/odds

响应: `data -> bookmakers[] -> {bookmaker, markets: {match_odds: {home/draw/away: {opening, last_seen}}, btts, total_goals, ...}}`

博彩公司: Betfair Exchange, Pinnacle, Bet365。代码优先用 `last_seen`，回退 `opening`。

## 回填管线架构

### Phase 1: 并发分页抓取基础Matches

对每个赛事ID，先调第1页获知 `total_pages`，再并发批量拉取剩余页面:

```python
# First discover page counts per competition
with ThreadPoolExecutor(max_workers=17) as ex:
    fut_map = {ex.submit(fetch_page, cid, 1): cid for cid in COMP_IDS}

# Then batch-fetch remaining pages with 20 concurrent workers
for i in range(0, len(remaining_pages), batch_size):
    batch = remaining_pages[i:i+batch_size]
    with ThreadPoolExecutor(max_workers=20) as ex:
        for cid, p in batch:
            ex.submit(fetch_page, cid, p)
    time.sleep(0.05)  # gentle rate limiting
```

**关键**: 用 `competition_id=CID`（单数参数），不是 `competition_ids=CID1,CID2`。复数参数被服务器静默忽略。

### Phase 2: 并行 Stats + Odds 详情

并发 20 线程批量处理，断点续传（检查 CSV 已有的 match_id）:

```python
with ThreadPoolExecutor(max_workers=20) as ex:
    fm = {ex.submit(fetch_stats_odds, m["id"]): m for m in batch}
    for f in as_completed(fm):
        m = fm[f]
        rec = f.result()
        # Merge base match data into rec
        rec["home_team"] = m.get("home_team", {}).get("name", "")
        rec["away_team"] = m.get("away_team", {}).get("name", "")
        rec["home_score"] = m.get("score", {}).get("home", "")
        rec["away_score"] = m.get("score", {}).get("away", "")
```

### SPF 格式归一化 (关键坑)

TheStatsAPI 返回 `H/D/A` 格式，但训练脚本期待 `3/1/0`（中国竞彩格式）。未归一化时所有 `H/D/A` 记录被错误映射为 `label=0`(客胜)，导致标签污染。

```python
SPF_MAP = {'H':'3', 'D':'1', 'A':'0'}
for m in data:
    m['spf_result'] = SPF_MAP.get(str(m['spf_result']), str(m['spf_result']))
```

验证: 合并后检查 spf_result 值域必须只有 3/1/0:
```python
Counter(str(m['spf_result']) for m in data)
```

### market_implied_prob 字段归一化

旧格式用单个 `market_implied_prob`(scalar pip from 500.com)，新格式用 `market_implied_h/d/a` 三件套(Betfair/Pinnacle)。训练脚本读 `m.get('market_implied_prob', 0.0)`，须归一化:

```python
for m in data:
    if not m.get('market_implied_prob'):
        h = m.get('market_implied_h')
        d = m.get('market_implied_d')
        a = m.get('market_implied_a')
        m['market_implied_prob'] = d or h or 0.0
```

### 断点续传

- Phase 2 启动时读 CSV 已有 match_id → `processed_ids` set
- `to_do = [m for m in all_matches if m.id not in processed_ids]`
- 幂等: CSV 用 append mode, 不覆盖已有行

## 整合到训练管线

### 合并流程

1. 读 CSV 转为 training_data_with_odds.json 格式字段
2. 按 `(date[:10], home_en, away_en)` 三元组去重
3. 旧数据优先保留（有更多特征），新 xG/odds 覆盖旧空值
4. 按日期排序输出

### 重训结果 (2026-06-15)

在 2,528 场（1,530 训练 / 656 验证）上重训 nat 11维 XGBoost:

| 指标 | 旧模型 (491场) | 新模型 (2,528场) | 改善 |
|------|--------------|-----------------|------|
| 验证准确率 | 52.6% | **61.9%** | **+9.3pp** |
| LogLoss | 1.2618 | **0.8158** | -35% |
| Brier | 0.3905 | **0.2919** | -25% |

Top-5 特征: lam_diff(16%) > dc_a(13.6%) > op_a(12%) > dc_h(11.1%) > market_implied(10%)

### 输出文件

- `/root/wc_2026_upgrade/base_matches_thestats.json` — 原始匹配数据 (2,289条)
- `/root/wc_2026_upgrade/training_data_thestats.csv` — 回填CSV (21列)
- `/root/wc_2026_upgrade/phase1_fetch.py` — Phase 1 并发抓取
- `/root/wc_2026_upgrade/phase2_backfill.py` — Phase 2 并行详情
- `/root/wc_2026_upgrade/retrain_nat.py` — 合并归一化 + XGBoost 重训 (统一入口)
- `/root/wc_2026_upgrade/merge_training_data.py` — 数据合并去重

## 已知局限

- xG 覆盖不均: WCQ CONCACAF 最高(48%), 世界杯/EURO/Copa America 为0
- Odds 覆盖 ~8%, 主Betfair Exchange, 无Bet365/Pinnacle
- 队名含特殊字符 (Côte d'Ivoire, Türkiye), 需映射到 Elo 键名
- 并发 Phase 2 有 ~95% 错误率 (stats/odds 端点对多数国家队比赛返回 non-200)，不影响基础数据
