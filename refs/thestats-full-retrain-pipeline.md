# TheStatsAPI 全量重训管线 (2026-06-15 建立)

## 概述

三步 Pipeline 将 DC+XGBoost 训练数据从 ~2,500 场扩展到 **32,001 场** (712 队伍, 22 赛事, 5 年跨度):

```
pull_training_data.py  →  thestats_training_data.json (32K 条)
       ↓
retrain_dc_model.py    →  dc_model.pkl (712 队 Dixon-Coles)
       ↓
retrain_poisson_elo.py →  poisson_elo_prior.json (712 Elo + 609 λ)
```

## 拉取数据 (pull_training_data.py)

### 正确赛事 ID (2026-06-15 从 /competitions 端点确认)

旧写死的 `comp_3040/3041/...` 全部返回 HTTP 400。
正确 ID 必须从 `/football/competitions?per_page=100` 端点获取:

| 赛事 | 正确 ID | 场次 |
|------|---------|------|
| Premier League | comp_3039 | 2,125 |
| LaLiga | comp_8814 | 2,125 |
| Bundesliga | comp_4643 | 1,730 |
| Ligue 1 | comp_0256 | 1,909 |
| Liga Portugal | comp_8385 | 1,749 |
| Eredivisie | comp_3809 | 1,736 |
| Championship | comp_8321 | 3,082 |
| MLS | comp_9799 | 2,762 |
| J1 League | comp_6240 | 1,956 |
| K League 1 | comp_1646 | 1,256 |
| Brasileirão | comp_4795 | 2,189 |
| FIFA World Cup | comp_6107 | 76 |
| International Friendly | comp_29967 | 1,673 |
| EURO | comp_2949 | 102 |

### 特征格式

```json
{
  "match_id": "mt_200652802",
  "date": "2026-05-29",
  "comp_name": "Ligue 1",
  "home": "Nice", "away": "Saint-Étienne",
  "h_score": 4, "a_score": 1,
  "neutral": false,
  "elo_h": 1515.4, "elo_a": 1413.3,
  "have_elo": true,
  "lambda_h": 1.1191, "lambda_a": 2.0536,
  "have_lambda": true
}
```

### 全覆盖指标

- Elo 覆盖率: **100%** (712 队)
- Poisson λ 覆盖率: **99.5%** (609 队)

### 断点续传

每完成一个赛事写入 checkpoint + 阶段性保存 JSON。

## DC 模型重训 (retrain_dc_model.py)

### joblib 序列化坑

直接 joblib dump 自定义类会报 `Can't get attribute 'DixonColes' on <module '__main__'>`。
**修复**: 将 DixonColes 类放入独立模块 `dc_model_definition.py`，保存时设置:

```python
dc.__class__.__module__ = 'dc_model_definition'
joblib.dump(dc, OUTPUT_MODEL)
```

### 训练结果

| 参数 | 值 |
|------|------|
| 球队数 | 712 |
| 全球场均进球 | 1.350 |
| 主场优势 γ | 0.0104 |
| ρ (Dixon-Coles) | ~0 |
| Host Bonus | +0.0422 |
| 衰减半衰期 | 540 天 |

### 攻击 Top 5

Brazil(1.55), France(1.53), Germany(1.43), Spain(1.40), Belgium(1.39)

## 先验集成

`predict_match_legacy()` 优先查 `_lookup_prior_elo/lambda()`，命中标记 `prior_poisson` (含 n_matches 统计)，未命中降级 `legacy_poisson`。加载日志输出 "✅ 加载全量 Elo+Poisson 先验: {N} 队 Elo, {M} 队 λ"。
