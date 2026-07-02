# 模型改进机会 — FifaWorldCupPreview 对比分析

来源: https://github.com/baibiao258/FifaWorldCupPreview
分析日期: 2026-06-03

## 项目概况

完整的机器学习世界杯预测管线，使用 scikit-learn HistGradientBoosting：
- 训练数据: 49,353 场国际比赛（1872-2025，同 Kaggle martj42 数据集）
- 特征: 15 维（Elo差值 + 滚动形式 + 射手特征）
- 模型: HistGradientBoostingClassifier + Isotonic 校准
- 验证: 时间分裂（pre-2018 / post-2018）
- 部署: FastAPI
- 准确率: 60.0%（超基线 12.7pp）
- 优势: 工程干净、特征无泄漏、校准好

## 对我们模型的 4 个改进点

### 1. 滚动形式特征（最高回报，最易加）

FifaWorldCupPreview 有的：
```python
home_form_points  # 近5场场均积分
away_form_gf      # 近5场场均进球
away_form_ga      # 近5场场均失球
form_points_diff  # 两队积分差
```

我们有的 23 维特征全是 Elo/λ/赔率，**没有球队近期状态**。Elo 是长期实力（100 场窗口），形式特征是短期动量（5 场窗口），两者互补。

一个球队连续 5 场不胜 vs 5 连胜，Elo 可能只差 20 点但状态天差地别。

**实现方式**：从 `international_results.json` 按时间序逐场计算滚动平均，shift(1) 保证无泄漏。数据现成可用，无需新数据源。

### 2. 射手多样性特征

FifaWorldCupPreview 从 `goalscorers.csv` 提取：
```python
scorer_diversity  # 场均射手数（10场窗口）
pen_share         # 点球占总进球比例（10场窗口）
```
一支队 10 场 8 球全是同一人进的 (diversity=1.0) vs 6 人进的 (diversity=1.6)，攻击可持续性完全不同。

**数据需求**：Kaggle martj42 数据集的 `goalscorers.csv` 文件。该文件每行一个进球（含 scorer, penalty, own_goal 字段），可用于每支球队的 10 场滚动窗口统计。

### 3. 概率校准

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
core = train[train["date"] < calibrate_cutoff]
calib = train[train["date"] >= calibrate_cutoff]
model.fit(core[FEATURES], core["target"])
calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
calibrated.fit(calib[FEATURES], calib["target"])
```
XGBoost 的原始概率输出倾向于极端化（太接近 0 或 1）。Isotonic 校准可以修正系统偏差，尤其是平局概率往往偏低的问题。

**注意**：校准需要独立的 validation set（时间分裂），不能使用测试集。校准后 LogLoss 通常改善 0.02-0.05。

### 4. FastAPI 部署

FifaWorldCupPreview 用 FastAPI + team_state.json（缓存的球队最新状态）实现即查即用 API：
```python
GET /teams → 336 支可查询球队
POST /predict {home_team, away_team, neutral} → {probabilities, most_likely}
```
每次预测无需重跑训练管线，直接查 `team_state.json` + 加载序列化模型即可。

## 我们 vs FifaWorldCupPreview 对比表

| 维度 | 我们 | FifaWorldCupPreview | 差距 |
|------|:---:|:------------------:|:----:|
| 训练量 | 4.9K A级赛(2021-2026) | 49K 全量(1872-2025) | 各有优势 |
| 模型 | DC+Poisson+XGB | HistGradientBoosting | 我们更专业 |
| 特征维度 | 23 维足球专用 | 15 维通用 | 我们丰富 |
| 滚动形式 | ❌ 无 | ✅ 5场窗口 | **弱项** |
| 射手特征 | ❌ 无 | ✅ 10场窗口 | **弱项** |
| 市场赔率 | ✅ Odds API | ❌ 无 | 我们的优势 |
| 蒙特卡洛 | ✅ 50K-200K | ❌ 无 | 我们的优势 |
| 概率校准 | ❌ 未显式 | ✅ Isotonic | 可借鉴 |
| 东道主加成 | ✅ 精细分档 | ❌ 统一处理 | 我们的优势 |
| API | ❌ 无 | ✅ FastAPI | 可借鉴 |

## 建议优先级

1. **Phase 1**: 加滚动形式特征（现有数据就能算，0 新依赖）
2. **Phase 2**: 加射手多样性 + 点球占比（需 goalscorers.csv）
3. **Phase 3**: XGBoost 概率校准（独立 validation set）
4. **Phase 4**: FastAPI 封装（取决于实际使用场景）
