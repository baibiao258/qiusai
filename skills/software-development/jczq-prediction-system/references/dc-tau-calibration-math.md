# Dixon-Coles ρ 校准数学原理 (2026-06-30 落地于 daily_jczq.py)

## 问题

纯 Poisson 假设两队进球独立，低比分场次（0-0/0-1/1-0/1-1）实际可观测相关性被低估。
当 λ_home ≈ λ_away ≈ 1.0 时，双 Poisson 卷积的峰值天然落在 2 球——"锚定 2 球"偏见。

## 修正公式 (Dixon & Coles, 1997)

联合概率：`P(X=x, Y=y) = τ(x,y) · Pois(λ_h, x) · Pois(λ_a, y)`

τ 仅在低比分修正：

| (x, y) | τ(x, y, λ_h, λ_a, ρ) | ρ<0 时 |
|--------|----------------------|--------|
| (0, 0) | 1 - ρ·λ_h·λ_a | **放大** 0-0 概率 |
| (0, 1) | 1 + ρ·λ_h | 缩小 0-1 |
| (1, 0) | 1 + ρ·λ_a | 缩小 1-0 |
| (1, 1) | 1 - ρ | **放大** 1-1 |
| other | 1.0 | 不变 |

## 代码位置

- `dc_tau()`: daily_jczq.py line ~1016
- 被调用处：`compute_rq_probs`, `compute_goals_distribution`, `compute_score_topn`
- ρ 来源：`fit_dc_model()` 在 retrain_dc_model.py 中 Stage 2 网格搜索，范围 [-0.30, 0.0]
- 传递路径：模型参数 `p.get('rho', 0.0)` → `build_prediction_bundle` → `rho=dc_rho`

## 验证

当 ρ=0 时，τ=1，退化为纯 Poisson。负 ρ 越大，低比分修正效果越强。
典型 ρ 范围 (Dixon & Coles 原论文): -0.13 ~ -0.03

## 已知陷阱

- daily_jczq.py .bak 遗留了两个 `compute_score_topn` 定义（旧版 line ~997 无 rho，新版 line ~1112 带 rho）。Python 使用最后一个，功能不受影响，但 patch 时务必确认修改的是新版。
- patch 后必须运行 `ast.parse()` 语法检查 + `grep "def function_name"` 查重。此技能的两个定义是旧患，非本修复引入。
