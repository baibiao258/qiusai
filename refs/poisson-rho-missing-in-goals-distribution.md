# DC ρ 低分修正实施记录

**发现日期**: 2026-06-18
**修复日期**: 2026-06-18
**状态**: ✅ 已修复

## 问题

`daily_jczq.py` 中所有基于 Poisson λ 的分布函数使用纯独立 Poisson 卷积，没有引入 DC 模型的 ρ 低分修正。

**根因**: DC 模型的 `predict_lambda()` 只返回 `(lam_h, lam_a)`，不暴露 `self.rho_`。固定 λ 后下游独立计算分布时 ρ 丢失。

## 修复内容

### 新增 `dc_tau()` 辅助函数 (daily_jczq.py 614-629行)

使用与 `wc_2026_phase1.py` 中 `predict_proba()` 完全一致的 τ 公式：

```
0-0: τ = 1 + ρ × exp(-λh-λa)
1-0: τ = 1 + ρ × λh × exp(-λh-λa)
0-1: τ = 1 + ρ × λa × exp(-λh-λa)
1-1: τ = 1 + ρ × λh × λa × exp(-λh-λa)
```

⚠️ **关键**: 不能用标准 DC 论文公式（`1 - ρ × λh × λa`）。现有模型拟合 NLL（`retrain_dc_model.py` line 253-255）用的是 `exp(-λh-λa)` 基底，ρ 的符号和取值范围都与论文版不同。直接用论文版公式会给已拟合好的模型引入不一致。

### 修改 4 个 compute_* 函数

| 函数 | 参数变更 | 所属 |
|------|---------|------|
| `compute_goals_distribution()` | 新增 `rho=0.0` | 总进球 13 档 |
| `compute_score_topn()`（×2副本） | 新增 `rho=0.0` | 比分排名 |
| `compute_rq_probs()` | 新增 `rho=0.0` | 让球概率 |

`compute_htft_topn()` 未修——内部委托给 `predict_half_full_probs()`（外部模块），ρ 修正需单独改。

### 4 个 λ 来源注入 rho

| 来源 | rho 值 | 路由 |
|------|--------|------|
| `_try_hybrid_predict()` | `_dc_model.rho_` (国际DC) | 国际赛主力 |
| `_try_club_predict()` | `_dc_club.rho_` (俱乐部DC) | 俱乐部赛 |
| `predict_match_legacy()` | 0.0 (无DC) | 纯Poisson回退 |
| `market_fallback` | 0.0 (无DC) | 无训练数据 |

### `build_prediction_bundle()` 传导

```python
dc_rho = p.get('rho', 0.0)
# 传入所有 compute_* 调用
goals_dist = compute_goals_distribution(lambda_home, lambda_away, rho=dc_rho)
rq_probs = compute_rq_probs(lambda_home, lambda_away, handicap, rho=dc_rho)
score_top8 = compute_score_topn(lambda_home, lambda_away, 8, rho=dc_rho)
```

## 验证结果

使用 λ_home=1.5, λ_away=1.2 测试，ρ=0.13（大赛典型值）：

| ρ | 0球 | 1球 | 2球 | 0-2合计 |
|---|------|-----|-----|--------|
| 0.0 | 6.73% | 18.17% | 24.53% | 49.42% |
| +0.13 | 6.76% | 18.30% | 24.60% | 49.66% |
| -0.13 | 6.70% | 18.04% | 24.47% | 49.21% |

正 ρ → 低比分增加，负 ρ → 低比分减少，与现有 `predict_proba()` 行为一致。总概率归一化后严格 1.0。

## 待办

- `compute_htft_topn()` 内部委托的 `predict_half_full_probs()`（`half_full_model.py` 和 `wc_2026_upgrade/half_full_model.py`）也在独立 Poisson 卷积，需要单独导入 `dc_tau` 加 ρ

## 关联

- 修复 PR: 直接 patched daily_jczq.py (6 个 patch 块 + 1 个函数新增)
- 后续验证: 样本达到 50+ 后 `evaluate_brier.py --ab` 对比 Goals Acc 提升
