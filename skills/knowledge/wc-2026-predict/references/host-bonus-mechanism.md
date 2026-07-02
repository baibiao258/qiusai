# host_bonus 机制：东道主主场优势修正

## 背景

2026 世界杯有三个东道主（加拿大、墨西哥、美国），共 **9 场实际主场**比赛（其余 95 场均为中性场）。

DC 模型在全局数据上拟合时，`gamma` 系数被大量中性场稀释至 ~0.009（几乎为零），导致东道主 9 场主胜概率系统性低估。

## v2.0 解决方案：per-team host_bonus + 淘汰赛衰减

### Stage 4 估计

在 `wc_2026_final.py` 的 `DixonColes.fit()` 后新增 **Stage 4**：

1. `fit()` 完成后冻结 attack/defense/gamma 参数
2. 筛选训练数据中 host teams（Canada/Mexico/USA）的非中性主场比赛
3. 仅优化一个参数 `host_bonus`：只加给主队 λ，客队不受影响

```python
# predict_lambda 中的计算公式
lh = exp(attack_[home] + defense_[away] + gamma + host_bonus)  # 主队有 bonus
la = exp(attack_[away] + defense_[home] + gamma)               # 客队无 bonus
```

DC 全局估计值：**host_bonus = 0.1445**（基于 103 场东道主主场比赛）

### 分拆为 per-team 值（2026-06-02 更新）

加拿大/Mexico 样本太少，不应与 USA 共享同一值。

```python
# wc_2026_final.py
HOST_BONUS_BY_TEAM = {
    'United States': 0.1445,  # 68场大样本: 70%得分率，保留全局估计值
    'Mexico': 0.10,           # 17场小样本: 79%得分率(0负)，保守下调
    'Canada': 0.07,           # 18场极小样本: 75%得分率，大幅下调
}
# 淘汰赛衰减系数 (小组赛满值 * 该系数)
KO_HOST_DECAY = 0.5  # 晋级淘汰赛后减半
```

### 淘汰赛衰减机制

淘汰赛（R32→Final）跨城市流动，东道主不一定在本土作战，不应享受满值加成。

因子计算公式（淘汰赛时）：
```
full_factor = exp(host_bonus)       # e^0.1445 ≈ 1.1555 for USA
ko_factor   = 1.0 + (full_factor - 1.0) * KO_HOST_DECAY
```

`KO_HOST_DECAY=0.5` 时因数插值到半程：

| 球队 | host_bonus | 满值因子 | 淘汰赛因子 |
|------|-----------|---------|-----------|
| United States | 0.1445 | 1.1555 | 1.0777 |
| Mexico | 0.10 | 1.1052 | 1.0526 |
| Canada | 0.07 | 1.0725 | 1.0363 |

### 代码实现

`_sim_worker` 接受 `host_bonus_by_team` 字典而非单 float：

```python
def _sim_worker(mc_cache_dict, elo, seed, n_sims, teams, groups,
                host_teams=None, host_bonus_by_team=None, ko_decay=0.5):
```

内部 `_sim_match` 新增 `ko` 参数：

```python
def _sim_match(mc, elo, h, a, ko=False):
```

调用方式：
- **小组赛**：`_sim_match(mc, elo, t1, t2)` — `ko=False`（默认），使用满值因子
- **淘汰赛**：`_sim_match(mc, elo, t1, t2, ko=True)` — 使用插值衰减因子

主机因子计算：
```python
hf = _host_factor_cache.get(h, 1.0)
if ko:
    hf = 1.0 + (hf - 1.0) * ko_decay
lam_h *= hf
cdf_h = _build_cdf(lam_h)
```

### 灵敏度测试

`wc_2026_final.py` 内置三组 uniform 灵敏度测试（所有东道主使用同一值做基准对比）：

```python
for bonus_val in HOST_BONUS_SENSITIVITY:  # [0.0, 0.07, 0.1445]
    uniform_dict = {t: bonus_val for t in HOST_TEAMS}
    ct, rt, tt = run_mc_for_bonus(..., uniform_dict)
```

**主 MC（200K）则使用 per-team dict：**
```python
executor.submit(_sim_worker, ..., HOST_TEAMS, HOST_BONUS_BY_TEAM, KO_HOST_DECAY)
```

### 对冠军概率的影响（50K 验证，2026-06-02）

| 球队 | 旧版(统一0.1445) | 新版(per-team+KO decay) | 变化 |
|------|-----------------|------------------------|------|
| Spain | 11.40% | 11.24% | -0.16pp (噪声) |
| Mexico | 5.79% | 5.70% | -0.09pp |
| Canada | ~3.6% | 4.14% | +0.5pp |
| United States | ~3.0% | 3.34% | +0.3pp |

加拿大和美国主场上涨是因为旧版用 USA 0.1445 复制给所有队；新版各自独立后，加/美在小组赛的满值加成实际更高，叠加淘汰赛衰减后仍为正收益。

## 使用方式

### predict_match.py --home 专用模式

```bash
python3 predict_match.py "Mexico" "South Africa" --home
```

东道主场次执行混合模型（2026-06-02 修复，不再纯 DC）：

- **DC 模型**：传递 per-team host_bonus（从 `HOST_BONUS_BY_TEAM` 取），neutral=False，正确提升主队 λ
- **XGBoost**：用增强后的 λ 重建 23 维特征 → 重新 predict_proba → 正常参与混合（DC×0.4 + XGB×0.6）
- **输出**：包含 `host_bonus_applied: true` 和 `host_bonus_val` 字段
- 验证：`Canada --home` 输出 `fin_h=23.0%`，其中 DC=17.4% XGB=26.7% 混合=23.0%

### 9 场东道主比赛清单

| Group | 主队 | 对手 | 场馆（国家） |
|:-----:|:---|:---|:---|
| A | Mexico | South Africa (11 Jun) | Estadio Azteca, Mexico City 🇲🇽 |
| A | Mexico | Korea Republic (18 Jun) | Estadio Akron, Guadalajara 🇲🇽 |
| A | Mexico | Czechia (24 Jun) | Estadio Azteca, Mexico City 🇲🇽 |
| B | Canada | Bosnia & Herzegovina (12 Jun) | BMO Field, Toronto 🇨🇦 |
| B | Canada | Qatar (18 Jun) | BC Place, Vancouver 🇨🇦 |
| B | Canada | Switzerland (24 Jun) | BC Place, Vancouver 🇨🇦 |
| D | United States | Paraguay (12 Jun) | SoFi Stadium, Los Angeles 🇺🇸 |
| D | United States | Australia (19 Jun) | Lumen Field, Seattle 🇺🇸 |
| D | United States | Türkiye (25 Jun) | SoFi Stadium, Los Angeles 🇺🇸 |

## 修复历程（四阶段收敛）

| 阶段 | 方法 | Mexico vs SA 主胜 | 问题 |
|:---:|:---|:---:|:---|
| 原始 | gamma=0.009（双方同加） | 48.8% | 严重低估 |
| hack 1 | gamma 强制 0.22（双方同加） | 49.1% | 客队 λ 被错误抬升 |
| hack 2 | +12pp 后处理偏移 | 61.1% | 数字正确但 pipeline 不干净 |
| **v1.0** | **host_bonus 只加主队（Stage 4 自动估计）** | **58.3%** | **干净、数据驱动、零 hack** |
| **v2.0** | **per-team HOST_BONUS_BY_TEAM + KO_HOST_DECAY** | **58.3%** | **三队独立、淘汰赛衰减** |

## 代码位置

- `wc_2026_final.py` 顶部常量：`HOST_BONUS_BY_TEAM`, `KO_HOST_DECAY`, `HOST_TEAMS`
- `wc_2026_final.py`：`_sim_worker` / `_sim_match` — MC 模拟中应用 per-team + KO 衰减
- `wc_2026_final.py`：`build_golden20_feat_full` 中检查东道主并传递 `host_bonus`（此路径目前仍使用 DC 全局 `dc.host_bonus_`，仅用于 XGB 训练特征，不影响 MC 模拟）
- `predict_match.py`：--home 分支中调用 `dc.predict_proba(..., host_bonus=host_bonus)`

## 推荐纪律

在输出冠军票建议前，必须做 host_bonus sensitivity 分析：
- `wc_2026_final.py` 内置灵敏度输出（bonus=0, 0.07, 0.1445，三组 uniform 基准对比）
- 若 bonus=0 时东道主冠军概率 <2%，即使 bonus>0 出正 EV 也不推荐
- 加拿大/墨西哥类长赔率票必须标注："概率来自东道主加成，非真实实力"
- 输出 EV 表时必须标注当前使用的 HOST_BONUS_BY_TEAM 值
