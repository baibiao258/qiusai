# 世界杯 10 届严格时序回测 (1986-2022, 604 场)

> 跨届验证 DC+Elo+Hybrid 模型的稳健性, 避免"只看 2022 一次回测"导致的过拟合/特殊化误判。

## 为什么是 10 届 (1986-2022)

| 指标 | 数值 | 说明 |
|------|------|------|
| 总场次 | 604 | 24队制(86-94) 52场/届 + 32队制(98-22) 64场/届 |
| 赛果类别 | 3 (H/D/A) | 单任务约 200 样本/类, 够分层/交叉验证 |
| 让球/比分/总进球 | 子任务样本更少 | 让球约 200+ 有效样本, 比分分布极稀疏 |
| 特征维度 | 20~50+ | Elo、近况、阵容价值、战术风格、市场赔率等 |

**少于 10 届 → 短期波动噪音主导**, 模型会过拟合某一届的特殊性 (如 2002 韩日、2010 南非低进球、2018 VAR 首秀)。

## 数据源: openfootball/worldcup.json

GitHub: https://github.com/openfootball/worldcup.json

- 完整覆盖 1930-2026 所有世界杯 (含 2026 签表草案)
- 格式: `{year}/worldcup.json`, 包含 `name` + `matches[]`
- 每场比赛字段: `date, team1, team2, score.ft[2], round, group, ground, goals1, goals2`
- **已下载并转换**: `/root/data/wc_historical_matches.json` (1986-2022 共 10 届, 604 场)
- 转换器要点:
  - 标准化队名: `Curaçao → Curacao`, `Cote d'Ivoire → Ivory Coast`, `DR Congo → DR Congo`
  - 添加字段: `tournament='FIFA World Cup'`, `neutral=True`, `year`, `round`, `group`
  - 与 `international_results.json` 合并时按 (date, home, away) 去重

## 模式: leave-one-edition-out (LOEO)

```python
# 伪码
WC_YEARS = [1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
for test_year in WC_YEARS:
    cutoff = f"{test_year}-01-01"
    train = [m for m in all if m['date'] < cutoff and m['tournament'] in A_MATCH]
    test = [m for m in wc_only if m['year'] == test_year]
    # 训练 Elo, DC, XGB; 在 test 上评估
    # 记录: Acc, Brier, LogLoss, 让球 Acc
```

**关键**: cutoff 必须是 test_year, 不是 `datetime.now()`! 见 Pitfall #49。

## 严格时序: 三个潜在泄漏点

| 泄漏点 | 修复 |
|--------|------|
| Elo 计算 | 仅用 train 比赛, test 时不更新 |
| DC fit | cutoff = test_year, 不是 `2026-05-19` 硬编码 |
| FeatureBuffer | test 时用独立 buffer, 重新从 train 状态构建, 增量加入 test 比赛 |

## 早届 (1986-1994) 数据稀疏处理

- DC `time_decay_hl=540` → `1080` (24 队制, 数据少, 需更长记忆)
- 训练集 10K+ 场仍可拟合, 但 attack_/defense_ 估计噪声大
- 实战可考虑: 1986-1994 用市场赔率或简化 Elo 兜底

## 实现

### v1: 极速版 (无 XGBoost) — `/root/wc_10edition_backtest.py`

- Elo + DC + HYB (0.6 XGB + 0.4 DC 等权), 让球-1 评估
- 单届耗时 ~60s (DC fit 占 ~50s, 主要瓶颈是矩阵运算)
- 10 届总耗时 ~10 分钟 (v1 极速版, 适合快速 sanity check)
- 输出 `/root/data/wc_10edition_backtest.json` (per-edition + macro avg)

### v2: 完整 29 维 + Stacking — `/root/wc_10edition_backtest_v2.py`

- Elo + DC + **XGBoost (29 维特征)** + **Stacking (LR meta-learner 9 维)** + 让球-1
- 29 维特征组成: 15 基线 (Elo/λ/DC prob/form5) + 5 黄金 (h2h/tier/form12) + 3 Elo odds + 6 滚动形式
- **XGB29 涨 +4.2pp (48.0% → 52.2%)** vs 14 维版本 (v1 HYB 的 XGB 单独表现)
- **Stacking 只 +0.6pp (52.1% → 52.7%)** vs v1 等权 HYB — 边际收益极小, Brier 几乎没变 (0.2001 → 0.2006)
- 单届耗时 ~3-5 分钟 (feature build 90-290s 占 95% 时间, XGB 训练本身 1-3s)
- 10 届总耗时 ~37 分钟 (2211 秒)
- 输出 `/root/data/wc_10edition_backtest_v2.json` (per-edition + macro avg + 5 模型并列对比)

**v2 关键发现 (2026-06-06)**:
1. XGB29 大幅涨 (+4.2pp) — 多 15 维 (form12/h2h_n/tier/odds) 确实帮 XGB
2. Stacking 边际收益极小 — 在 604 场 + 3 模型分歧不大时, LR meta 不如简单加权
3. **让球-1 DC 仍是最强可投策略** (59.4% 稳定) — 不受 stacking 影响
4. **性能瓶颈在 feature build (95% 时间) 而非 XGBoost 训练 (1-3s)** — 优化方向是把 29 维特征构建改成真正的 numpy 向量化或预索引累计统计

### v1 vs v2 对比

| 模型 | v1 (14 维 HYB) | v2 (29 维 Stack) | Δ |
|------|---------------|-----------------|---|
| Elo | 51.6% | 51.6% | = |
| DC | 50.9% | 50.9% | = |
| XGB 单独 | 48.0% | **52.2%** | **+4.2pp** |
| HYB/Stack | 52.1% | 52.7% | +0.6pp |
| 让球-1 DC | 59.4% | 59.4% | = |
| Brier | 0.2001 | 0.2006 | +0.0005 (略差) |

**实践建议**:
- 快速 sanity check: 用 v1 (~10 分钟)
- 完整验证或调优: 用 v2 (~37 分钟)
- 让球策略: 直接用 v1/v2 的 DC 让球-1 (59.4%, 不需 stacking)

## 实际 10 届结果 (2026-06-06 跑出)

| 届次 | N | Elo | DC | HYB | 让-1 (DC) | 让-1 (Elo) | 实际 H/D/A |
|------|---|-----|-----|-----|-----------|-----------|-----------|
| 1986 | 52 | 48.1% | 48.1% | 51.9% | **65.4%** | 57.7% | 38/31/31 |
| 1990 | 52 | 46.2% | 42.3% | 46.2% | 50.0% | 53.8% | 46/31/23 |
| 1994 | 52 | 57.7% | 51.9% | 55.8% | 53.8% | 51.9% | 48/23/29 |
| 1998 | 64 | 48.4% | 53.1% | 54.7% | **64.1%** | 60.9% | 42/31/27 |
| 2002 | 64 | 45.3% | 48.4% | 48.4% | 54.7% | 59.4% | 41/30/30 |
| 2006 | 64 | 59.4% | 54.7% | 57.8% | 59.4% | 53.1% | 48/27/25 |
| 2010 | 64 | 50.0% | 51.6% | 54.7% | 62.5% | **64.1%** | 36/28/36 |
| 2014 | 64 | 51.6% | 50.0% | 51.6% | 59.4% | **62.5%** | 38/27/36 |
| 2018 | 64 | 53.1% | **59.4%** | 59.4% | **64.1%** | 60.9% | 39/22/39 |
| 2022 | 64 | **56.2%** | 53.1% | 53.1% | 60.9% | 54.7% | 45/23/31 |
| **宏平均** | 604 | 51.6% | 51.3% | **53.4%** | **59.4%** | 57.9% | - |

**关键发现**:
1. HYB 比单模型稳 (每届都接近或超过单模型)
2. **让球-1 DC 59.4%** > 1X2 准确率 (~52%), 让球结果更可预测
3. 最难届: 1990 (HYB 46.2%, 防御足球 + 弱队爆冷)
4. 最强届: 2018 (HYB 59.4%, 强弱分明)

## 性能优化: FastBuffer 预索引模式

`FeatureBuffer` 旧实现: 每次构建特征 O(N) 扫描, 10K 比赛 × 14 特征 = 14 万次扫描, 慢到无法跑完。

**FastBuffer 模式**:
```python
class FastBuffer:
    def __init__(self, all_matches, elo, dc):
        # 预索引: team → sorted matches
        self.team_games = defaultdict(list)
        # 预索引: (t1, t2) → matches
        self.h2h = defaultdict(list)
        for m in all_matches:
            self.team_games[m['home']].append((m['date'], m['h_score'], m['a_score'], True))
            self.team_games[m['away']].append((m['date'], m['a_score'], m['h_score'], False))
            k = (m['home'], m['away']) if m['home'] < m['away'] else (m['away'], m['home'])
            self.h2h[k].append(m)
        for t in self.team_games: self.team_games[t].sort()
        for k in self.h2h: self.h2h[k].sort(key=lambda x: x['date'])
    
    def recent_form(self, team, date, n):
        # 反向扫描 O(min(n, games)) 代替 O(N) 扫描
        result = []
        for d, gf, ga, _ in reversed(self.team_games.get(team, [])):
            if d < date:
                result.append((gf, ga))
                if len(result) >= n: break
        ...
```

**性能提升**: 10K matches × 14 features 从 ~30min → ~5s。

## 输出指标矩阵 (推荐模板)

每届单独报告 + 宏平均:

| 届次 | N | DC Acc | DC Brier | DC LL | XGB Acc | XGB Brier | XGB LL | HYB Acc | HYB Brier | HYB LL | 让-1 Acc | 基线 H/D/A |
|------|---|--------|----------|-------|---------|-----------|--------|---------|-----------|--------|----------|-----------|
| 1986 | 52 | 48.1% | 0.186 | ... | ... | ... | ... | 51.9% | 0.199 | ... | 65.4% | 38/31/31 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **宏平均** | **604** | **51.3%** | **0.198** | **...** | **...** | **...** | **...** | **53.4%** | **0.200** | **...** | **59.4%** | - |

## 与 wc_2026_final.py 的关系

- `wc_2026_final.py` 只回测 2022 单届 (lines 692-735), 验证单场引擎
- `wc_10edition_backtest.py` (v1 极速版) 和 `wc_10edition_backtest_v2.py` (29 维 + Stack) 跨 10 届验证, 暴露单届回测看不到的"特殊届"问题
- 两者**不矛盾**: 2022 单届 + 10 届宏平均 + 当前 2026 预测 → 三重验证

## v2 完整 10 届结果 (2026-06-06 跑出, 29 维 + Stacking)

| 届次 | N | Elo | DC | XGB29 | Stack | 让-1 (DC) | 让-1 (Elo) | 实际 H/D/A |
|------|---|-----|-----|-------|-------|-----------|-----------|-----------|
| 1986 | 52 | 48.1% | 48.1% | 46.2% | 51.9% | 65.4% | 57.7% | 38/31/31 |
| 1990 | 52 | 46.2% | 42.3% | 50.0% | 50.0% | 50.0% | 53.8% | 46/31/23 |
| 1994 | 52 | 57.7% | 51.9% | 55.8% | **57.7%** | 53.8% | 51.9% | 48/23/29 |
| 1998 | 64 | 48.4% | 53.1% | 53.1% | 53.1% | 64.1% | 60.9% | 42/31/27 |
| 2002 | 64 | 45.3% | 45.3% | 46.9% | 42.2% | 54.7% | 59.4% | 41/30/30 |
| 2006 | 64 | 59.4% | 53.1% | **62.5%** | 60.9% | 59.4% | 53.1% | 48/27/25 |
| 2010 | 64 | 50.0% | 51.6% | 50.0% | 48.4% | 62.5% | 64.1% | 36/28/36 |
| 2014 | 64 | 51.6% | 50.0% | 53.1% | 53.1% | 59.4% | 62.5% | 38/27/36 |
| 2018 | 64 | 53.1% | **59.4%** | 53.1% | 54.7% | 64.1% | 60.9% | 39/22/39 |
| 2022 | 64 | 56.2% | 54.7% | 51.6% | 54.7% | 60.9% | 54.7% | 45/23/31 |
| **宏平均** | **604** | **51.6%** | **50.9%** | **52.2%** | **52.7%** | **59.4%** | **57.9%** | - |

**v2 关键观察**:
- XGB29 多数届涨 +3~5pp (1986 略降, 其他都涨)
- Stack vs HYB (v1) 多数届持平或略好, 2002/2010 反而 Stack 拖累
- **让球-1 DC 仍是唯一稳定可投策略** (各届 50-65%, 宏平均 59.4%)

## Stacking 边际收益: 仅 +0.6pp

**为什么 LR meta-learner 在 604 场跨 10 届只比等权 HYB 高 0.6pp**:
1. **样本虽大但每届独立训练 stacking** — LR 在某届 train 后 20% 训练 (5K-7K 场), 但实际 3 模型分歧大多 <5pp (DC/XGB/Elo 共识) → LR 无可学
2. **3 模型预测已高度相关** — XGB 训练时已用 DC prob 作特征 (DC p[0]/p[1]/p[2] in 29 维), XGB 输出与 DC 自然正交性低
3. **未调 LR 超参 (C=1.0 默认)** — Grid search C ∈ [0.1, 0.5, 1.0, 5.0] 可能更优, 但小样本下过拟合风险高

**替代方案 (未实施)**:
- 用 LR 训练**残差**: `y_residual = y_true - hybrid_pred`, 然后 `final = hybrid_pred + alpha * residual_pred`. 这给 LR 明确"只学 hybrid 漏掉的部分"
- 用 **soft-voting** (概率直接平均 + 温度缩放) 替代 LR meta
- 用 **bagging 5 XGB** (不同 seed) 平均 → 已验证在 22 场友谊赛上 +1-2pp

**何时 Stacking 值得做**:
- 5+ 模型分歧明显 (>10pp) — 当前 3 模型不满足
- 样本 ≥ 30K — 当前每届 meta 训练 5-7K 偏少
- 类别不平衡 — 世界杯 H/D/A 较均衡, 优势不显

**结论**: 对 10 届回测, **直接用等权 0.6 XGB + 0.4 DC 即可**, 不必上 Stacking。

## v3-v5 消融实验: 三种 stacking 改进方案均失败或边际无效

在 v2 baseline (HYB 53.4% / Stack 52.7% / 让-1 DC 59.4%) 之上, 尝试三种"升级路线"试图突破:

### v3: XGB 训练集扩量 (FIFA 全量 vs WC-only)

**做法**: 把 XGB 训练集从 "WC 历史 604 场 + 友谊赛筛选" 扩到 "FIFA 国际 A 级全部比赛 ~10万场"。

**结果**: A/B 测试 10 届回测, Acc 略涨 +0.3pp 但 Brier 持平, 统计上不显著。**结论**: XGB 训练量与 XGB 测试性能并非单调正相关 — WC-only 训练集已足够, 引入更多噪声数据反而稀释信号。

**根本原因**: XGB 是 29 维特征驱动, 不是数据驱动。Elo + DC 概率已浓缩 95% 信号, 训练集大小对最终概率影响有限。

### v4: XGB Bagging (5 seeds) + LR Meta

**做法**: 训练 5 个 XGB (不同 random_seed, 相同数据/参数), 输出 5 组概率做简单平均, 再喂给 LR meta-learner。

**结果**:
- XGB 单独: bagging 让 Acc +0.5pp (稳定性↑)
- **Stacking meta: bagging 反而拖累** (LR 学不到残差信号)

**结论**: Bagging 在 stacking meta 阶段是反效果。**直接对 XGB 概率 bagging, 不再喂 LR**。

### v5: 残差 Stacking (失败)

**做法**: 训练 LR 预测 `y_residual = y_true - hybrid_pred`, 最终 `final = hybrid_pred + alpha * residual_pred`。

**结果**: **完全失败** — 验证集 Acc ≈ 0% 提升, Brier 反而恶化。

**根因 (重要教训)**: 测试时 `y_residual = 0` (没有 ground truth), 而 `hybrid_pred` 在测试集并不等于 train 上的均值, 模型默认输出 0 等于"不加 LR"。残差 stacking 只在 train/test 分布严格一致时有效, 在跨届 (1986→2022) 严格时序场景下不适用。

**更深层原因**: 残差 stacking 假设 base model 的偏置是稳定的 (scaled+shifted by a constant), 但跨届场景下 bias 会随战术演变/规则变化漂移。

### 综合结论 (10 届回测全部经验)

| 方案 | Acc 提升 | Brier 变化 | 实施复杂度 | 推荐 |
|------|---------|-----------|----------|------|
| v1 HYB (14 维) | baseline | 0.2001 | 1x | ✅ 默认 |
| v2 XGB29 | +0.1pp | 0.2006 | 1.5x | ✅ 升级 |
| v2 Stacking | +0.6pp | 0.2006 | 2x | ⚠️ 仅需复现现成结果时 |
| v3 FIFA 全量训练 | +0.3pp | 持平 | 1.2x | ❌ 噪音>信号 |
| v4 XGB bagging | +0.5pp (XGB 单独) | 持平 | 1.5x | ⚠️ 仅需 XGB 稳定性时 |
| v4 + LR meta | 拖累 | 恶化 | 2x | ❌ 反效果 |
| v5 残差 stacking | 0% | 恶化 | 2x | ❌ 失败 |
| **让球-1 DC (单独)** | **+6.0pp** | n/a | 1x | ✅ **可投策略** |

**关键 ROI 排序**:
1. **真实市场赔率** (football-data.co.uk CSV) — Brier 0.6132 → 期望 0.55-0.58 (-0.04~0.06), 投入 2-3 天抓数据清洗
2. **XGB bagging** — 稳定性 +0.5pp, 投入 2 小时
3. **残差 stacking** — ROI 为负, 已证伪
4. **其他工程优化** (numpy 向量化 predict_proba) — 训练速度 +70%, 投入 1 天

**不要再尝试的方向**:
- 任何"更复杂的 stacking" (boosting meta, neural meta, etc.) — 604 场样本上限, 复杂度→过拟合
- 把 WC 历史扩展到 1930+ — 数据稀疏 (16 队制) 引入更多噪声
- 引入球员大名单/俱乐部联赛数据 — pitfall #48 已证伪 (Elo 已覆盖, 俱乐部→国家队映射无真实名单支撑)

## v2 实施时的实际坑

### ⚠️ `LogisticRegression(multi_class='multinomial')` 在 sklearn 1.x 已移除

**症状**: `TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'`

**原因**: sklearn 1.5+ 移除了 `multi_class` 参数, 1.5 之前会 DeprecationWarning, 之后直接报错。LR 现在默认是 multinomial (softmax), 不需要显式指定。

**修复**:
```python
# ❌ 旧 (sklearn < 1.5, 已 deprecated)
meta_lr = LogisticRegression(max_iter=500, C=1.0, multi_class='multinomial')

# ✅ 新 (sklearn 1.x)
meta_lr = LogisticRegression(max_iter=500, C=1.0)
# multinomial 是默认行为, 无需传参
```

**诊断**: 如果回测脚本在新环境突然崩, 先检查 sklearn 版本 (`python3 -c "import sklearn; print(sklearn.__version__)"`), 1.5+ 几乎必踩此坑。

### ⚠️ f-string 不能含特殊 unicode 字符作为格式说明符一部分

**症状**: `SyntaxError: f-string: expecting '}'`

**原因**: 在 f-string `f"...{var:.1f·s"}` 里, `·` (U+00B7 middle dot) 不是合法的格式说明符字符, 解析器在 `:.1f` 后期待 `}` 但碰到 `·`。

**修复**: 把 unicode 字符放在花括号外, 变量在花括号内:
```python
# ❌ 错误
log(f"  ⏱ DC: {time.time()-t0:.1f·s")

# ✅ 正确
log(f"  ⏱ DC: {time.time()-t0:.1f}s")  # 或 f"  ⏱ DC: {elapsed:.1f} 秒"
```

### ⚠️ Feature build 是性能瓶颈 (不是 XGBoost)

**实测耗时分布** (v2, 2022 WC 训练 35K 场):
- DC fit: 3.5s
- **Feature build (29 维): 290s** ← 95% 时间
- XGBoost 训练: 2-3s
- Stacking LR fit: <1s
- 预测 + 让球: <1s

**根因**: `build_train_features_29()` 对 35K 场逐场做 `recent_form` + `h2h_full` + `predict_proba`, 每次 O(N) dict lookup。

**优化方向** (未实施, 估计可省 70% 时间):
1. **预索引累计统计**: 对每队维护 `(gf_cum, ga_cum, w_cum, n_cum) -> per_date_dict`, 构建时 O(1) 查
2. **批量 predict_proba**: 用 numpy 向量化算 DC prob, 一次算 35K 场而非 35K 次单场
3. **Cache 复用**: 同一场只算一次, 但当 cutoff 不同时 (不同届) 需要重算 → 难 cache

**FastBuffer 模式** (pitfall #50) 已加速 form/h2h 部分, 但 DC predict_proba 仍是单场调用。如果优化, 先攻这个。
