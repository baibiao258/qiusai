# 双轨隔离路由架构 (2026-06-15 确立)

## 背景: 训练数据分布偏移

TheStatsAPI 全量训练数据 32,001 场:

| 类型 | 场次 | 占比 |
|------|------|------|
| 俱乐部比赛 | 31,600 | 98.7% |
| 国际比赛 | 401 | 1.3% |
| 其中世界杯 | 76 | 0.2% |

**DC 模型攻防参数是按球队独立估计的**, 不受训练集分布影响 → 国际赛表现稳定。
**XGBoost 特征权重受训练集分布主导** → 俱乐部数据训练的 XGBoost 拖累国际赛预测。

## 实证数据 (50 场世界杯回测)

| 模型 | 命中率 | 平均 Brier | 说明 |
|------|--------|-----------|------|
| DC 模型 (纯泊松) | **60.0%** | **0.1921** | 基线最佳 |
| DC+XGB (11维 nat) | 57.1% | 0.2070 | 恶化 |
| DC+XGB (17维 +form) | 46-52% | 0.21+ | 显著恶化 |
| 随机基线 | 33.3% | 0.222 | — |

DC 模型单独优于任何 XGBoost 变体。**结论: 国际赛应当跳过 XGBoost。**

## 实现: 双轨隔离

### `_try_hybrid_predict()` 函数签名

```python
def _try_hybrid_predict(home, away, league='', thestats_match_id=None, is_intl=None):
```

### is_intl 检测 (auto-detect)

```python
INTL_KEYWORDS = ['世界杯', 'World Cup', '欧洲杯', 'EURO', 'Copa America',
                 '非洲杯', 'AFCON', '亚洲杯', 'AFC Asian Cup', 'Gold Cup',
                 '国际', '友谊', 'Friendly', 'International',
                 '预选', 'Qualification', 'Qualifier', 'Nations League']

if is_intl is None:
    is_intl = any(kw.lower() in (league or '').lower() for kw in INTL_KEYWORDS)
```

### 路线 A: 国际赛 (is_intl=True)

```python
# 1. DC 模型概率
hybrid = dc_ado.copy()  # [A, D, H]

# 2. Pinnacle 市场校正 (仅当有赔率且分歧 >15%)
if pinn_prob_h > 0:
    divergence = np.max(np.abs(pinn_probs - hybrid))
    if divergence > 0.15:
        market_weight = 0.15   # 从原先30%降至15% (验证发现30%过度扭曲)
        hybrid = (1-market_weight)*hybrid + market_weight*pinn_probs

# 3. 平局膨胀因子 (Elo差<100时上调10-15%)
if elo_diff < 100:
    hybrid = _apply_intl_draw_boost(hybrid, elo_h, elo_a, is_knockout)

# 4. 战意不足补丁 (小组赛第三轮已出线强队概率削减15%)
if group_stage_last_round and team_already_qualified:
    hybrid[2] *= 0.85   # 削减主胜(降客队/平局)
    total = sum(hybrid.values())
    for k in range(3): hybrid[k] /= total

model_name = 'dc_pinnacle'
```

### 验证结果 (2026-06-15 全量测试)

| 测试 | 场次数 | 覆盖率 | 路由正确率 | 说明 |
|------|--------|--------|-----------|------|
| OOV 修复前 | 12 世界杯 | 7/12 (58%) | — | Netherlands/Japan/Ivory Coast 等跳过 |
| OOV 修复后 | 12 世界杯 | **12/12 (100%)** | **100%** | _TEAM_SYNONYMS + _fuzzy_team_lookup |
| dry-run 全联赛 | 7场混测 | — | 100% | 世界杯→dc_pinnacle, 联赛→hybrid |
| 平局膨胀 | Netherlands/Japan | — | — | draw: 22% → 25% |
| 平局膨胀 | Brazil/Morocco | — | — | draw: 26% → 29% |

### 路线 B: 俱乐部/联赛 (is_intl=False)

```python
# 1. DC 模型
# 2. 构造 11 维特征向量 (elo_diff, lam_h, lam_a, lam_diff, lam_ratio, dc_a, dc_d, dc_h, op_h, op_a, market_implied)
# 3. XGBoost 预测
xgb_p = _xgb_nat.predict_proba(feat_nat)[0]
# 4. 动态 DC+XGB 融合
xgb_w, dc_w, _ = compute_dynamic_xgb_weight(xgb_p)
hybrid = dc_w * dc_ado + xgb_w * xgb_p
# 5. Draw Correction
model_name = 'hybrid'
```

## 验证方法

### 干跑

```python
r = _try_hybrid_predict("Sweden", "Tunisia", "世界杯", None)
assert r['model'] == 'dc_pinnacle'     # 路线 A

r = _try_hybrid_predict("Arsenal", "Chelsea", "英超", None)
assert r['model'] == 'hybrid'            # 路线 B
```

### 返回格式

```python
{
    'probs': {'H': 0.69, 'D': 0.16, 'A': 0.16},
    'model': 'dc_pinnacle',         # 或 'hybrid'
    'routing': {'is_intl': True, 'market_corrected': True},
    'lambda_ft': {'home': 1.86, 'away': 0.54},
}
```

## 已知局限

1. **100% 俱乐部训练数据**: XGBoost 对俱乐部队的预测优势理论上存在, 但缺乏独立俱乐部测试集的验证
2. **Pinnacle 开盘赔率 vs 收盘赔率**: 当前使用 `opening` odds。收盘赔率 (last_seen) 包含更多市场信息, 但需要比赛开始后的快照, 无法在 03:00 UTC 预加载时获取
3. **is_intl 检测依赖 league 参数**: 如果 league 为空字符串或不含关键词, 默认走路线 B (俱乐部路径)。这可能导致遗漏的国际比赛走了错误路线
