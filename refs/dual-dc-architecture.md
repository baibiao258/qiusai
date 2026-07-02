# 双 DC 模型架构 (2026-06-14 建立, 2026-06-15 更新)

## 问题

原单核 DC 模型 (`dc_model.pkl`) 只覆盖 **226 支国家队**。验证集中 86% 的比赛是俱乐部赛事，DC 无法预测 → 特征退化为均匀分布 → 最新 fold 准确率降至 24.8% (vs baseline 89.2% 主场胜)。

## 架构

```
推理时 DC 查询链:
  1. dc_model (226 国家队)  ─── predict_lambda(home, away, neutral=True)
     ↓ 失败
  2. dc_club (2174 俱乐部)  ── predict_lambda(home, away, neutral=False)
     ↓ 失败
  3. 均匀概率 (1/3, 1/3, 1/3) + 全局平均λ=1.5
```

### 训练数据对比

| 维度 | dc_model (国家队) | dc_club (俱乐部) |
|------|-----------------|-----------------|
| 数据源 | international_results.json | 500_history_backfill.csv |
| 训练样本 | 225,000+ 场 (历史数据) | 18,102 场 (2026-01~06) |
| 球队数 | 226 | 2,174 |
| 队名语言 | 英文 | 中文 (500.com 直接使用) |
| 时间衰减 | 540 天 | 180 天 |
| 主场优势 (γ) | ~0.12 | 0.0 (500 数据 γ 不显著) |
| ρ | -0.25~0.0 | 0.0 |
| 筛选条件 | — | league≥200场 + team≥10场 |

### 俱乐部 DC 的局限性

1. **来自 500.com 的同批次数据**：训练集年份与验证集重叠 (2026-01~06)，有信息泄露风险
2. **低级别联赛为主**：勒沃库森、切尔西、巴萨等豪门因出现<10次被过滤
3. **γ=0**：无主场优势参数 → neutral=False 与 neutral=True 输出相同
4. **ρ=0**：射门相关性修正未生效（低比分校正）

### 覆盖提升

| DC 模型 | 验证集覆盖 | 增量 |
|---------|-----------|------|
| 仅国家队 (226队) | 73/170 = **43%** | — |
| + 俱乐部 (2,174队) | +43 = **116/170 = 68%** | **+25pp** |
| 仍无覆盖 | 54/170 = **32%** | 低频球队/小众联赛 |

## 🔴 关键坑: DC 概率顺序必须对齐 (2026-06-15 发现)

**问题**: `predict_proba()` 返回 `[ph, pd, pa]`(Home/Draw/Away 顺序), 但 XGBoost 训练的特征向量中 DC 概率是 `[p_a, p_d, p_h]`(Away/Draw/Home 顺序, 匹配标签类别 0=A/1=D/2=H)。推理代码如果误用 `[2],[1],[0]` 取索引, 会把 Home 填入 Away 的位置, 产生完全错误的预测。

**典型错误示例** (曾导致 Germany vs Curaçao 从 97.1% 变成 9.6%):

```python
# ❌ 错误: 索引取反
dc_pred = model.predict_proba(home, away)  # [ph, pd, pa]
dc_probs = np.array([dc_pred[2], dc_pred[1], dc_pred[0]])  # → [pa, pd, ph] ✅ 正确
# 但如果后续用 dc_probs[0] 时以为是 Away 其实是 ph(H)...

# 特征向量传入:
feat = [..., dc_probs[2], dc_probs[1], dc_probs[0], ...]  
# ❌ dc_probs[2]=ph 填在了 away 的位置 → 全部预测反了!
```

**修复后的正确做法**:

```python
# dc_pred = predict_proba() → [ph, pd, pa] (Home, Draw, Away)
# 目标: dc_feat = [p_a, p_d, p_h] 匹配特征向量位置
dc_probs = np.array([dc_pred[2], dc_pred[1], dc_pred[0]])  # → [pa, pd, ph]

# 特征向量: 索引5=away, 6=draw, 7=home
feat = [..., dc_probs[0], dc_probs[1], dc_probs[2], ...]  
# ✅ dc_probs[0]=pa (away), dc_probs[1]=pd (draw), dc_probs[2]=ph (home)
```

**验证方法**: 对比训练脚本 (`train_clean_xgb.py`) 和预测脚本 (`predict_wc_today.py`) 中 DC 概率传入特征向量的顺序是否一致。

训练代码中 `compute_dc_probs()` 返回的是 `[p_a, p_d, p_h]` (Away-first), 因为内部手动用 Poisson 计算了概率。而 `dc_model.predict_proba()` 返回 `[ph, pd, pa]` (Home-first)。两者相差一个索引反转。

### 防御性检查

所有新建的推理/预测脚本在第一次运行前, 必须在 `deterministic unit tests` 上验证:
- 取一场 DC 明显倾向的比赛 (如 Germany vs Curaçao)
- 预期: H > 90%
- 如果输出 H < 50%, 怀疑特征顺序有误

## Winsorize 截断 (2026-06-15 添加)

DC 模型在某些边缘比赛上给出接近 0/1 的极端概率, 使 XGBoost 特征空间扭曲。所有 DC 概率在入特征前做 Winsorize 截断:

```python
dc_probs = np.clip(dc_probs, 0.01, 0.99)
lam_h = max(0.1, min(5.0, lam_h))
lam_a = max(0.1, min(5.0, lam_a))
```

这个操作在 `train_clean_xgb.py`、`retrain_xgb_with_odds.py`、`predict_wc_today.py` 三处同步维护。

## TheOddsAPI 与 football-data.org 的 DC 补充 (2026-06-15 尝试)

### TheOddsAPI
- 免费版 500 次/月
- 世界杯期间有 64 场实时赔率, 但**不可用于训练历史 DC 模型**
- `/scores` 端点只返回近期完成的比赛
- 可用于**赔率特征注入** (存入 `wc_2026_odds_today.json`)

### football-data.org (不推荐用于 DC 训练)
- 拉取 8 大联赛 2,743 场赛果, 队名为英文
- **队名格式不兼容**: API 返回 "FC Bayern München", "Arsenal FC", "FC Barcelona" — 常见名称是 "Bayern Munich", "Arsenal", "Barcelona"
- 单赛季数据量不足以训练稳定 DC 参数 (Only 152 teams, 2,743 matches vs 2,174 teams, 18,102 matches from 500.com)
- 已训练 `dc_club_en.pkl` 但**不推荐生产使用**
- 正确的用法: 通过 football-data.org 的比赛 ID 关联, 而非队名字符串匹配

## 训练命令

```bash
# 俱乐部 DC (中文名, 推荐)
python3 wc_2026_upgrade/train_club_dc.py
# 输出: /root/data/dc_club.pkl

# 英文俱乐部 DC (不推荐生产)
python3 wc_2026_upgrade/train_club_dc_en.py
# 输出: /root/data/dc_club_en.pkl
```

筛选参数在 `train_club_dc.py` 顶部可调:
- `MIN_LEAGUE_MATCHES = 200`
- `MIN_TEAM_MATCHES = 10`
- `TIME_DECAY_HL = 180`

## 集成到训练管线

`compute_dc_probs(dc_model, home, away, dc_club=None)` 实现三层回退:

1. `dc_model.predict_lambda(home, away, neutral=True)` 优先
2. 若失败且 dc_club available: `dc_club.predict_lambda(home, away, neutral=False)` 回退
3. 都失败: return None → 调用方使用均匀概率

**注意**: `compute_dc_probs()` 在 `train_clean_xgb.py` 和 `retrain_xgb_with_odds.py` 中各自定义了一份独立实现。两处修改时必须同步更新。特征顺序 (A/D/H vs H/D/A) 是高频出错点。

## 待改进

- [ ] 集成到 daily_jczq.py 推理管线（当前只在训练中使用）
- [ ] 500.com 数据扩充到更多联赛（放宽 MIN_LEAGUE_MATCHES 并用球队过滤兜底）
- [ ] 俱乐部 DC 增加主场优势参数（需验证 500.com 数据是否有主场偏差）
- [ ] 统一 `compute_dc_probs()` 实现到共享函数而不是两份拷贝
