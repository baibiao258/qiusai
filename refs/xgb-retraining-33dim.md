# XGB 33维模型训练 (2026-06-11 Phase 2)

## 变更摘要

在29维基础特征上新增4维赛事阶段特征，训练得到34维（29+1市场赔率+4赛事阶段）的新影子模型。

## 修改的脚本

### prepare_training_data.py
- `merge_data()` 函数新增4个字段：`points_diff`, `rank_diff`, `is_knockout`, `round_num`
- 赛事类型从 `best_match['tournament']` 推断：检查 'World Cup'/'世界杯' 等关键词
- 淘汰赛标记：检查 'final'/'semi'/'quarter'/'knockout'（小写）
- 轮次推断：世界杯按月份（6月=第1轮, 7月=第2轮），非世界杯默认第1轮
- 输出文件：`/root/data/training_data_with_odds.json`（新增4个字段）

### retrain_xgb_with_odds.py
- `build_features()` 返回34维（原30维 + 4维赛事阶段）
- 新增 `stage_feat`：`[points_diff, rank_diff, is_knockout, round_num]`
- 特征名称列表新增4项
- 模型保存路径：`/root/data/xgb_model_33.pkl`（不再覆盖30维模型）
- 校准器保存路径：`/root/data/calibrators_v2.pkl`（覆盖旧校准器）

## 训练结果

```
训练数据: 263 场 (from 3248 kaijiang)
特征维度: 34
平均LogLoss: 0.6419
平均准确率: 78.5%

Top 5 特征重要性:
  1. op_a:          14.65%
  2. market_implied: 14.17%
  3. op_h:          11.39%
  4. dc_h:          11.27%
  5. elo_diff:      10.18%

赛事阶段特征重要性 (均为0%):
  points_diff: 0.00%
  rank_diff:   0.00%
  is_knockout: 0.00%
  round_num:   0.00%
```

**原因**：训练数据中的赛事阶段特征使用占位值（points_diff=0, rank_diff=0.333），模型未学习到这些特征的真实权重。推理时 `daily_jczq.py` 会通过 `TOURNAMENT_STATE_2026` 注入真实值，但模型本身尚未适应。

## daily_jczq.py 集成

`_load_shared_models()` 优先级：
1. `/root/data/xgb_model_33.pkl`（34维，含赛事阶段）
2. `/root/data/xgb_model_30.pkl`（30维，回退）
3. `/root/data/xgb_model_29.pkl`（29维，主模型）

推理时自动检测维度：
```python
feat_dim = _xgb_model.n_features_in_ if hasattr(_xgb_model, 'n_features_in_') else 29
if feat_dim == 29:
    feat = feat_33[:, :29]
else:
    feat = feat_33
```

## 后续步骤

**Phase 3: 用真实赛事状态数据重训**
- 从500.com抓取小组赛积分榜
- 从365scores获取实时排名
- 用真实 `points_diff`/`rank_diff` 替代占位值
- 重训后赛事阶段特征重要性应 > 1%
- 预期Brier Score再降0.02-0.03
