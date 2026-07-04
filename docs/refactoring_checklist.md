# docs/refactoring_checklist.md

# 重构完成后扫尾清单

大改动（拆分模块/重命名文件/更换特征维度）合并后，
容易被遗漏的不是核心代码，而是这些外围引用。

## 1. 文档同步扫描

重构后必须 grep 全仓库的旧路径/旧模块名，不能只看代码引用，
文档和注释一样会指向死代码。

```bash
grep -rn "旧文件名\|旧函数名" --include="*.md" --include="*.py" .
```

**案例**：`core/daily_jczq.py` 重构拆分后（`f4696b0`），README.md 三处仍标注
它为"主入口"并写着 `python3 core/daily_jczq.py` 作为运行命令（`4c9bab5`）。
若照文档操作，会误用残留在旧文件中的 8 维 xG 模型而非当前 12 维特征管线。

## 2. 唯一数据源检查

任何"结果标签"字段被多处写入时，必须确认所有写入点用的是同一套语义，
不能靠字段名一致就假设内容一致。

```bash
grep -rn "字段名" --include="*.py" . | grep -v "读取\|read"
```

**案例**：`actual_rq_result` 被两条路径分别写成 `H/D/A` 和 `让胜/让平/让负`，
字段名相同但语义不同，导致 RQ 准确率显示为 8%（实为标签错配，修复后为
37.9%）（`7ed2ba4`）。

## 3. 调用链去重确认

拆分模块后，检查是否有自包含的旧副本残留（同名函数在多处定义），
这类文件不会报错，但会在被误调用时静默使用过时逻辑。

```bash
grep -rln "def _build_xg_feat\|def predict_match" --include="*.py" .
```

若结果 > 1 处，需确认哪个是唯一权威版本，其余删除或明确废弃。

**案例**：`core/daily_jczq.py` 在被拆分出 7 个 pipeline 子模块后，
自身仍保留一份完整的 `_build_xg_feat`、`_try_club_predict` 和模型加载
逻辑（8 维、过时）。crontab 和系统调度均不经过它，但 README 将其标注
为"主入口"，构成了一个静默的错误路径。（`4c9bab5`）

## 4. 特征维度/Schema变更后重训确认

模型输入维度变化后，必须同步：

- 训练侧和推理侧的特征拼接顺序完全一致（顺序错位时 XGBoost 不会报错，
  但会学到错误的特征权重）
- 旧模型文件不能被新特征直接复用，需重新训练
- 变更前备份旧模型，保留可回滚路径

```bash
# 训练前备份
cp data/xgb_model_club.pkl data/xgb_model_club_37d_baseline.pkl.bak
# 训练后验证维度
python3 -c "import joblib; m=joblib.load('/root/data/xgb_model_club.pkl'); print(m.n_features_in_)"
```

## 5. 验证样本量声明

任何"待验证"的改动（新特征/新模型/新校正层），commit message
必须明确写出当前验证样本量和达到多少场才能下结论，
避免几周后误以为已经过实战验证。

```
feat: expand xG features 8->12 dims, retrain club model to 41d
...
NOTE: club model has 0 verified predictions in predictions_log.csv.
Real-world validation (HDA accuracy/Brier vs 37d baseline) pending
accumulation of 30+ verified club match results.
```

## 6. 删除即沟通（delete-by-default）

废弃的代码文件如果不再被任何人引用，直接删，不要留"DEPRECATED"软标记。
保留一份死代码等于给未来任何一个按文档操作的人设了一个静默陷阱——
"DEPRECATED"警告在实际操作中经常被忽略，尤其是当文档里明确写着
"这是运行命令"的时候。

**案例**：`core/daily_jczq.py` 删除时（`4c9bab5`），README.md 三处同步修正。
这是唯一正确的做法——保留文件 + 加 DEPRECATED 头注释只能降低被踩中的概率，
但不能消除。
