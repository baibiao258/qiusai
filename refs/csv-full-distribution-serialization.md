# CSV 完整概率分布序列化 (2026-06-09)

## 问题
`daily_jczq.py` 写入 `predictions_log.csv` 时只保存 top1 推荐（`pred_top_score`/`pred_top_htft`/`pred_top_goals`），
没有保存完整概率分布（如所有比分概率、所有半全场概率）。
读取端无法还原完整分布，导致输出只有一个"模型主推荐"。

## 根因
`record_prediction()` 调用 `backtest_jczq.py` CLI 写入 CSV，但 CLI 参数列表不包含完整分布字段。
backtest 脚本的 `FIELDS` 列表和 `elif` 映射链也不包含这些字段。

## 修复方案 (2026-06-10 更新)

### 2026-06-10: goals_full 改用完整分布而非 top5

`record_prediction()` 原写法用 `bundle.get('goals_top5', [])` 只保存 top5 进球档位。改为 `bundle.get('goals_all', [])` 保存全部 13 档 (0~12球)。

**修改行**: daily_jczq.py 第 1329 行
```python
# 旧: for g, pr in bundle.get('goals_top5', []):
# 新: for g, pr in bundle.get('goals_all', []):
```

`goals_all` 来自 `compute_goals_distribution()` 的 Poisson 卷积 (MAX_GOALS=6, max_total=12)，包含 13 个键值对。概率为 0.0 的档位也会写入 JSON。

**展示端注意**: CSV 可能有同场次多行（旧截断行 + 新完整行）。选行逻辑应优先完整分布（JSON 键数），而非 time/odds 等元数据字段。

**验证**:
```bash
python3 -c "import csv,json; rows=list(csv.DictReader(open('/root/data/predictions_log.csv'))); r=[r for r in rows if r.get('code')=='周四001' and r.get('goals_full')][0]; print(sorted(json.loads(r['goals_full']).keys()))"
# 应输出 ['0','1','2','3','4','5','6','7','8','9','10','11','12']
```

### 1. daily_jczq.py — 序列化完整分布
在 `record_prediction()` 的 `cmd` 列表追加 JSON 字符串参数：

```python
import json as _json

# 序列化完整概率分布
score_full = {}
for s, pr, _hg, _ag in bundle.get('score_top8', []):
    score_full[s] = round(pr, 4)

htft_full = {}
for label, pr in bundle.get('htft_top6', []):
    htft_full[HTFT_SHORT_MAP.get(label, label)] = round(pr, 4)

goals_full = {}
for g, pr in bundle.get('goals_top5', []):
    goals_full[str(g)] = round(pr, 4)

cmd += [
    '--score-full', _json.dumps(score_full, ensure_ascii=False),
    '--htft-full', _json.dumps(htft_full, ensure_ascii=False),
    '--goals-full', _json.dumps(goals_full, ensure_ascii=False),
]
```

### 2. backtest_jczq.py — 三处修改

**a) FIELDS 列表新增字段:**
```python
"score_full",    # 比分完整概率分布 (JSON)
"htft_full",     # 半全场完整概率分布 (JSON)
"goals_full",    # 总进球完整概率分布 (JSON)
```

**b) argparse 新增参数:**
```python
ap.add_argument('--score-full', type=str, default='')
ap.add_argument('--htft-full', type=str, default='')
ap.add_argument('--goals-full', type=str, default='')
```

**c) elif 映射链新增:**
```python
elif k == "score-full": row["score_full"] = v
elif k == "htft-full": row["htft_full"] = v
elif k == "goals-full": row["goals_full"] = v
```

## 数据格式

### 写入格式 (JSON 字符串)
```python
score_full = '{"1:0": 0.142, "0:0": 0.118, "2:0": 0.095, "1:1": 0.087}'
htft_full = '{"HH": 0.312, "HD": 0.118, "DH": 0.085}'
goals_full = '{"1": 0.254, "2": 0.301, "3": 0.218}'
```

### 读取格式
```python
import json
score_dist = json.loads(row['score_full'])  # → {'1:0': 0.142, ...}
htft_dist = json.loads(row['htft_full'])   # → {'HH': 0.312, ...}
goals_dist = json.loads(row['goals_full'])  # → {'1': 0.254, ...}
```

### 反向转换 (从旧格式 top1 恢复)
旧字段 `pred_top_score` 只保存最可能比分（如 "1:0"），无法还原完整分布。
修复后的新行同时保存 `pred_top_score` (top1) 和 `score_full` (完整分布)。

## 向后兼容
- 旧 CSV 行的 `score_full`/`htft_full`/`goals_full` 为空字符串
- 读取时应检查字段是否存在/非空，降级到使用 top1 字段
- `ensure_log_has_source_fields()` 已扩展以包含新字段（可选）
