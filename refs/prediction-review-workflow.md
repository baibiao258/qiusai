# 预测回顾分析流程 (2026-06-15 建立)

当用户问"昨天的预测怎么样"或"回顾一下预测"时，执行以下分析。

## 数据源

`/root/data/predictions_log.csv` — 每行一条预测，`settled_at` 非空且非 `missing` 表示已结算。

## 分析步骤

### 1. 筛选已结算比赛

```python
import csv
rows = list(csv.DictReader(open('/root/data/predictions_log.csv')))
yesterday = '2026-06-14'  # 或用户指定日期
settled = [r for r in rows if yesterday in (r.get('settled_at','') or '')]
```

### 2. 逐场计算命中

actual_hda 字段格式不统一（有的是 'H'/'D'/'A'，有的是 '胜'/'平'/'负'），需统一映射：

```python
hda_cn_to_code = {'胜': 'H', '平': 'D', '负': 'A'}
hda_map_r = {'主胜': 'H', '平局': 'D', '平': 'D', '客胜': 'A'}
```

- SPF: `hda_map_r[pred_spf_pick] == hda_cn_to_code.get(actual_hda, actual_hda)`
- RQ: `{'让胜':'H','让平':'D','让负':'A'}[pred_rq] == hda_cn_to_code.get(actual_rq, actual_rq)`
- HTFT: `{'胜胜':'HH',...}[pred_htft] == {'胜胜':'HH',...}[actual_htft]`
- Goals: `pred_goals == actual_goals`（字符串直接比较）

### 3. 统计汇总

输出: SPF命中率、RQ命中率、HTFT命中率、Goals命中率、平均Brier。

### 4. 模式识别

重点关注:
- 强队局 vs 冷门局的命中差异
- 模型对平局的预判能力
- 比分/进球预测的保守性偏差
- Brier Score vs 随机基线(0.25)的对比

### 5. 待结算比赛

筛选 `settled_at` 为空或 `missing` 的记录，显示预测值等待验证。

## 输出格式

结论先行（命中率），然后逐场明细（✓/✗标记），最后模式分析。不要只给数字不给解读。
