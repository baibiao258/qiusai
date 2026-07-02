# 5 市场独立校准记录 (2026-06-17 新增)

## 动机

之前系统只记录 SPF 的 Brier Score。让球 RQ、比分 Score、总进球 Goals、半全场 HTFT 四个衍生玩法没有校准依据，无法评估数学推导质量。

## 改动文件

**backfill_results.py** (~1197 行):

### 1. RESULT_FIELDS 扩展

追加 8 列（5 校准列 + 3 列尚为占位）:

```python
# 新增校准列
'brier_rq',       # 让球 Brier，计算方式同 SPF Brier
'acc_score_top1', # 比分 Top1 命中率（最高概率比分是否与实际一致）
'acc_goals_top1', # 总进球 Top1 命中率
'goals_mae',      # 总进球 Mean Absolute Error（平均偏差几球）
'acc_htft_top1',  # 半全场 Top1 命中率
```

### 2. 新增校准函数（4个）

```python
def compute_brier_rq(pred_win, pred_draw, pred_loss, actual_home, actual_away):
    """让球 RQ Brier: 将实际比分转为让球结果后计算 3-class Brier"""
    actual_hda = 'H' if actual_home + handicap > actual_away else ...  # 让球胜负判定
    return brier_spf(pred_win, pred_draw, pred_loss, actual_hda)

def check_score_accuracy(pred_dict, actual_home, actual_away):
    """比分 Top1 命中"""
    top_score = max(pred_dict, key=pred_dict.get)
    actual_score = f"{actual_home}:{actual_away}"
    return 1 if top_score == actual_score else 0

def check_goals_accuracy(pred_dict, actual_home, actual_away):
    """总进球 Top1 命中 + MAE"""
    total_goals = actual_home + actual_away
    top_goals = max(pred_dict, key=pred_dict.get)
    mae = abs(int(top_goals) - total_goals)
    return 1 if int(top_goals) == total_goals else 0, mae

def check_htft_accuracy(pred_dict, actual_ht_h, actual_ht_a, actual_ft_h, actual_ft_a):
    """半全场 Top1 命中"""
    ht_hda = ...  # 半场结果 (H/D/A)
    ft_hda = ...  # 全场结果 (H/D/A)
    actual_htft = f"{ht_hda}-{ft_hda}"  # 如 'H-H'
    top_htft = max(pred_dict, key=pred_dict.get)
    return 1 if top_htft == actual_htft else 0
```

### 3. backfill 循环改造

在原有 SPF Brier 计算块之后追加:

```python
# 让球 Brier
row['brier_rq'] = compute_brier_rq(rq_preds, hg, ag, handicap)

# 比分 Top1
row['acc_score_top1'] = check_score_accuracy(score_preds, hg, ag)

# 总进球 Top1 + MAE
acc_g, mae = check_goals_accuracy(goals_preds, hg, ag)
row['acc_goals_top1'] = acc_g
row['goals_mae'] = mae

# 半全场 Top1（需有半场比分）
if ht_available:
    row['acc_htft_top1'] = check_htft_accuracy(htft_preds, ht_h, ht_a, hg, ag)
```

### 4. 一次性迁移函数

`backfill_missing_new_columns()` 幂等执行:

```python
def backfill_missing_new_columns(csv_path):
    """为旧数据补充新校准列。幂等：已有值的不覆盖。"""
    rows = list(csv.DictReader(...))
    new_rows = []
    for row in rows:
        has_score = bool(row.get('actual_score', '').strip())
        if has_score:
            # 有赛果的行如果新列为空则重新计算
            if not row.get('brier_rq'):
                row['brier_rq'] = compute_brier_rq(...)
            ...
    # 全量写回
```

## 当前基线 (n=159 完赛回填, 2026-06-17)

| 市场 | 指标 | 值 | 备注 |
|------|------|----|------|
| SPF | Brier | 0.2422 | SPF 3-class |
| 让球 RQ | Brier | 0.1870 | 比 SPF Brier 低，说明让球预测更集中 |
| 比分 Score | Acc | 6.3% | 31 种选项，随机猜 ~3.2%，略好于随机 |
| 总进球 Goals | Acc | 29.6% | 13 档，MAE=1.6（平均偏差 1.6 球） |
| 半全场 HTFT | Acc | 30.1% | 9 种组合，随机 ~11.1%，显著优于随机 |

## 校准观测

- **RQ Brier (0.1870) < SPF Brier (0.2422)**: 让球预测相对于 SPF 更保守（概率分布更平），Brier 天然偏低。不能直接对 比，需按各自基线评估。
- **比分 Acc (6.3%)**: 低但合理——比分是 31 种选项的精确预测。随机准确率 ~3.2%。
- **总进球 MAE (1.6)**: 实际进球分布在 0-7，平均偏差 1.6 球，说明泊松参数 λ 的估计方向正确但不够尖锐。
- **半全场 Acc (30.1%)**: 9 种组合中随机猜 ~11.1%，30.1% 说明模型捕捉到了半全场相关性。
- **样本量 (n=159) 不足以下统计显著结论**: 需积累到 500+ 场。

## 查看命令

```bash
# 全 5 市场校准报告
python3 /root/evaluate_brier.py

# 仅统计覆盖
python3 /root/backfill_results.py --stats
```

## 关联文件

| 文件 | 职责 |
|------|------|
| `/root/backfill_results.py` | 赛果回填 + 5 市场校准 + 增量 Elo 更新 |
| `/root/evaluate_brier.py` | 5 市场校准概览打印 + 校准曲线 |
| `/root/data/predictions_log.csv` | 已含 5 校准列 (brier_rq, acc_score_top1, acc_goals_top1, goals_mae, acc_htft_top1) |
