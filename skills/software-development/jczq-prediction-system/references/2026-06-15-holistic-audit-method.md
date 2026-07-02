# 预测系统 4 层深度审计方法 (2026-06-15 建立)

## 审计流程模板

每次对 JCZQ/WC 预测系统做全面审计时，按以下 4 层 6 步执行。

### 步骤 0: 文件普查

```bash
# 模型文件
ls -la /root/data/xgb_model_*.pkl /root/data/dc_model*.pkl /root/data/elo*.pkl
# 校准器文件
ls -la /root/data/calibrat*.pkl
# 训练数据
wc -l /root/data/training_data_with_odds.json && python3 -c "import json; print(len(json.load(open('/root/data/training_data_with_odds.json'))))"
# 日志
wc -l /root/data/predictions_log.csv
# 管线
ls -la /root/daily_jczq.py /root/backfill_results.py
```

### 步骤 1: 层1 — 数据源与训练数据审计

```python
from collections import Counter
import json

d = json.load(open('/root/data/training_data_with_odds.json'))

# 1a. 标签类型混型检测
int_ct = sum(1 for m in d if isinstance(m.get('spf_result'), int))
str_ct = sum(1 for m in d if isinstance(m.get('spf_result'), str))
print(f'标签类型: int={int_ct}, str={str_ct}')
assert str_ct > int_ct, 'str 类型应占多数, 否则上游写入有问题'

# 1b. 标签污染计算
wrong = 0
for m in d:
    r = m.get('spf_result')
    if isinstance(r, int):
        hg, ag = m.get('ft_h',0), m.get('ft_a',0)
        true_label = 2 if hg > ag else (1 if hg == ag else 0)
        if true_label != 0:
            wrong += 1
print(f'标签污染: {wrong}/{int_ct} ({wrong/max(int_ct,1)*100:.0f}%)')

# 1c. 日期分布 + 空白检测
dates = sorted(set(m['date'][:10] for m in d if m.get('date')))
print(f'日期范围: {dates[0]} ~ {dates[-1]} ({len(dates)}天)')

# 检查空白月份
months_covered = set(d[:7] for d in dates)
all_months = set()
for y in range(2024, 2027):
    for m in range(1, 13):
        all_months.add(f'{y}-{m:02d}')
gaps = sorted(all_months - months_covered)
if gaps:
    print(f'数据空白月: {gaps}')

# 1d. DC 覆盖率 + 中文名比例
import joblib
dc = joblib.load('/root/data/dc_model.pkl')
cn = sum(1 for m in d if any('\u4e00' <= c <= '\u9fff' for c in m.get('home','') + m.get('away','')))
dc_cov = sum(1 for m in d if m.get('home_en','') in dc.team_idx_ and m.get('away_en','') in dc.team_idx_)
print(f'中文队名: {cn}/{len(d)}, DC覆盖: {dc_cov}/{len(d)}')
```

### 步骤 2: 层2 — 模型文件与特征健康度审计

```bash
# 2a. 模型时间戳一致性
echo "=== XGB模型 ==="
ls -la /root/data/xgb_model_*.pkl
echo "=== DC模型 ==="
ls -la /root/data/dc_model*.pkl
echo "=== 校准器 ==="
ls -la /root/data/calibrat*.pkl

# 2b. 死特征检测
python3 -c "
import joblib
for model_name in ['xgb_model_29', 'xgb_model_30', 'xgb_model_33', 'xgb_model_nat', 'xgb_model_club']:
    try:
        model = joblib.load(f'/root/data/{model_name}.pkl')
        imp = model.feature_importances_
        zero = sum(1 for v in imp if v == 0.0)
        print(f'{model_name}: {len(imp)}维, 死特征={zero}/{len(imp)} ({zero*100/len(imp):.0f}%)')
    except: pass
"
```

### 步骤 3: 层3 — 管线代码与算法配合审计

```bash
# 3a. 平局硬编码检测
grep -rn 'arr.*\[.*0.*1-' /root/*.py /root/wc_2026_upgrade/*.py 2>/dev/null | grep -v '\.pyc'
grep -rn 'draw.*=.*0\|, 0, 1-' /root/*.py /root/wc_2026_upgrade/*.py 2>/dev/null | grep -v '\.pyc'

# 3b. 校准器是否仍在调用链中
grep -rn 'calibrat.*predict\|_cal.*predict\|calibrators.*\.predict' /root/*.py /root/wc_2026_upgrade/*.py 2>/dev/null

# 3c. float/int 比较陷阱 (result == '3' 模式)
grep -rn "result.*== '" /root/wc_2026_upgrade/train_*.py 2>/dev/null
grep -rn "spf_result" /root/wc_2026_upgrade/train_*.py 2>/dev/null
# 确认都有 str() 包裹
```

### 步骤 4: 层4 — 回测与验证审计

```bash
# 4a. Brier 统计
python3 /root/backfill_results.py --stats

# 4b. model_route 填充率 + bet_action 区分度
python3 -c "
import csv
from collections import Counter
rows = list(csv.DictReader(open('/root/data/predictions_log.csv')))
total = len(rows)
routes = Counter(r.get('model_route','') for r in rows)
actions = Counter(r.get('bet_action','') for r in rows)
brier_ok = sum(1 for r in rows if r.get('brier_spf','').strip())
draw_pred = sum(1 for r in rows if float(r.get('pred_d',0)) > float(r.get('pred_h',0)) and float(r.get('pred_d',0)) > float(r.get('pred_a',0)))
print(f'总行: {total}, Brier: {brier_ok}/{total}, 平局首选: {draw_pred}')
print(f'路由: {dict(routes)}')
print(f'动作: {dict(actions)}')
# bet_action 命中率
for act in ['RECOMMEND','WATCH','WATCH_FRIENDLY']:
    subset = [r for r in rows if r.get('bet_action','') == act and r.get('brier_spf','').strip()]
    if subset:
        hits = sum(1 for r in subset if r.get('actual_hda','') == r.get('pred_spf_pick','')[:1])
        print(f'  {act}: {hits}/{len(subset)} = {hits/len(subset)*100:.0f}%')
"
```

### 步骤 5: 管线差异对比

```bash
# daily_jczq.py vs calibrated_predictor.py 差异审计
echo "=== daily_jczq 引用的 XGB ==="
grep 'xgb_model_' /root/daily_jczq.py | grep -o 'xgb_model_[a-z0-9_]*' | sort -u
echo "=== calibrated_predictor 引用的 XGB ==="
grep 'xgb_model_' /root/wc_2026_upgrade/calibrated_predictor.py | grep -o 'xgb_model_[a-z0-9_]*' | sort -u
echo "=== 校准器加载 ==="
grep -n 'calibrat' /root/daily_jczq.py | head -10
grep -n 'calibrat' /root/wc_2026_upgrade/calibrated_predictor.py | head -10
echo "=== 融合策略 ==="
grep -n 'blend\|entropy\|xgb_weight\|dc_weight\|compute_dynamic' /root/daily_jczq.py | head -5
grep -n 'blend\|entropy\|xgb_weight\|dc_weight' /root/wc_2026_upgrade/calibrated_predictor.py | head -5
```

### 步骤 6: 输出审计报告

按以下模板汇总:

```markdown
## 审计报告 {date}

### 层1 (数据源)
- 训练数据: {n}条, {range}
- 标签污染: {n}条 — P{0/1/2}
- DC覆盖: {n}%
- 数据空白: {months}

### 层2 (模型)
- 死特征: {modelA}: {n}/{m}, {modelB}: {n}/{m}
- 模型时间戳: {latest} for {model_name}
- 双管线XGB版本一致? {yes/no}

### 层3 (代码)
- 平局硬编码: {found/not found}
- 校准器使用: {active/stripped}
- 标签 str() 包裹: {all done/missing N}

### 层4 (验证)
- Brier: {avg} (n={n})
- model_route填充: {n}/{total}
- bet_action区分度: {RECOMMEND}/{WATCH}
- 平局首选: {n}

### 优先级排序
- P0: ...
- P1: ...
- P2: ...
```
