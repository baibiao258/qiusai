# 回测核验与非竞彩赛事处理 (2026-06-30 落地)

## 结论

`backtest_jczq.py` 的 `fetch_results(date)` 从 **500.com 开奖页** (`zx.500.com/jczq/kaijiang.php?playid=0&d={date}`) 拉取**竞彩官方开售**的赛果。  
**只有出现在该接口里的比赛，才能被回测核验**。

## 发现的不匹配案例 (2026-06-30)

| predictions_log 编码 | 日期 | 对阵 | 500.com 实际返回 |
|---------------------|------|------|-----------------|
| 周二077 科特迪瓦 vs 挪威 | 2026-06-28 | 预测: 客胜 49.5% | **无此场** |
| 周二078 法国 vs 瑞典 | 2026-06-28 | 预测: 主胜 81.2% | **无此场** |
| 周二077 科特迪瓦 vs 挪威 | 2026-06-29 | 预测: 客胜 50.1% | **无此场** |
| 周二078 法国 vs 瑞典 | 2026-06-29 | 预测: 主胜 81.3% | **无此场** |
| 周二079 墨西哥 vs 厄瓜多尔 | 2026-06-29 | 预测: 客胜 35.5% | **无此场** |
| 周三080 英格兰 vs 刚果(金) | 2026-06-29 | 预测: 主胜 79.2% | **无此场** |
| 周三081 比利时 vs 塞内加尔 | 2026-06-29 | 预测: 主胜 72.0% | **无此场** |
| 周三082 美国 vs 波黑 | 2026-06-29 | 预测: 主胜 78.5% | **无此场** |

500.com 实际返回：
- 2026-06-28: 周日073 南非 vs 加拿大 (世界杯)
- 2026-06-29: 周一074 巴西 vs 日本、周一075 德国 vs 巴拉圭、周一076 荷兰 vs 摩洛哥 (均世界杯)

**核心问题**：predictions_log 用「周二/周三」编码套用到「周日/周一」的日期，且对阵完全不同。这 8 场是**竞彩未开售的国际友谊赛**，被错误录入。

## 处理规则

### 1. 标记为「竞彩未开售」而非补录比分

```csv
# predictions_log.csv 修改
actual_score -> "NOT_OFFERED"
checked -> "1"
actual_hda -> ""
actual_rq_result -> ""
actual_goals -> ""
actual_htft -> ""
```

**理由**：
- 从国际友谊赛源补比分无法验证让球/半全场/总进球等竞彩玩法
- 保留审计痕迹：能看到"曾预测但竞彩未开售"
- 回测脚本 `backtest_jczq.py` 已对 `actual_score` 为空或非标准比分的行自动跳过

### 2. 预防：daily_jczq.py 写入前校验

在 `record_prediction` / `build_prediction_bundle` 阶段，**只记录 500.com trade 页有赔率的场次**。  
如果某场只在 365scores/友谊赛列表里，没有 500.com 赔率，**不要写入 predictions_log**。

### 3. 回测脚本兼容性

`backtest_jczq.py` 的 `load_log()` 已处理：
- `actual_score` 为空 / "NOT_OFFERED" / "CANCELLED" → `checked != '1'` 视为未核验
- 只有 `checked == '1' AND actual_score` 为标准比分 (如 "1:0") 才进入准确率分母

## 验证命令

```bash
# 查看某日期 500.com 实际开售场次
cd /root/.hermes/scripts && python3 -c "
from backtest_jczq import fetch_results
for d in ['2026-06-28', '2026-06-29']:
    print(f'=== {d} ===')
    for r in fetch_results(d):
        print(f'  {r[\"code\"]} {r[\"home\"]} vs {r[\"away\"]} {r[\"score_full\"]}')
"

# 检查 predictions_log 中无法核验的场次
python3 -c "
import csv
from datetime import date
today = date.today().isoformat()
with open('/root/data/predictions_log.csv') as f:
    rows = list(csv.DictReader(f))
for r in rows:
    if r.get('date','') < today and not r.get('actual_score','').strip() and r.get('checked') != '1':
        print(f'{r[\"date\"]} {r[\"code\"]} {r[\"home_cn\"]} vs {r[\"away_cn\"]} -> 需标记 NOT_OFFERED')
"
```

## 关联文件

- `/root/.hermes/scripts/backtest_jczq.py` — `fetch_results()`, `load_log()`, `cmd_record()`
- `/root/data/predictions_log.csv` — 底账，含 `actual_score`, `checked`, `actual_hda` 等列
- `/root/.hermes/scripts/backtest_runner.sh` — 每日自动回测入口

## Pitfall Checklist

- [ ] 写入 predictions_log 前确认 500.com trade 页有该场赔率
- [ ] 回测前用 `fetch_results` 核对当日实际开售场次
- [ ] 友谊赛/热身赛若无竞彩赔率 → 标记 NOT_OFFERED，勿补国际赛比分
- [ ] 编码 (周二/周三) 必须与 500.com 返回的编码 (周日/周一) 一致