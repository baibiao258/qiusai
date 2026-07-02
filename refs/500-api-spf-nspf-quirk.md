# 500.com API playid=269: spf/nspf 字段含义 (2026-06-08发现, 2026-06-09补充)

## Bug Summary (2026-06-08 发现, 2026-06-09 修复)

`fetch_500_market.py`（**已废弃**，参见 `references/async-scraper-architecture.md`）通过 playid=269 从 500.com API 拉取"胜平负"赔率时，`spf` 字段 **在有让球(rangqiu≠0)时不是标准1X2**，而是让球胜平负。真正的标准1X2在 `nspf` 字段。
`spf` 字段 **在有让球(rangqiu≠0)时不是标准1X2**，而是让球胜平负。
真正的标准1X2在 `nspf` 字段。

**影响**: 管线中所有30条 predictions_log 记录的 odds_h/d/a 全是让球赔率。
EV计算用模型的标准概率去对撞让球赔率，全部算错。

## 2026-06-09 修复：live.500.com 平均欧赔兜底

`apply_euro_fallback()` 因分析缓存冷启动问题几乎不生效。2026-06-09 改为 `_fetch_live_odds_map()`：

```python
def _fetch_live_odds_map():
    """从 live.500.com 获取平均欧赔兜底数据。
    返回: dict[code] -> {'h':float, 'd':float, 'a':float} 或 None
    """
    # 1. 请求 live.500.com (GBK编码)
    # 2. 提取 liveOddsList JS变量: fid -> {'0': [h,d,a]}
    # 3. 从HTML构建 code→fid映射: value="fid"/>周二201</td>
    # 4. 返回code→{'h','d','a'}字典
```

**原理**：`liveOddsList` 的 key `'0'` 是多家博彩公司对标准1X2的平均赔率，不受竞彩让球影响。

**验证方法**：
```bash
cd /root && python3 -c "
from daily_jczq import _fetch_live_odds_map
r = _fetch_live_odds_map()
print(f'{len(r)} 场兜底数据')
for k in ['周六005','周二203','周日009']:
    print(f'  {k}: {r[k][\"h\"]}/{r[k][\"d\"]}/{r[k][\"a\"]}')"
```

## 旧方案 (已废弃)

### 方案 A (推荐): 字段映射修正 (已包含在 daily_jczq.py 中)

### 方案 B: 欧赔回退 (已废弃，改用 live.500.com)

## 影响面 (2026-06-09 已修复)

当前预测管线中，当 nspf 为空时自动使用 `_fetch_live_odds_map()` 获取平均欧赔兜底。`std_odds_source` 字段标记为 `'live_euro_avg'` 表示使用了兜底方案。

### Evidence 1: 荷兰 2:1 乌兹别克 (周一201, shuju_id=1411007)

### 证据 1: 荷兰 2:1 乌兹别克 (周一201, shuju_id=1411007)

```
rangqiu = -2 (荷兰-2)
比分: 荷兰 2:1 乌兹别克 → 标准荷兰胜

spf (被当前代码当作1X2): {'0':'2.21', '1':'3.34', '3':'2.65'}
  '3'=2.65 = 荷兰-2让球胜 → 需要赢3+ → 实际2:1→不中 ✗
  如果这是标准1X2: 荷兰胜 at 2.65 → 中了但赔率不合理(排名7 vs 58)

nspf: {'0':'12.50', '1':'5.85', '3':'1.14'}
  '3'=1.14 = 荷兰标准胜 → 中了 ✓
  欧赔验证: 1.22 (500.com分析页百家欧赔)
  nspf 1.14 ≈ 欧赔 1.22  ← 合理!
```

### 证据 2: 法国 3:1 北爱尔兰 (周一202, shuju_id=1410357)

```
rangqiu = -2 (法国-2)
比分: 法国 3:1 北爱尔兰 → 标准法国胜, 让球-2→1:1=平

spf: {'0':'3.05', '1':'3.96', '3':'1.82'}
  '3'=1.82 = 法国-2让球胜 → 实际恰好赢2球→让球平→不中 ✗

nspf: {}  ← 空! 没有反向让球数据

欧赔: 1.12 (500.com分析页)
```

### 证据 3: 中国 vs 泰国 (周二201, rangqiu=-1)

```
spf: {'0':'1.82', '1':'3.35', '3':'3.55'}
  '3'=3.55 for 中国胜 (rank 90 vs 泰国 rank 97)
  看起来合理: 中国主场但泰国略强, 客场1.82是热门

nspf: {'0':'3.95', '1':'3.28', '3':'1.75'}
  '3'=1.75 for 中国-1让球胜 (更强烈的热门)
  注意: nspf 过round=112.9% = 标准竞彩抽水
```

**关键辨析**: 当 rangqiu=-1 时, spf 看起来还像标准1X2。
区别在 rangqiu≥2 时才明显 (因为让球大, 让球胜赔率会显著偏高)。

## 修复方案

### 方案 A (推荐): 字段映射修正

修改 `daily_jczq.py` 的 `scrape_500_odds_today()`:

```python
# 当前(错误):
'odds_h': to_float(spf.get('3')),   # 让球胜平负
'odds_d': to_float(spf.get('1')),
'odds_a': to_float(spf.get('0')),
'rq_h': to_float(nspf.get('3')),    # 反向让球
'rq_d': to_float(nspf.get('1')),
'rq_a': to_float(nspf.get('0')),

# 修复后:
_1x2 = nspf if (int(rq) != 0 and nspf) else spf  # 标准1X2来源
_hcap = spf if int(rq) != 0 else {}               # 让球来源

'odds_h': to_float(_1x2.get('3')),
'odds_d': to_float(_1x2.get('1')),
'odds_a': to_float(_1x2.get('0')),
'rq_h': to_float(_hcap.get('3')),
'rq_d': to_float(_hcap.get('1')),
'rq_a': to_float(_hcap.get('0')),
```

### 方案 B: 欧赔回退

对 nspf 为空的比赛(强队让2+球)：
- 阿根廷 vs 冰岛 rq=-2
- 西班牙 vs 佛得角 rq=-2
- 德国 vs 库拉索 rq=-3
- 卡塔尔 vs 瑞士 rq=+2
- 伊拉克 vs 挪威 rq=+2

使用 `scraper_500_analysis.py` 当前欧赔(current_euro_odds)作为标准1X2。
当 nspf 和 欧赔 都不可用时的兜底：用 `bet_math.py` 的 overround 反推。

## 影响面

当前 predictions_log.csv 中 **全部 30 条记录的 odds_h/d/a 都错了** (均 rq≠0)。
这意味着：
- 所有 EV 计算是错的
- 所有 CLV 计算是错的
- 所有 Kelly 仓位推荐是错的
- 所有 "价值投注" 推荐不靠谱

修复后需要重跑 `python3 daily_jczq.py` 生成新的预测日志。
