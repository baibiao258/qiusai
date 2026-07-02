# 训练数据重建管道: 500.com 多源合并

## 背景

`training_data_with_odds.json` 是带赔率特征的训练数据集，供应 `retrain_xgb_with_odds.py` 训练 v30/v33 模型。原始数据仅 263 条（止于 2024-11），缺口在 2025-2026。

## 三源数据流

```
Source A: historical_kaijiang.csv
  └─ 开奖记录 (3248场), 含 spf_sp / rqspf_sp / handicap / spf_result
  └─ Join → international_results.json (通过 team_name_mapping 中→英)
  └─ 输出: ~360 条 (2024年为主)

Source B: trade.500.com/jczq?playid=269&g=2&date=
  └─ 竞彩赔率 API (当天开盘场), 含 nspf_3/1/0 + spf_3/1/0 + handicap
  └─ 支持历史日期 (当前赛季), 2026-01~06 已验证可用
  └─ 注意: data-matchdate 与 URL 日期可能差1天
  └─ 输出: ~150 条 (2026年5-6月为主)

Source C: 500_history_backfill.csv
  └─ 从 wanchang.php 回填的赛果 (63490条, 2026-01~06)
  └─ 作为 Source B 的赛果配对方
  └─ 一个 fid 配一个赛果 (score_full)

合并流程:
  A (prepare_training_data.py) → 输出 kaijiang 格式
  B + C (build_training_from_500.py) → 配对赔率+赛果 → 输出 500 格式
  merge_training_data.py → 去重合并 → training_data_with_odds.json (510条)
```

## 关键脚本

| 脚本 | 功能 | 调用方式 |
|------|------|---------|
| `/root/wc_2026_upgrade/prepare_training_data.py` | kaijiang + intl join | `python3 wc_2026_upgrade/prepare_training_data.py` |
| `/root/scripts/build_training_from_500.py` | trade.500.com + wanchang 配对 | `--start 2026-01-01 --end 2026-06-13 --quick` |
| `/root/scripts/merge_training_data.py` | 合并两源 | 自动执行 |

## trade.500.com 参数细节

- **playid**: 269=nspf+spf(基础), 270=jqs, 271=bf, 272=bqc
- **支持历史**: 当前赛季所有日期 (2026赛季已验证 2026-01-01 起)
- **不支持更早**: 2024/2025 返回 0 场
- **限速**: ≥0.3s 间隔

## 数据匹配策略

```python
def match_odds_with_result(odds_match, results):
    """三步匹配法"""
    # Step 1: 用 data-matchdate + 正向队名
    key = (match_date, normalize(home), normalize(away))
    row = results.get(key)
    
    # Step 2: 主客交换
    if not row:
        row = results.get((match_date, normalize(away), normalize(home)))
    
    # Step 3: ±1 天宽容 (处理跨日赛)
    if not row:
        for offset in [1, -1]:
            adj_date = (match_date + timedelta(offset)).isoformat()
            row = results.get((adj_date, h, a)) or results.get((adj_date, a, h))
    
    return row
```

## 2025 年赔率缺口确认 (2026-06-15)

### 问题

training_data_with_odds.json 有 2024(339条) + 2026(171条)，但 **2025 全年赔率为零**。这是训练数据最大单一时段缺口。

### 三条路径探查结果

| 路径 | 结论 | 根因 |
|------|------|------|
| **kaijiang.csv** | 无 2025 数据 | historical_kaijiang.py 从 `zx.500.com/kaijiang.php?d=YYYY-MM-DD` 抓取，2025 日期返回空表（500.com 仅保留最近2赛季，2024 和 2026） |
| **trade.500.com** | 仅当前赛季 | `playid=269&g=2&date=YYYY-MM-DD` 对 2025 日期返回 0 场。500.com 竞彩赔率只保留当前赛季 (2026-01 起)。2025 赛季已归档不可查 |
| **sporttery.cn (webapi)** | WAF 拦截 | `https://webapi.sporttery.cn/gateway/lottery/getMatchResultV2.qry` 被腾讯 EdgeOne WAF 拦截，返回 **HTTP 567**（非 403/404）。所有请求路径均被识别为 bot 自动拦截 |
| **czl0325 API** | 有比赛元数据，无赔率 | `http://117.72.172.8:10008/schedule/list?type=2&afterDate=...` 返回 2024-2026 每日比赛列表（含联赛/队名/时间），但 `/odds/list` 对已结束比赛返回空列表。赔率仅赛前可查，赛后被清除 |

### 结论

**2025 年赔率数据暂时不可达**。三个独立数据源要么不保留该时段数据，要么被 WAF 封禁。无可用回填路径。

### 影响

- training_data_with_odds.json 的 2025 缺口**无法填补**
- xgb_model_33 (含 market_implied 特征) 只能从 2024 和 2026 学习，跳跃 2025 年的分布变化
- 场景价值: 2025 有大量国际赛（世预赛、欧国联决赛阶段、友谊赛）。模型对这些比赛的 market_implied 特征完全来自 2024/2026 的外推
- 缓解: 继续按日积累 2026 数据，当 2026 数据达到 400+ 场后，2024 的相对权重自然降低。2025 缺口通过 time-weighted sampling 缓解

## trade.500.com 正则解析陷阱

`build_training_from_500.py` 用纯正则解析 trade.500.com 的 HTML，以下陷阱已被确认修复:

| 陷阱 | 错误写法 | 正确写法 | 原因 |
|------|---------|---------|------|
| **Hyphen 属性名** | `(\w+)=` | `([\w-]+)=` | data-fixtureid, data-matchdate, data-simpleleague, data-matchnum 均含连字符 |
| **fixtureid 必须 required** | `(?:data-fixtureid="(\d+)")?` (optional) | `data-fixtureid="(\d+)"` (required) | optional 导致当某行无 fixtureid 时，整个 regex 匹配跳过，丢失该行后面的所有数据 |
| **赔率范围** | 只在 opening tr tag 内搜 | 在 `tr[after_open:close_tr]` 范围搜 | 赔率 `<p class="betbtn" data-type=...>` 在 `<tr>` 和 `</tr>` 之间，不在开头标签内 |
| **data-matchdate 位置** | 用 URL date 参数匹配 | 用 `data-matchdate` 属性值匹配 | trade.500.com 用开盘日期在 URL，但实际比赛日期在 `data-matchdate`，跨日赛 (00:00) 差 1 天 |
| **队名含排名前缀** | 直接比对队名 | 先 strip `[\\d+]` 前缀 | 500.com 队名含 FIFA 排名如 `[7]荷兰`，与 wanchang 或其他数据源的纯队名不匹配 |

### 解析代码骨架 (已验证)

```python
_ATTR_RE = re.compile(r'\s([\w-]+)="([^"]*)"')  # 注意 [\w-] not \w

_ROW_RE = re.compile(
    r'<tr[^>]*data-fixtureid="(\d+)"[^>]*>'
    r'(.*?)</tr>',
    re.DOTALL
)

def parse_trade_html(html):
    rows = _ROW_RE.findall(html)
    for fid, content in rows:
        attrs = dict(_ATTR_RE.findall(content))
        # attrs['data-type'] → 'nspf', 'spf', etc
        # attrs['data-value'] → '3', '1', '0'
        # attrs['data-sp'] → '2.74'
        type_key = attrs.get('data-type')
        val_key = attrs.get('data-value')
        sp_key = attrs.get('data-sp')
```

## 输出格式统一

500 trade 格式 → kaijiang 格式 转换:

| 500 字段 | kaijiang 字段 | 转换 |
|---------|-------------|------|
| nspf_3 | spf_sp | 直接映射 |
| spf_3/1/0 中最高值 | rqspf_sp | max(3,1,0) |
| spf_3/1/0 | — | 保留作为让球赔率 |
| nspf_3/1/0 → implied_prob | market_implied_prob | 1/o / sum(1/o) 去水 |

## 输出验证

```bash
python3 -c "
import json
d = json.load(open('/root/data/training_data_with_odds.json'))
print(f'{len(d)} 条')
from collections import Counter as C
print(C(s['date'][:4] for s in d))
"
```
