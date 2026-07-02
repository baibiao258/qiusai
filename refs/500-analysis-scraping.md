# 500.com 比赛分析数据爬虫 v2

## Purpose

从 500.com 指数中心抓取比赛的分析数据，注入到 daily_jczq.py 的预测输出中。
v2 重大升级：新增历史赔率矩阵（解决"合成赔率"问题的真实数据源）+ 完整未来赛程。

## Key Files

- **Module**: `/root/scraper_500_analysis.py` (管线集成, v2)
- **Cache**: `/root/data/500_analysis_cache.json` (1-hour TTL)
- **Standalone scraper**: `/root/scraper_500.py` (独立调试用, 非管线组件)

## URL Structure

```
https://odds.500.com/fenxi/shuju-{id}.shtml   # 数据分析页
```

- `{id}` 是 500.com 内部比赛 ID（如 1411007），不是竞彩编号
- 竞彩编号（如 周一201）在页面的 `hd_cz_box` 区域

## Encoding (CRITICAL)

500.com 使用 **gbk** 编码（不是纯 gb2312 —— 部分字符超出 gb2312 范围）。
必须按顺序尝试解码：`gb2312 → gbk → gb18030 → utf-8`

**Pitfall**: 直接用 `raw.decode('gb2312')` 会在 `position 128191` 附近崩溃（`illegal multibyte sequence`）。原因是页面含 GBK 扩展区字符。**必须用 fallback 链**。

## Match List Extraction

从任意比赛页的 `hd_cz_box` 区域提取同期全部竞彩比赛：

```python
pattern = (
    r'<span class="gray">(周[一二三四五六日]\d+)</span>.*?'
    r'href="/fenxi/shuju-(\d+)\.shtml"[^>]*>'
    r'.*?<em class="l">(.*?)</em>.*?<em class="r">(.*?)</em>'
)
```

返回 `dict[竞彩code] -> {'id': shuju_id, 'home': str, 'away': str}`

## v2 Extracted Fields (完整清单)

### 核心赔率 (CLV/EV 计算基础)

| 字段 | JSON key | 来源 | 说明 |
|------|----------|------|------|
| 当前欧赔 | `current_euro_odds` | bmatch行 `pub_table_pl` | `{home, draw, away}` float |
| 当前亚盘 | `current_asian_handicap` | bmatch行 `table_pl_center` | `{home_water, line, away_water}` |
| 历史欧赔 | `home_history[].euro_odds` | 数据分析表格 | 逐场 `{home, draw, away}` |
| 历史亚盘 | `home_history[].asian_handicap` | 数据分析表格 | 逐场 `{home_water, line, away_water, numeric}` |
| matchid | `matchid` | `<input name="matchid" value="65">` | 数据库关联主键 |
| hash | `hash` | `<input name="hash" value="55e090d">` | 页面校验 |

### 赛程元数据

| 字段 | JSON key | 来源 | 说明 |
|------|----------|------|------|
| 比赛时间 | `match_time` | `比赛时间2026-06-09 02:45` | |
| 联赛ID | `league_id` | `zuqiu-19472` | 500.com联赛编号 |
| 联赛名 | `league_name` | `26友谊赛六月` | |
| 未来赛事 | `future_fixtures[]` | 未来赛事区域 | `[{competition, date, home, away}]` |

### 战绩交叉验证

| 字段 | JSON key | 来源 | 说明 |
|------|----------|------|------|
| 主队历史 | `home_history[]` | team_zhanji1_1 | 10场逐场明细 |
| 客队历史 | `away_history[]` | team_zhanji1_0 | 10场逐场明细 |
| 战绩汇总 | `home_form` / `away_form` | bottom_info | `{team, wins, draws, losses, gf, ga}` |
| 赢盘率 | `home_record.cover_rate` | record_msg | 百分比 |
| 大球率 | `home_record.over_rate` | record_msg | 百分比 |
| 交战历史 | `h2h` | his_info | `{has_data, text}` |

### 历史战绩行结构 (每场)

```python
{
    "fid": "1393310",           # 500.com 比赛ID
    "league_id": "19472",       # 联赛ID
    "league_name": "友谊赛",
    "date": "26-06-04",
    "home": "荷兰",
    "away": "阿尔及利亚",
    "score": "0:1",             # None if not yet played
    "ht_score": "0:0",          # 半场比分
    "result": "负",             # 胜/平/负
    "panlu": "输",              # 赢/输/走 (盘路)
    "daxiao": "小",             # 大/小
    "euro_odds": {"home": 1.3, "draw": 5.2, "away": 8.66},
    "asian_handicap": {
        "home_water": 0.88,
        "line": "球半",
        "away_water": 0.90,
        "numeric": -1.5         # title属性提取
    }
}
```

### 辅助特征

| 字段 | JSON key | 说明 |
|------|----------|------|
| FIFA排名 | `fifa` | `{team: {rank, points}}` |
| 澳门推介 | `macau_tip` | 如 "和局" |
| 澳门理由 | `macau_reason` | 推介理由文字 |
| 首发阵容 | `home_lineup[]` | `[{number, name, position}]` |

## Integration Pattern

### Pipeline flow

```
main()
  ├─ scrape_500_odds_today()        # 赔率 (已有)
  ├─ scrape_500_analysis()          # 分析数据 (v2)
  │   ├─ _extract_match_list()      # 从500.com获取比赛列表
  │   ├─ _fetch(shuju_url)         # 逐场抓取分析页
  │   ├─ _parse_analysis(html)     # 解析HTML (v2: 含历史赔率)
  │   └─ _save_cache()             # 1小时缓存
  ├─ build_prediction_bundle()      # 构建预测bundle
  ├─ enrich_bundle_with_500()       # 注入分析数据 (v2: 含历史赔率+赛程)
  └─ print_match_bundle()           # 展示时输出分析行
```

### Enrichment function

`enrich_bundle_with_500(bundle, analysis)` 注入字段：
- 展示层: `fifa`, `home_form_500`, `away_form_500`, `home_record_500`, `away_record_500`, `h2h_500`, `macau_tip`, `macau_reason`, `home_lineup_500`, `asian_handicap_desc`
- **v2新增**: `home_history_500`, `away_history_500`, `matchid_500`, `current_euro_odds_500`, `current_asian_handicap_500`, `league_id_500`, `future_fixtures`

这些字段**不影响模型预测**，仅用于展示层和未来 CLV 回测。

### Display function

`format_500_analysis_lines(bundle)` 返回 `list[str]`，每行一个分析指标。
输出顺序: FIFA → 近10场 → 赢盘率/大球率 → 近3场赔率 → 交战 → 澳门 → 亚盘 → 世界杯赛程 → 首发

## Cache Strategy

- 文件: `/root/data/500_analysis_cache.json`
- TTL: 3600秒 (1小时)
- 缓存命中时终端显示: `📦 500.com分析缓存命中 (N场)`
- 清除缓存: `rm -f /root/data/500_analysis_cache.json`

## Pitfalls

1. **编码**: 必须用 `gb2312→gbk→gb18030→utf-8` fallback 链，不能只用 gb2312
2. **请求间隔**: 必须 ≥1.5秒，否则 500.com 可能封禁
3. **bmatch行提取**: 当前比赛行有 `style="display:none"`，但数据仍可提取 —— 用 `fid="{shuju_id}"` 定位而非 `class="bmatch"`
4. **FIFA积分正则**: 积分值(如1757)在独立的 `<tr>` 行中，不能和排名值(如7)用同一个 `<td>` 模式匹配。用 `section_m` 分段后再 `findall(r'<td>(\d{3,5})</td>')` 提取
5. **未来赛事section边界**: 用 `<h4>未来赛事</h4>` 作为起点，`<h4>` 或 `<div class="odds_msg">` 作为终点。不要用"预计阵容"作终点 —— 中间可能有"平均数据分析"等section
6. **友谊赛无联赛积分**: 赛前积分排名表为空，这是正常的
7. **历史赔率中缺失场次**: 部分友谊赛(如乌拉尔图)无赔率数据，`euro_odds` 为空dict，`asian_handicap.line` 为 "-"
8. **降级**: 分析数据缺失不影响预测输出，`enrich_bundle_with_500()` 在 analysis=None 时直接跳过

## 疲劳度特征 (fatigue_features.py)

**文件**: `/root/fatigue_features.py`
**集成**: 已接入 `daily_jczq.py` 两个分支 (友谊赛 + 联赛)

### 核心逻辑
- 输入: 未来赛事列表 (from 500.com `future_fixtures`) + 比赛日期 + 赛事类型
- 输出: `{home/away}_days_to_next`, `{home/away}_rotation_risk` (0-1), `{home/away}_fatigue` (0-1), `rotation_diff`
- 比赛重要度: 友谊赛=1, 预选赛/杯赛=2, 世界杯=3
- 轮换风险公式: 下一场重要度 × 距离衰减 (世界杯≤3天→95%, ≤7天→70%, ≤14天→40%)
- 概率调整: `rotation_diff ≥ 0.1` 时, 通过 `fatigue_adjustment()` 微调 H/D/A 概率 (最大±5pp)

### Pitfall: 队名前缀
500.com 的 `home_cn` 带 FIFA 排名前缀 (如 `"[7]荷兰"`)，而 `future_fixtures` 中是纯队名 (`"荷兰"`)。匹配时必须先 strip:
```python
import re
clean_home = re.sub(r'\[\d+\]', '', home_cn).strip()
```

### Pitfall: 日期格式
500.com 时间格式为 `"06-09 02:45"` (MM-DD HH:MM, 无年份)。`_parse_date()` 需要补年份:
```python
if len(date_str) == 11 and date_str[2] == '-' and date_str[5] == ' ':
    return datetime.strptime(f"2026-{date_str[:5]}", '%Y-%m-%d').date()
```

## CLV 回测 (clv_backtest.py)

**文件**: `/root/clv_backtest.py`
**用法**: `python3 clv_backtest.py --fetch` (抓取历史收盘赔率) 或 `python3 clv_backtest.py --report`

### 核心指标
- **CLV** = `(closing_odds / our_odds - 1)` — 正值=我们比市场更早发现价值
- **EV** = `prob × (odds - 1) - (1 - prob)` — 用我们的概率 vs 赔率
- 输出: 逐场明细 + 汇总统计 (平均CLV, 正CLV场次占比)

### 数据来源
- 我们的赔率: `predictions_log.csv` 的 `odds_h/d/a` (预测时抓取的500.com赔率)
- 收盘赔率: 通过 `search_500_match_id()` 搜索 + `fetch_match_closing_odds()` 抓取
- 缓存: `/root/data/500_historical_odds.json`

## 赛果回填 (backfill_results.py)

**文件**: `/root/backfill_results.py`
**用法**: `python3 backfill_results.py` (实际回填) 或 `--dry-run` (预览)

### 逻辑
1. 从 `predictions_log.csv` 找 `checked=0` 且 `date < today` 的比赛
2. 从 `500_analysis_cache.json` 获取 `shuju_id`
3. 抓取 500.com 页面提取比分 (`<p class="odds_hd_bf"><strong>2:1</strong>`)
4. 更新 CSV: `actual_score`, `actual_hda`, `actual_ht`, `actual_htft`, `actual_goals`, `actual_rq_result`, `checked=1`

### Pitfall: bmatch 行比分提取
已结束的比赛, bmatch 行的 `<em>` 标签内是比分 (如 `<em>0:1</em>`) 而非 "VS"。但 `style="display:none"` 不影响正则提取。
