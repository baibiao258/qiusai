# TheStatsAPI 第4数据源赛果回填 (2026-06-17, 2026-06-19 修复)

## 背景

`backfill_results.py` 原有3个赛果数据源（results JSON / kaijiang / 365scores），覆盖 ~75.7%。新增第4源 TheStatsAPI 作为超兜底，专门处理前3源都无法匹配的比赛。

## 数据流

```
backfill()  →  3源优先  →  thestats 源  →  Brier Score  →  Elo增量更新
                                  │
                                  ▼
                    GET /api/football/matches
                    ?date=YYYY-MM-DD&status=finished
                    Authorization: Bearer fapi_***
```

## 关键实现 (`match_from_thestats` + `_fetch_all_thestats_matches`)

### 全局缓存 + 翻页 (2026-06-19 重构)

**问题**: TheStatsAPI 默认每页仅返回 20 场比赛，且 `date` 参数失效（无论传什么日期都返回相同数据）。旧版逐条请求 + `date_window` 只能搜到前 20 场，遗漏大多数目标比赛。

**修复**: 全局缓存 `_fetch_all_thestats_matches()`:

```python
def _fetch_all_thestats_matches():
    """翻页获取全部比赛数据，全局缓存。"""
    all_matches = []
    for page in range(1, 50):           # 翻到空为止
        url = f"{BASE}/matches?per_page=100&page={page}"
        data = requests.get(url).json().get("data", [])
        if not data: break
        all_matches.extend(data)
    return all_matches
```

**关键发现**: TheStatsAPI 的 `date` 查询参数**静默失效**——任意日期返回完全相同的数据集（疑似 API 端 bug）。`per_page=100` 是有效最大页大小（`per_page=200/500` 返回 0）。`match_from_thestats()` 现在不再按日期过滤，而是从全局缓存中匹配队名 + `status=finished` 过滤。

### 双路线队名匹配

```python
def match_from_thestats(row, en_to_cn, _print_once=set()):
    """从 TheStatsAPI 全局缓存中匹配一场比赛。

    双路线:
    1. 中文名路线: cn_name → en_to_cn 反向映射 → 比对 games 的 home_en/away_en
    2. 英文名路线: eng_name 直比对 games 的 home_en/away_en（含 strip + lower）
    """
```

**为什么双路线？** CSV row 的 `home`/`away` 列可能混有中英文：
- 500.com 爬取 → 中文名（"塞内加尔"）
- TheStatsAPI 兜底 → 需要英文队名匹配
- `en_to_cn` 字典（`team_name_mapping.json` 的 value→key 反转）做反向映射

### 2026-06-19 新特性

1. **全局翻页缓存** — 替换了逐日请求 + `date_window`。单次调用翻 49 页拉取 4,900 场比赛（含 4,623 完赛），匹配命中率从 ~0 提升到全覆盖。
2. **`normalize_en_name()` 加变音符号去除** — 用 `unicodedata.normalize('NFKD', name).encode('ascii', 'ignore')` 处理 `Türkiye`→`turkiye`、`Côte d'Ivoire`→`cote d ivoire`、`Strømsgodset`→`stromsgodset`。同时处理 `&→and`。
3. **`_is_chinese()` 自动方向检测** — `load_team_name_map()` 不再强制所有条目为 `"中文": "英文"`。检测 key/value 哪边含中文，自动判断条目方向。反向条目 `"Czechia": "捷克"` 正确解析为 `en_to_cn["czechia"]="捷克"`。
4. **`normalize_en_name()` 扩展** — 统一英文格式：`&→and` + 变音符号剥离。消除了 `&` vs `and` 和 `ü` vs `u` 的匹配差异。

### 未命中警告（去重）

```python
def _print_once(msg, severity='info'):
    """避免同一比赛名在日志中反复打印同一警告。"""
```

每当比赛名在 `en_to_cn` 中找不到时，打印 `⚠️ [需补充字典] team_name_mapping.json 缺少 {eng_name}`。使用 `_print_once` 确保每条唯一消息只打印一次。

**典型未命中示例**（非竞彩球队，不影响核心覆盖）：
- Universidad de Chile, Sport Recife, Wydad Casablanca, AC Oulu

## 依赖

- `THE_STATS_KEY` 环境变量（Bearer token）
- 端点 `/api/football/matches?date=...&status=finished`
- Team name mapping: `/root/data/team_name_mapping.json` (152条)
- `normalize_en_name()` 函数 (`backfill_results.py`)

## 当前覆盖

- 修复前 (2026-06-18): en_to_cn 方向错误导致 Czechia→捷克、USA→美国、Bosnia & Herzegovina→波黑 等匹配全部静默失败
- 修复后 (2026-06-19): 方向检测修复，所有反向条目正确匹配

## 坑

1. **TheStatsAPI `date` 参数静默失效** (2026-06-19) — 无论传 `date=2026-06-14` 还是 `date=2026-06-20` 都返回完全相同的数据集。解决方案：已弃用按日期过滤，改用 `_fetch_all_thestats_matches()` 全局缓存 + `status=finished` 过滤。
2. **`per_page=100` 是页大小上限** — `per_page=200` 或 `per_page=500` 返回空数据。分页必须用 `per_page=100` 循环 `page=1..N` 直到空。
3. **需要翻 5+ 页** (2026-06-19) — 当前数据集约 4,900 场（49 页），其中 4,623 场完赛。默认 20 条/页只能覆盖前 20 场（≈0.4%），必须翻页。
4. **`_print_once` 作用域**：函数级 mutable default 参数 `_print_once=set()`。多次调用 `backfill()` 时不清空，但在单次回填中有效。注意 mutable default 参数的 Python 陷阱（set 累加）。
5. **同一天多场比赛**：`match_from_thestats()` 匹配第一个符合的，不处理同一日期同一组合的多场。如出现同一配对在同一日期有多个结果，只取第一个匹配。
6. **⚠️ 映射方向陷阱 (2026-06-18 → 2026-06-19 已修复)**: 旧版 `load_team_name_map()` 把所有 JSON 条目当 `"中文": "英文"` 解析，`"Czechia": "捷克"` 等反向条目的 en_to_cn 键变成中文。**已修复**: 新增 `_is_chinese()` 自动检测方向。`team_name_mapping.json` 现在可以自由混用两种格式。验证：`python3 -c "from backfill_results import load_team_name_map; _, en_to_cn = load_team_name_map(); print('czechia' in en_to_cn)"` 应返回 True。
7. **`backfill_results.py` 依赖 `import subprocess`**：脚本末尾调用 `retrain_poisson_elo.py incremental` (增量 Elo 更新) 用 `subprocess.run()`。新创建脚本或大幅重写时容易遗漏此 import，导致 `NameError: name 'subprocess' is not defined`。在文件顶部 import 区域加 `import subprocess`。
8. **变音符号差异** (2026-06-19): TheStatsAPI 返回 `Türkiye`（土耳其语）, mapping 存 `Turkey`（英语）。`normalize_en_name()` 通过 NFKD + ASCII 编码剥离变音符号解决。同样处理 `Côte d'Ivoire`→`Ivory Coast`, `Strømsgodset`→`Stromsgodset`。
9. **`per_page=100` 有时返回 0** — API 在浏览器/curl 中可用但在 Python requests 中返回空，通常是网关超时。加 `timeout=30` 和重试。`_fetch_all_thestats_matches()` 实际翻到第 49 页（4900 场）后返回空才停止。
