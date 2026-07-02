# 500.com 异步并发赔率抓取架构 (2026-06-09)

## 背景

旧方案使用 Node.js + jsdom (`fetch_500_market.py` → `scrape_500_market.js`)，按 playid 顺序调用 4 次（子进程），每次 ~2-3 秒，总耗时 8-12 秒。JS 脚本需维护独立打包的 jsdom 依赖。

新方案使用 Python aiohttp + BeautifulSoup，4 页面并发 ~2 秒完成，统一 DOM 解析，按 fixtureid 自动合并。

## 核心文件

- `/root/wc_2026_upgrade/async_500_scraper.py` — 异步并发爬虫
- `/root/daily_jczq.py` → `scrape_500_odds_today()` — 调用入口

## URL 体系

所有 URL 基于 `https://trade.500.com/jczq/`，参数：

| 参数 | 值 |
|------|-----|
| `playid` | 269(胜平负+让球), 270(总进球), 271(比分), 272(半全场) |
| `g` | 2 (混合过关) |
| `date` | YYYY-MM-DD |
| `_t` | 毫秒级时间戳（缓存穿透） |

## DOM 结构

竞彩页面 DOM 高度统一，核心属性：

| 属性 | 说明 |
|------|------|
| `tr class="bet-tb-tr"` | 赛事行容器 |
| `data-fixtureid` | 赛事唯一内码（合并主键） |
| `data-matchnum` | 场次编号（如 "201"） |
| `data-rangqiu` | 让球数（如 "-1"） |
| `data-matchdate` | 比赛日期 |
| `data-simpleleague` | 联赛名 |
| `data-sp` | 赔率值（float） |
| `data-type` | 玩法类型：nspf/spf/bf/jqs/bqc |
| `data-value` | 玩法选项：3/1/0, 1:0, 3-3, 0-7 等 |

`bet-more-wrap`：比分(bf)等复杂玩法在紧邻主行的 `<tr class="bet-more-wrap">` 中，通过 `row.find_next_sibling('tr', class_='bet-more-wrap')` 定位。

## 编码

500.com 页面使用 GBK 编码。aiohttp 返回原始 bytes，解码方式：
```python
try:
    html = raw.decode('gbk')
except UnicodeDecodeError:
    html = raw.decode('gbk', errors='replace')
```

## 解析流程

1. `_build_urls(date_str, playids)` — 生成 4-6 个带时间戳的 URL
2. `_fetch_one(session, url_info, sem)` — 带 3 次重试的异步请求（Semaphore 限流）
3. `_parse_html(html, playid)` — BeautifulSoup 统一解析：
   - 遍历所有 `tr.bet-tb-tr`
   - 提取 `data-fixtureid` / `data-rangqiu` / `data-matchnum`
   - 从 `td.td-team` 提取球队名（格式：`主队 VS 客队`）
   - 在主行和 bet-more-wrap 中统一搜索含 `data-sp` 属性的节点
   - 按 `data-type` 分组存入 `odds` 字典
4. `_merge_matches(base, incoming)` — 按 fixtureid 深度合并 odds
5. 输出兼容旧版格式的 list[dict]

## 与旧系统关键差异

| 维度 | 旧 Node.js | 新 Python |
|------|-----------|-----------|
| 依赖 | Node.js, jsdom 6MB+ | Python stdlib + aiohttp + bs4 + lxml |
| 请求方式 | 顺序 4 次子进程 | 并发 aiohttp |
| 耗时 | 8-12 秒 | ~2 秒 |
| 解析 | 各 playid 独立函数 | 统一 data-sp 遍历 |
| 合并 | Python 端多 markets 字典拼接 | 服务端按 fixtureid 自动合并 |
| 错误处理 | 单次失败=>整轮重试 | 每 URL 独立重试 3 次 |

## 下游数据消费

`scrape_500_odds_today()` 中从统一 dict 提取各玩法赔率时的 key 映射：

```python
odds = row.get('odds', {})
spf_raw = odds.get('spf', {})      # 让球胜平负 (handicap≠0) 或标准1X2 (handicap=0)
nspf_raw = odds.get('nspf', {})    # 标准1X2 (当 handicap≠0 时有值)
bf_data   = odds.get('bf', {})     # 比分, raw key: '1:0', '2:0', ...
jqs_data  = odds.get('jqs', {})    # 总进球, raw key: '0','1',...,'7'
bqc_data  = odds.get('bqc', {})    # 半全场, raw key: '3-3','3-1',...
```

后向兼容映射：
- 半全场: raw → 中文标签（`{'3-3':'胜胜', '3-1':'胜平', ...}`） via `HTFT_RAW_MAP`
- 总进球: raw → `{N}球`（`'0'→'0球'`） via `f"{int(k)}球"`
