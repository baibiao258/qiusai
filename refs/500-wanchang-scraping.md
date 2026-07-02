# 500.com wanchang.php 历史完场数据抓取

## 背景

500.com 的 `live.500.com/wanchang.php?e={date}` 提供指定日期的完场比分页面，覆盖全球所有足球联赛（含世界杯、五大联赛、亚洲联赛、低级联赛等），200KB+ 纯 HTML，**不是 SPA**。

## 前期错误诊断 (2026-06-13)

- **错误结论**: WAF 拦截 (HTTP 566), 纯 HTTP 无法获取
- **错误根因推断**: SPA WebSocket 动态加载
- **真实根因**: `python requests` 库对此站点超时 (30s hang)，但 `curl` 2s 出结果
- **教训**: 不要只凭一个工具失败就下结论。换 curl 验证再断言。

## 修复方案

### 关键改动

1. 用 `subprocess.run(['curl', ...])` 替代 `requests.get()`
2. 编码: `r.stdout.decode('gbk', errors='replace')` (不能用 `r.text` 或 `r.content.decode('utf-8')`)
3. 正则提取比分: 比分在 `<td align="center" class="red">X - Y</td>` 中 (有空格)
4. 半场比分在子行: `<tr parentid="a{fid}">...<font color="A52A2A">90分钟[X-Y]</font>`

### HTML 结构

```html
<tr id="a1359224" gy="世界杯,韩国,捷克" yy="世界杯,南韓,捷克" lid="110" class="">
    <td bgcolor="#0000FF" class="ssbox_01">...世界杯...</td>
    <td align="center">第1轮</td>
    <td align="center">06-12 10:00</td>              ← 日期时间
    <td align="center"><span class="red">完</span></td>
    <td align="right" ...><span class="mainName">韩国</span></td>   ← 主队
    <td align="center"><div class="pk">...让球盘...</div></td>
    <td align="left" ...><span class="clientName">捷克</span></td>  ← 客队
    <td align="center" class="red">0 - 0</td>        ← 全场比分!
</tr>
```

### 核心正则

```python
re.findall(
    r'<tr id="a(\d+)"[^>]*lid="(\d+)"[^>]*>.*?'
    r'<td align="center">([^<]*)</td>\s*'
    r'<td align="center">(\d{2}-\d{2}\s+\d+:\d+)</td>.*?'
    r'<span class="mainName[^"]*">([^<]+)</span>.*?'
    r'<span class="clientName[^"]*">([^<]+)</span>.*?'
    r'class="red"[^>]*>\s*(\d+)\s*-\s*(\d+)\s*</td>',
    html, re.DOTALL
)
```

### 半场比分

实际格式为 `90分钟[1-1]` (不是 `45 1-0`)。在 `parentid` 子行 `<tr parentid="a{fid}">` 中。但 **parentid 行极少出现**（2026-06-12 的 103 场中仅 2 场有），所以半场比分作为增强字段而非必需字段处理。

### 联赛过滤

页面左侧有 `<input type="checkbox" id="ckl{league_id}" value="{league_id}">`，但 500.com 的 league_id 映射与竞彩联赛体系不一致。建议: 所有比赛全拉，脚本端按 league_id 或队名过滤。

### 关键陷阱

| 陷阱 | 说明 |
|------|------|
| `requests` 超时 | 必须用 `subprocess.run(['curl', ...])`，requests 在此站点 hang |
| GBK 编码 | 必须 `raw.decode('gbk')`，不能用 `r.text` |
| lid 属性格式 | `lid="N"` 带等号和引号，不是 `lidN` |
| 比分格式 | `"2 - 1"` 含空格，`\s*\d+\s*-\s*\d+` |
| parentid 行罕见 | 半场比分在 `90分钟[1-1]` 格式，不是 `45 1-0` |
| 请求频率 | 批量回填时加 `time.sleep(0.5)` 防 429 |

## 脚本

`/root/scripts/fetch_500_wanchang.py` — 完整的单天/批量抓取 + CSV 保存 + 合并到 international_results.json。

### 命令行接口

```bash
# 单天预览
python3 scripts/fetch_500_wanchang.py --date 2026-06-12

# 按联赛过滤
python3 scripts/fetch_500_wanchang.py --date 2026-06-12 --leagues 110 558 557

# 批量回填 (2026-01-01 到 2026-06-13)
python3 scripts/fetch_500_wanchang.py --start 2026-01-01 --end 2026-06-13

# 批量 + 合并到 international_results.json
python3 scripts/fetch_500_wanchang.py --start 2026-01-01 --end 2026-06-13 --merge
```

### 验证结果

2026-06-12 测试: 99 场完场数据全部正确解析。

## 关联: trade.500.com 历史赔率 API

`trade.500.com/jczq/?playid={id}&g=2&date=YYYY-MM-DD` 支持**历史日期查询**（非必须当天）。返回当天的竞彩开盘场次赔率，包含 `nspf` (标准胜平负) 和 `spf` (让球胜平负) 的 `data-type/data-value/data-sp` 属性。

关键发现: 日期参数用**开盘日期**，但比赛 `data-matchdate` 属性是实际比赛日期（常差1天，如 00:00 跨日赛）。匹配赛果时要用 `data-matchdate` 而非 URL 日期，且需做 ±1 天宽容匹配。

HTML 结构:
```html
<tr class="bet-tb-tr" data-fixtureid="1412367" data-homesxname="保加利亚"
    data-awaysxname="黑山" data-matchdate="2026-06-02" data-rangqiu="1"
    data-simpleleague="国际赛" data-matchnum="周一001">
  <p class="betbtn" data-type="nspf" data-value="3" data-sp="2.74">...</p>
  <p class="betbtn" data-type="nspf" data-value="1" data-sp="2.65">...</p>
  <p class="betbtn" data-type="spf" data-value="3" data-sp="1.36">...</p>
</tr>
```

注意: trade.500.com 的 `fixtureid` 与 wanchang 的 `fid` 是不同的 ID 体系。配对要用 `(date, home, away)` 而非 ID。
