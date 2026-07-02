# 500.com wanchang 完场比分抓取 (2026-06-15 确认)

## URL

```
https://live.500.com/wanchang.php?e=YYYY-MM-DD
```

## 编码

**必须用 GBK 解码**, 不是 UTF-8。

```python
import subprocess
r = subprocess.run(['curl', '-s', url, '--max-time', '30',
    '-H', 'User-Agent: Mozilla/5.0'],
    capture_output=True)
html = r.stdout.decode('gbk', errors='replace')
```

## 与之前错误判断的关系

- **不是 SPA**: 页面是服务器端直接输出的静态 HTML，不含动态 JS 渲染
- **不是 WAF 阻拦**: 之前 `type=1` 参数返回不同页面; 直接用 `e=YYYY-MM-DD` + 标准 User-Agent 即可通过
- **requests 超时**: Python `requests` 包对该端点间歇性超时(30s+), 改用 `subprocess.run(['curl', ...])` 稳定(2-3s)

## 页面结构

### 主行 `<tr id="a{fid}" ... lid="{league_id}">`

```html
<tr id="a1359224" gy="世界杯,韩国,捷克" yy="世界杯,南韓,捷克" lid="110" class="">
  <td bgcolor="#0000FF">世界杯</td>          ← 联赛名
  <td>第1轮</td>                             ← 轮次
  <td>06-12 10:00</td>                       ← 日期时间
  <td><span class="red">完</span></td>       ← 比赛状态
  <td>...<span class="mainName">韩国</span>...</td>    ← 主队
  <td><div class="pk">...让球盘...</div></td>
  <td>...<span class="clientName">捷克</span>...</td>  ← 客队
  <td align="center" class="red">1 - 1</td>  ← 全场比分
</tr>
```

### 子行 (parentid)

部分比赛附带 `parentid` 子行, 包含扩展信息:

```html
<tr parentid="a1414609" style="text-align:center">
  <td colspan="9">
    <font color="#A52A2A">90分钟[1-1]</font>
    <font color="#A52A2A">120分钟[0-0]</font>
    <font color="#A52A2A">点球[5-4]</font>
  </td>
</tr>
```

注意: parentid 行在 `e=YYYY-MM-DD` 页面上**极少出现**(仅 ~2%), 大量比赛没有 parentid 行。

### 联赛过滤

页面左侧有 checkbox 列表:

```html
<input type="checkbox" id="ckl110" value="110"> 世界杯
```

每个联赛有唯一 `lid` (数值 ID)。

## 正则解析

### 主行匹配

```python
for m in re.finditer(
    r'<tr\s+id="a(\d+)"[^>]*\blid="(\d+)"[^>]*>(.*?)</tr>',
    html, re.DOTALL
):
    fid, lid, row = m.groups()
```

关键注意事项:
- `lid` 属性格式是 `lid="110"` 而非 `lid110`, 正则 `\blid(\d+)` 会失败, 必须用 `lid="(\d+)"`
- `class="red"` 是比分单元格。比分格式 `"2 - 1"` (含空格)。注意区分第一个 `class="red">完</span>` (状态) 和第二个 `class="red">X - Y</td>` (比分)

### 比分格式

```
"2 - 1" → split 得 [2, 1]
```

跳过含 `:` 的比分(未开始/进行中)。

## 生产脚本

`/root/scripts/fetch_500_wanchang.py` (2026-06-15 最终版)

CLI:
```bash
# 单天预览
python3 fetch_500_wanchang.py --date 2026-06-12

# 批量回填 (到 CSV)
python3 fetch_500_wanchang.py --start 2026-01-01 --end 2026-06-13

# 只抓指定联赛
python3 fetch_500_wanchang.py --start 2026-06-01 --end 2026-06-13 --leagues 110 558 557

# 合并到 international_results.json
python3 fetch_500_wanchang.py --start 2026-01-01 --end 2026-06-13 --merge
```

## 数据量

- 2026-06-12 单天: 103 场
- 2026-01-01 ~ 2026-06-13: 63,490 场（含各级别联赛）
- 周末(周六): 400-1300 场/天
- 非高峰: 60-200 场/天

## 与 365scores 对接

通过 `(date, home, away)` 三要素 JOIN。队名需经 `team_name_normalizer.py` 标准化。

## 训练数据断档 (training_data_with_odds.json)

**当前最大瓶颈**: 含竞彩 SPF 赔率的训练数据仅 263 场 (止于 2024-11)。
wanchang 提供赛果但不提供赔率。要补 2025-2026 有赔率的训练数据, 需:
1. 用 `fetch_500_wanchang.py` 回填赛果
2. 从 `odds.500.com/fenxi/shuju-{fid}.shtml` 拉历史赔率 (按 fid)
3. 按 fid 或 (date, team) JOIN 赛果+赔率
