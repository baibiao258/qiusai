# trade.500.com Regex 解析陷阱集 (2026-06-15)

## 背景

trade.500.com 交易页的 HTML 结构在竞彩抓取中用于获取赔率(nspf/spf/handicap)。以下是从 `build_training_from_500.py` 开发中总结的 regex 关键坑。

## Regex 要点

### 1. `<tr>` 行匹配: data-fixtureid 必须 required

```python
# ❌ 错误: data-fixtureid 写成可选 (?:...)? → 贪婪 [^>]* 会在首段过早关闭
r'<tr[^>]*(?:data-fixtureid="(\d+)")?[^>]*>(.*?)</tr>'

# ✅ 正确: data-fixtureid 作为必选部分
r'<tr[^>]*data-fixtureid="(\d+)"[^>]*>(.*?)</tr>'
```

**原理**: `[^>]*` 是贪婪的, 如果 data-fixtureid 写成可选组, 第一个 `[^>]*` 会匹配到第一个 `>` 就停止, 吞掉 id 属性。

### 2. 属性值正则: 必须支持连字符

```python
# ❌ 错误: \w 不匹配连字符, data-fixtureid 提取不到
_ATTR_RE = r'(\w+)="([^"]*)"'

# ✅ 正确: [\w-] 包含连字符
_ATTR_RE = r'([\w-]+)="([^"]*)"'
```

trade.500.com 的属性名: `data-fixtureid`, `data-homesxname`, `data-awaysxname`, `data-matchdate`, `data-rangqiu`, `data-simpleleague`, `data-matchnum` — 全部含连字符。

### 3. 赔率在内容体, 不在 opening tag

```python
# 正确提取范围: opening tag 的 > 后面到 </tr> 之间
match = re.match(r'<tr[^>]*data-fixtureid="(\d+)"[^>]*>(.*?)</tr>', row, re.DOTALL)
fid = match.group(1)
html_body = match.group(2)  # ← 这里才是 data-sp 所在的位置

# data-sp 在 <p class="betbtn" data-type="nspf" data-value="3" data-sp="2.74"> 中
sp_odds = re.findall(r'data-type="(\w+)"\s*data-value="(\d+)"\s*data-sp="([0-9.]+)"', html_body)
```

### 4. 半全场/总进球展开行

比分(bf)等展开玩法不在同一行, 需要用 `find_next_sibling` 或手动处理 `bet-more-wrap`:

```python
# BeautifulSoup 方式
for row in soup.find_all('tr', attrs={'data-fixtureid': True}):
    more = row.find_next_sibling('tr', class_='bet-more-wrap')
    if more:
        bf_odds = more.find_all(attrs={'data-sp': True})
```

### 5. 属性值含空格

部分属性值如 `data-simpleleague="国际赛"` 含中文, 匹配时用 `[^"]*`。

### 6. 完整提取示例

```python
import re

# 单行提取
ROW_RE = r'<tr[^>]*data-fixtureid="(\d+)"[^>]*>(.*?)</tr>'
ATTR_RE = r'([\w-]+)="([^"]*)"'
ODDS_RE = r'data-type="([\w]+)"\s*data-value="(\d+)"\s*data-sp="([0-9.]+)"'

for m in re.finditer(ROW_RE, html, re.DOTALL):
    fid = m.group(1)
    body = m.group(2)
    
    # 提取 tr 属性
    attrs = dict(re.findall(ATTR_RE, m.group(0)))
    home = attrs.get('data-homesxname')
    away = attrs.get('data-awaysxname')
    match_date = attrs.get('data-matchdate')
    handicap = int(attrs.get('data-rangqiu', '0'))
    
    # 提取赔率
    odds = {}
    for om in re.finditer(ODDS_RE, body):
        t, v, sp = om.groups()
        odds.setdefault(t, {})[v] = float(sp)
```

## 验证方法

对抓取的原始 HTML 文件测试:

```bash
# 保存测试页面
curl -s 'http://trade.500.com/jczq/?playid=269&g=2&date=2026-06-02' |
  iconv -f gbk -t utf-8 > /tmp/test_500.html

# 用 python 单行验证
python3 -c "
import re
html = open('/tmp/test_500.html').read()

# 测试行匹配
rows = re.findall(r'<tr[^>]*data-fixtureid=\"(\d+)\"[^>]*>(.*?)</tr>', html, re.DOTALL)
print(f'匹配行数: {len(rows)}')

# 检查属性提取含连字符
ATTR_RE = r'([\w-]+)=\"([^\"]*)\"'
first_row = html[:html.index('</tr>')+5]
attrs = dict(re.findall(ATTR_RE, first_row))
print(f'含连字符属性: {[k for k in attrs if \"-\" in k]}')
"

# 输出示例: 匹配行数: 7, 含连字符属性: ['data-fixtureid', 'data-homesxname', 'data-awaysxname', 'data-matchdate', 'data-rangqiu', 'data-simpleleague', 'data-matchnum']
```

## 关联

- 主 skill: `jczq-prediction-system` — 500.com 数据采集章节
- wanchang 抓取: `references/500-wanchang-scraping.md` — 与 trade 互补, 提供赛果而非赔率
- training data: `references/500-training-data-construction.md` — 配对赔率+赛果流程
