# 500.com 开奖页面 HTML 结构

## URL
```
zx.500.com/jczq/kaijiang.php?playid=0&d=YYYY-MM-DD
```
GBK 编码，请求间隔 ≥ 0.5s。

## 页面结构

所有比赛数据在 `<table class="ld_table">` 中。

### 表头行 (rows 0-8)
- 行 0: 15 个 `<th>` (赛事编号 | 赛事类型 | 比赛时间 | 主队 | 让球 | 客队 | 比分 | ... | 半全场)
- 行 1-8: 嵌套 `<table class="th_tb">` 定义各玩法的"彩果/奖金"子表头

### 数据行 (row 9+)
每行 **19 个 `<td>`**:

| 索引 | 内容 | 提取方式 |
|------|------|----------|
| [0] | 赛事编号 | `td.text.strip()` → "周二201" |
| [1] | 赛事类型 | `td.text.strip()` → "友谊赛" / "欧国联" / "荷乙" |
| [2] | 比赛时间 | `td.text.strip()` → "06-09 19:35" |
| [3] | 主队 | `td.text.strip()` → "荷兰" |
| [4] | 让球数 | `td.text.strip()` → "-2" / "+2" / "0" |
| [5] | 客队 | `td.text.strip()` → "乌兹别克斯坦" |
| [6] | 比分 | `td.text.strip()` → "(1:0) 2:1" |
| [7] | 分隔 | 空 |
| [8] | 让球胜平负彩果 | `td.text.strip()` → "胜"/"平"/"负" |
| [9] | 让球胜平负奖金 | `td.find('span', class_='red').text` → "2.21" |
| [10] | 分隔 | 空 |
| [11] | 胜平负彩果 | `td.text.strip()` → "胜"/"平"/"负" |
| [12] | 胜平负奖金 | `td.find('span', class_='red').text` → "1.14" |
| [13] | 分隔 | 空 |
| [14] | 总进球彩果 | `td.text.strip()` → "3" / "4" |
| [15] | 总进球奖金 | `td.find('span', class_='red').text` → "3.15" |
| [16] | 分隔 | 空 |
| [17] | 半全场彩果 | `td.text.strip()` → "胜胜"/"负负"/"胜平" |
| [18] | 半全场奖金 | `td.find('span', class_='red').text` → "1.48" |

## 解析示例

```python
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'lxml')
table = soup.find('table', class_='ld_table')
for row in table.find_all('tr'):
    tds = row.find_all('td')
    if len(tds) != 19:
        continue
    code = tds[0].text.strip()
    home, away = tds[3].text.strip(), tds[5].text.strip()
    score_raw = tds[6].text.strip()  # "(1:0) 2:1"
    spf_sp = tds[12].find('span', class_='red')
    spf_val = float(spf_sp.text.strip()) if spf_sp else 0.0
```

## 比分正则
```python
import re
SCORE_RE = re.compile(r'\((\d+):(\d+)\)\s*(\d+):(\d+)')
# "(1:0) 2:1" → ht_h=1, ht_a=0, ft_h=2, ft_a=1
```

## 彩果→内部编码映射
```python
RESULT_MAP_SPF = {'胜': '3', '平': '1', '负': '0'}
# [8] 让球胜平负彩果 → rqspf_result
# [11] 胜平负彩果 → spf_result (从比分算更可靠)
```

## 未开售处理
奖金显示 `--` 时，span.red 的 text 为 `"--"`，应赋值 0.0。
回测时需过滤 `sp <= 0` 的比赛。

## 日期遍历
- 支持 `2024-01-01` 至今
- 无赛事的日期返回空表（无 ld_table 行），标记为已完成避免重复抓取
- 建议 delay=0.5s，2024-01-01 至今约 282 天有赛事数据
