# 500.com 竞彩页面 HTML 结构逆向分析

## 数据结构（2026-06-14 逆向）

每场比赛是一个 `<tr class="bet-tb-tr">` 行，核心数据在 `data-*` 属性里：

```html
<tr class="bet-tb-tr"
    data-fixtureid="1359203"       ← 500.com 内部比赛ID（关联 odds.500.com）
    data-infomatchid="164902"      ← 信息比赛ID
    data-matchdate="2026-06-15"    ← 比赛日期
    data-matchtime="0400"          ← 比赛时间
    data-rangqiu="-1"              ← 让球数（-1=主让1球，0=平手，1=客让1球）
    data-homeid="18"               ← 主队ID
    data-awayid="29"               ← 客队ID
    data-matchid="110"             ← 赛事ID（110=世界杯）
    data-processid="376146"        ← 期次ID
    data-processdate="2026-06-14"  ← 销售日期
    data-processname="7010"        ← 期次名（周几+期号）
    data-matchnum="010"            ← 场次编号
    data-isend="0">               ← 0=未结束, 1=已结束
```

## 赔率提取规则

赔率在 `<td class="td-betbtn">` 里：

```html
<!-- 让球胜平负（nspf = 让球） -->
<div class="betbtn-row itm-rangB1">
  <p data-type="nspf" data-value="3" data-sp="1.86">1.86</p>  ← 主胜
  <p data-type="nspf" data-value="1" data-sp="3.38">3.38</p>  ← 平
  <p data-type="nspf" data-value="0" data-sp="3.38">3.38</p>  ← 客胜
</div>

<!-- 胜平负（spf = 标准） -->
<div class="betbtn-row itm-rangB2">
  <p data-type="spf" data-value="3" data-sp="4.00">4.00</p>   ← 主胜
  <p data-type="spf" data-value="1" data-sp="3.40">3.40</p>   ← 平
  <p data-type="spf" data-value="0" data-sp="1.71">1.71</p>   ← 客胜
</div>
```

## 关键语义区分

| 字段 | 含义 | 使用场景 |
|------|------|----------|
| `spf` | 标准胜平负（1X2） | 当 `data-rangqiu=0` 时是真正的标准赔率 |
| `nspf` | 让球胜平负 | 仅当让球≠0时有 |

**⚠️ 重要陷阱**：当 `handicap≠0` 且 `nspf` 为空时，`spf` 字段包含的是**让球后的赔率**，不是标准1X2。

## 让球值提取

```html
<p class="green itm-rangA2" title="主队让3球"> -3</p>  ← 文本内容是 "-3"
<p class="red itm-rangA2" title="客队让1球"> +1</p>    ← 文本内容是 "+1"
```

**注意**：`title` 属性是中文描述，实际数值在元素的**文本内容**中。

## 抓取代码

```python
import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://trade.500.com/',
    'Accept-Charset': 'gb2312,utf-8',
}

def fetch_500_odds(target_date, playid=269):
    url = f'https://trade.500.com/jczq/?playid={playid}g2&date={target_date}'
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.encoding = 'gb2312'  # 必须用 gb2312
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    rows = soup.select('tr.bet-tb-tr')
    
    results = []
    for row in rows:
        d = row.attrs
        
        # 让球值（从文本内容提取，不是title）
        rang_el = row.select_one('.itm-rangA2')
        rang_text = rang_el.get_text(strip=True) if rang_el else '0'
        rang_val = rang_text.replace('+', '')
        
        # 赔率提取
        odds = {'nspf': {}, 'spf': {}}
        for p in row.select('p[data-type]'):
            t = p.get('data-type')
            v = p.get('data-value')
            sp = p.get('data-sp')
            if t in odds and v and sp:
                odds[t][v] = float(sp)
        
        results.append({
            'fixture_id': d.get('data-fixtureid'),
            'home': d.get('data-homesxname', ''),
            'away': d.get('data-awaysxname', ''),
            'rang': float(rang_val),
            'spf_home': odds['spf'].get('3'),
            'spf_draw': odds['spf'].get('1'),
            'spf_away': odds['spf'].get('0'),
            'nspf_home': odds['nspf'].get('3'),
            'nspf_draw': odds['nspf'].get('1'),
            'nspf_away': odds['nspf'].get('0'),
            'has_nspf': bool(odds['nspf']),
        })
    
    return results
```

## 赔率来源关联

- `fixture_id`（如 `1359203`）关联 `odds.500.com/fenxi/shuju-{id}.shtml`
- `process_id`（如 `376146`）关联开奖页面 `kaijiang.500.com/jczq.shtml?issue=376146`
- `data-processname`（如 `7010` 表示周日第10场）对应竞彩彩票期次编号
