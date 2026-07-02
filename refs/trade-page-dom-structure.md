# trade.500.com 交易页 DOM 结构

## 核心行结构

```html
<tr class="bet-tb-tr"
    data-fixtureid="1411534"
    data-matchnum="周二201"
    data-matchdate="2026-06-09"
    data-rangqiu="-1"
    data-simpleleague="友谊赛">
  
  <!-- 球队名 td -->
  <td class="td-team">[世90]中国 VS 泰国 [世97]</td>
  
  <!-- 主要玩法赔率按钮 (胜平负/让球) -->
  <td class="td-betbtn">
    <p class="betbtn" data-type="nspf" data-value="3" data-sp="1.73">1.73</p>
    <p class="betbtn" data-type="nspf" data-value="1" data-sp="3.28">3.28</p>
    <p class="betbtn" data-type="nspf" data-value="0" data-sp="4.05">4.05</p>
    <p class="betbtn" data-type="spf" data-value="3" data-sp="3.66">3.66</p>
    <p class="betbtn" data-type="spf" data-value="1" data-sp="3.17">3.17</p>
    <p class="betbtn" data-type="spf" data-value="0" data-sp="1.85">1.85</p>
  </td>
</tr>

<!-- 展开行: 比分/总进球/半全场在下个tr -->
<tr class="bet-more-wrap">
  <td>
    <p class="sbetbtn" data-type="bf" data-value="1:0" data-sp="5.50">1:0</p>
    <p class="sbetbtn" data-type="bf" data-value="2:0" data-sp="7.00">2:0</p>
    <!-- ... -->
    <p class="sbetbtn" data-type="jqs" data-value="0" data-sp="11.00">0球</p>
    <p class="sbetbtn" data-type="jqs" data-value="1" data-sp="4.50">1球</p>
    <!-- ... -->
    <p class="sbetbtn" data-type="bqc" data-value="3-3" data-sp="2.90">胜胜</p>
    <!-- ... -->
  </td>
</tr>
```

## playid → URL 参数映射

| playid | 页面 | 包含玩法 | 展开行 |
|--------|------|---------|--------|
| 269 | 胜平负+让球 | spf, nspf | 无 (g=2 混合过关) |
| 270 | 总进球 | jqs | 有 (bet-more-wrap) |
| 271 | 比分 | bf | 有 (bet-more-wrap) |
| 272 | 半全场 | bqc | 有 (bet-more-wrap) |
| 312 | 单关入口 | spf, nspf, 部分bf | 不定 |

## 解析注意事项

1. **bet-more-wrap** 不是每行都有 — 只有 playid=270/271/272 且该行有展开数据时才存在
2. **data-rangqiu** 可为空 (无让球时为 "" 或 "0")
3. **队名处理**: home=`[世90]中国` → 去掉 `[世90]` 前缀; away=`泰国 [世97]` → 去掉 `[世97]` 后缀
4. **null/空值**: `data-sp` 为空字符串或 `-` 表示未开售, `data-value` 为空时跳过
