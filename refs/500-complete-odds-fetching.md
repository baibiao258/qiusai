# 500.com 完整赔率抓取技术文档

## 核心发现 (2026-06-14)

### 1. 页面结构：每场两个 `<tr>`

```html
<tr class="bet-tb-tr" data-fixtureid="1359200" data-subactive="...">
  <!-- 主行：队名 + 让球值 + nspf/spf 赔率 -->
</tr>
<tr class="bet-more-wrap hide">
  <!-- 展开行：bf/bqc/jqs 赔率 (CSS display:none，但HTML已存在) -->
</tr>
```

**关键**: 展开行通过CSS隐藏，但数据已在HTML中，requests可直接抓取，无需Playwright。

### 2. playid 只返回对应玩法的 data-type

| playid | 返回的 data-type | 说明 |
|--------|-----------------|------|
| 269 | spf, nspf | 胜平负+让球 |
| 270 | jqs | 总进球 |
| 271 | bf | 半全场 |
| 272 | bqc | 比分 |
| 312 | spf, nspf | ⚠️ 不含其他玩法 |

### 3. data-value 编码

**nspf/spf**: 3=主胜, 1=平, 0=客胜

**bqc (比分)**: `主队进球-客队进球`
- "3-3" = 主3:客3
- "3-1" = 主3:客1 (主胜)
- "1-3" = 主1:客3 (客胜)

**bf (半全场)**: 首位=半场, 次位=全场
- "33" = 半场主胜-全场主胜
- "31" = 半场主胜-全场平
- "10" = 半场平-全场客胜
- 编码: 3=主胜, 1=平, 0=客胜

**jqs (总进球)**: 0=0球, 1=1球, ..., 7=7+球

### 4. 关键属性

- `data-fixtureid`: 比赛唯一ID (用于跨playid合并)
- `data-subactive`: 玩法开关 (dg=单关, gg=过关, 1=有)
- `data-rangqiu`: 让球值 (可为空)
- `data-sp`: 赔率值 (1000.00=停售, 空=未开售)
- `data-homesxname` / `data-awaysxname`: 队名
- `data-matchdate`: 比赛日期
- `data-matchtime`: 比赛时间

### 5. 抓取架构

```python
# 多playid并发 + fixture_id合并
async def fetch_all_odds(date):
    tasks = [fetch_playid(pid, date) for pid in [269, 270, 271, 272]]
    results = await asyncio.gather(*tasks)
    
    # 按fixture_id合并
    merged = {}
    for result in results:
        for match in result:
            fid = match['fixture_id']
            if fid not in merged:
                merged[fid] = match
            else:
                merged[fid]['odds'].update(match['odds'])
    
    return list(merged.values())
```

### 6. 编码注意

- 500.com所有页面使用 **GB2312/GBK** 编码
- `resp.encoding = 'gb2312'` 必须设置
- BeautifulSoup解析时需正确处理中文队名

### 7. 与kaijiang数据JOIN

```python
# 通过 (日期 + 主队 + 客队) 三元组匹配
# fixture_id 在kaijiang中不存在，不能直接JOIN

for match in odds_data:
    home_norm = match['home'].replace(' ', '').lower()
    away_norm = match['away'].replace(' ', '').lower()
    
    for _, row in kaijiang_df.iterrows():
        k_home = row['home'].replace(' ', '').lower()
        k_away = row['away'].replace(' ', '').lower()
        
        if (home_norm in k_home and away_norm in k_away and
            match['match_date'] in row['date']):
            # 匹配成功
            break
```

### 8. 历史赔率回捞

```python
def backfill_history(start_date, end_date):
    """按日期回捞历史赔率"""
    current = start_date
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        odds = fetch_all_odds(date_str)
        # 只保存已结束的比赛
        ended = [m for m in odds if m.get('is_end')]
        save_to_json(ended, f'odds_{date_str}.json')
        time.sleep(1.5)  # 礼貌性延迟
        current += timedelta(days=1)
```

## 文件路径

- 抓取脚本: `/root/wc_2026_upgrade/fetch_500_complete.py`
- 集成工具: `/root/wc_2026_upgrade/integrate_500_odds.py`
- 输出数据: `/root/data/500_odds_complete_YYYYMMDD.json`
