# 训练数据清洗：剔除俱乐部比赛

## 背景

训练数据 `training_data_with_odds.json` 中混入了 ~96 场俱乐部比赛（意甲/英超/西甲/日职/挪超等），nat 模型（国家队专用）训练时会被这些俱乐部分布干扰。

## 清洗策略：子串匹配 blocklist + allowlist

### 核心原则
- **先 block, 后 allow**: 检查名称是否包含俱乐部关键词, 再检查是否包含国家队关键词
- **子串匹配**: 用 `keyword in name_lower` 而非 `name.startswith(keyword)`。例如 `'friendly' in 'international friendly'` = True（这是对的）, 但 `'international friendly'.startswith('friendly')` = False（这是 2026-06-16 犯过的错）
- **中英文覆盖**: 中文俱乐部名 (`意甲`, `日职`) + 英文 (`serie a`, `premier league`)

### Blocklist (已知俱乐部关键词)
```python
club_keywords = [
    '意甲', '英超', '西甲', '德甲', '法甲', '荷甲', '葡超',
    '英冠', '英甲', '荷乙', '德乙', '瑞超', '挪超',
    '日职', '韩职', '美职足', '澳超', '中超', '巴甲', '阿甲',
    '芬兰超级联赛', '沙特职业联赛',
    '法国杯', '解放者杯', '亚洲冠军乙级联赛', '欧协联',
    '欧冠', '欧联', '欧会杯', '欧罗巴',
    'serie a', 'premier league', 'la liga', 'bundesliga',
    'ligue 1', 'eredivisie', 'primeira liga',
    'championship', 'league one', 'league two',
    'j.league', 'k league', 'a-league',
    'mls ', 'copa libertadores', 'copa sudamericana',
    'caf champions league',
]
```

### Allowlist (国家队关键词)
```python
nt_keywords = [
    'world cup', '世界杯', 'friendly', '友谊赛', '国际赛',
    'uefa euro', 'euro qual', 'uefa nations league',
    'copa américa', 'copa america', 'afc asian cup',
    'africa cup of nations', 'african cup of nations',
    'concacaf gold cup', 'concacaf nations league',
    'wcq', 'world cup qualif', '世界杯预选',
    'nations league', 'olympic', '奥运会', '奥运',
    'confederations cup', 'finalissima', 'fifa series',
]
```

## 关键 bugs (2026-06-16 修复)

### Bug 1: `startswith` 漏掉 "International Friendly"
- 错误: `name.startswith('friendly')` → `'International Friendly'` 不匹配 (首字符 I ≠ f)
- 正确: `'friendly' in name_lower` → `'international friendly'` 匹配
- 后果: 780 场 `International Friendly` 被误判为俱乐部比赛剔除 → 训练集从 2528 → 1652 (丢失 876 场)

### Bug 2: `caf champions league` 在 allowlist 中导致误判
- CAF Champions League = 非洲冠军联赛 = 俱乐部比赛
- 但 `africa cup of nations` = 非洲国家杯 = 国家队比赛 (仅有 3 字母之差!)
- 必须在 blocklist 中显式排除 `caf champions league`, 避免 allowlist 中的 `africa cup` 无意中匹配

## 结果

```
Before: 2,528 场
After:  2,432 场 (移除 96 场俱乐部, 3.8%)
保留:  International Friendly 780 场 ✅
      WCQ 赛事 600+ 场 ✅
      其他国家队赛事 ✅
```
