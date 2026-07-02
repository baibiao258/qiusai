# 官方赛程提取流程

## 数据源

2026世界杯12×4赛程/分组数据来自 [WRooney108/World-Cup-Betting](https://github.com/WRooney108/World-Cup-Betting) 的 `prisma/seed.ts`。

## 提取方式

### seed.ts → 2026_groups.json

已有 `scripts/extract_official_groups.py` 可从 seed.ts 提取分组：
```bash
python3 /root/.hermes/skills/knowledge/wc-2026-predict/scripts/extract_official_groups.py
```

输出: `/root/data/2026_groups.json` — 12组×4队含FIFA排名

### seed.ts → 完整赛程（72场 + 匹配关系）

当需要完整的赛程（每场比赛的时间、场馆、对阵）时：

1. 从 seed.ts 提取 match 数组
2. 每个 match 包含: `roundNumber`, `groupLetter`, `home`, `away`, `date`
3. 场次按 roundNumber 排序（1-48 = 小组赛组内轮次，48+ = 淘汰赛）

### 验证：赛程 vs 分组一致性

提取赛程后必须验证所有72场对阵均在分组内：
```python
from wc_2026_final import GROUPS, TEAMS_2026
teams_in_schedule = set(extracted_teams)  # 从赛程提取
missing = TEAMS_2026 - teams_in_schedule
extra = teams_in_schedule - TEAMS_2026
assert not missing, f"赛程缺失球队: {missing}"
assert not extra, f"赛程多余球队: {extra}"
assert len(extracted_teams) == 48, "应恰好48支球队"
```

### 输出文件

- `/root/data/2026_matches_ref.json` — 完整赛程参考（含时间、场馆、轮次）
- 格式: `[{home_team, away_team, group, stage, date, venue, city}]`

## 队名映射陷阱

seed.ts 使用以下名称（需映射到代码内部名称）：

| seed.ts 名称 | 代码内部名 |
|:---|:---|
| USA | United States |
| Korea Republic | South Korea |
| Czechia | Czech Republic |
| Bosnia & Herzegovina | Bosnia |
| Türkiye | Turkey |

映射见 `wc_2026_final.py` 中的 `team_name_normalizer` 字典。
