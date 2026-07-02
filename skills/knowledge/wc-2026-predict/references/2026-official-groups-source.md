# 2026 世界杯官方分组 &赛程数据源

## 数据来源

GitHub 仓库 `WRooney108/World-Cup-Betting` 的 Prisma seed 文件包含 2026 世界杯完整分组和赛程数据：

```
https://github.com/WRooney108/World-Cup-Betting/blob/main/prisma/seed.ts
```

该 seed 文件 2026年5月31日更新，包含：

- 12 组 × 4 队 = 48 支参赛队
- 每队的世界排名（FIFA 排名）
- 每队主教练、简要介绍
- 全部 72 场小组赛 + R32 + R16 + QF + SF + 3rd + Final 的日期/时间/场地
- 关键词球员（每队 5 人）

## 分组结构 (2026年6月确认)

| 组 | 1 | 2 | 3 | 4 |
|----|---|---|---|---|
| A | Mexico (9) | South Africa (57) | South Korea (23) | Czech Republic (36) |
| B | Canada (40) | Bosnia (62) | Qatar (46) | Switzerland (15) |
| C | Brazil (5) | Morocco (13) | Haiti (89) | Scotland (34) |
| D | USA (11) | Paraguay (53) | Australia (26) | Turkey (28) |
| E | Germany (10) | Curaçao (101) | Ivory Coast (44) | Ecuador (31) |
| F | Netherlands (7) | Japan (18) | Sweden (22) | Tunisia (41) |
| G | Belgium (6) | Egypt (38) | Iran (24) | New Zealand (87) |
| H | Spain (8) | Cape Verde (68) | Saudi Arabia (56) | Uruguay (14) |
| I | France (2) | Senegal (21) | Iraq (76) | Norway (48) |
| J | Argentina (1) | Algeria (30) | Austria (25) | Jordan (71) |
| K | Portugal (4) | DR Congo (65) | Uzbekistan (70) | Colombia (12) |
| L | England (3) | Croatia (16) | Ghana (29) | Panama (45) |

括号内为 FIFA 官方排名（来自 seed）。

## 本地文件

- `/root/data/2026_groups.json` — 已从 seed.ts 同步，正式分组
- `/root/data/2026_groups_official.json` — 数据来源备份

## 赛制

12组×4队 = 72 场小组赛 → 每组前 2 + 8 个最佳第三 = 32 强淘汰赛 → R32 → R16 → QF → SF → Final。

## 与旧版差异 (2026-05-22 版 vs 2026-06-02 版)

旧版假设分组有重大错误: 
- 西班牙从 A 组（对伊朗）移到 H 组（对乌拉圭/佛得角/沙特）
- 法国从 C 组（对加拿大/乌兹别克）移到 I 组（对塞内加尔/伊拉克/挪威）
- 阿根廷从 B 组（对韩国/捷克）移到 J 组（对阿尔及利亚/奥地利/约旦）
- 加拿大从 C 组（对法国）移到 B 组（对瑞士/波黑/卡塔尔）
- 巴西从 F 组（对哥伦比亚/伊拉克）移到 C 组（对摩洛哥/海地/苏格兰）

全部 12 组均不相同，必须重新运行 MC。
