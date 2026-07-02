# 2026 官方淘汰赛签表（R32→Final）

## 数据来源

- **Repo**: https://github.com/openfootball/worldcup.git
- **文件**: `2026--usa/cup_finals.txt`
- **Star**: 602, 最近更新 2026-06-02
- **维护者**: openfootball 项目组（长期维护 FIFA/CONMEBOL/UEFA 赛事数据，数据可靠）

## 签表规则

### 32 强组成

- 12 个小组（A-L）的前两名 = 24 队
- 8 个最佳小组第三（从 12 个第三名中按积分/净胜球/进球排序选出）

### R32 配对（写死，不可调整）

```
(73) 2A  vs  2B
(74) 1E  vs  3{A/B/C/D/F}
(75) 1F  vs  2C
(76) 1C  vs  2F
(77) 1I  vs  3{C/D/F/G/H}
(78) 2E  vs  2I
(79) 1A  vs  3{C/E/F/H/I}
(80) 1L  vs  3{E/H/I/J/K}
(81) 1D  vs  3{B/E/F/I/J}
(82) 1G  vs  3{A/E/H/I/J}
(83) 2K  vs  2L
(84) 1H  vs  2J
(85) 1B  vs  3{E/F/G/I/J}
(86) 1J  vs  2H
(87) 1K  vs  3{D/E/I/J/L}
(88) 2D  vs  2G
```

第三名配对规则解读：`3{A/B/C/D/F}` = 来自 A/B/C/D/F 组中出线的最佳第三名。具体由第 79/80 位的第三名排序决定——FIFA 预先排好了各组第三名在签表中的落位组合。

### R16 配对

```
(89) W74 vs W77
(90) W73 vs W75
(91) W76 vs W78
(92) W79 vs W80
(93) W83 vs W84
(94) W81 vs W82
(95) W86 vs W88
(96) W85 vs W87
```

### QF 配对

```
(97) W89 vs W90
(98) W93 vs W94
(99) W91 vs W92
(100) W95 vs W96
```

### SF 配对

```
(101) W97 vs W98
(102) W99 vs W100
```

### 决赛 & 三四名

```
(103) L101 vs L102    — 三四名决赛
(104) W101 vs W102    — 决赛
```

## 场馆分配（按轮次）

- R32: Los Angeles, Boston, Monterrey, Houston, New Jersey, Dallas, Mexico City, Atlanta, San Francisco, Seattle, Toronto, Vancouver, Miami, Kansas City
- R16: Philadelphia, Houston, New Jersey, Mexico City, Dallas, Seattle, Atlanta, Vancouver
- QF: Boston, Los Angeles, Miami, Kansas City
- SF: Dallas, Atlanta
- 三四名/决赛: Miami / New Jersey

## 在模拟中使用官方签表 vs Elo 种子配对

| 维度 | Elo 种子（当前 wc_2026_final.py） | 官方签表（openfootball） |
|------|------|------|
| 配对逻辑 | Elo 1v32, 2v31... | 写死 2A vs 2B, 1E vs 3* 等 |
| 第三名处理 | 12个第三名选8个按Elo排序 | 第三名排序后分配到预设槽位 |
| 分半区 | 纯Elo分，强弱均匀 | 按小组位置分，可能半区强弱不均 |
| 准确性 | 近似值 | FIFA正式规则 |
| 何时可用 | 探索分析/原型 | 购买建议/正式报告 |

## 从 openfootball 同步

```bash
git clone --depth 1 https://github.com/openfootball/worldcup.git /tmp/openfootball-wc
cat /tmp/openfootball-wc/2026--usa/cup_finals.txt   # 签表
cat /tmp/openfootball-wc/2026--usa/cup.txt           # 小组赛程+官方队名映射
rm -rf /tmp/openfootball-wc
```

## 注意

- 第三名出线的 8 队排序方式：积分→净胜球→进球，前8晋级
- 第三名分配槽位由 FIFA 预先组合固定，不是随机分配
- 当前 `wc_2026_final.py` 和 `simulate_knockout.py` 使用 Elo 种子配对（未接入官方签表）
- 接入官方签表需修改 MC 循环中的淘汰赛配对逻辑，按出线后的小组位置映射
