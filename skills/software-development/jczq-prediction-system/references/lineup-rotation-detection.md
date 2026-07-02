# 阵容轮换检测 (Lineup Rotation Detection)

> 建立日期: 2026-06-16
> 关联: thestats_features.py v2 — detect_rotation + adjust_with_lineups

## 架构总览

```
fetch_team_squads.py (单次/每周cron)
  └─ 拉 48 队 WC 阵容 (TheStatsAPI /football/teams/{id}/players)
  └─ 标记 is_star / is_starter → star_players.json

fetch_thestats_features.py (每30分钟, 赛日8-22 UTC)
  └─ 拉当日比赛 Lineups (TheStatsAPI /football/matches/{id}/lineups)
  └─ 含 home_team / away_team 字段 → thestats_lineups.json

daily_jczq.py (每日预测)
  └─ model_prob → adjust_with_lineups(probs, home, away) → 调幅后概率
  └─ 注入点: line 1461-1473
```

## 数据库: star_players.json

**路径**: `/root/data/star_players.json` (385 KB)
**来源**: TheStatsAPI `/football/teams/{id}/players` 端点
**内容**: 48 支 WC 球队 × 18-26 人 = 1,169 名球员

### 球员评分公式

```python
def calc_star_score(player):
    # 位置权重: F=1.0, M=0.7, D=0.4, G=0.1
    age = player.get('age', 25)
    pos_w = POS_WEIGHT.get(player.get('position'), 0.3)

    # 年龄权重: 24-30 巅峰=1.0, 20-23/31-33=0.7, 
    #           17-19/34-36=0.4, 其他=0.2
    if 24 <= age <= 30: age_w = 1.0
    elif 20 <= age <= 23 or 31 <= age <= 33: age_w = 0.7
    ...

    score = pos_w * age_w * 0.7 + pos_w * 0.3
    return round(score, 3)

def is_star(player):
    # 绝对核心标记阈值 (收紧版 2026-06-16)
    if pos == 'F' and score >= 0.7: return True
    if pos == 'M' and score >= 0.6: return True
    if pos == 'D' and score >= 0.5: return True
    return False
```

**标记结果**: 466 核心 / 1169 总球员 (40%) — 每队 5-15 名核心。

### 数据结构

```json
{
  "tm_28735": {
    "name": "Mexico",
    "group": "A",
    "squad_size": 26,
    "star_ids": ["pl_xxx", "pl_yyy", ...],
    "starter_ids": ["pl_xxx", ...],
    "players": [
      {
        "id": "pl_30846417",
        "name": "Alexis Vega",
        "position": "M",
        "age": 28,
        "nationality": "Mexico",
        "club": "CD Toluca",
        "star_score": 0.700,
        "is_star": true,
        "is_starter": true
      },
      ...
    ],
    "by_position": {
      "F": ["Alexander Isak", ...],
      "M": ["Daniel Svensson", ...],
      "D": ["Isak Hien", ...],
      "G": ["Kristoffer Nordfeldt", ...]
    }
  }
}
```

## 核心算法: compute_rotation_penalty()

### 输入

- `team_name`: str — 球队名 (中英皆可, 自动匹配)
- `starter_names`: list[str] — 本场 11 名首发球员名
- `star_data`: dict — star_players.json 的解析结果

### 流程

1. 队名匹配: 精确 → 部分匹配 → 模糊匹配
2. 归一化首发名单: 去重音(Unicode NFKD) + 小写 + 去空格
3. 统计:
   - `n_stars`: 该队核心球员总数
   - `n_missing`: 不在首发的核心球员数
   - `excess_missing = max(0, n_missing - 2)` — 容忍 2 人正常轮换
4. 惩罚计算:
   ```python
   penalty = min(max(0, excess_missing * 0.035), 0.20)
   ```

### 惩罚曲线

| 缺阵核心 | excess_missing | penalty | 典型场景 |
|---------|---------------|---------|---------|
| 0-2 人 | 0 | 0% | 正常轮换/全主力 |
| 3 人 | 1 | 3.5% | 轻微保留 |
| 4 人 | 2 | 7.0% | 部分轮换 |
| 5 人 | 3 | 10.5% | 明显轮换 |
| 6 人 | 4 | 14.0% | 大幅轮换 |
| 7 人 | 5 | 17.5% | 近 B 队 |
| 8+ 人 | 6+ | 20.0% (上限) | B 队出战 |

## 调幅逻辑: adjust_with_lineups()

### 输入

- `probs`: dict `{'H':, 'D':, 'A':}` — 调幅前概率
- `home_team`, `away_team`: str — 队名

### 输出

- `(adjusted_probs, adjustments_list)`
- 无匹配/无轮换时返回原 probs + 空列表

### 匹配策略 (双路径)

```
路径1: lineup.home_team / lineup.away_team 直接匹配
  └─ fetch_thestats_features.py v2 已写入队名字段

路径2: _infer_team_from_names() — 球员名反向推断
  └─ 统计首发名单与 star_players 的交集
  └─ 命中 ≥3 个核心球员名 = 可靠匹配
  └─ 兼容旧 lineup 缓存 (无队名字段)
```

### 调幅应用

```python
# 主队轮换: 削减主胜, 按比例分配到平/客
if h_penalty > 0:
    shift = h_penalty * new_probs['H']
    new_probs['H'] = max(0.05, new_probs['H'] - shift)
    # redistribution 按原平/客权重比例分配
    ...

# 客队轮换: 削减客胜, 按比例分配到主/平
if a_penalty > 0:
    shift = a_penalty * new_probs['A']
    new_probs['A'] = max(0.05, new_probs['A'] - shift)
    ...

# 重新归一化确保 Σ = 1.0
total = sum(new_probs.values())
for k in new_probs: new_probs[k] /= total
```

## 队名匹配 (双路径匹配)

```python
def _infer_team_from_names(players_names, name2stars, name2id):
    """通过首发名单反向推断是哪支队"""
    norm_names = {_normalize_name(n) for n in players_names}
    best_team, best_score = None, 0
    for tname, star_set in name2stars.items():
        overlap = len(norm_names & star_set)
        if overlap > best_score:
            best_score = overlap
            best_team = tname
    return best_team if best_score >= 3 else None
```

名字归一化 (`_normalize_name`):
```
Unicode NFKD → ascii 子集 → 小写 → 去空格
"Alexis Vega" → "alexisvega"
"Viktor Gyökeres" → "viktorgyokeres"
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `/root/data/star_players.json` | 48 队阵容数据库 (385 KB) |
| `/root/data/thestats_lineups.json` | 当日 lineup 缓存 |
| `/root/wc_2026_upgrade/fetch_team_squads.py` | 阵容拉取脚本 |
| `/root/wc_2026_upgrade/thestats_features.py` | 旋转检测引擎 (v2) |
| `/root/wc_2026_upgrade/fetch_thestats_features.py` | Lineup 抓取 (v2, 含队名) |
| `/root/wc_2026_upgrade/recalc_on_lineup.py` | **赛前重推 (2026-06-16)** — 方向/EV预警 |

## 赛前重推 (recalc_on_lineup.py)

**痛点**: `daily_jczq.py` 在 03:00 UTC 运行，当日首发在 15:00~21:00 UTC 才公布。`adjust_with_lineups()` 在 03:00 时无数据 → NOP。

**方案**: 每次 lineup 抓取后立即运行 `recalc_on_lineup.py`，对比原始预测与调幅后概率。

**执行链路**:
```
thestats-lineup-fetch cron (每30min)
  ├── python3 fetch_thestats_features.py --lineups-only
  │     → thestats_lineups.json 更新
  └── python3 recalc_on_lineup.py
        → 加载 predictions_log.csv (filter 未完结行)
        → 加载 thestats_lineups.json (filter confirmed=True)
        → cn2en 中英队名匹配 (team_name_mapping.json)
        → 调用 adjust_with_lineups() 获取调幅后概率
        → 比较:
            ├── 最佳方向 (max prob) 是否改变?
            └── EV 是否从正变负?
        → 方向变/EV翻转 → ⚠️ [赛前急报] 终端输出
        → 方向不变/p>0但<2% → 📊 蓝色信息
        → 无惩罚 → 静默
        → cron 捕获 ⚠️ 行 → 推送 Telegram
```

**队名匹配细节**:
```python
# 方法1 (主): cn2en[中文] → lineup[英文] 精确匹配
# 方法2 (备): strip_ranking() 去[NN]前缀后查 mapping
# 方法3 (回退): en2cn[英文] → CSV match_key 解析
```

**predictions_log.csv 字段**:
- `result_status`: 空 = 未完结 (等待中), 'filled' = 已完结
- `match_key`: 格式 `2026-06-08|友谊赛|西班牙|佛得角|06-16 00:00` — 第2:3位是干净中文队名
- `home_cn`/`away_cn`: 可能带 `[N]` 排名前缀, 用 `strip_ranking()` 去除

## Cron

| Job ID | 定时 | 说明 |
|--------|------|------|
| thestats-squad-refresh | 周日 05:00 UTC | 每周阵容刷新 (cfefb24b00a9) |
| thestats-lineup-fetch | 每 30min (赛日 0-23 UTC) | 首发实时轮询 + recalc_on_lineup.py 重推推送 |

## 验证

```bash
# 阵容健康度
python3 -c "
import json
d = json.load(open('/root/data/star_players.json'))
print(f'球队: {len(d)}')
total = sum(len(v[\"players\"]) for v in d.values())
stars = sum(len(v[\"star_ids\"]) for v in d.values())
print(f'球员: {total}, 核心: {stars} ({stars*100/total:.0f}%)')
"

# 旋转检测测试
python3 -c "
import sys; sys.path.insert(0, \"/root/wc_2026_upgrade\")
from thestats_features import compute_rotation_penalty, _load_star_data
sd = _load_star_data()
p, d = compute_rotation_penalty('Sweden', 
    ['Alexander Isak', 'Viktor Gyokeres', ...], sd)
print(f'penalty={p:.1%}, missing={d[\"stars_missing\"]}')
"

# 检查调用点
grep -n 'adjust_with_lineups' /root/daily_jczq.py

# 手动刷新阵容
cd /root/wc_2026_upgrade && python3 fetch_team_squads.py
```

## 坑

1. **core标记需收紧**: 初始 is_star 阈值过宽(847/1169人=72%都是核心)。2026-06-16 收紧至 F≥0.7/M≥0.6/D≥0.5 → 466 人 (40%)。回测不准时先检查此阈值。
2. **名称匹配依赖 Unicode NFKD**: 重音字符(Gyökeres, Côte d'Ivoire)必须归一化。`_normalize_name()` 使用了 unicodedata.normalize('NFKD')。
3. **Cache 跨请求残留**: `_load_star_data()` 有全局缓存 `_STAR_DATA_CACHE`。测试时必须 `force_reload=True` 或重启进程。
4. **旧 lineup 缓存无队名**: 2026-06-16 之前缓存的 lineup 数据没有 `home_team`/`away_team` 字段。回退推断机制 `_infer_team_from_names()` 通过球员名匹配，但需 ≥3 名核心球员命中才视为可靠。
5. **惩罚只降不升**: rotation 调幅只削减 H 或 A 概率，从不提升。rebalance 是比例分配，不体现任何"板凳深度"信号。弱队 B 队 vs 强队全主力时 penalty 不对称。
7. **recalc_on_lineup.py 只读 — 不修改 predictions_log.csv**: 是纯检测+预警脚本, 不覆盖概率。如需实际采纳调整后概率, 需手动判断或写写入版本。
8. **predictions_log.csv 有重复行**: 同一场比赛可能在多日预测中出现 (date 不同但 code/home/away 相同)。`recalc_on_lineup.py` 默认匹配所有未完结行中的第一条。多日重复时可能匹配到过时版本。
