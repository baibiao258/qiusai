# TheStatsAPI 特征后处理管线

> 建立日期: 2026-06-15
> 关联 skill: jczq-prediction-system — TheStatsAPI 特征后处理管线 章节

## 整体架构

后处理层在 daily_jczq.py 的 `main()` 函数末尾（模型推理 + CSV 写入完成后）注入：

```
XGBoost 推理 → predictions_log.csv → apply_thestats_features() → 修正展示输出
                                                    ↑
                                        thestats_team_stats.json (缓存)
                                        thestats_lineups_cache.json (缓存)
```

## 代码入口

### daily_jczq.py 注入 (line 1461-1473, v2 2026-06-16)

**注入时机**: 确保所有模型推理和 CSV 写入**已完成**之后。后处理不修改 CSV, 只影响展示。

### print_match_bundle 展示增强

`print_match_bundle()` 中新增字段:

- `thestats_spf_pick` / `thestats_spf_h/d/a` — 修正后胜平负推荐
- `thestats_rq_pick` / `thestats_rq_win/draw/loss` — 修正后让球推荐
- `thestats_signal_summary` — 显示使用的信号 (stats/lineup/none)

当 thestats 字段存在时, 覆盖原始推荐行。展示添加:
```
📊 TheStats: 主队进攻1.8 vs 客队0.9 (场均进球差+0.9, 权重30%)
```

回退: `bundle.get('thestats_spf_pick', bundle['spf_pick'])`

## Team Stats 数据

### 拉取方式

`fetch_thestats_features.py`:

```python
# 对 17 个赛事 ID 的每支球队发起请求
# 20 并发, 带超时和断点续传
competition_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 136]
for cid in competition_ids:
    team_list = thestats_get('/team/stats/', {'competition_id': cid})
    for team in team_list:
        team_id = team['id']
        stats = thestats_get('/team/stats/', {'competition_id': cid, 'team_id': team_id})
```

输出: `/root/data/thestats_team_stats.json` — ~930 队, 2,750+ 条记录。

### 数据结构

```json
{
  "team_id_123": {
    "team_id": 123,
    "competition_id": 1,
    "team_name": "Brazil",
    "att": "WWDLW[WW]",
    "def": "WWDLW[DL]",
    "avg_scored_home": 2.1,
    "avg_scored_away": 1.5,
    "avg_conceded_home": 0.8,
    "avg_conceded_away": 1.2
  }
}
```

### WWDLW 陷阱

`att`/`def` 字段格式:
- 有值: `WWDLW[WW]` = 最近5场结果(W=赢,D=平,L=输) + 方括号内最近2场的比赛类型
- ⚠️ **不是数值**。`float(att)` 会崩溃。
- 替代: 用 `avg_scored_home` - `avg_scored_away` 计算场均进球差

### 信号计算

```python
def _estimate_goal_diff(home_stats, away_stats):
    """从 avg_scored/conceded 估算净胜球差"""
    home_attack = home_stats.get('avg_scored_home', 0) or home_stats.get('avg_scored', 0)
    away_attack = away_stats.get('avg_scored_away', 0) or away_stats.get('avg_scored', 0)
    home_defense = home_stats.get('avg_conceded_home', 0) or home_stats.get('avg_conceded', 0)
    away_defense = away_stats.get('avg_conceded_away', 0) or away_stats.get('avg_conceded', 0)
    # 进攻差 × 0.6 + 防守差 × 0.4
    attack_diff = home_attack - away_attack
    defense_diff = away_defense - home_defense  # 防守越好失球越少
    return 0.6 * attack_diff + 0.4 * defense_diff
```

### 权重策略

```python
def _get_stats_weight(goals_diff):
    abs_diff = abs(goals_diff)
    if abs_diff < 0.5: return 0.0
    weight = 0.30 + 0.10 * (abs_diff - 0.5) / 0.5
    return min(weight, 0.40)

def _adjust_probs_with_stats(base_probs, goals_diff, weight):
    direction = 1 if goals_diff > 0 else 0  # H=2, A=0
    adjustment = (base_probs[direction] + (1 - base_probs[1]) * weight)
    adjustment = max(base_probs[direction], min(adjustment, 0.85))
    # 贝叶斯裁剪
    bayes_clip = 0.25
    adjusted = min(adjustment, base_probs[direction] + bayes_clip)
    adjusted = max(adjusted, base_probs[direction] - bayes_clip)
    # 归一化
    new_probs = list(base_probs)
    new_probs[direction] = adjusted
    total = sum(new_probs)
    return [p / total for p in new_probs]
```

## 阵容轮换检测 (Replaced by new System 2026-06-16)

> Lineups 数据驱动的旋转检测已迁移到独立子系统：`references/lineup-rotation-detection.md`
>
> **变更**: 不再使用 `apply_thestats_features()`。daily_jczq.py line 1461-1473 直接调用 `adjust_with_lineups()` (thestats_features v2)。
>
> 新架构:
> - `/root/data/star_players.json` — 48队 1169球员 466核心 (球员数据库)
> - `/root/data/thestats_lineups.json` — 当日首发 (含队名字段)
> - `thestats_features.adjust_with_lineups(probs, home, away)` → 调幅后概率
> - 惩罚: 缺阵 3 核心 → 3.5%, 8+ → 20% (上限)

## 过滤器参数一览

## 过滤器参数一览

```python
FILTER_THRESHOLDS = {
    'min_stats_weight': 0.10,
    'max_pull_pct': 0.40,
    'min_goal_diff_for_full_weight': 1.0,
    'lineup_probability_clip': 0.30,
    'stats_bayes_clip': 0.25,
    'output_fields': [
        'spf_pick', 'rq_pick',
        'spf_h', 'spf_d', 'spf_a',
        'rq_win', 'rq_draw', 'rq_loss'
    ]
}
```

## Cron Jobs

### thestats-team-stats

- **类型**: 📜 pure-script (no_agent=True)
- **定时**: 每日 09:00 UTC (17:00 BJT)
- **脚本**: `scripts/fetch_thestats_features.py`
- **行为**: Team Stats 全量拉取, 覆盖 17 赛事 × 各队, 20 并发
- **失败场景**: API 超时/无新数据 → 使用缓存旧数据
- **输出文件尺寸**: ~2,750+ 条, ~150KB

### thestats-squad-refresh (NEW 2026-06-16)

- **类型**: 🤖 agent-driven
- **定时**: 周日 05:00 UTC
- **prompt**: 运行 fetch_team_squads.py 刷新 star_players.json
- **输出**: `/root/data/star_players.json` (385 KB, 48 teams, 1169 players)

### thestats-lineup-fetch

- **类型**: 📜 pure-script (no_agent=True)
- **定时**: 每 30 分钟 (赛日 08:00-22:00 UTC)
- **脚本**: `scripts/fetch_thestats_features.py` — fetch_today_lineups()
- **v2 变更 (2026-06-16)**: lineup 缓存新增 `home_team`/`away_team` 字段, 支持直接队名匹配
- **输出**: `/root/data/thestats_lineups.json` (合并写入)
- **当前限制**: 赛前 1h 才释放阵容, 历史比赛不保留

## 验证方式

```bash
# 检查 Team Stats 缓存
python3 -c "
import json
d = json.load(open('/root/data/thestats_team_stats.json'))
...

# 检查明星球员数据库
python3 -c '
import json
d = json.load(open(\"/root/data/star_players.json\"))
print(f\"球队: {len(d)}, 球员: {sum(len(v[\"players\"]) for v in d.values())}\")
'

# 检查 daily_jczq.py 注入点 (v2)
grep -n 'adjust_with_lineups' /root/daily_jczq.py
print(f'球队数: {len(teams)}')
print(f'记录数: {len(d)}')
# 检查 att 字段类型
wwdlw = sum(1 for v in d.values() if isinstance(v.get('att'), str) and v['att'] and not v['att'][0].isdigit())
print(f'WWDLW格式 att: {wwdlw}/{len(d)}')
"

# 检查 Lineups 缓存
python3 -c "
import json
d = json.load(open('/root/data/thestats_lineups_cache.json'))
filled = {k:v for k,v in d.items() if v}
print(f'非空阵容: {len(filled)}/{len(d)}')
"

# 检查 daily_jczq.py 注入点
grep -n 'thestats_features\|apply_thestats\|thestats_spf' /root/daily_jczq.py
```

## 诊断命令

```bash
# 手动跑 Team Stats 拉取
cd /root && python3 scripts/fetch_thestats_features.py 2>&1 | tail -5

# 手动跑 Lineups 拉取
cd /root && python3 scripts/thestats_lineup_fetch.py 2>&1 | head -20

# 检查特定比赛的 Team Stats
python3 -c "
import json
d = json.load(open('/root/data/thestats_team_stats.json'))
for k, v in d.items():
    name = v.get('team_name', '')
    if 'brazil' in name.lower() or 'argentina' in name.lower():
        print(f'{name}: att={v.get(\"att\")}, def={v.get(\"def\")}, scored_h={v.get(\"avg_scored_home\")})')
"
```
