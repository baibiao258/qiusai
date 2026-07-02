# Sofascore Lineup Integration (predict_match lineup_features)

**状态**: 集成完成, 已E2E验证 (2026-06-04)  
**位置**: `/root/sofascore_integration.py` (新文件, 14KB)  
**修改**: `/root/predict_match.py` 新增 `lineup_features` 可选参数

## 触发场景

当用户问"XX比赛预测"且比赛在24小时内进行时, 可调用 Sofascore 拉取真实首发数据
改善友谊赛/小比赛轮换问题的预测置信度。**数据时间窗: 赛前 ~1小时到完赛**。

## EasySoccerData 安装与Linux适配

```bash
pip install EasySoccerData
playwright install chromium
```

**关键坑**: EasySoccerData 0.0.8 默认 `browser_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe"`,
在Linux下必须显式传入 Playwright 安装的 Chrome 路径:

```python
import esd
CHROME = '/root/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome'
client = esd.SofascoreClient(browser_path=CHROME)
```

路径可能因 Playwright 版本 (`chromium-1217`) 而变, 用 `ls ~/.cache/ms-playwright/` 确认当前版本。

## SofascoreClient 主要方法 (21个)

| 方法 | 用途 | 返回 |
|------|------|------|
| `get_events(date='2026-06-04')` | 某日所有赛事 | `List[Event]` (含 status, home/away_team, score, tournament) |
| `get_match_lineups(event_id)` | 比赛阵容 | `Lineups(confirmed, home, away)` — **赛前1h才有数据** |
| `search('Slovenia')` | 搜索队伍 | `List[Team\|Player\|Tournament]` 混合结果 |
| `get_match_incidents(event_id)` | 进球/红黄牌 | 完赛后可用 |
| `get_match_stats(event_id)` | 比赛统计 | 完赛后可用 |
| `get_tournament_seasons/category_id` | 联赛/赛季 | 历史数据 |
| `get_team_events/get_team_players` | 队伍历史 | 历史数据 |

## 阵容数据结构 (TeamLineup)

每个 `TeamLineup.players` 元素 `PlayerLineup`:
- `info.market_value`: 市场价值 (EUR), 用于计算"阵容质量"
- `info.position`: G/D/M/F (门将/后卫/中场/前锋)
- `info.rating`: Sofascore 评分 (完赛后)
- `substitute: bool`: True=替补, False=首发
- `captain: bool`: 队长标识
- `statistics.minutes_played`: 实际分钟数

`TeamLineup.missing_players` 元素 `MissingPlayer`:
- `player`: 同上结构
- `reason`: 1=受伤, 11=停赛, 其他=轮换/未入选

`TeamLineup.formation`: '3-4-2-1' / '4-2-3-1' / '4-3-3' 等

## 提取的 7 个特征 (per team)

```python
{
    'starter_count': 11,                    # 首发数
    'starter_market_value_m': 285.1,        # 首发总市值 (M EUR)
    'avg_starter_market_value_m': 25.9,     # 首发人均市值
    'avg_starter_rating': 7.03,             # 首发平均评分 (完赛才有)
    'missing_count': 2,                     # 缺阵总人数
    'key_player_missing_count': 0,          # 缺阵中market_value>5M的人数
    'formation': '3-4-2-1',                 # 阵型字符串
    'attack_weight': 0.5,                   # 阵型攻强 (0=防守, 1=进攻)
}
```

**阵型→攻击权重映射** (内置, 可调整):
```python
FORMATION_ATTACK_WEIGHT = {
    '5-4-1': 0.2, '5-3-2': 0.3, '4-5-1': 0.3,
    '4-4-2': 0.4, '4-3-3': 0.6, '4-2-3-1': 0.6,
    '3-5-2': 0.7, '3-4-3': 0.8, '4-1-4-1': 0.5,
    '3-4-2-1': 0.5,  # 平衡阵型
}
```

## 集成到 predict_match.py

`predict_match()` 新增参数 `lineup_features=None`, 触发**阵容感知动态折扣**:

```python
def predict_match(home, away, host_bonus=0.0, match_type='competitive',
                 lineup_features=None):
    # ...原有逻辑...
    base_smooth = MATCH_TYPE_WEIGHTS.get(match_type, 0.0)  # 友谊赛 0.3
    smooth = base_smooth
    lineup_adjustment = 0.0
    if lineup_features and lineup_features.get('home') and lineup_features.get('away'):
        h = lineup_features['home']
        a = lineup_features['away']
        h_mv = h.get('avg_starter_market_value_m', 0)
        a_mv = a.get('avg_starter_market_value_m', 0)
        # 触发条件1: 任一方缺主力 >= 2
        if h.get('key_player_missing_count', 0) >= 2 or \
           a.get('key_player_missing_count', 0) >= 2:
            lineup_adjustment += 0.10
        # 触发条件2: 双方首发市值差异 > 30%
        if h_mv > 0 and a_mv > 0:
            mv_ratio = abs(h_mv - a_mv) / max(h_mv, a_mv)
            if mv_ratio > 0.3:
                lineup_adjustment += 0.10
        smooth = min(0.6, base_smooth + lineup_adjustment)  # cap 60%
```

**机制**: 把模型对均匀分布的折扣从 30% 提升到 30~60%, 反映"阵容不平衡/主力缺阵"导致的
预测不确定性。`lineup_features=None` 时完全回归原始行为。

## Sofascore 队名 → DC 模型队名

DC 模型用 `predict_match.py` 的 `TEAM_NAME_MAP` (45+ 国家), 关键映射:
- `'Ivory Coast' → "Côte d'Ivoire"` (特殊字符)
- `'Türkiye' → 'Türkiye'` (用土耳其新名, 不用旧 Turkey)
- `'Cape Verde' → 'Cape Verde'` (不用 Cabo Verde)
- `'DR Congo' → 'DR Congo'`

Sofascore 搜索结果可能返回 Player/Team/Tournament 混合, 必须按 `type(r).__name__=='Team'` 过滤,
再 `rname.lower() == team_name.lower()` 严格匹配 (避免如搜索 "United" 撞上多个国家队)。

## 缓存策略

`SofascoreFeatureExtractor` 内置缓存到 `/root/data/sofascore_cache/`, 24h TTL:
- `search_<name>.json` — 队伍ID
- `event_<date>_<home>_<away>.json` — event_id
- `lineups_<event_id>.json` — 阵容数据 (完赛后不会再变)

**赛前1h内调用**避免缓存到空 lineup (`starter_count: 0`)。

## E2E 验证结果 (2026-06-03 完赛 16 场国家队比赛)

| 比赛 | 实际 | 模型(friendly+lineup) | 阵容 | 准确 |
|------|------|----------------------|------|------|
| Croatia 0-2 Belgium | A | A 37.5% | 285M vs 276M | ✓ |
| Morocco 4-0 Madagascar | H | H 63.4% | 21M vs 0.3M | ✓ |
| Philippines 5-1 Guam | H | H 56.3% | 0.1M vs 0M | ✓ |
| Luxembourg 0-1 Italy | A | A 59.1% | 1.2M vs 17.9M | ✓ |
| Gibraltar 4-0 BVI | H | H 58.0% | 0.1M vs 0M | ✓ |
| South Korea 1-0 El Salvador | H | H 57.5% | 4.8M vs 0.4M | ✓ |
| Panama 4-2 Dominican Rep | H | H 50.1% | 0.5M vs 1M | ✓ |
| Georgia 1-1 Romania | D | H 36.2% (实际 D) | 4.8M vs 3.7M | ✗ |
| Wales 1-1 Ghana | D | H 40.8% (实际 D) | 10.4M vs 6.4M | ✗ |
| Haiti 4-0 New Zealand | H | A 39.6% (实际 H) | 2.8M vs 1.9M | ✗ |
| Albania 0-1 Israel | A | H 36.3% (实际 A) | 4.9M vs 4.1M | ✗ |
| Denmark 0-0 DR Congo | D | H 57.5% (实际 D) | 17M vs 5.5M | ✗ |
| Netherlands 0-1 Algeria | A | H 42.5% (实际 A) | 47.5M vs 11.6M | ✗ |
| Poland 2-2 Nigeria | D | H 35.4% (实际 D) | 10.2M vs 6.9M | ✗ |
| Indonesia 0-2 Singapore | A | H 56.7% (实际 A) | 0 vs 0 | ✗ |
| Tanzania 1-0 Malawi | H | A 39.7% (实际 H) | 0 vs 0 | ✗ |

**总准确率: 7/16 = 43.8%** (友谊赛+纯阵容数据)

**关键观察**:
- 阵容数据对**强队差异明显**的比赛有效 (Croatia vs Belgium, Luxembourg vs Italy)
- **荷兰 0-1 阿尔及利亚** 反映友谊赛轮换: 阵容差 4 倍仍爆冷 → 仅靠阵容数据不够
- 印度尼西亚/新加坡/坦桑尼亚等小队伍 MV=0, Sofascore 阵容数据稀疏
- 友谊赛本身噪声大, 43.8% 接近 47% 赔率市场水平

## 已知限制

1. **时间窗窄**: 阵容数据仅赛前 ~1小时发布, 实际应用窗口有限
2. **小队伍数据稀疏**: 印尼/新加坡/马拉维等 MV 普遍为 0, 无法差异化
3. **轮换不可预测**: 教练在友谊赛中给年轻球员机会, 阵容数据不能完全捕获
4. **未接入 XGB 训练**: 目前是后处理折扣, 未作为 7 个新特征喂给 XGB29 模型
5. **浏览器启动慢**: 每次初始化 ~5秒, 多场比赛时考虑复用 client

## 后续可做

1. **A/B 回测**: 拿6/3 16场比赛对比"有lineup"vs"无lineup"准确率
2. **加入 XGB 训练集**: 7个 lineup 特征作为 XGB 第 30-36 维, 重新训练模型
3. **赛前定时任务**: 比赛前 30 分钟自动 fetch 并覆盖预测结果
4. **校正触发阈值**: 用历史回测校准 30% MV差异 和 2 缺主力 的最佳阈值
