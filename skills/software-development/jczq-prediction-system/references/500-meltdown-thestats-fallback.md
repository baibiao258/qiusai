# 500.com 熔断 TheStatsAPI 兜底 (P0#3, 2026-06-16)

## 架构

```
scrape_500_odds_today()
  ├── (1) async_500_scraper.py 并发抓取 4 playid (timeout 45s)
  │    失败 ↓
  ├── (2) _load_fallback_odds()  → odds_history.json (stale 历史)
  │    也失败 ↓
  └── (3) _thestats_list_todays_matches()
          → TheStatsAPI /football/matches?competition_id=comp_6107
          → _500_MELTDOWN = True
```

## 新增组件 (`_thestats_list_todays_matches()`)

**位置**: `/root/daily_jczq.py` (模块全局, 紧接 imports)

**数据源**: TheStatsAPI (`competition_id=comp_6107` = 世界杯)

**输出格式**: 兼容 500.com `_500_odds` 结构, 但所有赔率字段全为 0:
```python
{
    'code': 'TSMo001',           # 唯一 code
    'home_cn': 'Spain',          # 英文队名 (来自 star_players.json)
    'away_cn': 'Cape Verde',
    'time': '16:00',
    'league': 'World Cup 2026',
    'source': 'thestats_fallback',
    'odds_h': 0, 'odds_d': 0, 'odds_a': 0,   # 赔率全 0
    'handicap': 0,
    'rq_h': 0, 'rq_d': 0, 'rq_a': 0,
    'bf_odds': {}, 'zjq_odds': {}, 'htft_odds': {},
    'nspf_empty': True,
    'is_fallback': True,
}
```

## 降级效果

| 组件 | 熔断时行为 |
|------|-----------|
| **EV 计算** | 跳过 (`_500_MELTDOWN` 分支, `ev_h/d/a = ''`) |
| **市场权重融合** | 跳过 (odds=0 → 条件 `odds_h > 1` 不满足) |
| **bet_action** | 强制 `WATCH_NO_ODDS` |
| **500 分析页面** | 跳过 (无 shuju_id) |
| **bet_math 汇总** | WATCH_NO_ODDS 场次从 `all_analyses` 过滤 |
| **概率输出** | 正常运行 (模型仍输出胜平负) |

## 关键设计点

1. **不翻译赔率**: TheStatsAPI 回传的是 Betfair 1X2 (国际盘), 与 500.com 竞彩 SP 有系统性差异。直接使用会导致 EV 计算偏差。所以**赔率全置 0**, 宁可不出 EV 也不出错误的 EV。
2. **队名英文→中文**: TheStatsAPI 返回英文队名, `star_players.json` 键是 team_id, 值含英文 name。兜底函数直接使用英文名, 不走 `_resolve_name()` (因为 500.com 熔断时无中文名输入)。`use_500_only` 路径中 `predict_match_wrapper` 可接受英文名。
3. **只覆盖世界杯**: 指定 `competition_id=comp_6107`, 不覆盖 500.com 上的其他联赛赛事。联赛赛事在熔断时无兜底。

## 联调验证

- `_500_MELTDOWN` 全局标志在 `daily_jczq.py` 模块级别定义, 初始 `False`
- 由 `scrape_500_odds_today()` 在完全熔断后设置 `_500_MELTDOWN = True`
- `main()` 中检测 `not _500_odds and _500_MELTDOWN` → 调用兜底
- `build_prediction_bundle()` 中检测 `_500_MELTDOWN` → 跳过 EV + 强制 bet_action
