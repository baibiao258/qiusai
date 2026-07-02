# status-code-system.md

## bet_action 所有状态码及含义

| 状态码 | 中文标注 | 含义 | 触发条件 |
|--------|---------|------|---------|
| RECOMMEND | 推荐投注 | 可投注 | 所有规则通过 |
| BET | 可投注(低确信) | 低确信度但仍可投 | margin_pp ≥ 10 |
| SKIP | 跳过(确信度不足) | 跳过 | margin_pp < 10 |
| DATA_INSUFFICIENT | 跳过[两队数据不足] | form_state 中缺少任一队近期数据 | 模型无法获取form特征 |
| SKIP_LEAGUE | 跳过[赛事类型排除] | 历史ROI低 | UEFA Nations League |
| WATCH | 仅观察[市场兜底] | EV循环论证 | market_fallback 路由 |
| WATCH_FRIENDLY | 仅观察[友谊赛过拟合] | 友谊赛不确定性高 | 友谊赛类型 |
| WATCH_INTL | 仅观察[非主流国际赛] | 退化数据 | dc_pinnacle 路由 + 非WC/非预选 |
| WATCH_NO_ODDS | 有概率无赔率[500.com熔断] | 500.com全量熔断, 有概率无赔率 | `_500_MELTDOWN=True` |
| SKIP_WORLD_CUP_FALLBACK | 跳过[世界杯爆冷高风险] | WC+market_fallback 组合 | WC + market_fallback |
| PREDICTION_STALE | 观望[模型数据过时] | 模型文件超过7天未更新 | 模型mtime > 7天 |

## 过滤链

在 `main()` 价值投注汇总时, 使用 `SKIP_STATUSES` 元组过滤:

```python
SKIP_STATUSES = (
    'SKIP_LEAGUE', 'WATCH', 'WATCH_FRIENDLY', 'WATCH_INTL',
    'WATCH_NO_ODDS', 'SKIP_WORLD_CUP_FALLBACK', 'DATA_INSUFFICIENT',
    'PREDICTION_STALE', 'SKIP',
)
```

只有不在 SKIP_STATUSES 中的场次才进入 RECOMMEND 汇总。

## 终端输出

```python
BET_ACTION_LABELS = {
    'RECOMMEND': '推荐投注',
    'BET': '可投注(低确信)',
    'SKIP': '跳过(确信度不足)',
    'DATA_INSUFFICIENT': '跳过[两队数据不足]',
    'SKIP_LEAGUE': '跳过[赛事类型排除]',
    'WATCH': '仅观察[市场兜底]',
    'WATCH_FRIENDLY': '仅观察[友谊赛过拟合]',
    'WATCH_INTL': '仅观察[非主流国际赛]',
    'WATCH_NO_ODDS': '有概率无赔率[500.com熔断]',
    'SKIP_WORLD_CUP_FALLBACK': '跳过[世界杯爆冷高风险]',
    'PREDICTION_STALE': '观望[模型数据过时]',
}
```

## 模型过时检测

```python
def _check_model_staleness(max_age_days=7):
    """检查模型文件是否过时。返回 (is_stale, newest_file, age_days)。"""
    model_files = [
        '/root/data/xgb_model_nat.pkl',
        '/root/data/xgb_model_33.pkl',
        '/root/data/dc_model.pkl',
        '/root/data/elo_ratings.pkl',
        '/root/data/poisson_elo_prior.json',
    ]
    # 取最新mtime, 超过 max_age_days 返回 stale
```

## 新增/修改状态码时的同步点

1. `daily_jczq.py` `compute_bet_action()` — 业务逻辑
2. `daily_jczq.py` `build_prediction_bundle()` — bet_action 赋值 (line 2462-2466)
3. `daily_jczq.py` `print_match_bundle()` — BET_ACTION_LABELS 字典 (line 2038-2049)
4. `daily_jczq.py` `main()` — SKIP_STATUSES 元组 (line 2850-2852)
5. `backtest_jczq.py` FIELDS — CSV 列定义
6. `telegram_bot.py` 过滤逻辑 — 只推 RECOMMEND
7. 本 skill 的 `bet_action 标签系统` 表格
