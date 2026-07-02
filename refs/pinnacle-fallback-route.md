# Pinnacle 赔率兜底路由 (L4) — 2026-06-20

## 动机

500.com 是唯一的 SPF/RQ 赔率源，熔断时系统降级到无赔率状态。TheStatsAPI 已有 Pinnacle 赔率获取逻辑（`_get_odds()`），且配额充裕（500万次/月）。将其作为第四层路由引入，消除赔率单点依赖。

## 架构位置

```
predict_match_wrapper → club → intl → [main pipeline]
  ├─ L3: fallback_market_predict(market_row)  → market_fallback
  └─ L4: market_fallback_pinnacle(home, away, league) → market_fallback_pinnacle
```

L4 从 L3 的调用点级联: 当 `fallback_market_predict(m5)` 返回 None 时，立即调用 `market_fallback_pinnacle()`。

代码位置: `daily_jczq.py` main() 函数 use_500_only 循环 (L3020-3027)

## 函数清单

### `_pinnacle_to_jczq_prob(p_h, p_d, p_a, jczq_vig=0.89)`

将 Pinnacle 隐含概率（已去抽水，三项和为 1.0）转换为竞彩可比概率。步骤：
1. 归一化（去除 Pinnacle 自身微量 vig）
2. 施加竞彩返奖率 (0.89)

**为什么需要 vig 修正**: Pinnacle 接近无抽水（~102%），竞彩 SPF 返奖率约 89%。直接代入会系统性高估所有概率 3-5%。经过 vig 修正后，概率和从 1.0 降至 0.89，与 500.com 赔率可比。

验证:
```python
>>> _pinnacle_to_jczq_prob(0.50, 0.25, 0.25)
(0.4450, 0.2225, 0.2225)  # sum=0.890
```

### `_thestats_search_match_id(home, away, target_date=None)`

通过 TheStatsAPI 搜索 (主队, 客队) 的 match_id。
- 查今日/指定日期的国际赛事
- 用队名模糊匹配（检查 name/name_cn 字段）
- 无缓存（当前调用频率低，200万次/月配额充裕）

### `market_fallback_pinnacle(home, away, league)`

第四层路由函数。

**流程**:
1. `_thestats_search_match_id()` → 获取 match_id
2. `get_all_advanced_features(match_id)` → 13维特征向量
3. 提取前 3 维 (Pinnacle 隐含概率)
4. `_pinnacle_to_jczq_prob()` → vig 归一化
5. 从概率反推 Poisson λ（与 `fallback_market_predict` 一致）
6. 计算最可能比分
7. 返回 dict 兼容 `build_prediction_bundle()` 输入格式

**返回格式** (与 `fallback_market_predict` 一致):
```python
{
    'probs': {'H': ..., 'D': ..., 'A': ...},
    'score': '2-1',
    'result': 'H/D/A',
    'min_odds': {'H': ..., 'D': ..., 'A': ...},
    'matches_data': (0, 0),
    'lambda_ft': {'home': ..., 'away': ...},
    'rho': 0.0,
    'model': 'market_fallback_pinnacle',   # 独立标签便于 Brier 分组
}
```

## bet_action 规则

```python
if model_type == 'market_fallback_pinnacle':
    return 'WATCH_PINNACLE'
```

L4 是最后的保险层，数据质量未经充分验证，先观察积累 Brier 后再决定是否升级为 RECOMMEND。

## 监控

```bash
# 检查 L4 被触发
grep "Pinnacle 兜底" /root/logs/daily_jczq.log

# 积累后分 model 看 Brier
python3 /root/evaluate_brier.py --new-only | grep pinnacle
```

## 已知限制

1. **仅覆盖国家队赛事**: `_thestats_search_match_id()` 查询 TheStatsAPI 国际赛事。俱乐部赛无 Pinnacle 兜底。
2. **队名匹配模糊**: `_thestats_search_match_id()` 用 name/name_cn 精确小写匹配，部分球队在 TheStatsAPI 中名称与 500.com/team_name_mapping 不一致时可能漏匹配。
3. **无 RQ 赔率**: Pinnacle 不提供竞彩让球盘口。`market_fallback_pinnacle` 返回的让球赔率由 Poisson λ 推导（同 market_fallback 模式）。
4. **Pinnacle 开盘时间**: 部分赛事 Pinnacle 赔率在赛前 24-48h 才开出。距离比赛时间太远时可能无赔率可用。

## Brier 预期

当前尚未积累有效样本。预计 `market_fallback_pinnacle` 的 Brier 应介于 `market_fallback` (0.2198) 和 `xgb_dc_nat_11d` (0.2393) 之间，因为:
- Pinnacle 赔率质量高于 500.com 平均赔率 (市场深度更大)
- vig 归一化消除了系统性偏移
- 但 Poisson λ 推导引入额外误差
