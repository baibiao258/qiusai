# 预测管线架构 (Pipeline Architecture)

## 双轨预测架构: Club vs International

`predict_match_wrapper()` 是统一入口，按优先级走两条并行管线：

```
predict_match_wrapper(home, away)
 ├── ① _try_club_predict      ← 俱乐部 DC+XGB (英超/西甲/德甲等)
 │     命中条件: 球队在 form_club.json + 有近期战绩
 │     故障模式: 中文队名未映射→TEAM_ALIASES 缺失→SILENT FALLBACK
 │
 ├── ② _try_hybrid_predict    ← 国际赛 DC+XGB (世界杯/友谊赛/预选赛)
 │     命中条件: Elo + DC lambda 可用 (form 缺失可降解)
 │     回退: fallback_market_predict (纯赔率反推)
 │
 └── ③ fallback_market_predict (纯市场赔率)
       当 _try_hybrid_predict 的 Elo/DC 不可用时触发
```

## 国际赛 Form 状态降解 (2026-07-01)

**_try_hybrid_predict** 原有一段硬返回 None 的过滤，当球队不在 form state 中时直接放弃。改为零填充 Form 特征后继续推理。

**改动**：删除 `if h not in fs or a not in fs... return None` 硬拦截。

**原理**：
- `pm_recent_form()` 对未知球队返回 `[0.5, 0.0, 0.0, 0.0]`，不抛异常
- DC lambda + Elo 独立于 form 状态可计算
- XGBoost 特征维度 29 维不变（form_feat 全零），model inference 不报错
- `form_gap` 标志保留，下游 `bet_action = 'SKIP_DATA'` 标注低置信度

**典型效果**：比利时 vs 塞内加尔从 `market_fallback` 主45% → `hybrid` 主58% (SKIP_DATA)
阿根廷 vs 佛得角从 SPF全0% → `hybrid` 主95% (SKIP_DATA)

## Standings 积分榜集成 (2026-07-01)

离线缓存 7 大联赛 standings，提供 3 维排名特征。

- **数据源**：TheStatsAPI `/competitions/{id}/seasons/{sid}/standings`
- **缓存脚本**：`/root/pull_standings_cache.py` → `/root/data/standings_cache.json` (43KB, 136队)
- **查询模块**：`/root/standings_lookup.py`
  - `load_standings_cache()` — lazy 加载缓存
  - `lookup_both(home, away)` → `(home_info, away_info, features)`
  - `features` = `[rank_diff/max_pos, pt_diff/max_pts, gd_diff/50]`
  - 内置 30+ 短名映射（Man City → Manchester City, PSG → Paris Saint-Germain 等）
- **展示**：仅在 `_try_club_predict` 路径生效，国家队比赛静默跳过
- **Phase 2**：3 维追加到 17 维特征向量 → 20 维 → 重训 xgb_model_club.pkl

### Short name map maintenance

在 `standings_lookup.py` 的 `_SHORT_NAME_MAP` 添加。当 `lookup_both()` 对某队返回 None 时，
先检查 TheStatsAPI 中该队的官方全名（通过 `curl /football/teams?search=XXX` 获取），
然后添加入口。

## team_name_normalizer 维护

`TEAM_ALIASES` 字典在 `/root/team_name_normalizer.py:7` 定义。

**排查**：新赛事中球队名未被英文映射时：
```python
from team_name_normalizer import TEAM_ALIASES, normalize_team_name
normalize_team_name('塞内加尔')  # 返回中文→没映射
TEAM_ALIASES.get('塞内加尔', 'NOT FOUND')  # 确认缺失
```

**修复**：追加 `'中文名': 'EnglishName'` 到 `TEAM_ALIASES`。

**典型缺失**（2026-07-01 发现）：
塞内加尔/Senegal, 埃及/Egypt, 佛得角/Cape Verde
