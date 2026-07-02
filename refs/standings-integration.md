# 联赛积分榜 (Standings) 集成到每日预测管线

## 架构

```
TheStatsAPI → pull_standings_cache.py → standings_cache.json ─┐
                                                               ├→ standings_lookup.py → _try_club_predict → bundle
form_club.json → _load_club_models() ──────────────────────────┘
```

## 组件

### pull_standings_cache.py
- 拉取 7 大联赛 (`comp_3039/4643/8814/0256/8385/3809/8321`) 的 standings
- 使用 TheStatsAPI 的 `current_season_id`（休赛期不变）
- 输出: `/root/data/standings_cache.json` (136队, ~43KB)

### standings_lookup.py
- `load_standings_cache()` — 惰性加载缓存，构建 `{comp_id: [rows]}` + 扁平索引
- `lookup_team(name)` — 模糊匹配：短名映射 → 精确匹配 → 子串匹配
- `lookup_both(home, away)` — 返回 `(home_info, away_info, [rank_diff/38, pt_diff/85, gd_diff/50])`
- `_SHORT_NAME_MAP` — 40+ 常见 football-data.org 短名→官方名映射

### daily_jczq.py 集成点
- `_try_club_predict()`: 预测成功后调用 `standings_lookup.lookup_both(h, a)`，结果挂到 `standings` 字段
- `build_prediction_bundle()`: 透传 `p.get('standings')` 到 bundle
- `print_match_bundle()`: 在 365 基本面后输出 `🏆 联赛排名: 主#1 85pts GD+44 | 客#5 60pts GD+10`

## 匹配策略

`_try_club_predict` 从 `normalize_match_pair()` 接收的 team name → standings_lookup 的队名匹配链：

1. **短名映射**: `_SHORT_NAME_MAP` (`"man city" → "Manchester City"`, `"psg" → "Paris Saint-Germain"`)
2. **归一化匹配**: 去空格、去 `FC/AFC/SC` 等后缀，去变音符号
3. **子串回退**: 双向子串匹配兜底

## 覆盖率验证 (2026-07-01)

- 136 队中 128 队精确匹配到 form_club 键
- 8 队因命名差异需要 `_SHORT_NAME_MAP` 或子串回退
- 跨联赛（如葡/荷/英球队名可能重复）通过 `comp_id` 交叉验证

## Phase 2 (模型重训)

当前 standings 特征以元数据形式存在 bundle 中，未进入 XGB 特征向量（维数固定 17）。重训时追加 3 维：
```python
gold = [h2h_gd, 0, 0, fh12_gf_diff, fa12_gf_diff, rank_diff/38, pt_diff/85, gd_diff/50]
```

## Pitfalls

- 休赛期的 standings 是上赛季最终数据，新赛季前 6 轮开始有用
- TheStatsAPI 的 team name 是短名（"Liverpool"），form_club.json 是长名（"Liverpool FC"），需模糊匹配
- 不同联赛的积分区间不同（英超 38 轮 114pts, 英冠 46 轮 138pts），归一化需按 `matches_played` 而非固定值
- `_SHORT_NAME_MAP` 需要随 football-data.org shortName 变化维护
