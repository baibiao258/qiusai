# 淘汰队追踪 (Elimination Tracking)

## 为什么需要

WC 冠军概率 MC 模拟是从小组赛开始的，所有 48 队都有非零概率。但锦标赛进行到淘汰赛阶段后，已被淘汰的球队不应出现在冠军概率输出表中。

## 数据源

- `/root/data/tournament_state.json` — 小组积分/排名/轮次，含 `eliminated` 字段
- `/root/data/wc_completed_results.json` — 所有已完成比赛赛果（accumulate_results.py 维护）

## 淘汰判定规则

`/root/update_elimination_status.py` 实现两条规则：

1. **完赛场次 >= 4**：小组赛每队只打 3 场，4+ 场说明已晋级淘汰赛并被淘汰
2. **小组排名 >= 3 且轮次 >= 3**：小组垫底且所有比赛已打完，数学上无法出线

## 输出过滤

`wc_2026_final.py` 第 1135 行附近：

- 加载 `tournament_state.json`，读 `eliminated` 字段
- 中文队名 → 英文队名映射 (`_cn_to_en` dict)
- 遍历 `champs_sorted`，跳过已淘汰队，只显示活跃队 Top 15

## 手动维护

每天运行 `accumulate_results.py` 后执行：

```bash
python3 /root/update_elimination_status.py
```

这会自动更新 `tournament_state.json` 的 `eliminated` 字段，供下一次 `wc_2026_final.py` 运行使用。

## 已知限制

- `_cn_to_en` 映射字典在 `wc_2026_final.py` 中是手写硬编码的，未复用 `team_name_normalizer`
- 只能判断"已淘汰"，不预测"理论出线可能"（如 1 分 2 场但数学上仍有微乎出线可能，仍标记为活跃）
