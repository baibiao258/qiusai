# 预测输出审计清单

`daily_jczq.py` 运行后，检查终端输出的每一项以发现潜在问题。

## 1. 模型概率 vs 市场赔率分歧检测

对于每个 RECOMMEND 场次，对比模型 SPF 概率与市场赔率隐含概率（去水后）：

```python
def implied_from_odds(h, d, a):
    ih, ida, ia = 1/h, 1/d, 1/a
    margin = ih + ida + ia
    return ih/margin, ida/margin, ia/margin
```

**警告阈值**：
- EV 超过 +30% — 模型与市场严重分歧，需追溯模型路由
- EV 超过 +50% — 极可能模型校准偏移或特征缺失，不直接采纳

**诊断步骤**（以美国 vs 澳大利亚 EV=+79.1% 为例）：
1. 确认模型路由：检查 `_try_hybrid_predict` 走了路线A（DC+Pinnacle）还是路线B（DC+XGB）
2. 检查 `thestats_adv_feat[0]` Pinnacle 市场概率是否加载（=0 意味着无市场校正）
3. 检查 DC 模型对该两队的 λ 是否合理（非均匀分布）
4. 如果路线 A/B 都无数据，检查是否走 market_fallback 导致 EV 循环论证

## 2. SPF 推荐区分度检查

当胜平负三项概率的最高值 - 次高值 < 2pp 时，SPF 推荐几乎无意义：

| 场景 | 模型输出 | 判断 |
|------|---------|------|
| 主36.7% / 平27.1% / 客36.2% | 差距 0.5pp | 实质均势，推荐方向不可靠 |
| 主39% / 平29% / 客32% | 差距 7pp | 勉强可用，但需看 EV |

处理：区分度低时在审计中标注，建议用户人工判断或 skip。

## 3. 365scores 数据覆盖检查

逐场确认是否有 `365scores公众投票` 行：

```
365scores公众投票: 主89.5% / 平5.6% / 客4.9% (n=145306)
```

**缺失原因排查**：
1. 未来场次（周五/周六的赛程）— 365scores CSV 可能还未包含，正常
2. 今日场次缺失 — 检查队名是否在 `score365_map` 中
   - 用 `_resolve_name()` + `normalize_match_pair()` 双重查找
   - 打印 365scores CSV 中该日所有足球比赛，对比队名
3. 小众联赛（芬兰超等）— 365scores 不覆盖，正常

**匹配成功但仍有问题的模式**：
- 公众投票 vs 模型概率显著矛盾（如公众 63.8% 支持 vs 模型 36.7%）→ 标记

## 4. 总进球分布一致性检查（回退模式特有）

所有 `market_fallback` 场次的泊松总进球分布仅取决于 λ_total：

```
P(goals=k) = e^(-λ_total) * λ_total^k / k!
```

当同一联赛/阶段的所有 fallback 场次使用相同的 `STAGE_LAM` 值时（如芬兰超均用 group=2.55），
**总进球分布必然相同**。这是泊松卷积的数学性质，不是 bug。

验证：检查比分分布（如 `1:0(10.9%)` vs `2:0(12.5%)`）是否不同 — 比分分布使用 λ_home/λ_away 独立计算，应当不同。

## 5. EV 来源核验

高 EV 出现时，追溯其来源：

```
玩法          推荐        赔率      概率     EV
总进球 1球    5.45      25.6%   +39.7%
```

- 检查 `is_sane_bet()` 过滤是否生效（赔率>30? 概率<15%?）
- 检查模型类型是否为 `market_fallback`（回退场次的 EV 是循环论证）
- 检查赔率是否来自 nspf 还是欧赔兜底（`euro_odds_ref` 不参与 EV 计算）

## 6. 赛程日期标签与赔率可用性

- 提前数天的场次（如周四/周五的 World Cup）通常有 SPF 赔率
- 极端强弱对话（让球≥2）可能 nspf 未开售 → SPF 赔率为 0 → `apply_euro_fallback` 标记
- 输出中检查 `SPF市场赔率` 行是否存在

## 7. 市场分歧交叉验证

```
市场分歧: 比分市场倾向=2:0 | 总进球市场倾向=3球
```

- 比分市场倾向 vs 模型推荐是否在同一方向？
- 总进球市场倾向 vs 模型推荐是否一致？
- 大量分歧出现可能意味着模型在所有市场维度上都偏离市场价格

## 快速审计命令

```bash
# 提取所有 RECOMMEND 场次的 EV 分布
python3 -c "
import csv
from collections import Counter
rows = list(csv.DictReader(open('/root/data/predictions_log.csv')))
today = '2026-06-17'
recs = [r for r in rows if r.get('date')==today and r.get('bet_action')=='RECOMMEND']
print(f'RECOMMEND: {len(recs)}')
# 检查 365scores 覆盖
has_vote = sum(1 for r in recs if r.get('s365_home_winrate','').strip())
print(f'365scores基本面覆盖: {has_vote}/{len(recs)}')
# 检查 model_route
routes = Counter(r.get('model_route','') for r in recs)
print(f'模型路由: {dict(routes)}')
"
```

## 逐场检查清单

- [ ] SPF 三项概率之和 ≈ 100%
- [ ] SPF 推荐的最高选项与次高差距 > 2pp（否则标记低区分度）
- [ ] EV 值 < +50%（否则标记异常）
- [ ] 让球概率：让胜+让平+让负 ≈ 100%
- [ ] 比分 Top3 累积概率合理（非极端集中）
- [ ] 半全场 9 项累积概率 ≈ 100%
- [ ] 365scores 基本面：公众投票总和 ≈ 100%
- [ ] 市场分歧数量 ≤ 2（过多表示模型全面偏离市场）
- [ ] model_route 非 'unknown'（否则 ah_probs 覆盖 bug 可能复发）
