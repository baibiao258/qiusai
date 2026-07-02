# bet_math.py — EV 计算 + Kelly Criterion

## 核心公式

### Expected Value (EV)
```
EV = P_win × (Odds - 1) - (1 - P_win)
```
- EV > 0 → 有价值下注 (Value Bet)
- EV > 0.05 → 强价值 (✅)
- EV > 0.10 → 极强价值 (🔥)

### Kelly Criterion
```
f* = (p × b - q) / b
其中 b = Odds - 1, q = 1 - p
```
- f* > 0 → 建议下注
- 实战使用 **Half-Kelly** (f*/2) 或 **Quarter-Kelly** (f*/4)
- 单日总仓位上限 15%（防爆仓）

### Edge (概率优势)
```
Edge = 模型概率 - 隐含概率
隐含概率 = 1 / Odds
```

## 文件位置
- `/root/bet_math.py` — 核心算法模块
- 已集成到 `/root/daily_jczq.py` 的 `build_prediction_bundle()` 和 `print_match_bundle()`

## 使用方式

### 独立调用
```python
import bet_math
analysis = bet_math.analyze_match(home, away, predictions, odds)
print(bet_math.format_ev_table(analysis, min_ev=0.05))
print(bet_math.format_value_summary([analysis], min_ev=0.05))
```

### daily_jczq.py 集成
- `build_prediction_bundle()` 自动构建 `_predictions` 和 `_odds` 字典
- 调用 `bet_math.analyze_match()` 生成全玩法分析
- `print_match_bundle()` 输出价值投注摘要行
- `main()` 末尾调用 `format_value_summary()` 输出全局汇总

## 数据结构

### predictions 字典
```python
{
    'spf': {'h': 0.637, 'd': 0.228, 'a': 0.135},
    'rq': {'rq_win': 0.417, 'rq_draw': 0.264, 'rq_lose': 0.319},
    'score': [{'score': '1:0', 'prob': 0.15}, ...],
    'total_goals': [{'goals': 2, 'prob': 0.28}, ...],
    'half_full': [{'hf': '胜-胜', 'prob': 0.12}, ...],
}
```

### odds 字典
```python
{
    'spf': {'h': 2.02, 'd': 3.30, 'a': 3.02},
    'rq': {'rq_win': 1.85, 'rq_draw': 3.50, 'rq_lose': 2.10},
    'score': {'1:0': 5.50, '0:0': 8.00, ...},
    'total_goals': {'0': 11.00, '1': 4.40, '2': 3.25, ...},
    'half_full': {'胜-胜': 2.80, ...},
}
```

## 输出格式

### format_ev_table() — 单场分析表
```
  💰 赔率分析: Arsenal vs Chelsea
  玩法     推荐      赔率   概率   隐含   Edge    EV   Kelly 1/4Kelly
  🔥胜平负   主胜      2.02 63.7% 49.5% +14.2% +28.7%  14.1%    7.0%
  🎯 最佳: 胜平负 主胜 (EV=+28.7%, Half-Kelly=14.1%, 建议仓位=7.0%)
```

### format_value_summary() — 全局汇总
```
  💎 价值投注汇总 (EV > 5%)
  比赛                   玩法     推荐      赔率   概率     EV    Kelly
  🔥荷兰 vs 乌兹别克    胜平负    主胜      2.02  63.7% +28.7%  14.1%
  💼 Quarter-Kelly 建议总仓位: 7.3% (上限 15% 单日)
```

## model_type 线索传递 (2026-06-09)

`BetScenario` 新增 `model_type` 字段（默认空串），值为 `hybrid` / `market_fallback` / `legacy_poisson`。

传递路径：
```
daily_jczq.py: model_type = p.get('model', '')
  → bet_math.analyze_match(..., model_type)
    → analyze_scenario(..., model_type)
      → BetScenario(model_type=model_type)
```

`format_value_summary()` 和 per-match 显示都通过 `is_sane_bet()` 使用此字段做过滤。

## 长尾偏差风控：is_sane_bet() (2026-06-09)

### 问题
market_fallback 场次的 Poisson 外推会产生荒谬的高 EV：
- 德国vs库拉索 比分0:3 → 赔率900.00, 概率17.7%, EV +15800%
- 原因：主模型失明 → 欧赔反推 → Poisson 概率未惩罚弱队进攻 → 乘数效应

### 三道保险过滤
```python
def is_sane_bet(s: BetScenario) -> bool:
    # 保险1: 赔率 > 30 倍一律不碰 (数字海市蜃楼)
    if s.odds > 30.0:
        return False
    # 保险2: 概率 < 15% 一律不碰 (低信心不上榜)
    if s.prob < 0.15:
        return False
    # 保险3: market_fallback 场次禁推比分/半全场 (泊松外推不可信)
    if s.model_type == 'market_fallback' and s.play in ('比分', '半全场'):
        return False
    return True
```

### 应用位置
1. **per-match 显示** (`print_match_bundle`): `value_bets = [s for s in ba.scenarios if s.ev > 0.02 and bet_math.is_sane_bet(s)]`
2. **全局汇总** (`format_value_summary`): 在 `ev >= min_ev` 判断后额外调用 `is_sane_bet()`
3. 两处统一过滤，输出显示过滤计数 `(过滤N个)`

### 效果 (2026-06-09 实测)
- 修复前: 79 个价值投注, 最高 EV +15800%
- 修复后: 36 个价值投注 (过滤43个), 最高 EV +280%
- 德国vs库拉索/西班牙vs佛得角的虚假比分推荐全部消失
- 保留的推荐均为 hybrid 模型产出、赔率<30、概率>15%

## 注意事项
1. Kelly 公式假设独立下注，串关不适用
2. 赔率来源需去水（1/odds 归一化），bet_math 内部不处理去水
3. Quarter-Kelly 是保守策略，Full Kelly 风险极高
4. EV 阈值建议：SPF > 5%，比分 > 10%，总进球 > 8%
5. **长尾偏差 (Longshot Bias)**：高赔率+低概率的"天价EV"几乎都是数学幻觉，必须用 `is_sane_bet()` 过滤
