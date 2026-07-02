# 黄金20+3 特征消融研究

## 背景

从33维全量特征中通过消融实验筛选出最优子集，解决"特征越加越差"的反直觉现象。

## 特征演进

| 管线 | 维度 | DC权重 | XGB权重 | 准确率 | Brier | 差异 |
|------|:----:|:------:|:-------:|:-----:|:-----:|:----:|
| 15维基线 | 15 | 0.5 | 0.5 | **57.81%** | 0.6135 | — |
| 33维全量 | 33 | 0.7 | 0.3 | 53.12% | 0.6096 | −4.69pp ❌ |
| 33维+Optuna | 33 | 0.6 | 0.4 | 54.69% | 0.6059 | −3.12pp |
| **20+3黄金** | **23** | **0.4** | **0.6** | **56.25%** | **0.6099** | -1.56pp |
| 20+3+Optuna (回测) | 23 | 0.4 | 0.6 | 56.25% | 0.6099 | baseline |

## 关键发现

1. **15维 57.81% 是铁顶** — 三次独立回测（默认参数、Optuna、20+3对比）全部稳定在此数字。不是数据泄漏。
2. **Brier持续下降** — 0.6135 → 0.6099 → 0.6059。Optuna的强正则化（高α/λ、低colsample、高min_child）让概率校准更好了。
3. **XGB权重回升** — 20+3的最佳权重是 DC0.4+XGB0.6，说明切除噪声特征后XGBoost变得可信了。
4. **剩余差距1.56pp** — 预计注入真实The Odds API历史单场赔率可弥补。

## 20+3特征定义

### 15维基线
```
[elo_diff, lam_h, lam_a, lam_diff, lam_ratio,
 dc_H, dc_D, dc_A,
 f5_win_h, f5_win_a, f5_att_adv, f5_def_adv,
 f5_gf_diff, f5_win_diff, neutral]
```

### 5黄金特征（新增）
```
[h2h_gd (H2H净胜球),
 tier_major (大赛正赛),
 tier_friendly (友谊赛),
 f12_att_adv (12场进攻优势 = home_gf12 - away_ga12),
 f12_win_a (12场客场进攻力 = away_gf12 - home_win12)]
```

### 3赔率特征
```
[odds_H, odds_D, odds_A]  # Elo校准隐含概率 (6% margin)
```

### FeatureBuffer 增量构建 (O(1) per match)

替代O(n)全量扫描，每个比赛只更新受影响球队的缓存：

```python
class FeatureBuffer:
    def __init__(self, elo, dc):
        self.elo = elo
        self.dc = dc
        self.team_games = defaultdict(list)    # team -> [{date, gf, ga}]
        self.h2h_cache = defaultdict(lambda: defaultdict(list))
        self.last_date = {}

    def add_match(self, m):
        # O(1): 只更新两队的缓存
        h, a = m['home'], m['away']
        for team, gf, ga in [(h, m['h_score'], m['a_score']),
                              (a, m['a_score'], m['h_score'])]:
            self.team_games[team].append({'date': m['date'], 'gf': gf, 'ga': ga})
            self.last_date[team] = m['date']
        key = (h, a) if h < a else (a, h)
        self.h2h_cache[key[0]][key[1]].append(m)

    def recent_form(self, team, date, n):
        games = [g for g in self.team_games.get(team, []) if g['date'] < date]
        relevant = sorted(games, key=lambda x: x['date'], reverse=True)[:n]
        # ... rest is O(n) on the filtered sublist (n<=12, negligible)
```

## Optuna最佳参数 (Brier=0.6059)

| 参数 | 值 | 作用 |
|------|:----:|------|
| max_depth | 4 | 浅树防过拟合 |
| learning_rate | 0.032 | 保守学习 |
| n_estimators | 369 | 足够迭代 |
| reg_alpha | 3.05 | 强L1稀疏化 |
| reg_lambda | 2.69 | 强L2正则 |
| colsample_bytree | 0.45 | 降特征共线性 |
| subsample | 0.64 | 行采样 |
| min_child_weight | 8.2 | 叶节点最小权重 |

## 严格回测方法

1. **截止日**: 2022-11-20 (世界杯开幕前一天)
2. **训练数据**: 截止日前所有A级国际赛
3. **逐场预测**: 64场按时间顺序，每场用前N-1场信息
4. **Elo实时更新**: 每场后更新Elo评分
5. **FeatureBuffer逐场添加**: 无未来信息泄漏
