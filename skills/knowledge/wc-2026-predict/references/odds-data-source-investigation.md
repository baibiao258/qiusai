# 历史博彩赔率数据源调查

## 结论：放弃真实历史赔率，使用Elo校准

**投产比分析**：为了最多 +1-2pp 的微弱增益去获取历史赔率，从工程ROI看极不划算。

## 已调查的数据源

### 1. The Odds API（已启用，免费层）

- **夺冠赔率**：✅ 可用（`soccer_fifa_world_cup_winner` 端点）
- **未来单场H2H赔率**：✅ 可用（`soccer_fifa_world_cup` 端点）
- **历史单场赔率**：❌ **付费功能**（~$50/月，Professional计划）
  - 尝试 `/v4/historical/` 端点返回：`HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN`
- **免费额度**：500次/月，实时消耗467次（每日cron约18次）
- **数据用途**：H2H赔率校准单场预测，夺冠赔率校准冠军概率

### 2. football-data.co.uk

- **联赛数据**：✅ 22个联赛CSV（E0, D1, I1, F1, SP1等），含Bet365/威廉希尔等多家赔率
- **世界杯数据**：❌ `WC.csv` 重定向到 `EC.csv`（English Conference），非World Cup
- **所有路径尝试**：
  - `/mmz4281/{season}/WC.csv` → 301跳转（不存在）
  - `/new/WC.csv` → 404
  - `/worldcup.php` → iframe嵌入（无直接CSV链接）
  - 搜索结果：football-data.co.uk 未托管世界杯独立CSV
- **状态**：❌ 不可用

### 3. GitHub 公开数据集

| 仓库 | 数据 | 状态 |
|------|------|:----:|
| stephenhillphd/worldcup22betviz | 小组出线概率，非H2H赔率 | ❌ 无H2H |
| mattymajestic/fifa-world-cup-2022 | 金靴/个人奖项赔率 | ❌ 无关 |
| ewenme/world-cup-2022 | odds.csv 404 | ❌ 404 |
| jalapic/WorldCup | 404 | ❌ |
| openfootball/world-cup | 赛果数据（无赔率） | ❌ 无赔率 |

### 4. Kaggle

- 有"FIFA World Cup 2022 Dataset"包含赔率
- **需Kaggle API Key**（用户无私藏Key）
- **状态**：❌ 无权限访问

### 5. GitHub Code Search

- 搜索模式：`World Cup 2022 odds B365H`（Bet365赔率列名）
- 结果：部分仓库含赔率CSV但链接已失效
- **状态**：❌ 无可用数据集

## 当前方案：Elo校准赔率

```python
def make_odds(eh, ea):
    """Elo → 公平赔率 (6% bookmaker margin)"""
    e_h = 1.0 / (1 + 10**((ea - eh) / 400))
    e_d = 0.26 * math.exp(-((eh - ea) / 200)**2)
    o = np.array([e_h * (1-e_d), e_d, (1-e_h) * (1-e_d)])
    o /= o.sum()
    return o  # [H, D, A] 无抽水概率
```

**验证结果**：
- 2022 WC严格回测 Brier=0.6132（×3归一化后）
- 与Golden20基准0.6099仅差0.0033
- **结论**：Elo校准赔率作为fallback已足够

## 未来可行路径（需资源投入）

1. **升级The Odds API**：Professional计划 ~$50/月 → 获取2022 WC历史H2H赔率
2. **爬取Oddsportal**：有历史赔率但需云服务器+反爬绕过
3. **买Kaggle数据集**：便宜但一次性，需维护更新
