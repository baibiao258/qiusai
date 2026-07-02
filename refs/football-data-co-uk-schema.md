# football-data.co.uk 数据格式参考

## URL 模式

```
https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv
```

- `season_code`: `2324` (2023-24), `2425` (2024-25), `2526` (2025-26)
- `league_code`: 见下方联赛代码表

## 联赛代码

| 代码 | 联赛 | 每赛季场次 |
|------|------|-----------|
| E0   | 英超 (English Premier League) | 380 |
| E1   | 英冠 (EFL Championship) | 552 |
| SP1  | 西甲 (La Liga) | 380 |
| D1   | 德甲 (Bundesliga) | 306 |
| I1   | 意甲 (Serie A) | 380 |
| F1   | 法甲 (Ligue 1) | 380 |
| SC0  | 苏超 (Scottish Premiership) | 228 |
| N1   | 荷甲 (Eredivisie) | 306 |
| P1   | 葡超 (Liga Portugal) | 306 |
| B1   | 比甲 (Belgian Pro League) | ~240 |

## 关键列映射

### 赛果
| 列名 | 含义 | 值 |
|------|------|----|
| FTHG | 主队全场进球 | int |
| FTAG | 客队全场进球 | int |
| FTR  | 全场结果 | H/D/A |
| HTHG | 主队半场进球 | int |
| HTAG | 客队半场进球 | int |
| HTR  | 半场结果 | H/D/A |

### 赔率（多家博彩公司）
| 列前缀 | 博彩公司 |
|--------|---------|
| B365H/D/A | Bet365 |
| BWH/D/A | Bet&Win |
| GBH/D/A | Gamebookers |
| IWH/D/A | Interwetten |
| PSH/D/A | Pinnacle |
| SBH/D/A | Sportingbet |
| WHH/D/A | William Hill |
| SJH/D/A | Stan James |
| VCH/D/A | VC Bet |
| Bb1X2 | 博彩公司数量统计 |
| MaxH/D/A | 最高赔率 |
| AvgH/D/A | 平均赔率 |

### 比赛统计
| 列名 | 含义 | 格式 |
|------|------|------|
| HS/AS | 主/客射门 | int |
| HST/AST | 主/客射正 | int |
| HF/AF | 主/客犯规 | int |
| HC/AC | 主/客角球 | int |
| HY/AY | 主/客黄牌 | int |
| HR/AR | 主/客红牌 | int |
| HWO/AWO | 主/客半场角球 | int (部分赛季) |

### 季度差异
- 2023-24: 约 106 列（基础赔率 + 统计）
- 2024-25: 约 128 列（增多家博彩公司 + xG 信息）
- 2025-26: 132 列（最完整，含预计进球等高级统计）

## 已验证的 HTTP 访问 (2026-06-15)

```bash
# 直接 curl 全部返回 200 OK
curl -sI "https://www.football-data.co.uk/mmz4281/2425/E0.csv" | head -1
# HTTP/2 200

# pandas 直接读取
python3 -c "import pandas as pd; df=pd.read_csv('https://www.football-data.co.uk/mmz4281/2526/D1.csv'); print(len(df), list(df.columns)[:20])"
# 306 ['Div','Date','Time','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR',...]
```

## 与竞彩预测系统的数据整合

### 对齐字段
```python
mapping = {
    'home_en': 'HomeTeam',
    'away_en': 'AwayTeam',
    'ft_h': 'FTHG',
    'ft_a': 'FTAG',
    'spf_result': lambda r: {'H':'3','D':'1','A':'0'}[r['FTR']],
    'date': lambda r: f"{year}-{r['Date'][:2]}-{r['Date'][3:]}",
    'market_odds_h': 'B365H',
    'market_odds_d': 'B365D',
    'market_odds_a': 'B365A',
}
```

### 注意事项
1. **未来比赛**: CSV 包含尚未进行的比赛（赔率为预发布），需过滤 `FTR != ''`
2. **队名标准化**: football-data.co.uk 使用传统英文队名（如 `Nott'm Forest`、`Sheffield Utd`），需清洗
3. **赔率来源**: Bet365 赔率最常用，但少数比赛可能缺失（建议用 AvgH/D/A 兜底）
4. **xG 数据**: 2025-26 赛季部分比赛含 `xG[xGH/xGA]` 列，但覆盖率 ~40%
