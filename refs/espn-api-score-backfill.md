# ESPN API 赛果回填 (WC 2026)

**发现日期**: 2026-06-27  
**用途**: 当标准回填管线（kaijiang / 365scores / TheStatsAPI）缺失 WC 2026 实际赛果时，用 ESPN API 作为补充回填源。

## 端点

```
https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=YYYYMMDD
```

- **认证**: 无需 API Key，免费开放
- **格式**: JSON，无需代理
- **限速**: 未观察到严格限速（实测并发无问题）

## 数据结构

```
events[]:
  - shortName: "SWI @ BOS" (显示名)
  - status.type.name: "STATUS_FULL_TIME" | "STATUS_SCHEDULED" | "STATUS_IN_PROGRESS"
  - competitions[0].competitors[]:
      - team.displayName: "Switzerland" (队名)
      - score: "4" (进球数)
```

## 用法示例

```python
import requests, json

def espn_fetch_scoreboard(date_str: str) -> list[dict]:
    """
    date_str: YYYYMMDD format
    Returns: [{home, away, home_score, away_score, status, shortName}]
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={date_str}"
    resp = requests.get(url, timeout=15)
    data = resp.json()
    matches = []
    for e in data.get('events', []):
        sc = e.get('competitions', [{}])[0].get('competitors', [])
        if len(sc) >= 2:
            status = e.get('status', {}).get('type', {}).get('name', '')
            n1 = sc[0].get('team', {}).get('displayName', '')
            n2 = sc[1].get('team', {}).get('displayName', '')
            s1 = sc[0].get('score', '?')
            s2 = sc[1].get('score', '?')
            matches.append({
                'home': n1, 'away': n2,
                'home_score': s1, 'away_score': s2,
                'status': status,
            })
    return matches
```

## 与 predictions_log.csv 的匹配逻辑

predictions_log.csv 存储中文队名（如 `瑞士`、`波黑`）。ESPN 返回英文队名（如 `Switzerland`、`Bosnia-Herzegovina`）。匹配需用 `team_name_mapping.json` 做 cn→en 映射。

```python
# 双向映射: 中文 → 英文
mapping = json.load(open('/root/data/team_name_mapping.json'))
cn_to_en = {v: k for k, v in mapping.items()}  # 按需扩展

# 匹配示例
cn_home = '瑞士'
en_home = cn_to_en.get(cn_home, cn_home)
# 遍历 ESPN matches, 找 en_home == match['home'] 且 en_away == match['away']
```

注意 ESPN 英文队名格式可能与 mapping 略有差异（如 `Bosnia-Herzegovina` vs `Bosnia & Herzegovina`）。匹配失败时打印 `⚠️ 队名不匹配` 并跳过。

## 已知局限

1. **仅限 FIFA World Cup**: 端点路径含 `fifa.world`。其他联赛需换 league slug（如 `uefa.euro`、`eng.1`）
2. **无历史回滚**: API 只返回近期的比赛。2025 年比赛不可查
3. **队名不标准**: 与 TheStatsAPI / football-data.org 的队名版本可能不同
4. **实时状态**: STATUS_FULL_TIME 表示已完赛有比分，STATUS_SCHEDULED 表示未开赛
