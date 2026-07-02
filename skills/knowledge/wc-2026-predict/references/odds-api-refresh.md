# Odds API 刷新参考

## API Key

- 存储在 `/root/.bashrc` 第103行: `export THE_ODDS_API_KEY=425a7c...a11`
- **重要**：`.bashrc` 首行有 `[ -z "$PS1" ] && return` 守卫
  - 非交互式shell（cron、`bash -c`、subprocess）中 `source .bashrc` **无效**
  - 必须直接传API Key值，不可依赖环境变量

## 夺冠赔率 API 调用

```bash
API_KEY="425a7cb6604fe89fcbd46a524ac08a11"
curl -s -D /tmp/odds_headers.txt \
  -o /tmp/fresh_odds.json \
  "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup_winner/odds/?apiKey=${API_KEY}&regions=uk,eu,us&oddsFormat=decimal"
```

### 捕获响应头（获取剩余额度）

用 `-D /tmp/odds_headers.txt` 将响应头写入文件（不要用 `-D -` + stdout 解析，容易因 subprocess text 编码问题失败），从中提取：
- `x-requests-remaining` — 今日剩余API调用次数
- `x-requests-used` — 今日已用次数

```bash
grep -i 'x-requests-remaining' /tmp/odds_headers.txt
# x-requests-remaining: 467
```

## 响应结构

API返回 **包含1个元素的数组**（不是每队独立条目）：

```json
[{
  "id": "...",
  "sport_key": "soccer_fifa_world_cup_winner",
  "bookmakers": [
    {
      "key": "draftkings",
      "title": "DraftKings",
      "markets": [
        {
          "key": "outrights",
          "outcomes": [
            {"name": "Spain", "price": 5.3},
            {"name": "France", "price": 5.7},
            {"name": "England", "price": 6.5},
            ...
          ]
        }
      ]
    }
  ]
}]
```

## 完整刷新脚本（可复用）— 使用临时文件分离 header/body

**不推荐用 `curl -D -` + `\r\n\r\n` 在 stdout 中解析，subprocess text=True 解码环境差异会导致 json.loads 失败（已踩坑）。改用 `-D <file>` + `-o <file>` 分离。**

```python
import json, subprocess, os, tempfile

API_KEY = "425a7cb6604fe89fcbd46a524ac08a11"
ODDS_FILE = "/root/data/theodds_api_data.json"

url = f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup_winner/odds/?apiKey={API_KEY}&regions=uk,eu,us&oddsFormat=decimal"

with tempfile.NamedTemporaryFile(prefix="odds_hdr_", suffix=".txt", delete=False) as hf, \
     tempfile.NamedTemporaryFile(prefix="odds_body_", suffix=".json", delete=False) as bf:
    header_path = hf.name
    body_path = bf.name

try:
    subprocess.run(["curl", "-s", "-D", header_path, "-o", body_path, url],
                   capture_output=True, timeout=30, check=True)

    with open(body_path) as f:
        raw = json.load(f)

    # 提取剩余额度
    remaining = 0
    with open(header_path) as f:
        for line in f:
            if line.lower().startswith("x-requests-remaining:"):
                remaining = int(line.split(":")[1].strip())
                break

    # 提取winner odds（取bookmaker最低价）
    winner_odds = {}
    for entry in raw:
        for bm in entry.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market["key"] == "outrights":
                    for outcome in market["outcomes"]:
                        t, p = outcome.get("name"), outcome.get("price")
                        if t and p and (t not in winner_odds or p < winner_odds[t]):
                            winner_odds[t] = p

    # 合并旧数据的 upcoming H2H 赔率
    with open(ODDS_FILE) as f:
        old = json.load(f)

    result_data = {
        "winner_odds": winner_odds,
        "upcoming": old.get("upcoming", {}),
        "remaining_credits": remaining,
    }

    with open(ODDS_FILE, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"OK: {len(winner_odds)} teams, {remaining} credits")
finally:
    for p in [header_path, body_path]:
        try: os.unlink(p)
        except OSError: pass
```

## 注意事项

- 每日1000次免费额度（50个联赛刷新+1个夺冠赔率），每日刷新约耗20-30次
- `x-requests-last` 响应头显示本次调用所耗次数（通常为3，因使用3个regions）
- 夺冠赔率每天06:00 UTC后刷新效果最佳
- 仅刷新夺冠赔率不会影响已有H2H赔率（保存在 `upcoming` 中）
