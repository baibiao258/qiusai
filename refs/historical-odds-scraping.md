# Historical Odds Scraping (kaijiang page)

## Overview

The `historical_kaijiang.py` script scrapes historical closing SP odds from 500.com's kaijiang (开奖) page, providing the ground truth for real odds backtesting.

## URL Pattern

```
https://zx.500.com/jczq/kaijiang.php?playid=0&d=YYYY-MM-DD
```

- `playid=0`: All play types
- `d`: Date in YYYY-MM-DD format
- Supports date range iteration for batch scraping

## HTML Structure

The page contains a `table.ld_table` with match data:

### Data Row Format (19 cells per row)

| Cell | Content | Example |
|------|---------|---------|
| [0] | Match code | 周二201 |
| [1] | League | 友谊赛 |
| [2] | Kickoff time | 06-09 19:35 |
| [3] | Home team | 中国 |
| [4] | Handicap | -1 |
| [5] | Away team | 泰国 |
| [6] | Score (HT/FT) | (1:0) 2:1 |
| [7] | Separator | - |
| [8] | rqspf result | 平 |
| [9] | rqspf SP (span.red) | 3.40 |
| [10] | Separator | - |
| [11] | spf result | 负 |
| [12] | spf SP (span.red) | 2.21 |
| [13] | Separator | - |
| [14] | jqs result | 3 |
| [15] | jqs SP (span.red) | 3.40 |
| [16] | Separator | - |
| [17] | bqc result | 胜胜 |
| [18] | bqc SP (span.red) | 4.10 |

### Key Parsing Rules

1. **Result encoding**: 胜=3, 平=1, 负=0 (for spf/rqspf)
2. **SP extraction**: Always from `span.red` within the cell
3. **Missing odds**: Show as `--` in HTML, parse as 0.0
4. **Score format**: `(HT:HT) FT:FT` e.g., `(1:0) 2:1`

## Output Format

CSV with columns:
```csv
date,code,league,time,home,away,handicap,handicap_str,ht_h,ht_a,ft_h,ft_a,spf_result,spf_sp,rqspf_result,rqspf_sp,total_goals,jqs_result,jqs_sp,bqc_result,bqc_result_cn,bqc_sp
```

## Usage

```bash
# Single day test
python3 historical_kaijiang.py --single 2026-06-08

# Full historical backfill (2024-01-01 to yesterday)
python3 historical_kaijiang.py --start 2024-01-01 --delay 0.5

# Resume interrupted scrape (uses progress file)
python3 historical_kaijiang.py --start 2024-01-01
```

## Integration with Backtest

The scraped data is used by `real_odds_backtest.py`:

1. Load `historical_kaijiang.csv` (3248 matches, 2024-01-01 to 2026-06-08)
2. Merge with `international_results.json` using team name mapping
3. Match on `[home_en, away_en, date ±2 days]`
4. Use closing SP for EV calculation

## Team Name Mapping

Chinese team names from 500.com need mapping to English names in international_results.json:

```json
{
  "阿根廷": "Argentina",
  "巴西": "Brazil",
  "法国": "France",
  "德国": "Germany",
  "英格兰": "England",
  "荷兰": "Netherlands",
  "西班牙": "Spain",
  "葡萄牙": "Portugal",
  "意大利": "Italy",
  "克罗地亚": "Croatia"
}
```

Full mapping: `/root/data/team_name_mapping.json` (101 entries)

## Pitfalls

1. **J-League clubs**: 500.com includes club matches (J-League, K-League, etc.) that are NOT in international_results.json. These are automatically filtered out during merge.

2. **Encoding**: Page uses GBK encoding. Always decode with `raw.decode('gbk', errors='replace')`.

3. **Rate limiting**: Add 0.5s delay between requests to avoid being blocked.

4. **Progress tracking**: Script saves progress to `historical_kaijiang_progress.json` for resume capability.

5. **Date range**: 500.com kaijiang pages go back to at least 2024-01-01. Older data may not be available.
