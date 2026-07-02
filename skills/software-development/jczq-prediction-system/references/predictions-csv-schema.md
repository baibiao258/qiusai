# predictions_log.csv 字段规范

## 核心字段

| 字段 | 类型 | 说明 |
|------|------|------|
| code | str | 场次代码 (如 周三201) |
| date | str | **预测运行日期** (YYYY-MM-DD, 非比赛日, 是 daily_jczq.py 运行的当天) |
| match_date | str | **实际比赛日期** (YYYY-MM-DD, 从 500.com data-matchdate 或 TheStatsAPI utc_date 提取) |
| time | str | 比赛时间 (UTC) |
| home_cn / away_cn | str | 中文队名 |
| league | str | 联赛类型 |
| rq | int | 让球数 (负=让球, 正=受让) |

## 模型预测 (百分比值, 66.0=66%)

| 字段 | 说明 |
|------|------|
| pred_h / pred_d / pred_a | 胜平负概率 (%) |
| pred_rq_win / pred_rq_draw / pred_rq_loss | 让球胜平负概率 (%) |
| pred_top_score / pred_top_goals / pred_top_htft | 最可能比分/进球/半全场 |
| pred_spf_pick / pred_rq_pick / pred_htft_pick / pred_goals_pick / pred_score_pick | 各玩法推荐 |

## 市场赔率

| 字段 | 说明 |
|------|------|
| odds_h / odds_d / odds_a | SPF 市场赔率 |
| ev_h / ev_d / ev_a | EV 值 |
| direction | 方向判断串 |

## 365scores 数据

| 字段 | 类型 | 说明 |
|------|------|------|
| vote_h / vote_d / vote_a | float | 公众投票 (%) |
| vote_count | int | 投票人数 |
| vote_fusion_alpha | float | 投票融合权重 (0.05-0.30) |
| pop_rank_home / pop_rank_away | int | 人气排名 (越低越热门) |
| pop_rank_diff | int | 人气排名差 (away-home) |
| trend_win_rate_home / trend_win_rate_away | float | 近5场胜率 (0-1) |
| trend_win_rate_diff | float | 胜率差 (home-away) |
| s365_home_winrate / s365_away_winrate | float | 365scores 近5场胜率 (2026-06-10) |
| s365_home_fifa / s365_away_fifa | int | FIFA排名 (2026-06-10) |
| s365_rank_diff | int | FIFA排名差 (away-home, 正=主队更强) |
| s365_popularity_diff | int | 人气差 (home-away) |

## 赛果回填

| 字段 | 说明 |
|------|------|
| actual_score / actual_ht | 全场/半场比分 |
| actual_hda | 胜平负彩果 (H/D/A) |
| actual_rq_result | 让球彩果 |
| actual_goals | 总进球数 |
| actual_htft | 半全场彩果 |
| brier_spf | 单场 Brier Score |
| brier_rq | 让球 Brier Score |
| acc_score_top1 | 比分 Top-1 命中 |
| acc_goals_top1 | 总进球 Top-1 命中 |
| goals_mae | 总进球 MAE |
| acc_htft_top1 | 半全场 Top-1 命中 |
| result_status | missing/filled/conflict/postponed |
| settled_at | 回填完成时间 (ISO) |
| backfill_source | 回填数据来源 |

## 模型元数据

| 字段 | 说明 |
|------|------|
| source_tag | 数据源标签 (500+365) |
| model_version | 预测版本 |
| bet_action | 赛事过滤标签 (RECOMMEND/WATCH/SKIP_LEAGUE) |
| model_route | 模型路由 (xgb_dc_nat_11d/market_fallback/club) |
| match_key | 稳定主键: `date|league|home|away|time` |
| kelly_pct | Quarter-Kelly 推荐仓位 (小数) |
| simple_pred / simple_conf | 并行模型预测 |
| pred30_h / pred30_d / pred30_a | A/B: 30维模型概率 (%) |
| score_full / htft_full / goals_full | 完整概率分布 (JSON) |

## 重要陷阱

1. **`date` 列是预测运行日, 不是比赛日**: `backtest_pipeline.py --verify` 用 `date >= today` 跳过新生成的预测, 即使比赛已结束。正确做法是用 `match_date` 列或解析 `time` 列。`backfill_results.py` 也有相同的保守行为。
2. **过滤比赛按 `match_date` 而非 `code`**: `code` 列如"周四025"中的"周四"是中文星期, 不唯一对应实际日期。用 `df[df['match_date'] == '2026-06-18']` 而非 `df['code'].str.contains('周四')`。
3. **概率是百分比值**: `pred_h=66.0` 表示 66%，不是 0.66。下游代码直接取用，不要 ×100。
4. **CSV 新增字段三处同步**: backtest_jczq.py FIELDS + cmd_record() + daily_jczq.py record_prediction() cmd 列表, 三者缺一不可。
5. **backfill 不覆盖已有值**: 只填 result_status=missing 的记录。
6. **match_key 含 date (预测运行日)**: 因此同一场比赛在不同预测日会生成不同的 match_key。去重时应用 `code|home_cn|away_cn` 三元组而非 match_key。

7. **JSON 字段逗号导致列错位 (CRITICAL)**: `score_full`/`htft_full`/`goals_full` 是嵌入 CSV 的 JSON 字符串（含逗号），当引号不规范时，`csv.DictReader` 会把 JSON 内部的逗号当成字段分隔符，导致后续列（pred_h/d/a/model_route）读到错误位置。具体表现为 `pred_h` 出现 `'market_fallback'` 等非数值值触发的 `ValueError`。

   识别方法：某行逗号数 ≠ 表头逗号数。以下命令批量检测：
   ```
   hc=$(head -1 predictions_log.csv | grep -o ',' | wc -l)
   while IFS= read -r l; do
     [ "$(echo "$l" | grep -o ',' | wc -l)" -ne "$hc" ] && echo "BAD LINE"
   done < predictions_log.csv
   ```

   修复方向：写入端确保 JSON 字段用标准 CSV 双引号引用（Python csv.writer 默认行为）。读取端在 float(r.get('pred_h', 0)) 等位置加 try/except (ValueError, TypeError) 跳过损坏行。

   当前状态 (2026-06-28)：偶发。写入端 record_prediction() 使用 subprocess 传参而非 csv.writer，JSON 字符串引号转义可能不一致。已损坏行只能通过读取端跳过。
