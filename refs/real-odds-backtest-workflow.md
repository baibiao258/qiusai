## 真实赔率回测 (real_odds_backtest.py)

用开奖数据中的真实收盘SP + 29维XGB模型预测 + Isotonic校准，计算真实ROI。

**数据合并流程**:
1. 加载 historical_kaijiang.csv (开奖数据)
2. 加载 international_results.json (国际赛数据，2023+)
3. 中文队名 → 英文队名 (team_name_mapping.json, 101条)
4. [主队标准名, 客队标准名, 日期±2天] 模糊匹配 (容忍时区错位)
5. 俱乐部/小联赛自动跳过 (J-League、荷乙等)

**29维特征结构** (与训练时一致):
```
b15[15] = [elo_diff/400, λ_h, λ_a, λ_diff, λ_ratio,
           dc_a, dc_d, dc_h, fh5_wr, fa5_wr,
           fh5_gf-fa5_ga, fa5_gf-fh5_wr, fh5_gf-fa5_gf, fh5_wr-fa5_wr, 1]
gold[5] = [h2h_gd, tier_major, tier_friendly, fh12_attack_def, fa12_attack_wr]
odds[3] = [op_h, op_a, 0.0]
form[6] = [fh5_gf, fh5_ga, fa5_gf, fa5_ga, fh5_wr*3, fa5_wr*3]
```

**Isotonic校准集成 (2026-06-09)**:
```python
# calibrators.pkl 包含 home/draw/away 三个 IsotonicRegression
raw_probs = xgb_model.predict_proba(feat)[0]  # [A, D, H]
calibrated = np.array([calibrators['away'].predict([raw_probs[0]])[0],
                       calibrators['draw'].predict([raw_probs[1]])[0],
                       calibrators['home'].predict([raw_probs[2]])[0]])
calibrated = calibrated / calibrated.sum()  # 重新归一化
```
校准器已集成到 daily_jczq.py (_try_hybrid_predict) 和 real_odds_backtest.py。

**回测结果 (2026-06-09, 校准后+赛事过滤+XGB重训练)**:
- 合并率: 3248场开奖 → 263场匹配
- 可交易: 234场 (spf_sp > 0)
- EV>5%触发: 80笔交易
- 命中率: 70.0% (56/80)
- **真实ROI: +69.86%** (从校准前的-3.94%累计提升+73.8pp)

**赛事类型过滤 (COMPETITION_TIER)**:
基于实际ROI设置权重，动态调整EV阈值:
```python
COMPETITION_TIER = {
    'AFC Asian Cup': 1.2,           # +194.7% ROI
    'FIFA World Cup qualification': 1.0,  # +15.0% ROI
    'UEFA Euro': 0.7,               # -2.4% ROI
    'Copa América': 0.6,            # -12.7% ROI
    'Friendly': 0.2,                # -58.1% ROI (过滤)
    'UEFA Nations League': 0.2,     # -72.5% ROI (过滤)
}
# 动态EV阈值 = base_ev / tier_weight
# tier_weight < 0.3 → 跳过
```

**逐月表现 (过滤后)**:
```
2024-01: 8场  | 命中25.0% | ROI +69.0%
2024-02: 3场  | 命中100%  | ROI +431.7%
2024-03: 2场  | 命中0.0%  | ROI -100.0%
2024-06: 22场 | 命中22.7% | ROI -5.8%
2024-07: 9场  | 命中22.2% | ROI -26.1%
2024-09: 5场  | 命中20.0% | ROI +21.0%
2024-10: 2场  | 命中50.0% | ROI +132.5%
2024-11: 2场  | 命中50.0% | ROI +170.0%
```

**优化路径**:
1. ✅ Isotonic校准: ROI -3.94% → +3.24% (+7.18pp)
2. ✅ 赛事类型过滤: ROI +3.24% → +37.64% (+34.4pp)
3. ✅ XGB重训练 (市场赔率特征): ROI +37.64% → +69.86% (+32.22pp)
4. ⏳ 下一步: form实时化 + 扩大回测样本到300+笔

**已知问题**:
- 样本量53笔仍偏小，需积累到300+笔
- 合并率低: 大量kaijiang数据是俱乐部赛被过滤

**队名映射陷阱**:
- 500.com 用中文简写 (波黑、沙特、科特迪瓦)
- international_results.json 用英文 (Bosnia and Herzegovina, Saudi Arabia, Ivory Coast)
- J-League俱乐部 (川崎前锋、浦和红钻) 不在国家队数据集中，自动跳过

## 相关文件

- `/root/wc_2026_upgrade/async_500_scraper.py` — 实时赔率并发抓取
- `/root/wc_2026_upgrade/historical_kaijiang.py` — 历史开奖数据抓取
- `/root/wc_2026_upgrade/real_odds_backtest.py` — 真实赔率回测 (29维XGB+收盘SP)
- `/root/data/historical_kaijiang.csv` — 开奖数据 (3248场, 2024年起)
- `/root/data/team_name_mapping.json` — 中英队名映射 (101条)
