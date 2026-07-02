# 双 DC 模型 + 校准预测器 (2026-06-14)

## 架构概览

```
比赛 (home, away, 市场赔率)
  │
  ▼ 队名标准化 (USA→United States, Bosnia & Herzegovina→Bosnia and Herzegovina)
  │
  ┌────┴────┐
  │ 在dc_model │  (226 国家队)
  │ 中吗?     │
  └────┬────┘
       │
  ┌────┴────┐           ┌─────────────┐
  │ YES     │           │ NO          │
  │ Pipeline A          │ Pipeline B  │
  │ xgb_model_nat.pkl   │ dc_club.pkl │
  │ + dc_model          │ + Elo       │
  │ + DC置信度加权       │ + Market    │
  └─────────┘           └─────────────┘
                           Fallsback
                              │
                        ┌─────┴─────┐
                        │ C: Elo+Market │
                        └───────────┘
```

## 三个 DC 模型

| 模型 | 文件 | 球队数 | 数据源 | 队名语言 |
|------|------|--------|--------|---------|
| 国家队 | dc_model.pkl | 226 | international_results.json | 英文 |
| 俱乐部(中) | dc_club.pkl | 2,174 | 500_history_backfill.csv (63K场) | 中文 |
| 俱乐部(英) | dc_club_en.pkl | 152 | football-data.org (2,743场) | 英文 |

## 训练脚本

| 脚本 | 位置 | 说明 |
|------|------|------|
| 国家队XGB | /root/wc_2026_upgrade/train_national_xgb.py | 只用英文名数据, 过滤96场中文名, 64.4% val acc |
| 俱乐部DC | /root/wc_2026_upgrade/train_club_dc.py | 从500_history_backfill.csv训练, 2,174队 |
| 英文DC | /root/wc_2026_upgrade/train_club_dc_en.py | 从football-data.org训练, 队名格式不兼容问题 |
| 清理XGB | /root/wc_2026_upgrade/train_clean_xgb.py | 11维干净特征, 去掉14个死特征 |
| 预测器 | /root/wc_2026_upgrade/calibrated_predictor.py | 双管线+置信度加权+队名标准化 |

## 已发现的 Bug

### Bug 1 — 特征顺序反转 (dc_a/dc_h 互换)
**症状**: 强队预测为极低胜率 (Germany H=9.6%, 市场92.8%)
**根因**: predict_wc_today.py 中 feature array 的 DC 概率索引写反
**修复**: dc_probs已经是[A,D,H]顺序, 直接填入对应位置即可

### Bug 2 — 显示映射反转
**症状**: 所有预测主客颠倒 (91.7% H 显示为 7.7%)
**根因**: hy = [A, D, H] 但输出赋值 results['h'] = hy[0] (实际是A)
**修复**: results['h'] = hy[2]; results['a'] = hy[0]

### Bug 3 — Isotonic 校准器过度压低强信号
**症状**: 99.7% H → 90.6% H (Germany vs Curaçao)
**根因**: 校准器在275场训练数据上过拟合
**应对**: 置信度加权用 xgb_weight = 0.5 + 0.3*conf 保持 XGB 主导

### Bug 4 — 队名不匹配降级
**症状**: "USA" 在 dc_model 中没有 (存为 "United States")
**修复**: _normalize_team() 映射覆盖 USA, Türkiye, Bosnia, Côte d'Ivoire

## 置信度加权

出场数 → 置信度: ≥200=1.0, ≥100=0.9, ≥50=0.8, ≥20=0.7, ≥10=0.5, ≥5=0.3, <5=0.1

融合: final = dc_conf * hy + (1-dc_conf) * base
base = 0.3*Elo + 0.7*Market

## 数组顺序约定

全局统一的内部约定: **DC 概率存储为 [A, D, H]**
- dc_model.predict_proba() 返回 [H, D, A] → 立刻转为 [A, D, H] 存储
- XGBoost predict_proba 返回 [P(A=0), P(D=1), P(H=2)] = [A, D, H] ✓
- hy = DC_weight * dc_ado + XGB_weight * xgb_p → [A, D, H] ✓
- Market probs: [rel_a, 0, rel_h] → [A, D, H] ✓
- 仅在展示时转回 [H, D, A]

## 关键文件

| 文件 | 路径 |
|------|------|
| 国家队XGB | /root/data/xgb_model_nat.pkl |
| 国家队校准器 | /root/data/calibrators_nat.pkl |
| 俱乐部DC(中) | /root/data/dc_club.pkl |
| 俱乐部DC(英) | /root/data/dc_club_en.pkl |
| 世界杯赔率 | /root/data/wc_2026_odds_today.json |
| 最终预测JSON | /root/data/wc_final_predictions.json |
| 最终预测文本 | /root/data/wc_final_predictions.txt |
| 预测器代码 | /root/wc_2026_upgrade/calibrated_predictor.py |
