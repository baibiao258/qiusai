# WC2026 可复现实验与审计 Runbook

## 目标
将“重训→验收→72场复算→证据打包”固定为一次可重复执行的审计流程，便于 Hermes / 人工 / Grok 交叉验证。

## 固定脚本
- `/root/run_group72_repro.py`：基于最新模型复算 72 场小组赛（输出 JSON/TXT）
- `/root/run_audit.sh`：一键执行审计管线，失败自动 `exit 1`

## 执行
```bash
cd /root
./run_audit.sh
```

## 关键门槛（脚本内置）
1. 输入文件齐全且非空：
   - `/root/wc_2026_final.py`
   - `/root/run_group72_repro.py`
   - `/root/data/international_results.json`
   - `/root/data/theodds_api_data.json`
   - `/root/data/2026_groups.json`
2. 重训日志必须命中：
   - `赛事过滤: 49257 → 4944`
   - `DC: ρ=0.2500`
   - 验证指标行（Acc/NLL/Brier）
   - `2022 WC 回测`
   - `保存: /root/data/final_results.json`
3. 硬阈值：
   - 验证集 `Brier <= 0.465`
   - `Hybrid_Brier_2022 <= DC_Brier_2022`
   - 日志中不得出现 `Traceback`
4. 72场复算：
   - `total_matches == 72`
   - 每场 `prob_home+prob_draw+prob_away` 近似 1（误差<=0.01）
5. 产物完整：
   - `/root/data/group_stage_scoreline_top5.json` 必须存在

## 输出证据包
脚本会生成：
- `/root/data/wc2026_audit_bundle_<timestamp>.tar.gz`

包含：
- `run_wc2026_audit.log`
- `final_results.json`
- `group_stage_predictions.json`
- `group_stage_predictions.txt`
- `group_stage_scoreline_top5.json`
- `2026_groups.json`
- `theodds_api_data.json`

## 典型人工验收摘录项
- DC 参数：`ρ / host_bonus / γ`
- 验证集：`Acc / Brier / NLL`
- 2022 回测：`Hybrid Brier`
- 72场复算：`matches=72`
- 审计包路径：`wc2026_audit_bundle_*.tar.gz`
