# P1 进球分布校准 (2026-06-30)

## 问题

总进球预测锚定"2 球"。回测 167 场中仅 27 场正确 (16.2%)，错误记录中大量 pred=2 而实际分布为 0-6。

## 根因

DC 双泊松模型的经典问题：双方 λ 接近 1 时泊松卷积峰值天然落在 2 球。同时 compute_goals_distribution() 和 compute_score_topn() 缺少 rho 修正，低比分概率偏高。

## 修复

### 改动文件

/root/daily_jczq.py

### 新增函数

dc_tau(): 从 wc_predictor_v3.py 搬入 (P0 compute_rq_probs 依赖但未定义)

### 函数签名变更

compute_goals_distribution(lambda_home, lambda_away, rho=0.0)
compute_score_topn(lambda_home, lambda_away, topn=8, rho=0.0)

### 循环逻辑

if rho != 0.0: p *= dc_tau(hg, ag, lambda_home, lambda_away, rho)

### 调用点 (build_prediction_bundle)

goals_dist = compute_goals_distribution(..., rho=dc_rho)
score_top8 = compute_score_topn(..., rho=dc_rho)
score_all = compute_score_topn(..., topn=999, rho=dc_rho)

### 数据流

fit_dc_model() → p['rho'] → build_prediction_bundle → rq/进球/比分函数

## 验证

python3 -c "import ast; ast.parse(open('/root/daily_jczq.py').read()); print('OK')"
