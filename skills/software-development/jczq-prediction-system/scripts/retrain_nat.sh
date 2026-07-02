#!/bin/bash
# retrain_nat.sh — 一站式纳模型重训
# 用法: ./retrain_nat.sh [api_key]
# 如果提供api_key则先拉取TheStatsAPI最新数据再合并重训
# 不提供则直接用现有 training_data_with_odds.json 重训

set -e
cd /root/wc_2026_upgrade

if [ -n "$1" ]; then
    echo "=== Phase 1: Fetch latest TheStatsAPI data ==="
    THE_KEY="$1" python3 phase1_fetch.py 2>&1 | tail -20
    
    echo "=== Phase 2: Fetch stats+odds ==="
    THE_KEY="$1" python3 -u phase2_backfill.py 2>&1 | tail -5
    
    echo "=== Merge ==="
    python3 merge_training_data.py 2>&1 | tail -10
fi

echo "=== Retrain ==="
python3 retrain_nat.py 2>&1 | tail -20

echo "=== Verify model loaded ==="
python3 -c "
import joblib
m = joblib.load('/root/data/xgb_model_nat.pkl')
print(f'Model: {type(m).__name__}')
print(f'Features: {m.n_features_in_}')
"