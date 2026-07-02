#!/bin/bash
# run_pipeline.sh — 每日竞彩预测全管线
# 顺序: tournament_state → daily_jczq → telegram_bot
set -e
cd /root

echo "=== 🏟️ 管线启动: $(date '+%Y-%m-%d %H:%M') ==="

echo ""
echo "=== Step 1/3: 更新赛事状态 ==="
python3 /root/update_tournament_state.py --force || echo "⚠️ 赛事状态更新跳过（非致命）"

echo ""
echo "=== Step 2/3: 竞彩预测 ==="
python3 /root/daily_jczq.py

echo ""
echo "=== Step 3/3: Telegram 推送 ==="
python3 /root/telegram_bot.py

echo ""
echo "=== ✅ 管线完成: $(date '+%Y-%m-%d %H:%M') ==="
