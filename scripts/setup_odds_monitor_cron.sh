#!/bin/bash
# 赛前赔率监控 Cron 配置脚本
# 在世界杯期间每30分钟检查一次即将开始的比赛

echo "配置赛前赔率监控 Cron Job..."

# 创建 cron job: 每30分钟运行一次检查
CRON_CMD="*/30 * * * * cd /root && python3 /root/scripts/pre_match_odds_refresh.py >> /root/data/odds_refresh.log 2>&1"

# 检查是否已存在
if crontab -l 2>/dev/null | grep -q "pre_match_odds_refresh.py"; then
    echo "⚠️ Cron job 已存在"
else
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ Cron job 已添加: 每30分钟检查盘口变化"
fi

echo ""
echo "当前 Cron 配置:"
crontab -l | grep "odds"
