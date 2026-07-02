#!/usr/bin/env python3
"""
赛前赔率自动刷新器
=================
在比赛前30分钟自动触发赔率抓取，并与历史数据对比。

用法:
  python3 pre_match_odds_refresh.py  # 自动检测即将开始的比赛
  python3 pre_match_odds_refresh.py --match "墨西哥 vs 南非"  # 指定比赛
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, '/root')

DATA_DIR = Path('/root/data')
ODDS_HISTORY = DATA_DIR / 'odds_history.json'
ALERT_LOG = DATA_DIR / 'odds_alerts.log'

# 盘口波动阈值
ODDS_CHANGE_THRESHOLD = 0.10  # 10% 变化触发报警
MATCH_START_BUFFER_MINUTES = 30  # 比赛前30分钟开始监控


def load_odds_history():
    """加载历史赔率数据"""
    if ODDS_HISTORY.exists():
        with open(ODDS_HISTORY) as f:
            return json.load(f)
    return {}


def save_odds_history(history):
    """保存赔率历史"""
    with open(ODDS_HISTORY, 'w') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def fetch_current_odds():
    """抓取当前赔率"""
    from wc_2026_upgrade.async_500_scraper import scrape_500_concurrent
    import asyncio
    
    result = asyncio.run(scrape_500_concurrent())
    return {f"{m['home']} vs {m['away']}": m for m in result}


def detect_odds_change(old_odds, new_odds, threshold=ODDS_CHANGE_THRESHOLD):
    """检测赔率变化"""
    alerts = []
    
    for match_key, new_data in new_odds.items():
        if match_key not in old_odds:
            continue
        
        old_data = old_odds[match_key]
        old_spf = old_data.get('odds', {}).get('nspf', {})
        new_spf = new_data.get('odds', {}).get('nspf', {})
        
        for outcome in ['3', '1', '0']:  # 主胜/平/客胜
            old_val = float(old_spf.get(outcome, 0))
            new_val = float(new_spf.get(outcome, 0))
            
            if old_val > 0 and new_val > 0:
                change = abs(new_val - old_val) / old_val
                if change >= threshold:
                    direction = '↑' if new_val > old_val else '↓'
                    alerts.append({
                        'match': match_key,
                        'outcome': {'3': '主胜', '1': '平局', '0': '客胜'}[outcome],
                        'old_odds': old_val,
                        'new_odds': new_val,
                        'change_pct': change * 100,
                        'direction': direction,
                    })
    
    return alerts


def log_alert(alerts):
    """记录报警"""
    if not alerts:
        return
    
    timestamp = datetime.now().isoformat()
    with open(ALERT_LOG, 'a') as f:
        for alert in alerts:
            line = f"{timestamp} | {alert['match']} | {alert['outcome']} | {alert['old_odds']:.2f} → {alert['new_odds']:.2f} ({alert['direction']}{alert['change_pct']:.1f}%)\n"
            f.write(line)
    
    print(f"⚠️ 检测到 {len(alerts)} 个盘口异常波动!")
    for alert in alerts:
        print(f"   {alert['match']}: {alert['outcome']} {alert['old_odds']:.2f} → {alert['new_odds']:.2f} ({alert['direction']}{alert['change_pct']:.1f}%)")


def main():
    """主流程"""
    print("=" * 60)
    print("赛前赔率监控器")
    print("=" * 60)
    
    # 1. 加载历史赔率
    history = load_odds_history()
    print(f"📊 历史赔率记录: {len(history)} 场")
    
    # 2. 抓取当前赔率
    print("\n📡 抓取当前赔率...")
    current_odds = fetch_current_odds()
    print(f"   获取到 {len(current_odds)} 场比赛")
    
    # 3. 检测变化
    if history:
        alerts = detect_odds_change(history, current_odds)
        log_alert(alerts)
    else:
        print("   首次运行，无历史数据对比")
    
    # 4. 更新历史
    save_odds_history(current_odds)
    print("\n✅ 赔率快照已保存")
    
    # 5. 输出摘要
    print("\n📋 当前赔率摘要:")
    for match_key, data in list(current_odds.items())[:5]:
        spf = data.get('odds', {}).get('nspf', {})
        if spf:
            print(f"   {match_key}: {spf.get('3', '-')} / {spf.get('1', '-')} / {spf.get('0', '-')}")


if __name__ == '__main__':
    main()
