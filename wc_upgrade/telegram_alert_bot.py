#!/usr/bin/env python3
"""
telegram_alert_bot.py — Telegram 竞彩推送机器人
==============================================
功能:
  1. --daily   : 扫描 predictions_log.csv, 过滤 RECOMMEND, 发送每日精华
  2. --rotation: 接收 recalc_on_lineup.py 输出, 发送缺阵急报
  3. --status   : 发送每日系统状态简报

读取:
  TG_BOT_TOKEN  — Telegram Bot API Token (环境变量 > .env)
  TG_CHAT_ID    — 推送目标 Chat ID (环境变量 > .env)

挂载方式:
  cron 03:15 UTC → telegram_alert_bot.py --daily
  线上 lineups cron → 捕获 recalc 输出 → telegram_alert_bot.py --rotation
"""

import os, sys, csv, json, re, requests
from datetime import datetime, date, timezone

# ════════════════════════════════════════
# 1. Telegram 连接配置
# ════════════════════════════════════════

def _load_env():
    """从 .env 加载环境变量"""
    env_path = os.path.expanduser("~/.hermes/.env")
    if not os.path.exists(env_path):
        return {}
    vals = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                vals[k.strip()] = v.strip()
    return vals

def get_tg_config():
    """读取 TG_BOT_TOKEN 和 TG_CHAT_ID"""
    env_vals = _load_env()
    token = os.environ.get('TG_BOT_TOKEN') or env_vals.get('TELEGRAM_BOT_TOKEN') or ''
    chat_id = os.environ.get('TG_CHAT_ID') or env_vals.get('TELEGRAM_HOME_CHANNEL') or env_vals.get('TELEGRAM_ALLOWED_USERS') or '5568846786'
    if not token:
        print("❌ TG_BOT_TOKEN 未配置，请设置环境变量或在 ~/.hermes/.env 中添加")
        print("   export TG_BOT_TOKEN='your_token_here'")
        sys.exit(1)
    return token, chat_id.strip()

def send_telegram(token, chat_id, text, parse_mode='HTML'):
    """发送 Telegram 消息 (HTML 格式)"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data.get('ok'):
                print(f"  ✅ Telegram 发送成功 (msg_id={data['result']['message_id']})")
                return True
            else:
                print(f"  ❌ Telegram API 错误: {data.get('description','')}")
        else:
            print(f"  ❌ HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"  ❌ 发送失败: {e}")
    return False

# ════════════════════════════════════════
# 2. 每日精华推送 (--daily)
# ════════════════════════════════════════

def send_daily_summary():
    """扫描今日 predictions_log.csv, 筛选 RECOMMEND 场次, 发送精排消息"""
    csv_path = "/root/data/predictions_log.csv"
    if not os.path.exists(csv_path):
        print(f"  ❌ {csv_path} 不存在")
        return False

    # 读取并去重 (同一场比赛取最后一条)
    from collections import OrderedDict
    match_map = OrderedDict()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            ba = r.get('bet_action', '').strip()
            actual = r.get('actual_hda', '').strip()
            if ba == 'RECOMMEND' and not actual:
                # 用 home+away+league 去重 (取最后出现的 = 最新预测)
                key = (r.get('home_cn',''), r.get('away_cn',''), r.get('league',''))
                match_map[key] = r

    matches = list(match_map.values())
    today_str = date.today().isoformat()

    if not matches:
        print("  ℹ️ 今日无 RECOMMEND 场次，跳过推送")
        return True

    # 构建 HTML 消息
    lines = []
    lines.append(f"⚽ <b>竞彩日报</b>  {today_str}")
    lines.append(f"━━━━━━━━━━━━━━━━━━")
    lines.append("")

    for m in matches:
        home = m.get('home_cn', '')
        away = m.get('away_cn', '')
        league = m.get('league', '')
        time = m.get('time', '')
        pred_h = m.get('pred_h', '')
        pred_d = m.get('pred_d', '')
        pred_a = m.get('pred_a', '')
        ev_h = m.get('ev_h', '')
        ev_d = m.get('ev_d', '')
        ev_a = m.get('ev_a', '')
        spf_pick = m.get('pred_spf_pick', '')
        rq_pick = m.get('pred_rq_pick', '')
        htft_pick = m.get('pred_htft_pick', '')
        goals_pick = m.get('pred_goals_pick', '')
        score_pick = m.get('pred_score_pick', '')

        # 格式化 EV
        def fmt_ev(v):
            try:
                f = float(v)
                return f"{'🟢' if f > 0.15 else '🟡' if f > 0 else '🔴'}{f:+.2f}"
            except (ValueError, TypeError):
                return '—'

        lines.append(f"🏆 <b>{home} vs {away}</b>")
        lines.append(f"   📺 {league}  ⏰ {time}")
        lines.append(f"   📊 <b>SPF</b> {pred_h}% / {pred_d}% / {pred_a}%")
        lines.append(f"   🎯 <b>推荐:</b> SPF→{spf_pick}  RQ→{rq_pick}  HTFT→{htft_pick}")
        lines.append(f"   🎲 比分→{score_pick}  总进球→{goals_pick}球")
        lines.append(f"   💰 EV: H {fmt_ev(ev_h)}  D {fmt_ev(ev_d)}  A {fmt_ev(ev_a)}")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append("📌 赛果需 90 分钟内裁定")
    lines.append("⚠️ 统计参考，非投注建议")

    msg = '\n'.join(lines)
    return send_telegram(*get_tg_config(), msg)


# ════════════════════════════════════════
# 3. 缺阵急报推送 (--rotation)
# ════════════════════════════════════════

def send_rotation_alert():
    """
    读取 recalc_on_lineup.py 的输出,
    提取缺阵急报信息, 发送紧急 Telegram 通知
    """
    # 优先读文件 (cron 和终端都可用)
    content = None
    alert_path = "/root/data/lineup_alert_latest.txt"
    if os.path.exists(alert_path):
        with open(alert_path) as f:
            content = f.read()

    # 回退: 从 stdin 读取 (支持管道)
    if not content and not sys.stdin.isatty():
        content = sys.stdin.read()

    if not content:
        print("  ℹ️ 无缺阵信息")
        return True

    # 提取急报段落
    alerts = []
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        if '赛前急报' in lines[i] or '推荐方向变更' in lines[i] or 'EV 翻转' in lines[i]:
            # 收集从当前行开始的完整急报段落 (向上3行 header + 向下到空行)
            start = max(0, i-3)
            block = lines[start:i+1]
            i += 1
            while i < len(lines) and lines[i].strip():
                block.append(lines[i])
                i += 1
            alerts.append('\n'.join(block))
            break
        i += 1

    if not alerts:
        print("  ℹ️ 无方向变更/急报内容")
        return True

    # 构建紧急消息
    now = datetime.now(timezone.utc).strftime('%H:%M UTC')
    header = f"🚨 <b>赛前急报</b>  {now}\n━━━━━━━━━━━━━━━━\n"
    body = '\n\n'.join(alerts)
    footer = "\n━━━━━━━━━━━━━━━━\n⚠️ 请核实后决策"

    msg = header + body + footer
    return send_telegram(*get_tg_config(), msg)


# ════════════════════════════════════════
# 4. 系统状态简报 (--status)
# ════════════════════════════════════════

def send_status_report():
    """发送今日系统状态"""
    csv_path = "/root/data/predictions_log.csv"
    stats = {'total': 0, 'recommend': 0, 'watch': 0, 'filled': 0}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                stats['total'] += 1
                ba = r.get('bet_action', '')
                if r.get('actual_hda', '').strip():
                    stats['filled'] += 1
                elif ba == 'RECOMMEND':
                    stats['recommend'] += 1
                elif 'WATCH' in ba:
                    stats['watch'] += 1

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    msg = (
        f"📊 <b>系统状态</b>  {now}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📄 predictions_log: {stats['total']} 场\n"
        f"✅ 已完结: {stats['filled']} 场\n"
        f"🏆 RECOMMEND: {stats['recommend']} 场\n"
        f"👀 WATCH: {stats['watch']} 场\n"
        f"━━━━━━━━━━━━━━━━"
    )
    return send_telegram(*get_tg_config(), msg)


# ════════════════════════════════════════
# Main
# ════════════════════════════════════════

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        print("用法:")
        print("  python3 telegram_alert_bot.py --daily      # 每日精华推送")
        print("  python3 telegram_alert_bot.py --rotation   # 缺阵急报推送")
        print("  python3 telegram_alert_bot.py --status     # 系统状态")
        sys.exit(0)

    mode = sys.argv[1]

    if mode == '--daily':
        print("📤 每日精华推送...")
        send_daily_summary()
    elif mode == '--rotation':
        print("📤 缺阵急报推送...")
        send_rotation_alert()
    elif mode == '--status':
        print("📤 系统状态...")
        send_status_report()
    else:
        print(f"❌ 未知模式: {mode}")
        print("   支持: --daily, --rotation, --status")
        sys.exit(1)


if __name__ == '__main__':
    main()
