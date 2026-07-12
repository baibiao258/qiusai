#!/usr/bin/env python3
"""
telegram_bot.py — 每日竞彩预测 Telegram 自动推送
===============================================
读取 daily_jczq.py 刚刚写入的 predictions_log.csv,
提取 推荐 场次, 格式化 Markdown 消息并推送.

使用:
  python3 telegram_bot.py                  # 正常推送
  python3 telegram_bot.py --dry-run         # 仅打印, 不发送
  python3 telegram_bot.py --force           # 强制发送 (忽略 enabled=false)

执行顺序 (run_pipeline.sh):
  1. python3 update_tournament_state.py --force
  2. python3 daily_jczq.py
  3. python3 telegram_bot.py
"""

import csv, json, os, sys, requests
from datetime import date, datetime
from pathlib import Path

# ── 路径 ──
CONFIG_PATH = '/root/telegram_config.json'
LOG_PATH = '/root/data/predictions_log.csv'

# ── Quarter-Kelly 计算 (与 daily_jczq.py 一致) ──
def calc_kelly_pct(ev_str, odds_str):
    """从 CSV 中的 EV 和赔率计算 Quarter-Kelly 百分比 (小数)"""
    if not ev_str or not odds_str:
        return 0.0
    try:
        ev = float(ev_str)
        odds = float(odds_str)
        if odds <= 1 or ev <= 0:
            return 0.0
        kelly_f = ev / (odds - 1)
        return max(0.0, kelly_f / 4.0)
    except (ValueError, ZeroDivisionError):
        return 0.0


# ── 读取配置 ──
def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ 配置文件不存在: {CONFIG_PATH}")
        print(f"   请创建 {CONFIG_PATH} 并填入 bot_token 和 chat_id")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    # --dry-run 模式下允许占位 token (仅打印预览)
    is_dry_run = '--dry-run' in sys.argv
    if not is_dry_run:
        if not cfg.get('bot_token') or cfg['bot_token'] == 'YOUR_BOT_TOKEN':
            print("⚠️ Telegram 未配置 (bot_token 为空或占位值)")
            print(f"   请编辑 {CONFIG_PATH} 填入实际 token")
            sys.exit(0)
    return cfg


# ── 读取今日预测 ──
def load_today_predictions():
    """从 CSV 读取今天的 推荐 比赛"""
    if not os.path.exists(LOG_PATH):
        print(f"❌ 预测日志不存在: {LOG_PATH}")
        return []

    today_str = date.today().isoformat()
    recommends = []

    with open(LOG_PATH, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 只取今天的场次
            row_date = row.get('date', '').strip()
            if row_date != today_str:
                continue

            bet_action = row.get('bet_action', '').strip()
            if '推荐' not in bet_action:
                continue

            # 提取字段
            home = row.get('home_cn', '')
            away = row.get('away_cn', '')
            league = row.get('league', '')
            time_str = row.get('time', '')
            spf_pick = row.get('pred_spf_pick', '')

            # 赔率 (选取推荐方向对应的)
            odds_map = {
                '主胜': row.get('odds_h', ''),
                '平': row.get('odds_d', ''),
                '客胜': row.get('odds_a', ''),
            }
            ev_map = {
                '主胜': row.get('ev_h', ''),
                '平': row.get('ev_d', ''),
                '客胜': row.get('ev_a', ''),
            }
            odds_val = odds_map.get(spf_pick, '')
            ev_val = ev_map.get(spf_pick, '')

            # Kelly 计算 (优先从 CSV 的 kelly_pct 列读取)
            kelly_str = row.get('kelly_pct', '').strip()
            if kelly_str:
                try:
                    kelly = float(kelly_str)
                except ValueError:
                    kelly = calc_kelly_pct(ev_val, odds_val)
            else:
                kelly = calc_kelly_pct(ev_val, odds_val)

            # 概率
            probs = {
                '主胜': row.get('pred_h', '0'),
                '平': row.get('pred_d', '0'),
                '客胜': row.get('pred_a', '0'),
            }
            prob_pick = probs.get(spf_pick, '0')

            # 比分/总进球推荐
            top_score = row.get('pred_top_score', '')
            top_goals = row.get('pred_top_goals', '')

            recommends.append({
                'home': home,
                'away': away,
                'league': league,
                'time': time_str,
                'spf_pick': spf_pick,
                'odds': odds_val,
                'ev': ev_val,
                'kelly': kelly,
                'prob': prob_pick,
                'top_score': top_score,
                'top_goals': top_goals,
                'code': row.get('code', ''),
            })

    return recommends


# ── 格式化 Markdown 消息 ──
def format_message(recommends):
    if not recommends:
        return "🏳️ *今日无符合 EV 和安全边际的推荐赛事*，建议观望。"

    lines = []
    today_cn = ['一', '二', '三', '四', '五', '六', '日'][date.today().weekday()]
    lines.append(f"📊 *每日竞彩预测推送*  {date.today().isoformat()} 周{today_cn}")
    lines.append(f"")

    total_kelly = 0.0

    for i, r in enumerate(recommends, 1):
        # 清理队名中的排名数字 e.g. "[2]法国" → "法国"
        home_clean = r['home'].split(']')[-1] if ']' in r['home'] else r['home']
        away_clean = r['away'].split(']')[-1] if ']' in r['away'] else r['away']

        ev_str = f"+{r['ev']}" if r['ev'] and float(r['ev']) > 0 else r['ev']
        kelly_pct = r['kelly'] * 100
        total_kelly += r['kelly']

        lines.append(f"*{i}. {home_clean} vs {away_clean}*")
        lines.append(f"   🏆 {r['league']}  |  ⏰ {r['time']}")
        lines.append(f"   🎯 推荐: *{r['spf_pick']}* (概率 {r['prob']}%)")
        lines.append(f"   💰 赔率 {r['odds']}  |  EV {ev_str}")
        lines.append(f"   📊 建议仓位: *{kelly_pct:.1f}%* 总资金")
        lines.append(f"   🔮 参考比分: {r['top_score']}  |  总进球: {r['top_goals']}")
        lines.append("")

    # 总仓位
    lines.append(f"───")
    lines.append(f"💰 *建议总仓位: {total_kelly*100:.1f}%* (Quarter-Kelly)")

    if total_kelly * 100 > 15:
        lines.append(f"⚠️ *当日并发总仓位超过 15% 上限*，建议按比例缩减投注额！")

    lines.append("")
    lines.append("🤖 *Hermes 量化预测系统*")
    lines.append("⚽ 数据源: TheStatsAPI + 365scores + 500.com")
    lines.append("📐 模型: DC泊松 + Pinnacle市场校正 + Quarter-Kelly资金管理")
    lines.append("📢 *本推送基于统计数据，不构成投注建议，请理性购彩*")

    return "\n".join(lines)


# ── 发送 Telegram 消息 ──
def send_telegram(bot_token, chat_id, message):
    """通过 Telegram Bot API 发送 Markdown 消息"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            result = resp.json()
            if result.get('ok'):
                print(f"✅ 推送成功: {result['result']['message_id']}")
            else:
                print(f"❌ 推送失败: {result.get('description', '未知错误')}")
        else:
            print(f"❌ HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"❌ 发送异常: {e}")


# ── 主入口 ──
def main():
    dry_run = '--dry-run' in sys.argv
    force = '--force' in sys.argv

    # 加载配置
    cfg = load_config()
    bot_token = cfg['bot_token']
    chat_id = cfg['chat_id']
    enabled = cfg.get('enabled', True)

    if not enabled and not force and not dry_run:
        print("ℹ️ Telegram 推送已禁用 (enabled=false), 使用 --force 强制发送")
        return

    # 读取今日推荐
    recommends = load_today_predictions()
    print(f"  推荐: {len(recommends)} 场")

    # 格式化消息
    message = format_message(recommends)
    print(f"\n{'='*50}")
    print("推送内容预览:")
    print(f"{'='*50}")
    print(message)
    print(f"{'='*50}")

    if not dry_run:
        label = f" ({len(recommends)} 场推荐)" if recommends else " (无推荐场次通知)"
        print(f"\n📤 正在推送 Telegram{label}...")
        send_telegram(bot_token, chat_id, message)
    else:
        print(f"\n🏁 --dry-run 模式, 未发送")

    print(f"\n🏁 telegram_bot.py 完成")


if __name__ == '__main__':
    main()
