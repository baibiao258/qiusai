#!/usr/bin/env python3
"""
daily_alert.py — 每日竞彩价值投注预警
=====================================
运行 daily_jczq.py, 提取价值投注摘要, 输出邮件内容.
"""
import subprocess
import sys
import re
from datetime import date

def main():
    today = date.today().isoformat()
    wd = ['一','二','三','四','五','六','日'][date.today().weekday()]

    # 运行 daily_jczq.py
    try:
        result = subprocess.run(
            ['python3', '/root/daily_jczq.py'],
            capture_output=True, text=True, timeout=300,
            cwd='/root'
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"⚠️ daily_jczq.py 超时 (300s)")
        return
    except Exception as e:
        print(f"⚠️ 运行失败: {e}")
        return

    lines = output.split('\n')

    # 提取 "💎 价值投注汇总" 到 "💎 购彩建议" 之间的内容
    in_summary = False
    summary_lines = []
    advice_lines = []

    for line in lines:
        if '💎 价值投注汇总' in line:
            in_summary = True
            summary_lines.append(line)
            continue
        if in_summary:
            if '💎 购彩建议' in line:
                in_summary = False
                advice_lines.append(line)
                continue
            summary_lines.append(line)
        if '💎 购彩建议' in line:
            advice_lines.append(line)

    # 提取场次统计
    match_count = 0
    value_count = 0
    for line in lines:
        m = re.search(r'(\d+)\s*场竞彩赛事', line)
        if m:
            match_count = int(m.group(1))
        m2 = re.search(r'价值投注:\s*(\d+)\s*个', line)
        if m2:
            value_count = int(m2.group(1))

    # 构建邮件内容
    email_lines = []
    email_lines.append(f"⚽ 竞彩价值投注预警 {today} 周{wd}")
    email_lines.append(f"{'='*50}")
    email_lines.append("")

    if summary_lines:
        for line in summary_lines:
            email_lines.append(line.rstrip())
    else:
        email_lines.append("⚠️ 今日无价值投注汇总")

    email_lines.append("")
    if advice_lines:
        for line in advice_lines[:20]:
            email_lines.append(line.rstrip())

    email_lines.append("")
    email_lines.append("⚠️ 基于统计模型, 不构成投注建议. 请理性购彩.")

    print('\n'.join(email_lines))


if __name__ == '__main__':
    main()
