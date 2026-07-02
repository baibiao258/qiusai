#!/usr/bin/env python3
"""500.com 竞彩足球主玩法抓取：胜平负/让球胜平负"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCRIPT = BASE / 'scrape_500_odds.js'

if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else ''
    cmd = ['node', str(SCRIPT)]
    if date:
        cmd.append(date)
    raise SystemExit(subprocess.call(cmd))
