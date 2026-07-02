#!/usr/bin/env python3
"""500.com 总进球抓取：原生赔率直出"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCRIPT = BASE / 'scrape_500_totalgoals.js'

if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else ''
    cmd = ['node', str(SCRIPT)]
    if date:
        cmd.append(date)
    raise SystemExit(subprocess.call(cmd))
