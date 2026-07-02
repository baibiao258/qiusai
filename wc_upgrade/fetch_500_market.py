#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCRIPT = BASE / 'scrape_500_market.js'

if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else ''
    playid = sys.argv[2] if len(sys.argv) > 2 else '269'
    g = sys.argv[3] if len(sys.argv) > 3 else '2'
    cmd = ['node', str(SCRIPT)]
    if date:
        cmd.append(date)
    else:
        cmd.append('')
    cmd.append(playid)
    cmd.append(g)
    raise SystemExit(subprocess.call(cmd))
