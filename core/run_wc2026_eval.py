#!/usr/bin/env python3
import sys, os, json
from datetime import datetime

# Import the final wc_2026_final.py so we can execute it without cron wait
# The file seems to be inside /usr/local/lib/hermes-agent/models/ or /root/
sys.path.extend(['/usr/local/lib/hermes-agent/strategy'])
try:
    from wc_2026_final import run_pipeline
except ImportError:
    print("Cannot import wc_2026_final. Searching...")

