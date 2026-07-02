#!/usr/bin/env python3
"""Fix the predictions_log.csv file by restoring from backup and preserving recent data."""
import csv
import io
import os

BACKUP = '/root/data/predictions_log.csv.bak'
CURRENT = '/root/data/predictions_log.csv'

# Read backup
with open(BACKUP) as f:
    backup_content = f.read()

backup_reader = csv.DictReader(io.StringIO(backup_content))
backup_rows = list(backup_reader)
backup_fields = backup_reader.fieldnames

print(f"Backup: {len(backup_rows)} rows, {len(backup_fields)} fields")

# Read current file (try to extract what we can)
with open(CURRENT) as f:
    current_content = f.read()

# Count raw lines
raw_lines = current_content.split('\n')
print(f"Current: {len(raw_lines)} raw lines")

# Try to parse current
current_reader = csv.DictReader(io.StringIO(current_content))
current_rows = list(current_reader)
print(f"Current parseable: {len(current_rows)} rows")

# Find the latest date in backup
backup_dates = set()
for r in backup_rows:
    d = r.get('date', '')
    if d:
        backup_dates.add(d)
print(f"Backup dates: {sorted(backup_dates)}")

# Find dates in current (excluding corrupted ones)
current_dates = set()
for r in current_rows:
    d = r.get('date', '')
    if d and len(d) == 10 and d[4] == '-' and d[7] == '-':
        current_dates.add(d)
print(f"Current good dates: {sorted(current_dates)}")

# We want to keep backup data for up to June 15, and current data for June 16+
# But the current file might have some issues
# Let's just try to merge
new_dates = current_dates - backup_dates
print(f"New dates (in current but not backup): {sorted(new_dates)}")

# For simplicity: write a clean version by reading current with proper error handling
# and fixing any parsing issues
print("\n--- Attempting clean parse ---")
with open(CURRENT) as f:
    raw = f.read()

# The issue is likely that Python's csv.DictReader doesn't handle the quoting properly
# Let's use csv.reader to see each row as raw fields
reader = csv.reader(io.StringIO(raw))
all_rows_raw = list(reader)
print(f"Raw CSV rows: {len(all_rows_raw)}")
for i, row in enumerate(all_rows_raw):
    if len(row) != 76:
        print(f"Row {i}: {len(row)} fields (expected 76) — code={row[0] if row else 'EMPTY'}")
