#!/usr/bin/env python3
"""Merge backup (69 fields) into current (76 fields) to recover lost rows."""
import csv
import io
import os

BACKUP = '/root/data/predictions_log.csv.bak'
CURRENT = '/root/data/predictions_log.csv'

# Read backup
with open(BACKUP) as f:
    backup_reader = csv.DictReader(f)
    backup_rows = list(backup_reader)
    backup_fields = list(backup_reader.fieldnames) if backup_reader.fieldnames else []

print(f"Backup: {len(backup_rows)} rows, {len(backup_fields)} fields")

# Read current
with open(CURRENT) as f:
    current_reader = csv.DictReader(f)
    current_rows = list(current_reader)
    current_fields = list(current_reader.fieldnames) if current_reader.fieldnames else []

print(f"Current: {len(current_rows)} rows, {len(current_fields)} fields")

# New fields in current vs backup
new_fields = set(current_fields) - set(backup_fields)
missing_fields = set(backup_fields) - set(current_fields)
print(f"New fields in current: {new_fields}")
print(f"Missing from current: {missing_fields}")

# Get unique keys (code + date) in current
current_keys = set()
for r in current_rows:
    current_keys.add((r.get('code',''), r.get('date','')))
print(f"Current unique (code+date) combos: {len(current_keys)}")

# Find backup rows that are NOT in current
new_rows = []
for br in backup_rows:
    key = (br.get('code',''), br.get('date',''))
    if key not in current_keys:
        # Add new fields with empty values
        for f in current_fields:
            if f not in br:
                br[f] = ''
        new_rows.append(br)

print(f"Rows in backup but not current: {len(new_rows)}")
for r in new_rows[:5]:
    print(f"  {r.get('date','')} {r.get('code',''):10s} {r.get('home_cn',''):20s} vs {r.get('away_cn',''):20s}")
