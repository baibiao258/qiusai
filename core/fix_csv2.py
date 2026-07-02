#!/usr/bin/env python3
"""Attempt to reconstruct a clean predictions_log.csv"""
import csv
import io
import json

CURRENT = '/root/data/predictions_log.csv'

# Read raw lines
with open(CURRENT) as f:
    raw = f.read()

lines = raw.split('\n')
print(f"Total lines: {len(lines)}")

# Use csv.reader to get each row as a list of fields
reader = csv.reader(io.StringIO(raw))
all_rows = list(reader)
print(f"CSV-parsed rows: {len(all_rows)}")

# For each row, check the number of fields
header = all_rows[0]
expected = len(header)
print(f"Expected fields: {expected}")
print(f"Header: {header[:10]}...{header[-5:]}")

problem_rows = []
for i, row in enumerate(all_rows):
    if len(row) != expected:
        problem_rows.append((i, len(row), row[:5]))

print(f"\nProblem rows ({len(problem_rows)}):")
for i, nf, first5 in problem_rows:
    print(f"  Row {i}: {nf} fields (expected {expected})")
    print(f"    First 5: {first5}")
    # Try to see what the raw line looks like
    if i < len(lines):
        print(f"    Raw: {lines[i][:200]}...")
