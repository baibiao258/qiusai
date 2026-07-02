#!/usr/bin/env python3
"""Compare groups and copy official over old."""
import json

old = json.load(open('/root/data/2026_groups.json'))
new = json.load(open('/root/data/2026_groups_official.json'))

print("Group differences (旧→新):")
changes = 0
for g in sorted(set(list(old.keys()) + list(new.keys()))):
    o = old.get(g, [])
    n = new.get(g, [])
    if o != n:
        print(f"  {g}: {o} → {n}")
        changes += 1
print(f"Total group changes: {changes}")

# Copy official over
import shutil
shutil.copy('/root/data/2026_groups_official.json', '/root/data/2026_groups.json')
print("\n✅ Replaced 2026_groups.json with official groups")

# Also print team rankings from seed
import re
text = open('/root/repo_wcbetting/prisma/seed.ts').read()
print("\n--- World Rankings from seed ---")
for m in re.finditer(r'name:\s*"([^"]+)".*?worldRanking:\s*(\d+)', text):
    name = m.group(1)
    if name != 'TBD':
        rank = m.group(2)
        print(f"  {name:>25s}  #{rank}")
