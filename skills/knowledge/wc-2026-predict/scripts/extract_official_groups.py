#!/usr/bin/env python3
"""
Extract official 2026 World Cup groups from WRooney108/World-Cup-Betting prisma seed.

Usage:
  python3 /root/.hermes/skills/knowledge/wc-2026-predict/scripts/extract_official_groups.py

Updates /root/data/2026_groups.json with the latest group assignments from the seed.

The seed file is at /root/repo_wcbetting/prisma/seed.ts (cloned from:
https://github.com/WRooney108/World-Cup-Betting/blob/main/prisma/seed.ts)

Run this any time the seed.ts is updated to sync the groups file.
"""
import re, json, shutil, sys

SEED_PATH = '/root/repo_wcbetting/prisma/seed.ts'
OUTPUT_PATH = '/root/data/2026_groups.json'
BACKUP_PATH = '/root/data/2026_groups_official.json'

NAME_MAP = {
    'Korea Republic': 'South Korea',
    'Czechia': 'Czech Republic',
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
    "Côte d'Ivoire": 'Ivory Coast',
    'Cabo Verde': 'Cape Verde',
    'USA': 'United States',
    'Türkiye': 'Turkey',
    'IR Iran': 'Iran',
}

def main():
    with open(SEED_PATH) as f:
        text = f.read()

    teams_raw = re.findall(r'name:\s*"([^"]+)".*?groupName:\s*"([A-Z])"', text)
    if not teams_raw:
        print(f"❌ No group data found in {SEED_PATH}")
        sys.exit(1)

    groups = {chr(65 + i): [] for i in range(12)}
    for name, grp in teams_raw:
        if grp in groups and name != 'TBD':
            groups[grp].append(NAME_MAP.get(name, name))

    # validate
    for g, ts in groups.items():
        if len(ts) != 4:
            print(f"⚠️  Group {g} has {len(ts)} teams (expected 4): {ts}")

    # backup current
    try:
        shutil.copy(OUTPUT_PATH, BACKUP_PATH)
    except FileNotFoundError:
        pass

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)

    print(f"✅ Updated {OUTPUT_PATH} with {len(groups)} groups ({sum(len(v) for v in groups.values())} teams)")
    print(f"   Backup at {BACKUP_PATH}")

    # print diff summary
    old = {}
    try:
        old = json.load(open(BACKUP_PATH))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    changes = 0
    for g in sorted(set(list(old.keys()) + list(groups.keys()))):
        o = old.get(g, [])
        n = groups.get(g, [])
        if o != n:
            print(f"   Δ {g}: {o} → {n}")
            changes += 1
    if changes == 0:
        print("   No group changes detected (up to date)")

if __name__ == '__main__':
    main()
