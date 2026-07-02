#!/usr/bin/env python3
"""Extract official 2026 World Cup groups from prisma seed - use groupName field."""
import re, json

text = open('/root/repo_wcbetting/prisma/seed.ts').read()

# Find all teams in the seed data: lines with name and groupName
# Pattern: { name: "X", ..., groupName: "Y", ... }
teams_raw = re.findall(r'name:\s*"([^"]+)".*?groupName:\s*"([A-Z])"', text)

name_map = {
    'Korea Republic': 'South Korea',
    'Czechia': 'Czech Republic',
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
    "Côte d'Ivoire": 'Ivory Coast',
    'Cabo Verde': 'Cape Verde',
    'USA': 'United States',
    'Türkiye': 'Turkey',
    'IR Iran': 'Iran',
}

groups = {chr(65+i): [] for i in range(12)}  # A-L
for name, grp in teams_raw:
    if grp in groups and name != 'TBD':
        groups[grp].append(name_map.get(name, name))

print(json.dumps(groups, indent=2, ensure_ascii=False))

with open('/root/data/2026_groups_official.json', 'w') as f:
    json.dump(groups, f, indent=2, ensure_ascii=False)
print(f'\n✅ Saved')
