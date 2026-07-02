"""
clean_league_names.py — P2-⑧ 清洗 predictions_log.csv 历史赛事名
==============================================================
读取 daily_jczq.py 的 normalize_league_name 函数,
清洗 predictions_log.csv 的 league 列 + training_data_with_odds.json 的 tournament 列
"""

import csv, json, os, sys, shutil

# 从 daily_jczq.py 导入标准化函数
sys.path.insert(0, '/root')
from daily_jczq import normalize_league_name, LEAGUE_NORMALIZE_MAP

CSV_PATH = "/root/data/predictions_log.csv"
JSON_PATH = "/root/data/training_data_with_odds.json"

results = {}

# ════════════════════════════════════════
# 1. predictions_log.csv
# ════════════════════════════════════════
if os.path.exists(CSV_PATH):
    backup = CSV_PATH + ".bak"
    shutil.copy2(CSV_PATH, backup)
    
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    
    if 'league' in cols:
        before = set()
        changes = 0
        for r in rows:
            raw = r.get('league', '')
            if raw:
                before.add(raw)
                normalized = normalize_league_name(raw)
                if normalized != raw:
                    r['league'] = normalized
                    changes += 1
        
        after = set(r['league'] for r in rows if r['league'])
        
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        
        results['predictions_log.csv'] = {
            'total_rows': len(rows),
            'changes': changes,
            'unique_before': len(before),
            'unique_after': len(after),
            'before_values': sorted(before),
            'after_values': sorted(after),
            'backup': backup,
        }

# ════════════════════════════════════════
# 2. training_data_with_odds.json
# ════════════════════════════════════════
if os.path.exists(JSON_PATH):
    backup_j = JSON_PATH + ".bak.league"
    shutil.copy2(JSON_PATH, backup_j)
    
    with open(JSON_PATH) as f:
        data = json.load(f)
    
    before_t = set()
    changes_t = 0
    for m in data:
        raw = m.get('tournament', '')
        if raw:
            before_t.add(raw)
            normalized = normalize_league_name(raw)
            if normalized != raw:
                m['tournament'] = normalized
                changes_t += 1
    
    after_t = set(m['tournament'] for m in data if m['tournament'])
    
    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    results['training_data_with_odds.json'] = {
        'total_rows': len(data),
        'changes': changes_t,
        'unique_before': len(before_t),
        'unique_after': len(after_t),
        'before_values': sorted(before_t),
        'after_values': sorted(after_t),
        'backup': backup_j,
    }

# ════════════════════════════════════════
# Report
# ════════════════════════════════════════
print("=" * 60)
print("  P2-⑧ 赛事名称清洗报告")
print("=" * 60)
print()

for name, info in results.items():
    print(f"📄 {name}")
    print(f"  总行数: {info['total_rows']}")
    print(f"  改动量: {info['changes']} 行")
    print(f"  唯一值: {info['unique_before']} → {info['unique_after']}")
    print(f"  备份: {info['backup']}")
    print()
    if info['before_values'] != info['after_values']:
        print(f"  清洗前:")
        for v in info['before_values']:
            if v not in set(info['after_values']):
                print(f"    ❌ {v}")
        print(f"  清洗后:")
        for v in info['after_values']:
            print(f"    ✅ {v}")
    else:
        print(f"  无变化, 唯一值:")
        for v in info['after_values']:
            print(f"    {v}")
    print()

# 汇总
all_after = set()
for info in results.values():
    all_after.update(info.get('after_values', []))
print(f"📊 标准化后全系统赛事名称共 {len(all_after)} 种:")
for v in sorted(all_after):
    print(f"  • {v}")
