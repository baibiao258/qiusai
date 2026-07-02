"""
standardize_dates.py — P2-⑦ 时间字段标准化
=========================================
1. training_data_thestats.csv: utc_date → 补充 date 列 (YYYY-MM-DD)
2. 旧爬虫脚本: match_date → date (4个文件)
3. 验证核心管线已全部使用 date
"""

import os, re, csv, shutil

DATA_DIR = "/root/data"
WC_DIR = "/root/wc_2026_upgrade"

changes = []

# ════════════════════════════════════════
# 1. training_data_thestats.csv
# ════════════════════════════════════════
csv_path = f"{WC_DIR}/training_data_thestats.csv"
if os.path.exists(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    
    if "utc_date" in cols and "date" not in cols:
        # Add date column (first 10 chars of utc_date)
        new_cols = list(cols)
        # Insert 'date' right after 'utc_date'
        utc_idx = new_cols.index("utc_date")
        new_cols.insert(utc_idx + 1, "date")
        
        backup = csv_path + ".bak"
        shutil.copy2(csv_path, backup)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_cols)
            writer.writeheader()
            for r in rows:
                r['date'] = r.get('utc_date', '')[:10]
                writer.writerow(r)
        
        changes.append(f"📄 {csv_path}: added 'date' column from utc_date (→ {backup})")
    elif "date" in cols:
        changes.append(f"📄 {csv_path}: already has 'date' column ✅")
    else:
        changes.append(f"📄 {csv_path}: no date/utc_date column found ?")

# ════════════════════════════════════════
# 2. Old scraper scripts: match_date → date
# ════════════════════════════════════════
files_to_fix = [
    f"{WC_DIR}/async_500_scraper.py",
    f"{WC_DIR}/fetch_500_complete.py",
    f"{WC_DIR}/fetch_500_odds.py",
    f"{WC_DIR}/integrate_500_odds.py",
]

for fp in files_to_fix:
    if not os.path.exists(fp):
        changes.append(f"📄 {os.path.basename(fp)}: not found (already removed?)")
        continue
    
    with open(fp) as f:
        content = f.read()
    
    # Count occurrences
    total_before = content.count("match_date")
    adjusted_before = content.count("'match_date'")
    adjusted_after = content.count("'date'")
    
    if total_before == 0:
        changes.append(f"📄 {os.path.basename(fp)}: no match_date references ✅")
        continue
    
    # Replace dict key 'match_date' → 'date'
    new_content = content.replace("'match_date'", "'date'")
    new_content = new_content.replace('"match_date"', '"date"')
    
    # Also replace variable name match_date → match_dt to avoid confusion but keep readability
    # Only do this for the variable declaration pattern
    # Actually, for minimal change, just fix the dict key. Variable names can stay.
    
    count = new_content.count("'date'") - adjusted_after
    backup = fp + ".bak"
    with open(fp) as f_orig:
        orig = f_orig.read()
    if orig != new_content:
        shutil.copy2(fp, backup)
        with open(fp, 'w') as f_out:
            f_out.write(new_content)
        changes.append(f"📄 {os.path.basename(fp)}: 'match_date'→'date' ({count} changes, backup→{backup})")
    else:
        changes.append(f"📄 {os.path.basename(fp)}: no changes needed ✅")

# ════════════════════════════════════════
# 3. Verify core pipeline
# ════════════════════════════════════════
core_scripts = [
    "/root/daily_jczq.py",
    f"{WC_DIR}/retrain_nat.py",
    f"{WC_DIR}/merge_training_data.py",
    f"{WC_DIR}/thestats_features.py",
    f"{WC_DIR}/fetch_team_squads.py",
    f"{WC_DIR}/fetch_thestats_features.py",
]

core_issues = []
for fp in core_scripts:
    if not os.path.exists(fp):
        continue
    with open(fp) as f:
        content = f.read()
    refs = content.count("match_date")
    if refs > 0:
        core_issues.append(f"  ⚠️ {os.path.basename(fp)}: {refs} remaining match_date refs")

# ════════════════════════════════════════
# Report
# ════════════════════════════════════════
print("=" * 60)
print("  P2-⑦ 时间字段标准化报告")
print("=" * 60)
print()
for c in changes:
    print(f"  {c}")
print()

if core_issues:
    print("⚠️  核心管线仍有 match_date 引用:")
    for ci in core_issues:
        print(f"  {ci}")
else:
    print("✅ 核心管线 (daily_jczq.py, retrain_nat.py 等) 无 match_date 引用")
print()

# Verify all match_date in wc_2026_upgrade/
remaining = 0
for root, dirs, files in os.walk(WC_DIR):
    for fn in files:
        if fn.endswith('.py'):
            fp = os.path.join(root, fn)
            with open(fp) as f:
                c = f.read()
            cnt = c.count("match_date")
            if cnt > 0:
                remaining += cnt
                print(f"  剩余: {fn}: {cnt} refs")

if remaining == 0:
    print("✅ wc_2026_upgrade/ 目录无 match_date 引用残留")
else:
    print(f"\n⚠️  {remaining} 残留引用 (可能在 __pycache__ 或无关文件中)")
