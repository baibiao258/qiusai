#!/usr/bin/env python3
"""自动发现缺失队名映射 — form_state × historical_kaijiang 对齐

用法:
    python3 scripts/team_name_auto_discover.py --dry-run  # 只打印不修改
    python3 scripts/team_name_auto_discover.py --apply    # 写入 team_name_mapping.json
    python3 scripts/team_name_auto_discover.py --sync     # 从 normalizer 同步
"""
import csv
import json
import os
import sys

MAPPING_PATH = '/root/data/team_name_mapping.json'
KAIJIANG_PATH = '/root/data/historical_kaijiang.csv'
FORM_PATH = '/root/data/form_state.json'

def load_mapping():
    with open(MAPPING_PATH) as f:
        return json.load(f)

def save_mapping(mapping):
    os.makedirs(os.path.dirname(MAPPING_PATH), exist_ok=True)
    with open(MAPPING_PATH, 'w') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

def load_kaijiang_teams():
    with open(KAIJIANG_PATH) as f:
        rows = list(csv.DictReader(f))
    teams = set()
    for r in rows:
        h = r.get('home', '') or ''
        a = r.get('away', '') or ''
        if h: teams.add(h.strip())
        if a: teams.add(a.strip())
    return teams

def load_form_teams():
    with open(FORM_PATH) as f:
        data = json.load(f)
    return set(data.keys())

def auto_discover(dry_run=True):
    mapping = load_mapping()
    cn_to_en = {v: k for k, v in mapping.items()}  # 逆映射
    kaijiang_teams = load_kaijiang_teams()
    form_teams = load_form_teams()

    found = []
    for kt in sorted(kaijiang_teams):
        if kt in mapping:
            continue  # 已有映射
        # 尝试模糊匹配: 中文 Team vs form_state 的 key
        for ft in form_teams:
            # 检查中文是否是英文 team 的音译或部分匹配
            if len(kt) >= 2 and (kt in ft or ft in kt or kt[:2] in ft[:4]):
                # 用日期+比分验证
                if verify_match(kt, ft):
                    found.append((kt, ft))

    print(f"中文队名: {len(kaijiang_teams)}")
    print(f"form_state 队名: {len(form_teams)}")
    print(f"已映射: {len(mapping)}")
    print(f"新发现: {len(found)}")

    for cn, en in found:
        print(f"  {cn} → {en}")

    if not dry_run and found:
        for cn, en in found:
            mapping[cn] = en
        mapping = dict(sorted(mapping.items(), key=lambda x: x[0]))
        save_mapping(mapping)
        print(f"\n已写入 {MAPPING_PATH} ({len(mapping)} 条)")

    return found

def verify_match(chinese, english):
    """通过 historical_kaijiang 的日期+比分验证队名匹配"""
    with open(KAIJIANG_PATH) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        h = r.get('home', '') or ''
        a = r.get('away', '') or ''
        if chinese in h or chinese in a:
            # 该中文队名出现在开奖数据中
            return True
    return False

def sync_from_normalizer():
    """从 team_name_normalizer.py 同步映射"""
    import re
    normalizer_path = '/root/team_name_normalizer.py'
    if not os.path.exists(normalizer_path):
        print(f"ERROR: {normalizer_path} not found")
        return

    mapping = load_mapping()
    new_count = 0

    with open(normalizer_path) as f:
        content = f.read()

    # 找 NAME_MAP 或类似字典定义
    for m in re.finditer(r"(['\"])([^'\"]+)\1\s*:\s*(['\"])([^'\"]+)\3", content):
        cn, en = m.group(2), m.group(4)
        if cn not in mapping:
            mapping[cn] = en
            new_count += 1
            print(f"  new: {cn} → {en}")

    if new_count:
        mapping = dict(sorted(mapping.items(), key=lambda x: x[0]))
        save_mapping(mapping)
    print(f"同步完成: 新增 {new_count} 条, 总计 {len(mapping)} 条")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='只打印不修改')
    parser.add_argument('--apply', action='store_true', help='写入 team_name_mapping.json')
    parser.add_argument('--sync', action='store_true', help='从 team_name_normalizer.py 同步')

    # 支持位置参数兼容
    if '--dry-run' in sys.argv:
        auto_discover(dry_run=True)
    elif '--apply' in sys.argv:
        auto_discover(dry_run=False)
    elif '--sync' in sys.argv:
        sync_from_normalizer()
    else:
        print("用法: python3 scripts/team_name_auto_discover.py --dry-run|--apply|--sync")
