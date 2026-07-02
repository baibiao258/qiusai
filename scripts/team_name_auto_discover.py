#!/usr/bin/env python3
"""
team_name_auto_discover.py — 自动发现缺失的队名映射
====================================================
从 historical_kaijiang.csv (中文名) 和 form_state.json (英文名) 交叉关联，
自动补全 team_name_mapping.json 的缺失映射。

策略:
  1. 直接映射: form_state 的 key 是英文, kaiijang 只有中文 → 按比赛日期+比分对齐
  2. 模糊匹配: 中文名在已有 mapping 中找不到时, 尝试按联赛+对手+日期对齐
  3. 输出已确认但缺失的映射, 追加到 mapping

用法:
  python3 scripts/team_name_auto_discover.py [--apply] [--dry-run]
  
  --dry-run: 只打印不保存 (默认)
  --apply:   自动补充到 team_name_mapping.json
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

DATA_DIR = '/root/data'
KAJ_PATH = os.path.join(DATA_DIR, 'historical_kaijiang.csv')
FORM_PATH = os.path.join(DATA_DIR, 'form_state.json')
MAPPING_PATH = os.path.join(DATA_DIR, 'team_name_mapping.json')
INTERNATIONAL_RESULTS = os.path.join(DATA_DIR, 'international_results.json')


def load_current_mapping():
    try:
        with open(MAPPING_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def load_form_state():
    try:
        with open(FORM_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def load_kaijiang():
    matches = []
    with open(KAJ_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ft_h = int(row.get('ft_h', -1))
                ft_a = int(row.get('ft_a', -1))
                if ft_h < 0 or ft_a < 0:
                    continue
                matches.append({
                    'date': row.get('date', ''),
                    'code': row.get('code', ''),
                    'league': row.get('league', ''),
                    'home_cn': row.get('home', '').strip(),
                    'away_cn': row.get('away', '').strip(),
                    'ft_h': ft_h,
                    'ft_a': ft_a,
                })
            except (ValueError, TypeError):
                continue
    return matches


def load_international_results():
    """加载 international_results.json (英文名+比分)."""
    try:
        with open(INTERNATIONAL_RESULTS, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def discover_mappings():
    """自动发现新的映射."""
    mapping = load_current_mapping()
    form_state = load_form_state()
    kaijiang = load_kaijiang()
    intl_results = load_international_results()

    # 反向映射: en → cn (从已有 mapping 推断)
    en_to_cn = {v: k for k, v in mapping.items()}

    # ── 策略1: international_results.json → kaijiang 对齐 ──
    # 按 日期+比分+联赛 匹配
    print("📡 策略1: international_results.json ↔ historical_kaijiang 对齐...")
    new_mappings_1 = {}

    # 按日期+联赛建立 kaijiang 索引
    kaijiang_by_date_leauge = defaultdict(list)
    for m in kaijiang:
        key = (m['date'], m['league'])
        kaijiang_by_date_leauge[key].append(m)

    for m in intl_results:
        en_home = m.get('home', m.get('home_team', ''))
        en_away = m.get('away', m.get('away_team', ''))
        if not en_home or not en_away:
            continue
        try:
            hg = int(m.get('h_score', m.get('home_score', -1)))
            ag = int(m.get('a_score', m.get('away_score', -1)))
        except (ValueError, TypeError):
            continue
        if hg < 0 or ag < 0:
            continue
        date_key = m.get('date', '')[:10]
        # 尝试按 league 匹配
        tournament = m.get('tournament', '')
        if 'Friendlies' in tournament or 'Friendly' in tournament:
            league_key = '友谊赛'
        else:
            league_key = tournament

        candidates = kaijiang_by_date_leauge.get((date_key, '友谊赛'), [])
        candidates += kaijiang_by_date_leauge.get((date_key, '国际赛'), [])
        candidates += kaijiang_by_date_leauge.get((date_key, ''), [])

        for cm in candidates:
            if cm['ft_h'] == hg and cm['ft_a'] == ag:
                # 比分匹配! 检查队名
                for cn_name, en_name in [
                    (cm['home_cn'], en_home),
                    (cm['away_cn'], en_away),
                ]:
                    # cn_name 是中文, en_name 是英文
                    # 检查是否已映射
                    if cn_name not in mapping and en_name not in en_to_cn:
                        new_mappings_1[cn_name] = en_name

    print(f"   发现 {len(new_mappings_1)} 个新映射")

    # ── 策略2: form_state keys (英文) × kaijiang (中文) 对齐 ──
    print("📡 策略2: form_state.json ↔ historical_kaijiang 对齐...")
    new_mappings_2 = {}

    # form_state keys 全是英文 (或已知别名)
    en_teams = set(form_state.keys())

    # 对 kaijiang 中的每场比赛, 尝试找出未映射的中文名
    for m in kaijiang:
        for cn_name in [m['home_cn'], m['away_cn']]:
            if cn_name in mapping:
                continue
            # 检查是否已发现
            if cn_name in new_mappings_2:
                continue
            # 通过比分交叉验证找可能的英文名
            # 找同一日期+联赛下, 比分匹配的 form_state 英文队名
            matches_same_date_league = []
            for key, ml in kaijiang_by_date_leauge.items():
                if key[0] == m['date'] and key[1] == m['league']:
                    matches_same_date_league = ml
                    break
            # 找对手队的中文名 → 英文名
            opponent_cn = m['away_cn'] if cn_name == m['home_cn'] else m['home_cn']
            opponent_en = mapping.get(opponent_cn)
            if opponent_en:
                # 通过对手的英文名 + 比分 反过来找当前队的英文名
                # 在 intl_results 中找同时满足对手名 + 比分 + 日期的比赛
                for mi in intl_results:
                    try:
                        hi = int(mi.get('h_score', mi.get('home_score', -1)))
                        ai = int(mi.get('a_score', mi.get('away_score', -1)))
                    except (ValueError, TypeError):
                        continue
                    if hi < 0 or ai < 0:
                        continue
                    date_i = mi.get('date', '')[:10]
                    if date_i != m['date']:
                        continue
                    h_name = mi.get('home', mi.get('home_team', ''))
                    a_name = mi.get('away', mi.get('away_team', ''))
                    if not (hi == m['ft_h'] and ai == m['ft_a']):
                        continue
                    # 比分匹配! 判断哪个是当前队
                    if h_name == opponent_en:
                        # 当前队是客队
                        candidate_en = a_name
                    elif a_name == opponent_en:
                        # 当前队是主队
                        candidate_en = h_name
                    else:
                        continue
                    if candidate_en not in en_to_cn.values():
                        new_mappings_2[cn_name] = candidate_en

    print(f"   发现 {len(new_mappings_2)} 个新映射")

    # ── 合并去重 ──
    all_new = {}
    for d in [new_mappings_1, new_mappings_2]:
        for k, v in d.items():
            if k not in mapping:
                all_new[k] = v

    # ── 冲突检查 ──
    conflicts = []
    for cn_name, en_name in all_new.items():
        existing_en = mapping.get(cn_name)
        if existing_en and existing_en != en_name:
            conflicts.append(f"{cn_name}: [{existing_en}] vs [{en_name}]")

    print(f"\n📊 汇总:")
    print(f"   当前 mapping: {len(mapping)} 条")
    print(f"   新增可能: {len(all_new)} 条")
    if conflicts:
        print(f"   冲突: {len(conflicts)} 条 (跳过)")
        for c in conflicts[:10]:
            print(f"     ⚠️ {c}")

    return all_new


def main():
    apply = '--apply' in sys.argv
    dry_run = not apply

    print("=" * 60)
    print("  🏷️ 队名映射自动发现管线")
    print("=" * 60)

    new_map = discover_mappings()

    if not new_map:
        print("\n✅ 无新增映射")
        return

    if dry_run:
        print(f"\n📋 Dry-run: 以下 {len(new_map)} 条将被添加:")
        for cn, en in sorted(new_map.items()):
            print(f"  + '{cn}' → '{en}'")
        print(f"\n运行 --apply 以保存")
        return

    # ── 保存 ──
    mapping = load_current_mapping()
    mapping.update(new_map)
    with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存 {len(new_map)} 条新映射到 {MAPPING_PATH}")
    print(f"   新总量: {len(mapping)} 条")


if __name__ == '__main__':
    main()
