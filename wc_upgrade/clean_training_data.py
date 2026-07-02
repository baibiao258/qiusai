"""
clean_training_data.py — 从 merged JSON 剔除俱乐部比赛
========================================================
输出: 干净的 training_data_with_odds.json (直接覆盖)
"""

import json, os, shutil
from collections import Counter

SRC = "/root/data/training_data_with_odds.json"
BACKUP = "/root/data/training_data_with_odds.json.pre_clean"


def is_national_team(tournament_name):
    """判断是否为国家队赛事 — 子串匹配, 先block后allow"""
    if not tournament_name:
        return False

    name_lower = tournament_name.lower().strip()

    # ── Blocklist: 已知俱乐部联赛 ──
    club_keywords = [
        '意甲', '英超', '西甲', '德甲', '法甲', '荷甲', '葡超',
        '英冠', '英甲', '荷乙', '德乙', '瑞超', '挪超',
        '日职', '韩职', '美职足', '澳超', '中超', '巴甲', '阿甲',
        '芬兰超级联赛', '沙特职业联赛',
        '法国杯', '解放者杯', '南美解放者杯', '亚洲冠军乙级联赛', '欧协联',
        '欧冠', '欧联', '欧会杯', '欧罗巴',
        'serie a', 'premier league', 'la liga', 'bundesliga',
        'ligue 1', 'eredivisie', 'primeira liga',
        'championship', 'league one', 'league two',
        'j.league', 'k league', 'a-league',
        'mls ', 'copa libertadores', 'copa sudamericana',
        'caf champions league',  # 非洲冠军联赛 (俱乐部, 区别于非洲国家杯)
    ]
    for kw in club_keywords:
        if kw.lower() in name_lower:
            return False

    # ── Allowlist: 国家队赛事关键词 ──
    nt_keywords = [
        'world cup', '世界杯',
        'friendly', '友谊赛', '国际赛',
        'uefa euro', 'euro qual', 'euro 202',
        'uefa nations league',
        'copa américa', 'copa america',
        'afc asian cup',
        'africa cup of nations', 'african cup of nations',
        'concacaf gold cup', 'concacaf nations league',
        'wcq', 'world cup qualif', '世界杯预选',
        'confederations cup', 'finalissima', 'fifa series',
        'nations league',
        'olympic', '奥运会', '奥运',
        'african nations championship',
    ]
    for kw in nt_keywords:
        if kw.lower() in name_lower:
            return True

    return False


def main():
    if not os.path.exists(SRC):
        print(f"❌ {SRC} not found")
        return

    # Backup
    if not os.path.exists(BACKUP):
        shutil.copy2(SRC, BACKUP)
        print(f"📦 Backed up → {BACKUP}")

    with open(SRC) as f:
        data = json.load(f)

    total_before = len(data)
    print(f"Before: {total_before} matches")

    # 按 tournament 分组
    tourn_count = Counter(m.get('tournament', '?') for m in data)

    kept = []
    removed = []
    club_stats = Counter()

    for m in data:
        t = m.get('tournament', '?')
        if is_national_team(t):
            kept.append(m)
        else:
            removed.append(m)
            club_stats[t] += 1

    total_after = len(kept)
    removed_count = len(removed)

    print(f"After:  {total_after} matches")
    print(f"Removed: {removed_count} matches ({removed_count/total_before*100:.1f}%)")
    print()

    if removed_count > 0:
        print("=== 已剔除的比赛 ===")
        for t, n in club_stats.most_common():
            print(f"  {t:40s}: {n}")

    # 写回
    with open(SRC, 'w') as f:
        json.dump(kept, f, indent=2)
    print(f"\n✅ Clean data saved to {SRC}")

    # 按来源统计
    source_after = Counter(m.get('source','?') for m in kept)
    print("\n=== After (by source) ===")
    for s, n in source_after.most_common():
        print(f"  {s:20s}: {n}")

    dates = sorted(set(m.get('date', '')[:10] for m in kept if m.get('date')))
    if dates:
        print(f"\nDate range: {dates[0]} ~ {dates[-1]}")


if __name__ == '__main__':
    main()
