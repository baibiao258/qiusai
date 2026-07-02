#!/usr/bin/env python3
"""全面更新淘汰赛阶段的状态"""
import json
from collections import Counter, defaultdict

ts = json.load(open('/root/data/tournament_state.json'))
results = json.load(open('/root/data/wc_completed_results.json'))

# 按日期排序
results_sorted = sorted(results, key=lambda r: r.get('commence_time', r.get('date', '')))

# 队名英中映射
EN_TO_CN = {
    'Mexico': '墨西哥', 'South Korea': '韩国', 'Czech Republic': '捷克',
    'South Africa': '南非', 'Canada': '加拿大', 'Switzerland': '瑞士',
    'Bosnia & Herzegovina': '波黑', 'Qatar': '卡塔尔', 'Brazil': '巴西',
    'Morocco': '摩洛哥', 'Paraguay': '巴拉圭', 'Scotland': '苏格兰',
    'Curaçao': '库拉索', 'Germany': '德国', 'Japan': '日本',
    'Netherlands': '荷兰', 'Ivory Coast': '科特迪瓦',
    'Tunisia': '突尼斯', 'Egypt': '埃及', 'Cape Verde': '佛得角',
    'Iraq': '伊拉克', 'Algeria': '阿尔及利亚', 'Uzbekistan': '乌兹别克',
    'Croatia': '克罗地亚', 'Spain': '西班牙', 'Saudi Arabia': '沙特阿拉伯',
    'Ecuador': '厄瓜多尔', 'Iran': '伊朗', 'New Zealand': '新西兰',
    'France': '法国', 'Senegal': '塞内加尔', 'Norway': '挪威',
    'Austria': '奥地利', 'Jordan': '约旦', 'Argentina': '阿根廷',
    'Portugal': '葡萄牙', 'England': '英格兰', 'Ghana': '加纳',
    'Panama': '巴拿马', 'Belgium': '比利时', 'Sweden': '瑞典',
    'Turkey': '土耳其', 'Australia': '澳大利亚', 'Colombia': '哥伦比亚',
    'USA': '美国', 'Uruguay': '乌拉圭', 'Haiti': '海地',
    'DR Congo': '刚果(金)', 'Costa Rica': '哥斯达黎加',
}
CN_TO_EN = {v: k for k, v in EN_TO_CN.items()}

# 统计每队完赛场次 & 最后一场比赛
team_matches = Counter()
team_last_result = {}

for r in results_sorted:
    h, a = r['home'], r['away']
    team_matches[h] += 1
    team_matches[a] += 1
    team_last_result[h] = r
    team_last_result[a] = r

total_teams = len(team_matches)
print(f"总完赛场次: {len(results_sorted)}, 涉及球队: {total_teams}")

# 找出哪些队打了小组赛后的比赛
# 2026 WC: 12组×4队, 每组打6场 = 72场小组赛
# 先确定分组（从tournament_state的分组信息反推）
groups = defaultdict(list)
for cn_name, info in ts.items():
    r = info.get('round_num', 1)
    if r <= 3:
        # 还在小组赛阶段的信息，无法确定分组
        pass

# 更好的方法: 统计每个队的出场次数
# 小组赛: 每队打3场
# 淘汰赛: 赢的队继续打

# 已淘汰的队:
# 1. 打了≥3场且确定没进前2（小组排名3或4且所有场次已打完）
# 2. 打了淘汰赛并输了

# 找出所有淘汰赛比赛：总场次 - 72小组赛 = 淘汰赛场次
group_match_count = 72  # 12组 × 6场
ko_matches = results_sorted[group_match_count:]
ko_match_count = len(ko_matches)

print(f"\n小组赛: 72场")
print(f"淘汰赛: {ko_match_count}场")
print(f"\n淘汰赛对阵:")
eliminated_teams_en = set()

for r in ko_matches:
    h, a = r['home'], r['away']
    hs = int(r.get('home_score', 0) or 0)
    as_ = int(r.get('away_score', 0) or 0)
    
    if hs > as_:
        winner, loser = h, a
    elif as_ > hs:
        winner, loser = a, h
    else:
        # 平局=点球, 结果未知, 丢给≥4场逻辑处理
        print(f"  ⚠ {h} {hs}-{as_} {a} (平局, 点球结果未知)")
        continue
    
    eliminated_teams_en.add(loser)
    print(f"  {winner} {hs}-{as_} {loser} → {loser}淘汰")

# 平局淘汰赛: 不知道点球谁赢, 只能通过"≥4场且最后一场是平局=淘汰"来判断
# 但保守起见, 仅当该队最后一场打平且在淘汰赛列表中才标记
for r in ko_matches:
    h, a = r['home'], r['away']
    hs = int(r.get('home_score', 0) or 0)
    as_ = int(r.get('away_score', 0) or 0)
    if hs == as_:  # knockout draw
        # 如果该队在点球输了, 他们的最后一场比赛就是这个
        # 但点球赢家会继续打下一轮
        # 保守: 只标记打了4场且没出现在后续淘汰赛中的队
        pass  # 交给下面的≥4场逻辑处理

# 对于小组赛未出线的队: 如果该队打了3场且没进前2, 淘汰
en_group_rank = {}  # English name -> rank in group
for cn_name, info in ts.items():
    en_name = CN_TO_EN.get(cn_name, cn_name)
    rank = info.get('home_group_rank', info.get('away_group_rank', 99))
    round_num = info.get('round_num', 1)
    en_group_rank[en_name] = {'rank': rank, 'round': round_num, 'eliminated': info.get('eliminated', False)}

# 打了≥3场的队: 如果排名≥3且在round≥3(小组打完), 淘汰
for en_name, matches in team_matches.items():
    if matches >= 3:
        grank = en_group_rank.get(en_name, {}).get('rank', 99)
        ground = en_group_rank.get(en_name, {}).get('round', 1)
        if grank >= 3 and ground >= 3:
            eliminated_teams_en.add(en_name)

# 打了4场且最后一场输了 → 淘汰赛输了
# (不打平局, 因为点球我们不知道谁赢)
# 补充: 打了4场且最后一场是淘汰赛平局 → 说明点球淘汰了(无后续比赛)
for en_name, matches in team_matches.items():
    if matches >= 4 and en_name not in eliminated_teams_en:
        last = team_last_result.get(en_name)
        if last:
            h, a = last['home'], last['away']
            hs = int(last.get('home_score', 0) or 0)
            as_ = int(last.get('away_score', 0) or 0)
            # 输球 → 淘汰
            if en_name == h and hs < as_:
                eliminated_teams_en.add(en_name)
            elif en_name == a and as_ < hs:
                eliminated_teams_en.add(en_name)
            # 平局淘汰赛 → 点球输了
            # 判断: 最后一场是淘汰赛(在72场之后)且是平局
            elif hs == as_:
                idx = results_sorted.index(last)
                if idx >= 72:  # 淘汰赛比赛
                    eliminated_teams_en.add(en_name)

# 转中文名
eliminated_cn = set()
for t in eliminated_teams_en:
    cn = EN_TO_CN.get(t, t)
    eliminated_cn.add(cn)

# 存活球队
alive_cn = set()
for cn_name in ts:
    if cn_name not in eliminated_cn:
        alive_cn.add(cn_name)

print(f"\n=== 淘汰球队 ({len(eliminated_cn)}) ===")
for t in sorted(eliminated_cn):
    en = CN_TO_EN.get(t, t)
    print(f"  ❌ {t} ({team_matches.get(en, 0)}场)")

print(f"\n=== 存活球队 ({len(alive_cn)}) ===")
for t in sorted(alive_cn):
    en = CN_TO_EN.get(t, t)
    print(f"  ✅ {t} ({team_matches.get(en, 0)}场)")

# 更新 tournament_state.json
changes = 0
for cn_name in ts:
    should_be = cn_name in eliminated_cn
    current = ts[cn_name].get('eliminated', False)
    if should_be != current:
        ts[cn_name]['eliminated'] = should_be
        changes += 1

json.dump(ts, open('/root/data/tournament_state.json', 'w'), ensure_ascii=False, indent=2)
print(f"\n✅ 更新完成, {changes} 变更")

for team in ['日本', '德国']:
    en = CN_TO_EN.get(team, team)
    print(f"{team}: elim={ts[team].get('eliminated')}, {team_matches.get(en,0)}场")
