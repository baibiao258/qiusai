import pandas as pd

# 6/7 主流竞彩场次 (英文名匹配 365scores)
# 6/8 凌晨的友谊赛是主流
target_matches = [
    # K2 联赛 (小, 跳过)
    ("K2 联赛", "Cheonan City", "Suwon FC", "06-07 18:30"),
    # 6/8 凌晨主要友谊赛
    ("Friendly", "Croatia", "Slovenia", "06-08 02:45"),
    ("Friendly", "Morocco", "Norway", "06-08 03:00"),
    ("Friendly", "Greece", "Italy", "06-08 03:00"),
    ("Friendly", "Colombia", "Jordan", "06-08 07:00"),
]

df7 = pd.read_csv('/root/data/365scores/2026-06-07.csv')
df_all = pd.read_csv('/root/data/365scores/all_games.csv')

print(f"6/7 365scores 总场次: {len(df7)}")
print(f"all_games 累计: {len(df_all)} 场")
print()

for league, home, away, time_str in target_matches:
    print(f">>> {league} | {home} vs {away} | {time_str}")

    # 6/7 数据
    m7 = df7[
        ((df7['home'].str.contains(home, case=False, na=False)) |
         (df7['away'].str.contains(home, case=False, na=False))) &
        ((df7['home'].str.contains(away, case=False, na=False)) |
         (df7['away'].str.contains(away, case=False, na=False)))
    ]

    if len(m7) > 0:
        for _, row in m7.iterrows():
            print(f"   6/7: {row['home']} vs {row['away']} | {row['competition']} | {row['status']} | {row['time']}")
            if pd.notna(row.get('vote_count')) and row.get('vote_count', 0) > 0:
                print(f"      投票: H{row['vote_home']}% D{row['vote_draw']}% A{row['vote_away']}% n={row['vote_count']}")
            if pd.notna(row.get('trend_home_w')):
                print(f"      主近况: W{row['trend_home_w']} D{row['trend_home_d']} L{row['trend_home_l']} (胜率{row['trend_win_rate_home']}%)")
                print(f"      客近况: W{row['trend_away_w']} D{row['trend_away_d']} L{row['trend_away_l']} (胜率{row['trend_win_rate_away']}%)")
            if pd.notna(row.get('pop_rank_home')):
                print(f"      主人气:{row['pop_rank_home']} 客人气:{row['pop_rank_away']}")
    else:
        print(f"   6/7 未找到")

    # 6/7 也有其他包含 home/away 队名的场次 (历史交锋)
    h_matches = df7[df7['home'].str.contains(home, case=False, na=False) |
                     df7['away'].str.contains(home, case=False, na=False)]
    a_matches = df7[df7['home'].str.contains(away, case=False, na=False) |
                     df7['away'].str.contains(away, case=False, na=False)]
    print(f"   365scores 6/7 中 {home} 出现 {len(h_matches)} 次, {away} 出现 {len(a_matches)} 次")

    # 累计数据
    h_all = df_all[df_all['home'].str.contains(home, case=False, na=False) |
                    df_all['away'].str.contains(home, case=False, na=False)]
    a_all = df_all[df_all['home'].str.contains(away, case=False, na=False) |
                    df_all['away'].str.contains(away, case=False, na=False)]
    print(f"   累计 {home} 出现 {len(h_all)} 次, {away} 出现 {len(a_all)} 次")
    print()
