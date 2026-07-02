import pandas as pd

# 6/7 可买 jczq 主流场次 (从 500.com zqdc 提取)
target_matches = [
    ("K2 联赛", "天安城", "水原FC", "06-07 18:30"),
    ("K2 联赛", "首尔衣恋", "忠北清州", "06-07 18:30"),
    ("K2 联赛", "金浦市民", "全南天龙", "06-07 18:30"),
    ("友谊赛", "克罗地亚", "斯洛文尼亚", "06-08 02:45"),
    ("友谊赛", "摩洛哥", "挪威", "06-08 03:00"),
    ("友谊赛", "希腊", "意大利", "06-08 03:00"),
    ("友谊赛", "哥伦比亚", "约旦", "06-08 07:00"),
]

df7 = pd.read_csv('/root/data/365scores/2026-06-07.csv')
print(f"6/7 365scores 总场次: {len(df7)}")

# 也读历史累计数据
df_all = pd.read_csv('/root/data/365scores/all_games.csv')
print(f"all_games 累计: {len(df_all)} 场")

print("\n" + "=" * 80)
print("365scores 目标队伍数据")
print("=" * 80)

for league, home, away, time_str in target_matches:
    print(f"\n>>> {league} | {home} vs {away} | {time_str}")

    # 在 6/7 找
    matches_7 = df7[
        ((df7['home'].str.contains(home[:4], case=False, na=False)) |
         (df7['away'].str.contains(home[:4], case=False, na=False))) &
        ((df7['home'].str.contains(away[:4], case=False, na=False)) |
         (df7['away'].str.contains(away[:4], case=False, na=False)))
    ]

    # 累计找 (历史表现)
    matches_all = df_all[
        ((df_all['home'].str.contains(home[:4], case=False, na=False)) |
         (df_all['away'].str.contains(home[:4], case=False, na=False))) &
        ((df_all['home'].str.contains(away[:4], case=False, na=False)) |
         (df_all['away'].str.contains(away[:4], case=False, na=False)))
    ]

    if len(matches_7) > 0:
        for _, row in matches_7.iterrows():
            print(f"   6/7 找到: {row['home']} vs {row['away']} | {row['competition']} | {row['status']}")
            if pd.notna(row.get('vote_home')):
                print(f"      投票: H{row['vote_home']}% D{row['vote_draw']}% A{row['vote_away']}% n={row['vote_count']}")
            if pd.notna(row.get('trend_home_w')):
                print(f"      主近况: W{row['trend_home_w']} D{row['trend_home_d']} L{row['trend_home_l']} (胜率{row['trend_win_rate_home']}%)")
                print(f"      客近况: W{row['trend_away_w']} D{row['trend_away_d']} L{row['trend_away_l']} (胜率{row['trend_win_rate_away']}%)")
    else:
        print(f"   6/7 未找到精确匹配")

    # 历史交锋
    if len(matches_all) > 0:
        print(f"   历史累计 {len(matches_all)} 条记录")
        for _, row in matches_all.head(3).iterrows():
            print(f"      {row['date']} {row['home']} vs {row['away']} | {row['competition']} | {row['status']} | {row['score']}")
    else:
        print(f"   历史无交锋")
