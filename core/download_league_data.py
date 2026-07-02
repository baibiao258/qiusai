#!/usr/bin/env python3
"""下载 football-data.org 历史联赛数据"""
import os, sys, urllib.request, csv, time
from datetime import datetime

DATA_DIR = '/root/data/football_data_leagues'
os.makedirs(DATA_DIR, exist_ok=True)

# 关键联赛，含 WC 2026 主流球队
LEAGUES = {
    'E0': 'England Premier League',      # 英超
    'E1': 'England Championship',        # 英冠
    'D1': 'Germany Bundesliga',          # 德甲
    'D2': 'Germany 2. Bundesliga',       # 德乙
    'I1': 'Italy Serie A',               # 意甲
    'I2': 'Italy Serie B',               # 意乙
    'SP1': 'Spain La Liga',              # 西甲
    'SP2': 'Spain Segunda Division',     # 西乙
    'F1': 'France Ligue 1',              # 法甲
    'F2': 'France Ligue 2',              # 法乙
    'P1': 'Portugal Primeira Liga',      # 葡超
    'N1': 'Netherlands Eredivisie',      # 荷甲
    'B1': 'Belgium Pro League',          # 比甲
    'SC0': 'Scotland Premiership',       # 苏超
    'T1': 'Turkey Super Lig',            # 土超
    'G1': 'Greece Super League',         # 希腊超
}

SEASONS = ['2122', '2223', '2324', '2425']  # 最近 4 个赛季

BASE_URL = 'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'

def download_league(league_code, league_name):
    """下载单个联赛多个赛季"""
    all_rows = []
    for season in SEASONS:
        url = BASE_URL.format(season=season, league=league_code)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            raw = urllib.request.urlopen(req, timeout=20).read().decode('utf-8', errors='ignore')
            reader = csv.DictReader(raw.splitlines())
            for row in reader:
                row['Season'] = f'20{season[:2]}-20{season[2:]}'
                row['League'] = league_code
                all_rows.append(row)
            print(f"  ✅ {league_name} {season}: {len(all_rows)} 累计")
            time.sleep(0.5)
        except Exception as e:
            print(f"  ⚠️ {league_name} {season}: {e}")
    return all_rows

def main():
    all_data = []
    for code, name in LEAGUES.items():
        print(f"\n📥 下载 {name} ({code})...")
        rows = download_league(code, name)
        all_data.extend(rows)
        print(f"  {name} 总计: {len(rows)} 场")

    # 保存合并 CSV
    if all_data:
        # 统一列（不同赛季列可能不同，取并集）
        fieldnames = set()
        for r in all_data:
            fieldnames.update(r.keys())
        fieldnames = sorted(fieldnames)

        out_path = os.path.join(DATA_DIR, 'all_leagues_combined.csv')
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        print(f"\n✅ 合并保存: {out_path} ({len(all_data)} 场)")

        # 也分别保存每联赛
        for code, name in LEAGUES.items():
            league_rows = [r for r in all_data if r.get('League') == code]
            if league_rows:
                fnames = sorted(set().union(*[r.keys() for r in league_rows]))
                p = os.path.join(DATA_DIR, f'{code}_{name.replace(" ", "_")}.csv')
                with open(p, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fnames)
                    writer.writeheader()
                    writer.writerows(league_rows)
                print(f"  {name} -> {p} ({len(league_rows)} 场)")

if __name__ == '__main__':
    main()