#!/usr/bin/env python3
"""
一次性脚本: 从 all_games.csv 中提取足球数据到 football_games.csv
使用更全面的足球 competition 列表
"""
import csv
import os
from collections import Counter

SRC = '/root/data/365scores/all_games.csv'
DST = '/root/data/365scores/football_games.csv'

if not os.path.exists(SRC):
    print(f"❌ {SRC} 不存在")
    exit(0)

# 已知足球联赛名 (从实际数据中收集)
FOOTBALL_COMPS = {
    # 国际赛
    'Friendly International', 'FIFA World Cup',
    'UEFA Women WC Qualifiers', 'CONMEBOL Women\'s Nations League',
    'U21 Friendly International', 'U20 Friendly International',
    'U23 Friendly International', 'International Friendly (W)',
    'Friendly Women', 'Maurice Revello Tournament',
    'Euro U19 Qualification', 'Euro U17', 'Euro U17 Qualification',
    # 英格兰
    'Premier League', 'Championship', 'League One', 'League Two',
    'National League', 'National League Women', 'Premier League 2',
    'EFL Trophy', 'FA Cup', 'League Cup', 'FA Community Shield',
    # 西班牙
    'La Liga', 'LaLiga 2', 'Primera Division RFEF',
    'Segunda Division', 'Tercera Division A', 'Tercera División B',
    'Copa del Rey', 'Supercopa',
    # 德国
    'Bundesliga', '2. Bundesliga', '3. Liga',
    'DFB-Pokal', 'DFL-Supercup',
    'Regionalliga', 'Oberliga',
    # 意大利
    'Serie A', 'Serie B', 'Serie C', 'Serie D',
    'Coppa Italia', 'Supercoppa',
    # 法国
    'Ligue 1', 'Ligue 2', 'Championnat National',
    'Coupe de France',
    # 荷兰
    'Eredivisie', 'Eerste Divisie',
    # 葡萄牙
    'Primeira Liga', 'Segunda Liga',
    'Campeonato de Portugal',
    # 比利时
    'Pro League', 'Challenger Pro League',
    # 苏格兰
    'Premiership', 'Championship', 'League One',
    # 土耳其
    'Süper Lig', '1. Lig', '2. Lig', '3. Lig',
    # 希腊
    'Super League Greece',
    # 巴西
    'Brasileirão - Série A', 'Brasileirão - Série B',
    'Brasileiro U20', 'Copa do Brasil',
    'Paulista - Série A1', 'Paulista Serie B', 'Paulista U20',
    'Carioca', 'Carioca, Serie A2',
    'Catarinense - Serie B',
    'Baiano - Série B', 'Cearense 2', 'Paranaense 2',
    'Copa FGF', 'Copa do Nordeste', 'Copa Verde',
    'Copa Espirito Santo', 'Copa Sul-Sudeste',
    # 阿根廷
    'Liga Profesional', 'Primera Nacional',
    'Primera B Metropolitana', 'Primera C',
    'Federal A', 'Divisional C',
    'Copa de la Liga', 'Copa Argentina',
    'Copa Paraguay', 'Copa Simón Bolívar',
    # 日本
    'J1 League', 'J2 League', 'J3 League',
    'Emperor Cup', 'J.League YBC Levain Cup',
    # 韩国
    'K League 1', 'K League 2', 'K League 3',
    'FA Cup',
    # 美国/加拿大
    'MLS', 'MLS Next Pro', 'USL Championship', 'USL League One',
    'Canadian Premier League',
    # 其他
    'Liga MX', 'Liga de Expansión',
    'Eredivisie', 'Jupiler Pro League',
    'Premier Division', 'Premiere Division',
    'Primera División', 'Segunda División',
    'División Profesional', 'Division Intermedia',
    'Vysshaya Liga', 'Pershaya Liga',
    'OBOS-ligaen', 'Eliteserien', '1. divisjon', '2. divisjon',
    'Allsvenskan', 'Superettan', 'Damallsvenskan', 'Ettan',
    'Veikkausliiga', 'Ykkonen', 'Ykkosliiga', 'Suomen Cup',
    'Betrideildin', 'Formuladeildin',
    'Besta-deild karla', 'Inkasso Deildin', 'Icelandic Cup - Mjólkurbikarinn',
    'Meistriliiga', 'Esiliiga',
    'Virsliga', 'A Lyga', 'I Lyga',
    'Erovnuli Liga', 'Erovnuli Liga 2',
    'Premijer liga', 'Super Liga',
    'Super Ligi', 'Ligi Kuu Bara',
    'Liga 3', 'Liga 2',
    'Liga Uruguaya', 'Uruguayan Championship',
    'Liga Nacional de Honduras', 'Costa Rica Championship',
    'Liga FUTVE', 'Liga FUTVE 2',
    'Liga Profesional - Reserva',
    'Liga Portuguesa', 'Liga de ascenso',
    'Botola Pro', 'Botola 2',
    'Syrian Premier League', 'Yemeni League', 'Iraqi League', 'FA Iraqi Cup',
    'Baltic Cup', 'National Cup', 'Estonian Cup', 'Svenska Cupen',
    'NM Cupen', 'Lidl Starligue',
    'V-league', 'First Division', 'First Division B',
    'Champions League', 'Europa League', 'Conference League',
    'UEFA Super Cup',
    # 女足
    'Championship Women', 'Women League', 'Women\'s Super League',
    'Superliga Femenina', 'Campeonato Femenino', 'Campeonato Femenino A',
    'Alcides Márquez Cup (W)', 'NORCECA Final Four',
    'Superiores Primera (W)', 'Superiores Segunda (W)',
    'Paulista Women',
    # 澳超
    'A-League',
    # 其他青年/州级
    'U20 League', 'U19 League',
    'Promocional Amateur', 'Superiores Primera',
    'Superiores Segunda', 'APF Primera B', 'APF Primera C',
    'LPR Pro', 'Serie A2', 'Serie B',
}

# 反模式: 名字像足球但不是足球
NON_FOOTBALL_COMPS = {
    'ACB', 'LBF', 'LFB (W)', 'LNB', 'LNB Pro B', 'LNB Élite',
    'BSN', 'NBL', 'WNBA', 'CEBL', 'CBA', 'NBA', 'NBB', 'LBP',
    'LKL League', 'BNXT League', 'PBA Philippine Cup',
    'Basket League', 'Basket Liga', 'Lega A', '1ª FEB',
    'ABA League', 'VTB United League', 'Liga Nacional',
    'Liga Federal', 'LNB 2', 'Liga Mayor',
    'LMB', 'LMBP', 'NPB', 'KBO', 'MLB', 'CIBACOPA',
    'Lidl Starligue', 'Division de Honor', 'Liga de Honor Oro',
    'Liga de Honor Oro Women',
    'NHL', 'AIHL', 'NZIHL', 'Winner League',
    'Super Rugby', 'Top 14', 'URBA Top 14', 'NRL', 'CFL',
    'Volleyball Nations League', 'Volleyball Nations League (W)',
    'AVC Women\'s Cup', 'NORCECA Final Four (W)',
    'European League (M)', 'European League (W)',
    'Ilkley', 'Nottingham', 'Queen\'s', 'Stuttgart', 'Berlin', 'Bratislava',
    'Hertogenbosch', 'Poznan', 'Asunción', 'Dublin',
    'San Miguel de Tucuman',
    'Birmingham', 'Tyler', 'Bad Rappenau', 'Centurion',
    'Prostejov', 'Perugia', 'Makarska', 'Foggia',
    'Roland Garros - Men', 'Roland Garros - Women',
    'Roland Garros - Doubles (M)', 'Roland Garros - Doubles (W)',
    # 其他非足球
    'Liga Elite',  # 墨西哥棒球
    'Premiership Rugby (RU)',
    'Superliga', 'Pro A', '1st Division',
}

# 统计
comp_counter = Counter()
rows = []
with open(SRC, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        if None in row:
            # 跳过空列
            row = {k: v for k, v in row.items() if k is not None}
            comp_counter['(null column)'] += 1
            continue
        comp = row.get('competition', '').strip()
        comp_counter[comp] += 1
        rows.append(row)

print(f"📊 {SRC} 总行数: {len(rows)}")

# 分类
football_rows = []
unknown_rows = []

for row in rows:
    comp = row.get('competition', '').strip()
    if comp in FOOTBALL_COMPS:
        football_rows.append(row)
    elif comp in NON_FOOTBALL_COMPS:
        pass
    else:
        unknown_rows.append(row)

print(f"✅ 足球: {len(football_rows)} 行")
print(f"❌ 非足球: {len(rows) - len(football_rows) - len(unknown_rows)} 行")
print(f"❓ 未知: {len(unknown_rows)} 行")

if unknown_rows:
    print(f"\n❓ 未知比赛类型 (需要补全到 FOOTBALL_COMPS 或 NON_FOOTBALL_COMPS):")
    u_counter = Counter(r['competition'] for r in unknown_rows)
    for comp, cnt in u_counter.most_common():
        print(f"  {comp:45s} {cnt:5d}")

# 写入
with open(DST, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(football_rows)

os.rename(SRC, SRC.replace('.csv', '.raw.csv'))
print(f"\n✅ 写入 {DST}: {len(football_rows)} 行")
print(f"📦 原文件备份至: {SRC.replace('.csv', '.raw.csv')}")
print(f"\n⚠️  注意: {len(unknown_rows)} 行未知未纳入, 需手动判断")
