#!/usr/bin/env python3
"""
一次性迁移: 从 all_games.raw.csv 提取纯足球数据到 football_games.csv
2026-06-14 后 collect_365scores_daily.py 直接用 filter_sid=1 过滤, 无需再跑此脚本。
"""
import csv
import os
from collections import Counter

SRC = '/root/data/365scores/all_games.raw.csv'
DST = '/root/data/365scores/football_games.csv'

if not os.path.exists(SRC):
    print("all_games.raw.csv 不存在 (可能已迁移过), 跳过")
    exit(0)

FOOTBALL_COMPS = {
    'Friendly International', 'FIFA World Cup',
    'UEFA Women WC Qualifiers', 'CONMEBOL Women\'s Nations League',
    'U21 Friendly International', 'U20 Friendly International',
    'U23 Friendly International', 'International Friendly (W)',
    'Friendly Women', 'Maurice Revello Tournament',
    'Premier League', 'Championship', 'League One', 'League Two',
    'National League', 'National League Women',
    'La Liga', 'LaLiga 2', 'Primera Division RFEF',
    'Segunda Division', 'Tercera Division A', 'Tercera División B',
    'Bundesliga', '2. Bundesliga', '3. Liga',
    'Serie A', 'Serie B', 'Serie C', 'Serie D',
    'Ligue 1', 'Ligue 2', 'Eredivisie', 'Primeira Liga',
    'Premier Division', 'Premiere Division',
    'MLS', 'MLS Next Pro', 'USL Championship', 'USL League One',
    'Canadian Premier League',
    'J1 League', 'J2 League', 'J3 League',
    'K League 1', 'K League 2', 'K League 3',
    'Brasileirão - Série A', 'Brasileirão - Série B',
    'Brasileiro U20', 'Paulista Serie B', 'Paulista U20',
    'Carioca', 'Carioca, Serie A2',
    'Liga Profesional', 'Primera Nacional',
    'Primera B Metropolitana', 'Primera C', 'Federal A', 'Divisional C',
    'Copa de la Liga', 'Copa Simón Bolívar',
    'División Profesional', 'Division Intermedia',
    'Vysshaya Liga', 'Pershaya Liga',
    'OBOS-ligaen', 'Eliteserien',
    'Allsvenskan', 'Superettan', 'Damallsvenskan',
    'Veikkausliiga', 'Ykkonen', 'Ykkosliiga', 'Suomen Cup',
    'Meistriliiga', 'Esiliiga', 'Virsliga', 'A Lyga', 'I Lyga',
    'Erovnuli Liga', 'Erovnuli Liga 2',
    'Betrideildin', 'Formuladeildin',
    'Besta-deild karla', 'Inkasso Deildin',
    'Botola Pro', 'Botola 2',
    'Super Ligi', 'Ligi Kuu Bara',
    'Champions League', 'Europa League', 'Conference League',
    'FA Cup', 'DFB-Pokal', 'Coppa Italia', 'Emperor Cup',
    'NM Cupen', 'Svenska Cupen', 'Estonian Cup',
    'Icelandic Cup - Mjólkurbikarinn',
    'Baltic Cup', 'National Cup',
    'FA Iraqi Cup', 'Yemeni League', 'Syrian Premier League',
    'Liga MX', 'Liga de Expansión',
    'Liga FUTVE', 'Liga FUTVE 2', 'Liga Profesional - Reserva',
    'Liga 3', 'Liga 2', 'Primera División',
    'Uruguayan Championship', 'Liga Uruguaya',
    'Copa Paraguay', 'Copa FGF', 'Copa Verde', 'Copa do Nordeste',
    'Campeonato Femenino', 'Campeonato Femenino A',
    'Superliga Femenina', 'Women\'s Super League',
    'Superiores Primera (W)', 'Superiores Segunda (W)',
    'Promocional Amateur', 'Superiores Primera', 'Superiores Segunda',
    'APF Primera B', 'APF Primera C',
    'First Division', 'First Division B', 'Division 1',
}

NON_FOOTBALL_COMPS = {
    'MLB', 'LMB', 'LMBP', 'NPB', 'KBO', 'CIBACOPA',
    'WNBA', 'NBA', 'NBB', 'LBP', 'CBA',
    'BSN', 'NBL', 'CEBL', 'LNB', 'LNB Pro B', 'LNB Élite',
    'ACB', 'LBF', 'LFB (W)', 'LKL League', 'BNXT League',
    'PBA Philippine Cup', 'Basket League', 'Basket Liga',
    'Lega A', '1ª FEB', 'ABA League', 'VTB United League',
    'NHL', 'AIHL', 'NZIHL', 'Winner League',
    'Super Rugby', 'Top 14', 'URBA Top 14', 'NRL', 'CFL',
    'Volleyball Nations League', 'Volleyball Nations League (W)',
    'AVC Women\'s Cup', 'NORCECA Final Four (W)',
    'European League (M)', 'European League (W)',
    'Ilkley', 'Nottingham', "Queen's", 'Stuttgart', 'Berlin', 'Bratislava',
    'Hertogenbosch', 'Poznan', 'Asunción', 'Dublin',
    'San Miguel de Tucuman', 'Birmingham', 'Tyler', 'Bad Rappenau',
    'Centurion', 'Prostejov', 'Perugia', 'Makarska', 'Foggia',
    'Roland Garros - Men', 'Roland Garros - Women',
    'Roland Garros - Doubles (M)', 'Roland Garros - Doubles (W)',
    'Lidl Starligue', 'Division de Honor', 'Liga de Honor Oro',
    'Liga Elite', 'Premiership Rugby (RU)',
    'Superliga', 'Pro A', '1st Division',
}

rows = []
with open(SRC, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        if None in row: continue
        rows.append(row)

football = [r for r in rows if r.get('competition','') in FOOTBALL_COMPS]
unknown = [r for r in rows if r['competition'] not in FOOTBALL_COMPS and r['competition'] not in NON_FOOTBALL_COMPS]

print(f"足球: {len(football)} / 未知: {len(unknown)} / 总计: {len(rows)}")
if unknown:
    print("未知类型:")
    for c, n in Counter(r['competition'] for r in unknown).most_common():
        print(f"  {c}: {n}")

with open(DST, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(football)
print(f"写入 {DST}: {len(football)} 行")
