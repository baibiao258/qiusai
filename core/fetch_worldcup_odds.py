#!/usr/bin/env python3
"""Fetch 2026 World Cup winner odds from The Odds API — daily cron"""
import requests, json, os, sys
from datetime import datetime

DATA_DIR = '/root/data'
API_KEY = os.environ.get('THE_ODDS_API_KEY', '425a7cb6604fe89fcbd46a524ac08a11')
URL = (f"https://api.the-odds-api.com/v4/sports/"
       f"soccer_fifa_world_cup_winner/odds/"
       f"?apiKey={API_KEY}&regions=us,uk,eu")

TEAMS_2026 = [
    'Argentina', 'Australia', 'Austria', 'Belgium', 'Bosnia & Herzegovina',
    'Brazil', 'Canada', 'Chile', 'China', 'Colombia', 'Croatia',
    'Curacao', 'Czech Republic', 'Denmark', 'Ecuador', 'Egypt',
    'England', 'France', 'Germany', 'Ghana', 'Hungary', 'Iran',
    'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Mexico',
    'Morocco', 'Netherlands', 'New Zealand', 'Nigeria', 'North Korea',
    'Norway', 'Panama', 'Paraguay', 'Peru', 'Poland', 'Portugal',
    'Romania', 'Saudi Arabia', 'Senegal', 'Serbia', 'Slovakia',
    'Slovenia', 'South Korea', 'Spain', 'Switzerland', 'Tunisia',
    'Ukraine', 'USA', 'Uruguay', 'Uzbekistan', 'Venezuela', 'Wales',
]

def normalize_name(name):
    mapping = {
        'Côte d\'Ivoire': 'Ivory Coast', 'Cote d\'Ivoire': 'Ivory Coast',
        'USA': 'USA', 'United States': 'USA',
        'South Korea': 'South Korea', 'Korea Republic': 'South Korea', 'Korea': 'South Korea',
        'Bosnia-Herzegovina': 'Bosnia & Herzegovina', 'Bosnia': 'Bosnia & Herzegovina',
        'Netherlands': 'Netherlands', 'Holland': 'Netherlands',
        'New Zealand': 'New Zealand', 'DPR Korea': 'North Korea',
        'Switzerland': 'Switzerland', 'Serbia': 'Serbia',
        'Slovenia': 'Slovenia', 'Slovakia': 'Slovakia',
        'Iran': 'Iran', 'I.R. Iran': 'Iran',
        'Saudi Arabia': 'Saudi Arabia',
        'Curaçao': 'Curacao', 'Czech Republic': 'Czech Republic',
    }
    return mapping.get(name, name)

def fetch():
    print(f"🌐 GET {URL[:90]}...")
    resp = requests.get(URL, timeout=30)

    if resp.status_code == 401:
        print("❌ API Key 无效"); return None
    elif resp.status_code == 404:
        print("❌ 赛事未开放"); return None
    elif resp.status_code != 200:
        print(f"❌ HTTP {resp.status_code}: {resp.text[:200]}"); return None

    data = resp.json()
    winner_odds = {}
    bookmaker_count = 0

    for entry in data if isinstance(data, list) else [data]:
        for book in entry.get('bookmakers', []):
            bookmaker_count += 1
            for m in book.get('markets', []):
                if m.get('key') == 'outrights':
                    for o in m.get('outcomes', []):
                        name = normalize_name(o['name'])
                        price = o['price']
                        if name not in winner_odds or price < winner_odds[name]:
                            winner_odds[name] = price

    print(f"  📚 Bookmakers: {bookmaker_count}  |  🏆 队伍: {len(winner_odds)}")

    sorted_odds = sorted(winner_odds.items(), key=lambda x: x[1])
    total_implied = sum(1.0 / p for _, p in sorted_odds)
    for name, price in sorted_odds[:15]:
        prob = (1.0 / price) / total_implied * 100
        pps = '█' * int(prob) + '░' * max(0, 30 - int(prob))
        print(f"    {name:<20s} {price:>5.1f}  {prob:5.1f}%  {pps}")

    used = int(resp.headers.get('x-requests-used', 0))
    limit = int(resp.headers.get('x-requests-limit', 500))
    remain = limit - used
    print(f"  📊 已用 {used}/{limit}  剩余 {remain}")

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, 'theodds_api_data.json')
    with open(path, 'w') as f:
        json.dump({
            "winner_odds": winner_odds,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_count": used,
            "request_limit": limit,
        }, f, indent=2)
    print(f"\n✅ 已保存 {path}")
    return True

if __name__ == '__main__':
    sys.exit(0 if fetch() else 1)
