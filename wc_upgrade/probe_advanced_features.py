"""
probe_advanced_features.py — 探测 TheStatsAPI 高级端点数据结构
探测三个维度:
  1. 比赛统计: /football/matches/{mt_id}/statistics
  2. 国际赔率: /football/matches/{mt_id}/odds
  3. 裁判信息: 从主赛程获取 referee + 球队统计中的得牌数据
"""
import requests, json, os, sys
from datetime import datetime, timezone

KEY = os.environ.get('THE_KEY') or os.environ.get('THE_STATS_KEY')
if not KEY:
    KEY = 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww'

H = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
print(f"=== 探测高级端点 | {today} ===\n")

# 获取今天一场已完结的 WC 比赛作为测试样本
r = requests.get(f"{BASE}/matches?competition_id=comp_6107&status=finished&per_page=2", headers=H, timeout=15)
matches = r.json().get('data', [])
if not matches:
    print("❌ 无已完结的 WC 比赛")
    sys.exit(1)

m = matches[0]
mid = m['id']
ht = m.get('home_team', {}).get('name', '?')
at = m.get('away_team', {}).get('name', '?')
print(f"样本比赛: {ht} vs {at} (id={mid})\n")

# === 1. Statistics 端点 ===
print("=" * 60)
print("1. /football/matches/{id}/statistics")
print("=" * 60)
r1 = requests.get(f"{BASE}/matches/{mid}/statistics", headers=H, timeout=15)
print(f"   Status: {r1.status_code}")
if r1.status_code == 200:
    d = r1.json()
    data = d.get('data', d)
    if isinstance(data, dict):
        print(f"   Keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                print(f"   {k}: {json.dumps(v, indent=2, ensure_ascii=False)[:600]}")
            else:
                print(f"   {k}: {v}")
    elif isinstance(data, list):
        print(f"   Items: {len(data)}")
        if data:
            print(json.dumps(data[0], indent=2, ensure_ascii=False)[:600])
else:
    print(f"   Response: {r1.text[:300]}")

# === 2. Odds 端点 ===
print("\n" + "=" * 60)
print("2. /football/matches/{id}/odds")
print("=" * 60)
r2 = requests.get(f"{BASE}/matches/{mid}/odds", headers=H, timeout=15)
print(f"   Status: {r2.status_code}")
if r2.status_code == 200:
    d = r2.json()
    data = d.get('data', d)
    if isinstance(data, dict):
        print(f"   Keys: {list(data.keys())}")
        # 打印 Bet365 和 Pinnacle
        for bookmaker in ['bet365', 'pinnacle', 'betfair']:
            if bookmaker in data:
                print(f"   {bookmaker}: {json.dumps(data[bookmaker], indent=2, ensure_ascii=False)[:400]}")
    elif isinstance(data, list):
        print(f"   Items: {len(data)}")
        for item in data[:3]:
            print(f"   {json.dumps(item, indent=2, ensure_ascii=False)[:300]}")
else:
    print(f"   Response: {r2.text[:300]}")

# === 3. 比赛主端点 (裁判+球队统计) ===
print("\n" + "=" * 60)
print("3. 比赛主端点 (referee 字段)")
print("=" * 60)
r3 = requests.get(f"{BASE}/matches?competition_id=comp_6107&per_page=3", headers=H, timeout=15)
matches3 = r3.json().get('data', [])
print(f"   Status: {r3.status_code}")
for m3 in matches3:
    ht3 = m3.get('home_team', {}).get('name', '?')
    at3 = m3.get('away_team', {}).get('name', '?')
    ref = m3.get('referee', 'NO_REF')
    print(f"   {ht3} vs {at3}: referee={ref}")

# 球队统计 (含得牌)
print("\n" + "=" * 60)
print("4. 球队统计 /teams/{id}/stats (得牌数据)")
print("=" * 60)
# 取两支已知 WC 球队
for tid in ['tm_28735', 'tm_41775']:  # Mexico, South Korea
    r4 = requests.get(f"{BASE}/teams/{tid}/stats?season_id=sn_118868", headers=H, timeout=15)
    print(f"   Team {tid} Status: {r4.status_code}")
    if r4.status_code == 200:
        d = r4.json()
        data = d.get('data', d)
        if isinstance(data, dict):
            # 找得牌相关字段
            card_keys = [k for k in data.keys() if any(x in k.lower() for x in ['card', 'foul', 'yellow', 'red', 'disciplinary'])]
            if card_keys:
                for ck in card_keys:
                    print(f"   {ck}: {data[ck]}")
            else:
                print(f"   无得牌字段. All keys: {list(data.keys())[:20]}")
        print(f"   (truncated)")
    else:
        print(f"   Response: {r4.text[:200]}")