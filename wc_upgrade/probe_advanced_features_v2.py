"""
probe_advanced_features_v2.py — 更深入的端点探测
"""
import requests, json, os, sys
KEY = os.environ.get('THE_KEY') or os.environ.get('THE_STATS_KEY', 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww')
H = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

print("=== 深入探测 ===\n")

# 1. Odds 端点详情
print("=" * 60)
print("1. 赔率端点详情 /matches/{id}/odds")
print("=" * 60)
r = requests.get(f"{BASE}/matches/mt_209798753/odds", headers=H, timeout=15)
d = r.json().get('data', r.json())
bks = d.get('bookmakers', [])
print(f"   Bookmakers: {len(bks)}")
for bk in bks[:3]:
    name = bk.get('name', '?')
    odds = bk.get('odds', {})
    print(f"   {name}: {json.dumps(odds, ensure_ascii=False)[:300]}")
    
# 2. 单场信息 (裁判)
print("\n" + "=" * 60)
print("2. 单场信息 /matches/{id}")
print("=" * 60)
r2 = requests.get(f"{BASE}/matches/mt_209798753", headers=H, timeout=15)
print(f"   Status: {r2.status_code}")
if r2.status_code == 200:
    d2 = r2.json().get('data', r2.json())
    # 找所有可用字段
    for k, v in d2.items():
        if not isinstance(v, (dict, list)):
            print(f"   {k}: {v}")
    # 如果有 events
    if 'events' in d2:
        print(f"   events: {len(d2['events'])} items")
        for ev in d2['events'][:5]:
            print(f"      {json.dumps(ev, ensure_ascii=False)[:200]}")
    # lineup
    if 'lineup' in d2:
        print(f"   lineup: present")

# 3. 比赛统计 - 尝试不同路径
print("\n" + "=" * 60)
print("3. 统计端点探测")
print("=" * 60)
paths = [
    f"{BASE}/matches/mt_209798753/stats",
    f"{BASE}/matches/mt_209798753/statistics",
    f"{BASE}/matches/mt_209798753/events",
]
for p in paths:
    try:
        r3 = requests.get(p, headers=H, timeout=10)
        label = p.replace(BASE, '')
        print(f"   {label}: {r3.status_code}")
        if r3.status_code == 200:
            d3 = r3.json()
            print(f"      {json.dumps(d3, ensure_ascii=False)[:500]}")
    except Exception as e:
        print(f"   {label}: Error {e}")

# 4. 搜索比赛统计 (用 params)
print("\n" + "=" * 60)
print("4. 比赛统计 (查询参数探测)")
print("=" * 60)
# 有些 API 把 stats 放在 matches 查询里
for param in ['include=statistics', 'include=events', 'include=referee', 'include=all']:
    r4 = requests.get(f"{BASE}/matches/mt_209798753?{param}", headers=H, timeout=10)
    print(f"   ?{param}: {r4.status_code}")
    if r4.status_code == 200:
        d4 = r4.json().get('data', r4.json())
        extra = [k for k in d4.keys() if k not in ('id','competition_id','season_id','status','home_team','away_team','score','start_date','utc_date','matchday')]
        if extra:
            print(f"      Extra keys: {extra}")
        else:
            print(f"      No extra data")

# 5. 替代: 从 football-data.co.uk 获取统计数据 (已知可行)
print("\n" + "=" * 60)
print("5. 备选: football-data.co.uk 统计")
print("=" * 60)
try:
    import pandas as pd
    url = 'https://www.football-data.co.uk/mmz4281/2425/E0.csv'
    df = pd.read_csv(url)
    stat_cols = [c for c in df.columns if c in ('HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR')]
    print(f"   Available stat columns: {stat_cols}")
    if stat_cols:
        print(f"   Sample (1 row):")
        print(f"   {df[stat_cols].iloc[0].to_dict()}")
except ImportError:
    print("   pandas not available")
except Exception as e:
    print(f"   Error: {e}")