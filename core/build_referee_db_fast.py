#!/usr/bin/env python3
"""快速构建裁判数据库 (仅需 ~200 次 API 调用)"""
import requests, json, os
from collections import defaultdict

KEY = "fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww"
HDR = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api"

# 已知的世界杯裁判 (从 WC match details 获取)
wc_referees = set()
wc_matches = requests.get(f"{BASE}/football/matches?competition_id=comp_6107&status=finished&per_page=50", headers=HDR, timeout=30)
for m in wc_matches.json().get('data', []):
    try:
        det = requests.get(f"{BASE}/football/matches/{m['id']}", headers=HDR, timeout=10).json().get('data', {})
        ref = det.get('referee', {})
        if ref and ref.get('name'):
            wc_referees.add(ref['name'])
    except:
        pass

print(f"WC裁判: {sorted(wc_referees)}")

ref_data = defaultdict(lambda: {"matches": 0, "total_yc": 0, "total_rc": 0, "total_fl": 0})

# 从训练数据中扫描
with open('/root/data/thestats_training_data.json') as f:
    data = json.load(f)

# 取最新的200场
target = sorted(data, key=lambda x: x['date'], reverse=True)[:200]
print(f"待扫描: {len(target)} 场")

count = 0
for m in target:
    mid = m['match_id']
    try:
        r = requests.get(f"{BASE}/football/matches/{mid}", headers=HDR, timeout=10)
        det = r.json().get('data')
        if not det:
            continue
        ref = det.get('referee', {}).get('name', '')
        if not ref:
            continue
    except:
        continue

    try:
        sr = requests.get(f"{BASE}/football/matches/{mid}/stats", headers=HDR, timeout=10)
        st = sr.json().get('data', {})
        if not st:
            continue
        ov = st.get('overview', {}) or {}
    except:
        continue

    yc_h = (ov.get('yellow_cards', {}) or {}).get('all', {}) or {}
    rc_h = (ov.get('red_cards', {}) or {}).get('all', {}) or {}
    fl_h = (ov.get('fouls', {}) or {}).get('all', {}) or {}

    rd = ref_data[ref]
    rd["matches"] += 1
    rd["total_yc"] += (yc_h.get('home', 0) or 0) + (yc_h.get('away', 0) or 0)
    rd["total_rc"] += (rc_h.get('home', 0) or 0) + (rc_h.get('away', 0) or 0)
    rd["total_fl"] += (fl_h.get('home', 0) or 0) + (fl_h.get('away', 0) or 0)
    rd["name"] = ref
    count += 1

    if count >= 200:
        break

output = {}
for name, rd in ref_data.items():
    if rd["matches"] < 1:
        continue
    output[name] = {
        "avg_yellow": round(rd["total_yc"] / rd["matches"], 2),
        "avg_red": round(rd["total_rc"] / rd["matches"], 3),
        "avg_fouls": round(rd["total_fl"] / rd["matches"], 1),
        "match_count": rd["matches"],
    }

all_yc = [r["avg_yellow"] for r in output.values()]
all_rc = [r["avg_red"] for r in output.values()]
avg_yc = round(sum(all_yc) / len(all_yc), 2) if all_yc else 3.5
avg_rc = round(sum(all_rc) / len(all_rc), 3) if all_rc else 0.15
output["_default"] = {"avg_yellow": avg_yc, "avg_red": avg_rc, "avg_fouls": 20.0, "match_count": 0}

with open('/root/data/referee_strictness.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ 保存 {len(output)} 名裁判 (含默认值)")
print(f"全局均值: 黄牌={avg_yc}, 红牌={avg_rc}")
for name, d in sorted(output.items(), key=lambda x: -x[1].get('match_count', 0))[:8]:
    if name == '_default':
        continue
    print(f"  {name}: {d['match_count']}场, yc={d['avg_yellow']}, rc={d['avg_red']}")
