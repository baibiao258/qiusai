#!/usr/bin/env python3
"""超快速裁判数据库构建 (50场)"""
import requests, json, os
from collections import defaultdict

KEY = "fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww"
HDR = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api"

ref_data = defaultdict(lambda: {"matches": 0, "total_yc": 0, "total_rc": 0})
counted = 0

# 只用最新50场
with open('/root/data/thestats_training_data.json') as f:
    data = sorted(json.load(f), key=lambda x: x['date'], reverse=True)

for m in data[:50]:
    mid = m['match_id']
    # 并行获取 match + stats
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def get(path):
            r = requests.get(f"{BASE}{path}", headers=HDR, timeout=10)
            return r.json().get('data', {}) if r.status_code == 200 else {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut_det = ex.submit(get, f"/football/matches/{mid}")
            fut_st = ex.submit(get, f"/football/matches/{mid}/stats")
            det = fut_det.result()
            st = fut_st.result()
    except:
        continue

    ref = (det.get('referee', {}) or {}).get('name', '')
    if not ref:
        continue

    ov = st.get('overview', {}) or {}
    yc_all = (ov.get('yellow_cards', {}) or {}).get('all', {}) or {}
    rc_all = (ov.get('red_cards', {}) or {}).get('all', {}) or {}

    rd = ref_data[ref]
    rd["matches"] += 1
    rd["total_yc"] += (yc_all.get('home', 0) or 0) + (yc_all.get('away', 0) or 0)
    rd["total_rc"] += (rc_all.get('home', 0) or 0) + (rc_all.get('away', 0) or 0)
    rd["name"] = ref
    counted += 1

output = {}
for name, rd in ref_data.items():
    if rd["matches"] < 1: continue
    output[name] = {
        "avg_yellow": round(rd["total_yc"] / rd["matches"], 2),
        "avg_red": round(rd["total_rc"] / rd["matches"], 3),
        "match_count": rd["matches"],
    }

all_yc = [r["avg_yellow"] for r in output.values()]
avg_yc = round(sum(all_yc)/len(all_yc), 2) if all_yc else 3.5
avg_rc = round(sum(r["avg_red"] for r in output.values())/len(all_yc), 3) if all_yc else 0.15
output["_default"] = {"avg_yellow": avg_yc, "avg_red": avg_rc, "match_count": 0}
with open('/root/data/referee_strictness.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✅ {counted}场, {len(output)}裁判")
print(f"默认: yc={avg_yc}, rc={avg_rc}")
for n, d in sorted(output.items(), key=lambda x:-x[1]['match_count'])[:5]:
    if n != '_default':
        print(f"  {n}: {d['match_count']}场, yc={d['avg_yellow']}")
