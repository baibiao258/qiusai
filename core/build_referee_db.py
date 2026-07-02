#!/usr/bin/env python3
"""
build_referee_db.py — 从训练数据构建裁判出牌数据库
=============================================
扫描 TheStatsAPI 训练数据, 提取每位裁判的历史出牌数据.

输出: /root/data/referee_strictness.json
"""
import requests, json, os, time
from collections import defaultdict

API_KEY="fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww"
HDR={"Authorization":f"Bearer {API_KEY}"}
BASE="https://api.thestatsapi.com/api"

print(f"{'='*50}")
print(f"  构建裁判数据库")
print(f"{'='*50}")

# 加载训练数据
with open('/root/data/thestats_training_data.json') as f:
    data = json.load(f)
print(f"训练数据: {len(data):,} 场")

# 对每场比赛, 获取裁判信息 + 黄牌/红牌
# 使用 match detail + stats 端点
referee_data = defaultdict(lambda: {
    "name": "", "matches": 0,
    "total_yellow": 0, "total_red": 0,
    "total_fouls": 0, "total_cards": 0,
})
referee_matches = defaultdict(list)

# 只扫描一小部分 (最新500场) 和所有世界杯/国际比赛
sample = [m for m in data if any(k in m.get('comp_name','').lower() for k in ['world cup', 'euro', 'copa', 'international', 'friendly', 'championship', 'premier', 'liga', 'serie', 'bundesliga'])]
# 取最后2000场
sample = sorted(data, key=lambda x: x['date'], reverse=True)[:2000]
print(f"待扫描: {len(sample):,} 场 (最新2000场)")

counted = 0
errors = 0
for i, m in enumerate(sample):
    mid = m['match_id']
    
    # 获取 match detail (含 referee)
    try:
        r = requests.get(f"{BASE}/football/matches/{mid}", headers=HDR, timeout=10)
        if r.status_code != 200:
            errors += 1
            continue
        detail = r.json().get('data')
        if not detail:
            errors += 1
            continue
    except:
        errors += 1
        continue
    
    ref_info = detail.get('referee')
    if not ref_info:
        continue
    
    ref_name = ref_info.get('name', '')
    if not ref_name:
        continue
    
    # 获取 stats (含黄牌)
    try:
        sr = requests.get(f"{BASE}/football/matches/{mid}/stats", headers=HDR, timeout=10)
        if sr.status_code != 200:
            continue
        stats = sr.json().get('data')
        if not stats:
            continue
    except:
        continue
    
    overview = stats.get('overview', {})
    yc_all = overview.get('yellow_cards', {}).get('all', {}) if isinstance(overview, dict) else {}
    rc_all = overview.get('red_cards', {}).get('all', {}) if isinstance(overview, dict) else {}
    fouls_all = overview.get('fouls', {}).get('all', {}) if isinstance(overview, dict) else {}
    
    home_yc = yc_all.get('home', 0) if isinstance(yc_all, dict) else 0
    away_yc = yc_all.get('away', 0) if isinstance(yc_all, dict) else 0
    home_rc = rc_all.get('home', 0) if isinstance(rc_all, dict) else 0
    away_rc = rc_all.get('away', 0) if isinstance(rc_all, dict) else 0
    home_fl = fouls_all.get('home', 0) if isinstance(fouls_all, dict) else 0
    away_fl = fouls_all.get('away', 0) if isinstance(fouls_all, dict) else 0
    
    total_yc = (home_yc or 0) + (away_yc or 0)
    total_rc = (home_rc or 0) + (away_rc or 0)
    total_fl = (home_fl or 0) + (away_fl or 0)
    
    rd = referee_data[ref_name]
    rd["name"] = ref_name
    rd["matches"] += 1
    rd["total_yellow"] += total_yc
    rd["total_red"] += total_rc
    rd["total_fouls"] += total_fl
    rd["total_cards"] += total_yc + total_rc * 3  # 红牌折算3张黄牌
    
    counted += 1
    
    if (i+1) % 500 == 0:
        print(f"   进度: {i+1}/{len(sample)}, 裁判数={len(referee_data)}, 错误={errors}")

# 转换为可导出格式
output = {}
for name, rd in referee_data.items():
    if rd["matches"] >= 2:  # 至少执法2场
        avg_yellow = rd["total_yellow"] / rd["matches"]
        avg_red = rd["total_red"] / rd["matches"]
        avg_fouls = rd["total_fouls"] / rd["matches"]
        output[name] = {
            "avg_yellow": round(avg_yellow, 2),
            "avg_red": round(avg_red, 3),
            "avg_fouls": round(avg_fouls, 1),
            "avg_cards": round(rd["total_cards"] / rd["matches"], 2),
            "match_count": rd["matches"],
        }

print(f"\n✅ 数据库构建完成:")
print(f"   扫描场次: {counted}")
print(f"   裁判数量: {len(output)} (>=2场执法的)")
print(f"   典型裁判:")
# 按match count排序
top_refs = sorted(output.items(), key=lambda x: -x[1]['match_count'])[:5]
for name, rd in top_refs:
    print(f"     {name}: {rd['match_count']}场, avg_yc={rd['avg_yellow']}, avg_rc={rd['avg_red']}")

# 计算全局均值
all_yc = [r['avg_yellow'] for r in output.values()]
all_rc = [r['avg_red'] for r in output.values()]
print(f"   全局均值黄牌: {sum(all_yc)/len(all_yc):.2f}, 红牌: {sum(all_rc)/len(all_rc):.3f}")

# 添加默认值
output["_default"] = {
    "avg_yellow": round(sum(all_yc)/len(all_yc), 2) if all_yc else 3.5,
    "avg_red": round(sum(all_rc)/len(all_rc), 3) if all_rc else 0.15,
    "avg_fouls": 20.0,
    "avg_cards": 4.0,
    "match_count": 0,
}

with open('/root/data/referee_strictness.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n💾 保存 {len(output)} 名裁判数据到 referee_strictness.json")
