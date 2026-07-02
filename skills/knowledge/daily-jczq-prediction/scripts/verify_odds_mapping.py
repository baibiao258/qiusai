#!/usr/bin/env python3
"""
验证 500.com playid=269 的 spf/nspf 赔率映射是否正确。
检测 nspf 为空的比赛中，系统是否错误地将让球赔率用作标准赔率。

用法:
  python3 scripts/verify_odds_mapping.py [YYYY-MM-DD]
  
如果不传日期，默认使用今天。
"""
import json
import subprocess
import sys
from datetime import date

today = date.today().isoformat() if len(sys.argv) < 2 else sys.argv[1]

out = subprocess.check_output(
    ['node', '/root/wc_2026_upgrade/scrape_500_market.js', today, '269', '2'],
    text=True, timeout=30
)
data = json.loads(out)

total = len(data['result'])
ok = 0
affected = []

print(f"📡 验证 500.com 赔率映射  ({today})")
print(f"总计 {total} 场\n")

for r in data['result']:
    no = r['no']
    home = r.get('home', '')
    away = r.get('away', '')
    rq = r.get('rangqiu', '')
    handicap = int(rq or 0)
    spf = r['odds'].get('spf', {})
    nspf = r['odds'].get('nspf', {})
    has_nspf = bool(nspf and nspf.get('3'))

    if handicap != 0 and not has_nspf:
        # nspf 为空且 handicap≠0 → 让球赔率被误作标准赔率
        affected.append((no, home, away, handicap, spf, nspf))
        std = spf  # 系统误用的"标准赔率"
        print(f"❌ {no} {home} vs {away} (rq={handicap})")
        print(f"   spf(让球): 3={spf.get('3','-')} 1={spf.get('1','-')} 0={spf.get('0','-')}")
        print(f"   nspf(标准): 空 ⚠️")
        print(f"   系统误用: {std.get('3','?')}/{std.get('1','?')}/{std.get('0','?')} ← 这是让球赔率!")
    elif handicap == 0:
        ok += 1
        # 不让球，spf 就是标准赔率，没问题
    else:
        ok += 1
        # nspf 存在且正确
        std = nspf
        hcap = spf
        print(f"✅ {no} {home} vs {away} (rq={handicap})")
        print(f"   标准SPF: {std.get('3','-')}/{std.get('1','-')}/{std.get('0','-')}")
        print(f"   让球RQ:  {hcap.get('3','-')}/{hcap.get('1','-')}/{hcap.get('0','-')}")

print(f"\n{'='*50}")
print(f"✅ 正确: {ok} 场 | ❌ 受影响: {len(affected)} 场")
if affected:
    print(f"⚠️ {len(affected)} 场比赛的标准赔率被让球赔率污染!")
    for no, h, a, rq, spf, nspf in affected:
        print(f"  • {no} {h} vs {a} (rq={rq})")
        print("")
        print("原因: nspf 为空，代码将 spf(让球赔率) 赋值给 odds_h/d/a")
        print("修复: _fetch_live_odds_map() 自动从 live.500.com 获取平均欧赔兜底")
        print("  ✅ daily_jczq.py 的 scrape_500_odds_today() 已集成此修复")
        print("  ✅ 运行 daily_jczq.py 时看到 '🌐 live.500.com 平均欧赔兜底加载' 即表示生效")
        print("")
        print("如果兜底未生效，手动修复方法(已验证 2026-06-09):")
        print("  1. 从 trade 页提取 shuju_id:")
        print('     python3 -c "import re,urllib.request; r=urllib.request.urlopen(\"https://trade.500.com/jczq/?playid=269&g=2\",timeout=15); h=r.read().decode(\"gbk\",errors=\"replace\"); [print(f\"{m.group(1)} vs {m.group(2)} -> shuju-{m.group(3)}\") for m in re.finditer(r\"<tr[^>]*data-homesxname=\\\"([^\\\"]*)\\\"[^>]*data-awaysxname=\\\"([^\\\"]*)\\\"[^>]*>.*?shuju-(\\d+)\", h, re.DOTALL)]"')
        print("  2. 从分析页获取欧赔:")
        print('     python3 -c "import re,urllib.request; r=urllib.request.urlopen(\"https://odds.500.com/fenxi/shuju-{sid}.shtml\",timeout=10); h=r.read().decode(\"gbk\",errors=\"replace\"); m=re.search(\"<p class=\\\"pub_table_pl\\\"><span>([\\d.]+)</span><span>([\\d.]+)</span><span>([\\d.]+)</span></p>\",h); print(f\"欧赔: {m.group(1)}/{m.group(2)}/{m.group(3)}\" if m else \"未找到欧赔\")"')
        print("  3. 用欧赔替换 spf 值作为标准赔率")
else:
    print("✅ 今日所有赔率映射正确")
