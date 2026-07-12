#!/usr/bin/env python3
"""近期准确率波动分析 - v2"""
import csv, json
from collections import defaultdict

rows = []
with open('data/predictions_log.csv', 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

settled = [r for r in rows if r.get('result_status') == 'filled']
print(f"总记录: {len(rows)}, 已结算: {len(settled)}")

by_date = defaultdict(list)
for r in settled:
    by_date[r['date']].append(r)

# 最近5天趋势
dates_sorted = sorted(by_date.keys())
print(f"\n最近5天趋势:")
for d in dates_sorted[-5:]:
    recs = by_date[d]
    rec_items = [r for r in recs if r.get('bet_action') in ('推荐','推荐 [低置信]')]
    total_rec = len(rec_items)
    rec_correct = sum(1 for r in rec_items if r.get('pred_spf_pick') and r.get('actual_hda') and 
        ((r['pred_spf_pick'] == '主胜' and r['actual_hda'] == 'H') or
         (r['pred_spf_pick'] == '平' and r['actual_hda'] == 'D') or
         (r['pred_spf_pick'] == '客胜' and r['actual_hda'] == 'A')))
    
    total_all = len(recs)
    all_correct = sum(1 for r in recs if r.get('pred_spf_pick') and r.get('actual_hda') and 
        ((r['pred_spf_pick'] == '主胜' and r['actual_hda'] == 'H') or
         (r['pred_spf_pick'] == '平' and r['actual_hda'] == 'D') or
         (r['pred_spf_pick'] == '客胜' and r['actual_hda'] == 'A')))
    
    all_acc = all_correct/total_all*100 if total_all else 0
    rec_acc = rec_correct/total_rec*100 if total_rec else 0
    print(f"  {d}: 全部{all_correct}/{total_all}={all_acc:.0f}% | 推荐 {rec_correct}/{total_rec}={rec_acc:.0f}%")

# 完整汇总
all_rec = [r for r in settled if r.get('bet_action') in ('推荐','推荐 [低置信]')]
rec_correct_spf = sum(1 for r in all_rec if r.get('pred_spf_pick') and r.get('actual_hda') and 
    ((r['pred_spf_pick'] == '主胜' and r['actual_hda'] == 'H') or
     (r['pred_spf_pick'] == '平' and r['actual_hda'] == 'D') or
     (r['pred_spf_pick'] == '客胜' and r['actual_hda'] == 'A')))
print(f"\n===== 总汇总 =====")
print(f"推荐: {len(all_rec)}场, SPF正确: {rec_correct_spf}/{len(all_rec)}={rec_correct_spf/len(all_rec)*100:.1f}%")

# 分日期汇总每个推荐
print(f"\n===== 每日推荐明细 =====")
for d in sorted(by_date.keys()):
    recs = by_date[d]
    rec_items = [r for r in recs if r.get('bet_action') in ('推荐','推荐 [低置信]')]
    if not rec_items:
        continue
    
    wrong = []
    for r in rec_items:
        c = 0
        if r.get('pred_spf_pick') and r.get('actual_hda') and \
          ((r['pred_spf_pick'] == '主胜' and r['actual_hda'] == 'H') or
           (r['pred_spf_pick'] == '平' and r['actual_hda'] == 'D') or
           (r['pred_spf_pick'] == '客胜' and r['actual_hda'] == 'A')):
            c = 1
        if not c:
            wrong.append(r)
    
    correct = len(rec_items) - len(wrong)
    pct = correct/len(rec_items)*100
    marker = '✅' if pct >= 60 else ('⚠️' if pct >= 40 else '❌')
    print(f"\n  {d}: {marker} {correct}/{len(rec_items)}={pct:.0f}%")
    for r in wrong[:5]:
        h = float(r.get('pred_h',0))
        d = float(r.get('pred_d',0))
        a = float(r.get('pred_a',0))
        pick = r.get('pred_spf_pick','?')
        actual = r.get('actual_hda','?')
        score = r.get('actual_score','?')
        max_p = max(h,d,a)
        route = r.get('model_route','')
        print(f"    ❌ {r['home_cn']} vs {r['away_cn']}: 预测{pick}({max_p:.0f}%)→实际{actual}({score}) {route}")

# 推荐在近期的细分
late_recs = [r for r in all_rec if r.get('date') >= '2026-06-25']
late_correct = sum(1 for r in late_recs if r.get('pred_spf_pick') and r.get('actual_hda') and 
    ((r['pred_spf_pick'] == '主胜' and r['actual_hda'] == 'H') or
     (r['pred_spf_pick'] == '平' and r['actual_hda'] == 'D') or
     (r['pred_spf_pick'] == '客胜' and r['actual_hda'] == 'A')))
print(f"\n===== 近期(6/25起) =====")
print(f"推荐: {len(late_recs)}场, 正确: {late_correct}/{len(late_recs)}={late_correct/len(late_recs)*100:.1f}%")

# 6/20低谷日详细  
june20 = [r for r in settled if r.get('date') == '2026-06-20']
print(f"\n===== 6/20低谷日 =====")
for r in june20:
    h = float(r.get('pred_h',0))
    d = float(r.get('pred_d',0))
    a = float(r.get('pred_a',0))
    pick = r.get('pred_spf_pick','?')
    actual = r.get('actual_hda','?')
    score = r.get('actual_score','?')
    max_p = max(h,d,a)
    route = r.get('model_route','')
    bet = r.get('bet_action','')
    correct = '✅' if (
        (pick == '主胜' and actual == 'H') or
        (pick == '平' and actual == 'D') or
        (pick == '客胜' and actual == 'A')
    ) else '❌'
    print(f"  {correct} {r['home_cn']} vs {r['away_cn']}: 预测{pick}({max_p:.0f}%) 实际{actual}({score}) [{bet}] {route}")
