#!/usr/bin/env python3
"""
展示 predictions_log.csv 中指定日期的完整5玩法预测

用法:
  python3 _show_tomorrow.py                  # 今天所有场次
  python3 _show_tomorrow.py 2026-06-14       # 按日期
  python3 _show_tomorrow.py 周六              # 竞彩前缀
  python3 _show_tomorrow.py 周四002          # 场次编码
"""

import csv, json, sys
from datetime import date

HT_LABELS = {
    'HH': '胜胜', 'DH': '平胜', 'HD': '胜平', 'DD': '平平',
    'AA': '负负', 'DA': '平负', 'AD': '负平', 'HA': '胜负', 'AH': '负胜',
}


def match_p(row, arg):
    if not arg:
        return False
    code = row.get('code', '')
    row_date = row.get('date', '')
    return code == arg or code.startswith(arg) or row_date == arg


def format_full(rows_group):
    best = None
    best_s = -1
    for r in rows_group:
        s = 0
        if r.get('score_full', ''): s += 10
        gf = r.get('goals_full', '')
        if gf:
            try:
                gd = json.loads(gf)
                s += min(len(gd), 13)
            except:
                s += 10
        if r.get('htft_full', ''): s += 10
        if r.get('time', ''): s += 5
        if r.get('odds_h', ''): s += 3
        if s > best_s:
            best_s = s
            best = r
    if best is None:
        return

    r = best
    h = float(r.get('pred_h', 0))
    d0 = float(r.get('pred_d', 0))
    a = float(r.get('pred_a', 0))
    oh, od, oa = r.get('odds_h', ''), r.get('odds_d', ''), r.get('odds_a', '')
    rq_val = r.get('rq', '')
    rq = [float(r.get('pred_rq_win', 0)), float(r.get('pred_rq_draw', 0)), float(r.get('pred_rq_loss', 0))]
    code = r.get('code', '')
    home = r.get('home_cn', '')
    away = r.get('away_cn', '')
    tm = r.get('time', '')
    model = r.get('model_version', '')
    sep = "\u2500" * 60

    print(f'\n\u23f0 {code} {home} vs {away}  ({tm})')
    print(f'  {sep}')

    # 胜平负
    spf = sorted([('主胜', h), ('平局', d0), ('客胜', a)], key=lambda x: -x[1])
    print(f'  \u3010胜平负\u3011主{h:.1f}% / 平{d0:.1f}% / 客{a:.1f}%')
    print(f'  \u2192 推荐: {spf[0][0]}({spf[0][1]:.1f}%)')
    odds_str = f'{oh} / {od} / {oa}' if oh else '未开售'
    print(f'  市场赔率: {odds_str}')

    # 让球
    rq_sorted = sorted([('让胜', rq[0]), ('让平', rq[1]), ('让负', rq[2])], key=lambda x: -x[1])
    print(f'  \u3010让球(让{rq_val})\u3011让胜{rq[0]:.1f}% / 让平{rq[1]:.1f}% / 让负{rq[2]:.1f}%')
    print(f'  \u2192 推荐: {rq_sorted[0][0]}({rq_sorted[0][1]:.1f}%)')

    # 比分
    sf = r.get('score_full', '')
    if sf:
        scores = json.loads(sf)
        ss = sorted(scores.items(), key=lambda x: -float(x[1]))
        sc = ' '.join([f'{k}({float(v)*100:.1f}%)' for k, v in ss[:8]])
        print(f'  \u3010比分\u3011{sc}')
        print(f'  \u2192 推荐: {ss[0][0]}({float(ss[0][1])*100:.1f}%)')

    # 总进球
    gf = r.get('goals_full', '')
    if gf:
        goals = json.loads(gf)
        gs = sorted(goals.items(), key=lambda x: -float(x[1]))
        gs_str = ' '.join([f'{k}球({float(v)*100:.1f}%)' for k, v in gs])
        print(f'  \u3010总进球\u3011{gs_str}')
        print(f'  \u2192 推荐: {gs[0][0]}球({float(gs[0][1])*100:.1f}%)')

    # 半全场
    hf = r.get('htft_full', '')
    if hf:
        htft = json.loads(hf)
        hs = sorted(htft.items(), key=lambda x: -float(x[1]))
        hs_str = ' '.join([f'{HT_LABELS.get(k, k)}({float(v)*100:.1f}%)' for k, v in hs])
        print(f'  \u3010半全场\u3011{hs_str}')
        print(f'  \u2192 推荐: {HT_LABELS.get(hs[0][0], hs[0][0])}({float(hs[0][1])*100:.1f}%)')

    print(f'  模型: {model}')
    print()


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime('%Y-%m-%d')

    with open('/root/data/predictions_log.csv') as f:
        rows = list(csv.DictReader(f))

    grouped = {}
    for r in rows:
        grouped.setdefault(r.get('code', ''), []).append(r)

    matched = set()
    for r in rows:
        c = r.get('code', '')
        if c and match_p(r, arg):
            matched.add(c)

    if not matched:
        print(f'\u274c 未找到匹配 "{arg}" 的场次')
        return

    print(f'\u23f0 共 {len(matched)} 场匹配')
    for code in sorted(matched):
        format_full(grouped.get(code, []))

    print('所有预测按90分钟常规时间(含伤补)口径 | 数据: 500.com+365scores+DC+XGBoost')


if __name__ == '__main__':
    main()
