#!/usr/bin/env python3
"""
format_predictions_sorted.py
============================
Read predictions_log.csv and display each match's 5 betting categories
with probabilities sorted DESCENDING within each category.

Usage:
    python3 format_predictions_sorted.py                        # all matches
    python3 format_predictions_sorted.py --code 周三201         # single match

Output:
    Separator line per match, then:
      ① 胜平负  — 3 items sorted by prob, with odds + EV
      ② 让球    — 3 items sorted by prob
      ③ 半全场  — 9 items sorted by prob
      ④ 比分    — Top 15 sorted by prob
      ⑤ 总进球  — 13档 sorted by prob (filters <0.05%)

Requires:
    /root/data/predictions_log.csv
    pandas
"""

import pandas as pd
import json
import sys

CSV_PATH = '/root/data/predictions_log.csv'

HTFT_MAP = {
    'HH': '胜胜', 'DH': '平胜', 'AH': '负胜',
    'HD': '胜平', 'DD': '平平', 'AD': '负平',
    'HA': '胜负', 'DA': '平负', 'AA': '负负',
}

def load_csv(path=CSV_PATH):
    return pd.read_csv(path)

def fmt_prob(v):
    """Convert stored value (66.0 = 66%) to display string."""
    return f'{float(v):.1f}%'

def fmt_prob_dec(v):
    """Convert decimal-stored probability (0.405 = 40.5%) to display."""
    return f'{float(v)*100:.1f}%'

def safe_json(s):
    if not isinstance(s, str) or not s:
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}

def print_match(code, df):
    row = df[df['code'] == code]
    if len(row) == 0:
        print(f'[NOT FOUND] {code}')
        return
    row = row.iloc[-1]

    home = row['home_cn']
    away = row['away_cn']
    league = row.get('league', '')
    rq = row.get('rq', '')

    print('=' * 65)
    print(f'  {code}  {home} vs {away}')
    print(f'  联赛: {league}  |  让球: {rq}')
    print('=' * 65)

    # ① 胜平负 — sorted by prob descending
    spf = [
        (float(row['pred_h']), f'{home}胜'),
        (float(row['pred_d']), '平局'),
        (float(row['pred_a']), f'{away}胜'),
    ]
    spf.sort(key=lambda x: -x[0])
    oh, od, oa = float(row['odds_h']), float(row['odds_d']), float(row['odds_a'])
    odds_map = {f'{home}胜': oh, '平局': od, f'{away}胜': oa}

    print('\n① 胜平负（按概率降序）')
    print('  ' + '-' * 50)
    for i, (p, name) in enumerate(spf):
        odd = odds_map[name]
        if 0 < odd < 100:
            ev = (p / 100 * odd - 1) * 100
            label = '🏆' if i == 0 else '  '
            print(f'  {label} #{i+1} {name:<10s}  {p:.1f}%  @{odd:.2f}  EV={ev:+.1f}%')
        else:
            print(f'     #{i+1} {name:<10s}  {p:.1f}%  未开售')

    # ② 让球 — sorted by prob descending
    try:
        rq_items = [
            (float(row['pred_rq_win']), '让胜'),
            (float(row['pred_rq_draw']), '让平'),
            (float(row['pred_rq_loss']), '让负'),
        ]
    except (ValueError, TypeError):
        rq_items = [(0, '让胜'), (0, '让平'), (0, '让负')]
    rq_items.sort(key=lambda x: -x[0])

    # 让球显示: rq存储原始让球数 (负=让球, 正=受让)
    # 遵循 pitfall: rq_text已含前缀, 不重复添加
    try:
        rq_val = float(rq) if not isinstance(rq, (int, float)) else rq
        if rq_val == 0:
            rq_display = '0 (平手)'
        elif rq_val < 0:
            rq_display = f'主让{abs(int(rq_val))}'
        else:
            rq_display = f'主受让{int(rq_val)}'
    except (ValueError, TypeError):
        rq_display = '0 (平手)'
    print(f'\\n② 竞彩让球 ({rq_display})（按概率降序）')
    print('  ' + '-' * 50)
    for i, (p, name) in enumerate(rq_items):
        label = '🏆' if i == 0 else '  '
        print(f'  {label} #{i+1} {name:<10s}  {p:.1f}%')

    # ③ 半全场 — 9 items sorted by prob descending
    hf = safe_json(row.get('htft_full', ''))
    if hf:
        hf_sorted = sorted(
            [(v, HTFT_MAP.get(k, k)) for k, v in hf.items()],
            key=lambda x: -x[0]
        )
        print('\n③ 半全场（9项完整，按概率降序）')
        print('  ' + '-' * 50)
        for i, (p, name) in enumerate(hf_sorted):
            label = '🏆' if i == 0 else '  '
            print(f'  {label} #{i+1} {name:<8s}  {p*100:.1f}%')

    # ④ 比分 — Top 15 sorted by prob descending
    sf = safe_json(row.get('score_full', ''))
    if sf:
        sf_sorted = sorted(sf.items(), key=lambda x: -x[1])
        print('\n④ 比分（Top 15，按概率降序）')
        print('  ' + '-' * 50)
        for i, (score, prob) in enumerate(sf_sorted[:15]):
            label = '🏆' if i == 0 else '  '
            print(f'  {label} #{i+1} {score:<6s}  {prob*100:.1f}%')

    # ⑤ 总进球 — 13档 sorted by prob descending
    gf = safe_json(row.get('goals_full', ''))
    if gf:
        gf_sorted = sorted(
            [(float(k), v) for k, v in gf.items()],
            key=lambda x: -x[1]
        )
        print('\n⑤ 总进球（13档完整，按概率降序）')
        print('  ' + '-' * 50)
        for i, (goals, prob) in enumerate(gf_sorted):
            if prob < 0.0005:
                continue
            g = f'{int(goals)}球' if goals < 13 else f'{int(goals)}+球'
            label = '🏆' if i == 0 else '  '
            print(f'  {label} #{i+1} {g:<6s}  {prob*100:.1f}%')

    print()


def main():
    df = load_csv()
    
    if '--code' in sys.argv:
        idx = sys.argv.index('--code')
        codes = sys.argv[idx+1:]
    else:
        codes = df['code'].unique()

    for code in codes:
        print_match(code, df)


if __name__ == '__main__':
    main()
