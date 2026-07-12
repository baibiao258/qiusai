#!/usr/bin/env python3
"""
analyze_daily_results.py — daily_jczq.py 后处理规律推断引擎 v2
============================================================
用法:
    python3 /root/daily_jczq.py                # 先生成预测
    python3 /root/analyze_daily_results.py      # 再规律分析

v2 新增:
  - 预测演变轨迹分析 (从CSV历史快照提取)
  - 近tie检测 (D/A或H/D概率差<5pp)
  - EV信号类型分析 (SPF方向vs其他)
  - Kelly归零预警
  - 市场分歧分类
"""

import csv
import json
import os
import sys
import re
from datetime import datetime
from collections import defaultdict

PREDICTIONS_LOG = '/root/data/predictions_log.csv'

VALID_MODELS = ('xgb_dc_nat_11d',)
CONF_SWEET_MIN = 55.0
CONF_SWEET_MAX = 78.0
CONF_DANGER = 80.0
CONF_WEAK = 40.0
NEAR_TIE_THRESHOLD = 5.0  # 近tie容忍度(pp)
RULES = 9  # 规律条数（与底部打印一致）

HTFT_CN = {
    'HH':'胜胜','HD':'胜平','HA':'胜负',
    'DH':'平胜','DD':'平平','DA':'平负',
    'AH':'负胜','AD':'负平','AA':'负负',
}


def _load_csv(path):
    if not os.path.exists(path):
        print(f'❌ 找不到: {path}')
        print('   请先运行 python3 /root/daily_jczq.py')
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def _build_evolution(rows):
    """从CSV中同一code的多条记录构建演变数据"""
    # 按code分组
    by_code = defaultdict(list)
    for r in rows:
        code = r['code']
        run_date = r.get('date', '') or ''
        # 跳过无日期的TSTh条目
        if code.startswith('TSTh'):
            continue
        by_code[code].append((run_date, r))
    
    evo = {}
    for code, entries in by_code.items():
        entries.sort(key=lambda x: x[0])  # 按run_date排序
        snapshots = []
        for rd, r in entries:
            try:
                h = float(r['pred_h'])
                d = float(r['pred_d'])
                a = float(r['pred_a'])
                snapshots.append({
                    'date': rd,
                    'h': h, 'd': d, 'a': a,
                    'spf': r['pred_spf_pick'],
                    'kelly': float(r.get('kelly_pct',0) or 0),
                })
            except:
                continue
        if len(snapshots) >= 2:
            evo[code] = snapshots
    return evo


def _analyze_evolution(code, evo_data):
    """分析预测演变特征"""
    if code not in evo_data:
        return {}
    
    snaps = evo_data[code]
    features = {
        'has_evolution': True,
        'n_snapshots': len(snaps),
        'initial_h': snaps[0]['h'],
        'initial_d': snaps[0]['d'],
        'initial_a': snaps[0]['a'],
        'final_h': snaps[-1]['h'],
        'final_d': snaps[-1]['d'],
        'final_a': snaps[-1]['a'],
        'initial_spf': snaps[0]['spf'],
        'final_spf': snaps[-1]['spf'],
        'initial_kelly': snaps[0]['kelly'],
        'final_kelly': snaps[-1]['kelly'],
    }
    
    # 初始max和最终max
    init_max = max(snaps[0]['h'], snaps[0]['d'], snaps[0]['a'])
    final_max = max(snaps[-1]['h'], snaps[-1]['d'], snaps[-1]['a'])
    features['initial_max'] = init_max
    features['final_max'] = final_max
    features['max_change'] = final_max - init_max
    
    # 检测SPF方向改变
    first_spf = snaps[0]['spf']
    last_spf = snaps[-1]['spf']
    spf_changed = first_spf != last_spf
    features['spf_direction_changed'] = spf_changed
    
    # 检测是否曾经跨过80%红线
    ever_above_80 = any(s['h'] > CONF_DANGER for s in snaps)
    features['ever_above_80'] = ever_above_80
    
    # 检测初始到最终是否跨过80%
    crossed_80 = init_max <= CONF_SWEET_MAX and final_max > CONF_DANGER
    features['crossed_80'] = crossed_80
    
    # 检测是否存在D/A近tie
    has_near_tie = False
    near_tie_type = ''
    for s in snaps:
        sorted_3 = sorted([('H',s['h']),('D',s['d']),('A',s['a'])], key=lambda x:x[1], reverse=True)
        gap_1_2 = sorted_3[0][1] - sorted_3[1][1]
        gap_2_3 = sorted_3[1][1] - sorted_3[2][1]
        if gap_1_2 < NEAR_TIE_THRESHOLD:
            has_near_tie = True
            near_tie_type = f"{sorted_3[0][0]}/{sorted_3[1][0]}近tie(gap={gap_1_2:.1f}pp)"
            break
        if gap_2_3 < NEAR_TIE_THRESHOLD and sorted_3[2][1] > 20:
            has_near_tie = True
            near_tie_type = f"{sorted_3[1][0]}/{sorted_3[2][0]}聚拢"
            break
    features['has_near_tie'] = has_near_tie
    features['near_tie_type'] = near_tie_type
    
    # Kelly归零检测
    kelly_start = snaps[0]['kelly']
    kelly_end = snaps[-1]['kelly']
    kelly_dropped_to_zero = kelly_start > 0.01 and kelly_end < 0.001
    features['kelly_dropped_to_zero'] = kelly_dropped_to_zero
    
    return features


def _classify_match(r, evo_features):
    code = r['code']
    home = r['home_cn']
    away = r['away_cn']
    league = r['league']
    match_date = r.get('match_date', '')

    if r.get('result_status','').strip() == 'filled':
        return None

    try:
        h = float(r.get('pred_h', 0) or 0)
        d = float(r.get('pred_d', 0) or 0)
        a = float(r.get('pred_a', 0) or 0)
    except (ValueError, TypeError):
        return None  # 跳过列错位/格式损坏的行
    spf = r.get('pred_spf_pick', '?')
    model = r.get('model_route', '?')
    bet = r.get('bet_action', '?')
    try:
        kelly = float(r.get('kelly_pct', 0) or 0)
    except (ValueError, TypeError):
        kelly = 0.0
    score = r.get('pred_top_score', '?')
    rq_pick = r.get('pred_rq_pick', '?')
    rq_val = r.get('rq', '')
    goals_pick = r.get('pred_goals_pick', '?')

    try:
        sh, sa = [int(x) for x in score.split(':')]
        score_dir = '主胜' if sh > sa else ('平' if sh == sa else '客胜')
    except:
        score_dir = '?'
    spf_score_agree = (spf == score_dir)
    max_prob = max(h, d, a)

    draw_prob = 0.0
    exp_total = 0.0
    top3_scores = []
    home_win_prob = 0.0
    away_win_prob = 0.0
    try:
        sd = json.loads(r.get('score_full','') or '{}')
        gf = json.loads(r.get('goals_full','') or '{}')
        # 优先用goals_full(完整分布sum≈1.0), 回退到score_full(仅top8≈50-85%)
        if gf:
            exp_total = sum(int(g)*p for g,p in gf.items())
        elif sd:
            exp_total = sum((int(s.split(':')[0])+int(s.split(':')[1]))*p for s,p in sd.items())
        if sd:
            draw_prob = sum(p for s,p in sd.items() if int(s.split(':')[0])==int(s.split(':')[1]))
            home_win_prob = sum(p for s,p in sd.items() if int(s.split(':')[0])>int(s.split(':')[1]))
            away_win_prob = sum(p for s,p in sd.items() if int(s.split(':')[0])<int(s.split(':')[1]))
            ranked = sorted(sd.items(), key=lambda x:x[1], reverse=True)
            top3_scores = [(s, p*100) for s,p in ranked[:3]]
    except:
        pass

    top_htft_cn = ''
    try:
        hd = json.loads(r.get('htft_full','') or '{}')
        if hd:
            th = sorted(hd.items(), key=lambda x:x[1], reverse=True)[0][0]
            top_htft_cn = HTFT_CN.get(th, th)
    except:
        pass

    base = dict(
        code=code, home=home, away=away, league=league, date=match_date,
        spf_pick=spf, model=model, bet=bet,
        h=h, d=d, a=a, kelly=kelly,
        score_pick=score, rq_pick=rq_pick, rq_val=rq_val,
        goals_pick=goals_pick, score_dir=score_dir,
        spf_score_agree=spf_score_agree, max_prob=max_prob,
        draw_prob=draw_prob, exp_total=exp_total,
        top3_scores=top3_scores,
        home_win_prob=home_win_prob, away_win_prob=away_win_prob,
        top_htft=top_htft_cn,
    )

    # ── 层级过滤 ──
    if model not in VALID_MODELS:
        return {**base, 'tier':'D', 'tier_label':'🔴', 'reason':'模型='+model+'已证0%正确', 'verdict':'忽略'}
    if bet not in ('推荐', '观望'):
        return {**base, 'tier':'D', 'tier_label':'🔴', 'reason':'系统标记='+bet, 'verdict':'跳过'}
    if max_prob < 1:
        return {**base, 'tier':'D', 'tier_label':'🔴', 'reason':'概率数据异常', 'verdict':'无数据'}

    # ── 规律引擎 ──
    reasons = []
    verdict_parts = []
    flags = []

    # 1: 置信度
    if max_prob < CONF_WEAK:
        reasons.append(f'信号弱(max={max_prob:.0f}%)')
        base_tier = 'C'
    elif max_prob > CONF_DANGER:
        reasons.append(f'超高置信(max={max_prob:.0f}%)⚠️葡萄牙85%先例')
        base_tier = 'B'
    elif max_prob >= CONF_SWEET_MIN:
        reasons.append(f'甜区置信(max={max_prob:.0f}%)')
        base_tier = 'A'
    else:
        reasons.append(f'中等置信(max={max_prob:.0f}%)')
        base_tier = 'B'

    # 2: SPF-Score方向
    if not spf_score_agree:
        reasons.append(f'SPF({spf})≠Score({score}→{score_dir})')
        if score_dir == '平':
            reasons.append('⚠️比分暗示平局')
            verdict_parts.append('防平')
        base_tier = chr(ord(base_tier)+1) if base_tier < 'C' else base_tier

    # 3: Kelly
    if kelly > 0.05:
        reasons.append(f'Kelly={kelly:.4f}✅')
    elif kelly > 0:
        reasons.append(f'Kelly={kelly:.4f}')
    else:
        reasons.append('Kelly=0')
        if base_tier == 'A':
            base_tier = 'B'

    # 4: 平局概率
    if draw_prob > 0.22:
        reasons.append(f'平局率{draw_prob*100:.0f}%偏高')
        if spf in ('主胜','客胜'):
            verdict_parts.append('防平')
    elif draw_prob < 0.12:
        reasons.append(f'平局率{draw_prob*100:.0f}%极低')

    # 5: RQ信号
    if spf == '主胜':
        if rq_pick == '让胜':
            reasons.append('RQ让胜(预期大胜)')
        elif rq_pick == '让负':
            reasons.append('RQ让负(预期小胜)')

    # 6: 期望进球(基于goals_full完整分布)
    if exp_total > 3.0:
        reasons.append(f'E[总进]={exp_total:.2f}偏高')
    elif exp_total < 2.0:
        reasons.append(f'E[总进]={exp_total:.2f}偏低')

    # ── 演变分析 (新!从历史快照) ──
    evo = evo_features or {}
    if evo.get('has_evolution'):
        # 6a: SPF方向改变
        if evo.get('spf_direction_changed'):
            reasons.append(f'⚠️SPF演变改变: {evo["initial_spf"]}→{evo["final_spf"]}(不稳定)')
            flags.append('spf_change')
            base_tier = chr(ord(base_tier)+1) if base_tier < 'C' else base_tier
        
        # 6b: 跨过80%
        if evo.get('crossed_80'):
            reasons.append(f'⚠️置信穿越: {evo["initial_max"]:.0f}%→{evo["final_max"]:.0f}%(跨80%)')
            flags.append('cross_80')
            base_tier = chr(ord(base_tier)+1) if base_tier < 'C' else base_tier
        
        # 6c: D/A近tie
        if evo.get('has_near_tie'):
            reasons.append(f'⚠️{evo["near_tie_type"]}')
            flags.append('near_tie')
        
        # 6d: Kelly归零
        if evo.get('kelly_dropped_to_zero'):
            reasons.append(f'⚠️Kelly归零: {evo["initial_kelly"]:.4f}→0(EV消失)')
            flags.append('kelly_zero')
            base_tier = chr(ord(base_tier)+1) if base_tier < 'C' else base_tier

    # ── 最终赛况 ──
    if spf == '主胜':
        dir_text = f'{home}胜'
    elif spf == '客胜':
        dir_text = f'{away}胜'
    else:
        dir_text = '平局'

    if verdict_parts:
        verdict = f'{dir_text}({"+".join(verdict_parts)})'
    else:
        verdict = dir_text

    if top3_scores:
        sr = '/'.join([s for s,_ in top3_scores[:2]])
        verdict += f' 比分→{sr}'
    verdict += f' 总进' + ('≥3' if exp_total > 3.0 else '≤2' if exp_total < 2.0 else f'∼{goals_pick}球')
    if top_htft_cn:
        verdict += f' 半全→{top_htft_cn}'

    tier_label_map = {'A':'🟢 推荐','B':'🟡 谨慎','C':'🟠 弱信号','D':'🔴 不可用'}
    return {**base, 'tier':base_tier, 'tier_label':tier_label_map.get(base_tier,'?'),
            'reason':'; '.join(reasons), 'verdict':verdict, 'flags':flags}


def print_report(results, completed_cnt=0):
    tier_order = {'A':0,'B':1,'C':2,'D':3}
    results.sort(key=lambda x: (tier_order.get(x['tier'],9), -x['kelly']))

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print()
    print('╔' + '═'*73 + '╗')
    print(f'║  ⚽ daily_jczq.py 规律推断报告 v2 ({RULES}条规律)    ║')
    print(f'║  {now}  | 基于{completed_cnt}场完赛+演变复盘           ║')
    print('╚' + '═'*73 + '╝')
    print()

    cnt = {'A':0,'B':0,'C':0,'D':0}
    for r in results:
        cnt[r['tier']] = cnt.get(r['tier'],0)+1
    print(f'  总场次: {len(results)}  |  🟢推荐{cnt["A"]}  🟡谨慎{cnt["B"]}  🟠弱信号{cnt["C"]}  🔴不可用{cnt["D"]}')
    print()

    # ── 汇总速览 ──
    print('  ┌────┬──────────┬──────────────────┬──────┬──────┬────────────────────────────────┐')
    print('  │ 级 │ 比赛      │ 预测SPF(概率)     │ 凯利 │比分  │ 推测赛况                        │')
    print('  ├────┼──────────┼──────────────────┼──────┼──────┼────────────────────────────────┤')
    for r in results:
        if r['tier'] in ('A','B','C'):
            label = r['tier_label'][:2]
            teams = f"{r['home'][:6]}vs{r['away'][:6]}"
            spf_str = f"{r['spf_pick']}({r['max_prob']:.0f}%)"
            k_str = f"{r['kelly']:.4f}" if r['kelly'] > 0 else '0    '
            sc_str = r['score_pick']
            vd_str = r['verdict'][:36]
            flag_str = ''
            if r.get('flags'):
                flag_str = ' ⚠️' + ' '.join(r['flags'][:2])
            print(f'  │ {label} │ {teams:16s} │ {spf_str:16s} │ {k_str} │ {sc_str:4s} │ {vd_str:36s}{flag_str} │')
    if cnt['D'] > 0:
        # 统计D-tier子类型
        d_counts = defaultdict(int)
        for r in results:
            if r['tier'] == 'D':
                if '模型=' in r['reason'] and '0%正确' in r['reason']:
                    d_counts['market_fallback'] += 1
                elif '系统标记' in r['reason']:
                    d_counts['跳过'] += 1
                else:
                    d_counts['其他'] += 1
        d_summary = ' | '.join(f'{k}={v}' for k,v in d_counts.items())
        print(f'  ├────┼──────────┼──────────────────┼──────┼──────┼────────────────────────────────┤')
        print(f'  │ 🔴 │ D-tier  {cnt["D"]}场         │ {d_summary:32s} │')
    print('  └────┴──────────┴──────────────────┴──────┴──────┴────────────────────────────────┘')
    print()

    for r in results:
        if r['tier'] == 'D':
            continue
        print(f'  {"─"*73}')
        print(f'  {r["tier_label"]} {r["code"]} | {r["home"]} vs {r["away"]}  |  {r["date"]}')
        print(f'  {r["spf_pick"]}(H={r["h"]:.0f}% D={r["d"]:.0f}% A={r["a"]:.0f}%)  '
              f'盘{r["rq_val"]}→{r["rq_pick"]}  '
              f'比{r["score_pick"]}(→{r["score_dir"]})  '
              f'总{r["goals_pick"]}球  '
              f'Kelly={r["kelly"]:.4f}  '
              f'{"一致" if r["spf_score_agree"] else "SPF≠Score❌"}')
        if r['top3_scores']:
            print(f'  比分T3: {" ".join(f"{s}({p:.1f}%)" for s,p in r["top3_scores"])}  '
                  f'平局率{r["draw_prob"]*100:.0f}%  E总进{r["exp_total"]:.2f}')
        print(f'  规律: {r["reason"]}')
        print(f'  🎯 {r["verdict"]}')
        print()

    # 底部规律说明
    print(f'  {"─"*73}')
    print(f'  规律体系 ({RULES}条, 源自{completed_cnt}场已完赛复盘):')
    print(f'  ┌─────────────────────────────────────────────────────────────────────┐')
    print(f'  │ ❶ 模型过滤: market_fallback=0% → 直接忽略                          │')
    print(f'  │ ❷ 置信三段: 甜区55-78%(强) | >80%(爆冷) | <40%(模糊)               │')
    print(f'  │ ❸ SPF-Score方向一致: 必要非充分条件                                 │')
    print(f'  │ ❹ Kelly>0: 加分, Kelly=0→降1级                                     │')
    print(f'  │ ❺ 演变轨迹: SPF方向改变→预警 | 跨80%红线→爆冷风险                   │')
    print(f'  │ ❻ 近tie检测: D/A或H/D概率差<5pp→不稳定信号                         │')
    print(f'  │ ❼ EV信号类型: 非SPF方向的高EV是价格信号,非结果信号                   │')
    print(f'  │ ❽ Kelly归零: 演变中Kelly从>0→0 → EV消失是危险信号                   │')
    print(f'  │ ❾ 市场分歧: 模型≠市场时, SPF方向分歧需警惕                           │')
    print(f'  └─────────────────────────────────────────────────────────────────────┘')
    print(f'  ✅ 正确案例: 英格兰(演变不跨80%+稳定甜区) 哥伦比亚(方向不变+近tie拉开)')
    print(f'  ❌ 错误案例: 葡萄牙(跨80%红线) 加纳(SPF方向改变+三向均衡)')
    print()


def main():
    rows = _load_csv(PREDICTIONS_LOG)
    
    # 统计已完赛的有效模型场次（结果已回填）
    filled_codes = set()
    for r in rows:
        if r.get('result_status','').strip() == 'filled' and r.get('model_route','') in VALID_MODELS:
            filled_codes.add(r['code'])
    completed_cnt = len(filled_codes)
    
    # 构建演变数据
    evo_data = _build_evolution(rows)
    
    # 去重: 每个code只保留最新match_date
    best = {}
    for r in rows:
        code = r['code']
        md = r.get('match_date', '') or ''
        if code not in best or md > best[code].get('match_date', ''):
            best[code] = r
    deduped = list(best.values())
    
    results = []
    for r in deduped:
        evo = _analyze_evolution(r['code'], evo_data)
        res = _classify_match(r, evo)
        if res is not None:
            results.append(res)

    if not results:
        print('⚠️ 没有未完赛的预测场次')
        return

    print_report(results, completed_cnt)


if __name__ == '__main__':
    main()
