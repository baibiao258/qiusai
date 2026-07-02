#!/usr/bin/env python3
"""
recalc_on_lineup.py — 赛前重推: 首发阵容 → 概率修正 → 预警
==========================================================
读取管线:
  /root/data/predictions_log.csv     — 当日预测
  /root/data/thestats_lineups.json   — 最新首发
  /root/data/team_name_mapping.json  — 中英队名映射
  /root/data/star_players.json       — 核心球员名单 (用于调整)

逻辑:
  1. 遍历 thestats_lineups.json 中已确认的首发
  2. 对每场找到 predictions_log.csv 中对应的预测行
  3. 调用 adjust_with_lineups() 模拟调幅
  4. 如果 penalty>0 且方向/EV 变化 → 打印 ⚠️ 预警
  5. 如果无变化但 penalty>0 → 打印蓝色 INFO

输出: 终端彩色文本 + 可被 cron 捕获的文本
"""

import json, os, sys, csv, re
from datetime import datetime, timezone

DATA_DIR = "/root/data"
LOG_PATH = f"{DATA_DIR}/predictions_log.csv"
LINEUPS_PATH = f"{DATA_DIR}/thestats_lineups.json"
MAPPING_PATH = f"{DATA_DIR}/team_name_mapping.json"
STAR_PATH = f"{DATA_DIR}/star_players.json"

# ── 加载中英队名映射 ──
def load_name_mapping():
    """返回 (cn2en, en2cn) 两个字典"""
    cn2en = {}
    en2cn = {}
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH) as f:
            cn2en = json.load(f)
    for cn, en in cn2en.items():
        en2cn[en.lower()] = cn
        # 也存别名
        en_short = en.replace('United States', 'USA').replace('Czech Republic', 'Czechia')
        en2cn[en_short.lower()] = cn
    return cn2en, en2cn


def build_en2cn_from_star():
    """从 star_players.json 补充队名映射"""
    en2cn = {}
    if os.path.exists(STAR_PATH):
        with open(STAR_PATH) as f:
            star = json.load(f)
        for tid, v in star.items():
            en_name = v.get('name', '')
            if en_name:
                en2cn[en_name.lower()] = en_name
    return en2cn


def strip_ranking(name):
    """去掉队名中的 FIFA 排名前缀/后缀 [N]"""
    name = re.sub(r'^\[\d+\]', '', name).strip()
    name = re.sub(r'\[\d+\]$', '', name).strip()
    return name


def cn_name_from_csv(home_cn, away_cn):
    """从 CSV 的行提取干净中文队名"""
    h = strip_ranking(home_cn.strip())
    a = strip_ranking(away_cn.strip())
    return h, a


def find_prediction_row(lineups, cn2en, en2cn):
    """
    对 lineup 中的每场比赛, 在 predictions_log.csv 找对应行
    
    返回: list of (row_dict, lineup_entry, home_en, away_en)
    """
    if not os.path.exists(LOG_PATH):
        return []
    
    matches = []
    
    with open(LOG_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 只找未完结的行
    pending_rows = [r for r in rows if not r.get('result_status','').strip()]
    
    for mid, lu in lineups.items():
        if not lu.get('confirmed'):
            continue
        
        home_en = lu.get('home_team', '')
        away_en = lu.get('away_team', '')
        if not home_en or not away_en:
            continue
        
        # 找匹配的预测行
        matched_row = None
        for r in pending_rows:
            h_cn, a_cn = cn_name_from_csv(r.get('home_cn',''), r.get('away_cn',''))
            
            # 方法1: 中文队名 → 英文, 对比 lineup 队名
            h_en_from_cn = cn2en.get(h_cn, '').lower()
            a_en_from_cn = cn2en.get(a_cn, '').lower()
            
            if h_en_from_cn == home_en.lower() and a_en_from_cn == away_en.lower():
                matched_row = r
                break
            
            # 方法2: 英文队名 → 中文, 对比 CSV 中文队名
            h_cn_from_en = en2cn.get(home_en.lower(), '')
            a_cn_from_en = en2cn.get(away_en.lower(), '')
            if h_cn_from_en and a_cn_from_en:
                if h_cn_from_en == h_cn or a_cn_from_en == a_cn:
                    matched_row = r
                    break
                # 部分匹配 (考虑别名)
                if (h_cn in h_cn_from_en or h_cn_from_en in h_cn) and \
                   (a_cn in a_cn_from_en or a_cn_from_en in a_cn):
                    matched_row = r
                    break
            
            # 方法3: 从 match_key 提取队名对比 (无排名前缀)
            mk = r.get('match_key', '')
            parts = mk.split('|')
            if len(parts) >= 4:
                mk_home = parts[2].strip()
                mk_away = parts[3].strip()
                if mk_home == h_cn and mk_away == a_cn:
                    # 已从 h_cn/a_cn 查过一次, 再给一次机会
                    pass
        
        if matched_row:
            matches.append((matched_row, lu, home_en, away_en))
        else:
            print(f"  ⏭️  {home_en:25s} vs {away_en:25s} — 未找到匹配预测行")
    
    return matches


def apply_and_compare(row, home_en, away_en):
    """
    对预测行应用 lineup 调整, 比较前后变化
    
    返回: (changed, details_dict)
    """
    # 提取原始概率 (CSV 存的是百分比 0~100)
    try:
        orig_h = float(row.get('pred_h', 0) or 0) / 100.0
        orig_d = float(row.get('pred_d', 0) or 0) / 100.0
        orig_a = float(row.get('pred_a', 0) or 0) / 100.0
    except (ValueError, TypeError):
        return False, None
    
    orig_probs = {'H': orig_h, 'D': orig_d, 'A': orig_a}
    
    # 原始 EV
    try:
        ev_h = float(row.get('ev_h', 0) or 0)
        ev_d = float(row.get('ev_d', 0) or 0)
        ev_a = float(row.get('ev_a', 0) or 0)
    except (ValueError, TypeError):
        ev_h = ev_d = ev_a = 0
    
    orig_best = max(orig_probs.keys(), key=lambda k: orig_probs[k])
    orig_has_positive_ev = max(ev_h, ev_d, ev_a) > 0
    
    # 调用 lineup 调整
    sys.path.insert(0, '/root/wc_2026_upgrade')
    try:
        from thestats_features import adjust_with_lineups, compute_rotation_penalty, _load_star_data
        star_data = _load_star_data()
        adj_probs, adj_log = adjust_with_lineups(
            orig_probs, home_en, away_en
        )
    except Exception as e:
        return False, {'error': str(e)}
    
    if not adj_log:
        return False, {'no_adjust': True}
    
    # 提取调整信息
    h_penalty = max(a.get('home_rotation', 0) for a in adj_log)
    a_penalty = max(a.get('away_rotation', 0) for a in adj_log)
    h_missing = []
    a_missing = []
    for a in adj_log:
        h_missing.extend(a.get('stars_missing_home', []))
        a_missing.extend(a.get('stars_missing_away', []))
    
    # 新方向
    new_best = max(adj_probs.keys(), key=lambda k: adj_probs[k])
    
    # 新旧概率对比
    direction_changed = (new_best != orig_best)
    prob_changed = any(abs(adj_probs[k] - orig_probs[k]) > 0.02 for k in ['H','D','A'])
    
    if not direction_changed and not prob_changed:
        return False, {
            'minor': True,
            'h_penalty': h_penalty,
            'a_penalty': a_penalty,
            'h_missing': h_missing,
            'a_missing': a_missing,
            'orig': orig_probs,
            'adj': adj_probs,
        }
    
    # 分析 EV 变化 (需要赔率)
    odds_h = float(row.get('odds_h', 0) or 0)
    odds_d = float(row.get('odds_d', 0) or 0)
    odds_a = float(row.get('odds_a', 0) or 0)
    
    new_ev_h = round(adj_probs['H'] * odds_h - 1, 4) if odds_h else 0
    new_ev_d = round(adj_probs['D'] * odds_d - 1, 4) if odds_d else 0
    new_ev_a = round(adj_probs['A'] * odds_a - 1, 4) if odds_a else 0
    
    ev_flipped = False
    old_best_ev = max(ev_h, ev_d, ev_a)
    new_best_ev = max(new_ev_h, new_ev_d, new_ev_a)
    if old_best_ev > 0 and new_best_ev <= 0:
        ev_flipped = True
    if old_best_ev <= 0 and new_best_ev > 0:
        ev_flipped = True  # EV 从负变正也是重要信号
    
    # 方向映射
    dir_map = {'H': '主胜', 'D': '平局', 'A': '客胜'}
    
    result = {
        'direction_changed': direction_changed,
        'ev_flipped': ev_flipped,
        'h_penalty': h_penalty,
        'a_penalty': a_penalty,
        'h_missing': list(dict.fromkeys(h_missing))[:5],
        'a_missing': list(dict.fromkeys(a_missing))[:5],
        'orig_best': dir_map.get(orig_best, orig_best),
        'new_best': dir_map.get(new_best, new_best),
        'orig_probs': {k: round(v*100, 1) for k,v in orig_probs.items()},
        'new_probs': {k: round(v*100, 1) for k,v in adj_probs.items()},
        'old_ev': {'H': ev_h, 'D': ev_d, 'A': ev_a},
        'new_ev': {'H': new_ev_h, 'D': new_ev_d, 'A': new_ev_a},
    }
    
    return (direction_changed or ev_flipped), result


def main():
    print(f"\n{'='*65}")
    print(f"  ⚽ 赛前重推: 首发阵容调幅校验")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*65}")
    
    # 加载数据
    lineups = {}
    if os.path.exists(LINEUPS_PATH):
        with open(LINEUPS_PATH) as f:
            lineups = json.load(f)
    
    if not lineups:
        print("  ℹ️  无 lineup 数据, 跳过")
        return
    
    print(f"  阵容缓存: {len(lineups)} 场")
    
    cn2en, en2cn_base = load_name_mapping()
    en2cn_star = build_en2cn_from_star()
    # 合并 en2cn, star 为主
    en2cn = {**en2cn_base, **en2cn_star}
    
    # 找匹配
    matches = find_prediction_row(lineups, cn2en, en2cn)
    
    if not matches:
        print("  ℹ️  未找到匹配的未完结预测")
        return
    
    print(f"  匹配预测: {len(matches)} 场\n")
    
    changes = 0
    for row, lu, home_en, away_en in matches:
        # 获取开赛时间 (从 match_key)
        mk = row.get('match_key', '')
        kickoff = mk.split('|')[-1] if mk else '?'
        
        print(f"  📋 {home_en:25s} vs {away_en:25s}  (开赛 {kickoff})")
        
        changed, detail = apply_and_compare(row, home_en, away_en)
        
        if detail is None:
            print(f"     ❌ 无法计算\n")
            continue
        
        if detail.get('error'):
            print(f"     ❌ 错误: {detail['error']}\n")
            continue
        
        if detail.get('no_adjust'):
            print(f"     ℹ️  lineup 可用但无轮换信号\n")
            continue
        
        if detail.get('minor'):
            # 微调 (< 2%) — 仅当 penalty>0 时蓝色提示
            hp = detail.get('h_penalty', 0)
            ap = detail.get('a_penalty', 0)
            hm = detail.get('h_missing', [])
            am = detail.get('a_missing', [])
            if hp > 0 or ap > 0:
                print(f"     📊 轮换影响微弱 (<2%), 不改变推荐方向")
                if hm:
                    print(f"       🏠 {home_en} 缺阵: {', '.join(hm[:3])} (惩罚 {hp:.1%})")
                if am:
                    print(f"       🚌 {away_en} 缺阵: {', '.join(am[:3])} (惩罚 {ap:.1%})")
            else:
                print(f"     ℹ️  无显著轮换\n")
            continue
        
        # ⚠️ 重要变化
        changes += 1
        hp = detail['h_penalty']
        ap = detail['a_penalty']
        
        # 高亮警告
        warn_parts = [f"  {'⚠️' if changed else '📊'} [赛前急报]"]
        warn_parts.append(f"{home_en} vs {away_en}")
        
        if detail['direction_changed']:
            warn_parts.append(f"推荐方向变更: {detail['orig_best']} → {detail['new_best']}")
        
        if detail['ev_flipped']:
            warn_parts.append(f"EV 翻转: {detail['old_ev']} → {detail['new_ev']}")
        
        missing_parts = []
        if detail['h_missing']:
            missing_parts.append(f"{home_en} 缺阵 {', '.join(detail['h_missing'][:3])} (惩罚 {hp:.1%})")
        if detail['a_missing']:
            missing_parts.append(f"{away_en} 缺阵 {', '.join(detail['a_missing'][:3])} (惩罚 {ap:.1%})")
        
        # 概率变化
        old_p = detail['orig_probs']
        new_p = detail['new_probs']
        prob_line = f"      概率: H {old_p['H']:.0f}→{new_p['H']:.0f}% | D {old_p['D']:.0f}→{new_p['D']:.0f}% | A {old_p['A']:.0f}→{new_p['A']:.0f}%"
        
        ev_line = f"      EV:   H {detail['old_ev']['H']:.2f}→{detail['new_ev']['H']:.2f}  D {detail['old_ev']['D']:.2f}→{detail['new_ev']['D']:.2f}  A {detail['old_ev']['A']:.2f}→{detail['new_ev']['A']:.2f}"
        
        if detail.get('direction_changed') or detail.get('ev_flipped'):
            print(f"  {'⚠️' * 3} {'='*55}")
            print(f"  ⚠️  [赛前急报] {home_en} vs {away_en}")
            print(f"  ⚠️  开赛 {kickoff}")
            if detail['h_missing']:
                print(f"  ⚠️  🏠 {home_en} 核心缺阵: {', '.join(detail['h_missing'][:5])}")
            if detail['a_missing']:
                print(f"  ⚠️  🚌 {away_en} 核心缺阵: {', '.join(detail['a_missing'][:5])}")
            if detail['direction_changed']:
                print(f"  ⚠️  推荐方向变更: {detail['orig_best']} → {detail['new_best']}")
            if detail['ev_flipped']:
                print(f"  ⚠️  EV 翻转: 旧最佳 {detail['old_ev']:.2f} → 新最佳 {detail['new_ev']:.2f}")
            print(f"  ⚠️  {prob_line}")
            print(f"  ⚠️  {ev_line}")
            print(f"  {'⚠️' * 3} {'='*55}")
        else:
            print(f"     📊 概率微调 (未改变推荐)")
            if detail['h_missing']:
                print(f"       🏠 缺阵: {', '.join(detail['h_missing'][:3])} (惩罚 {hp:.1%})")
            if detail['a_missing']:
                print(f"       🚌 缺阵: {', '.join(detail['a_missing'][:3])} (惩罚 {ap:.1%})")
            print(f"       {prob_line}")
            print(f"       {ev_line}")
        
        print()
    
    if changes == 0:
        print("  📊 所有 lineup 已评估, 无方向性变化")


if __name__ == '__main__':
    main()
