#!/usr/bin/env python3
"""今日完整预测: 5 玩法 (HDA + 让球 + 半全场 + 比分 + 总进球数)

数据源: 500.com 今日赛程 + 500.com 实时赔率 (SPF + 让球 + 半全场)
模型: predict_match (P1+P2+P3+S1+P4 已就位) + 半全场 sidecar

用法:
    python3 /root/predict_today.py
    python3 /root/predict_today.py --json   # 原始 JSON
"""
import sys, os, json, argparse
sys.path.insert(0, '/root')
import numpy as np
from fetch_365scores import fetch_365scores_data, extract_games, parse_trend
from daily_jczq import scrape_500_odds_today
from team_name_normalizer import normalize_team_name
from predict_match import predict_match, mc_score_dist, _build_total_goals_recommendation, _load_form_state

# Manual CN→EN fallback (for known unmapped names from 500.com)
MANUAL_EN = {
    '斯洛文尼': 'Slovenia',
    '科特迪瓦': 'Ivory Coast',
    '斯洛文尼亚': 'Slovenia',
    '中国': 'China PR',
    '中国香港': 'Hong Kong',
    '美国': 'United States',
    '韩国': 'South Korea',
    '朝鲜': 'North Korea',
    '沙特': 'Saudi Arabia',
    '阿联酋': 'United Arab Emirates',
    '新西兰': 'New Zealand',
    '特立尼达和多巴哥': 'Trinidad and Tobago',
    '哥斯达黎加': 'Costa Rica',
    '洪都拉斯': 'Honduras',
    '危地马拉': 'Guatemala',
    '萨尔瓦多': 'El Salvador',
    '尼加拉瓜': 'Nicaragua',
    '巴拿马': 'Panama',
    '牙买加': 'Jamaica',
    '海地': 'Haiti',
    '多米尼加': 'Dominican Republic',
    '波多黎各': 'Puerto Rico',
    '古巴': 'Cuba',
    '冰岛': 'Iceland',
    '北爱尔兰': 'Northern Ireland',
    '苏格兰': 'Scotland',
    '威尔士': 'Wales',
    '英格兰': 'England',
    '捷克': 'Czech Republic',
    '斯洛伐克': 'Slovakia',
    '波黑': 'Bosnia and Herzegovina',
    '黑山': 'Montenegro',
    '北马其顿': 'North Macedonia',
}


def safe_normalize(cn_name):
    en = normalize_team_name(cn_name)
    if en and all(ord(c) < 128 for c in en.replace(' ', '')):
        return en
    if cn_name in MANUAL_EN:
        return MANUAL_EN[cn_name]
    return None


# ── 500.com 总进球 (playid=270) 抓取 ──
def scrape_500_total_goals():
    """从 500.com playid=270 抓取今日总进球赔率 (8 选项 0-7+)
    Returns: {code: [odds_0, odds_1, ..., odds_7p]}  or {} on failure
    """
    import subprocess, re
    try:
        result = subprocess.run(
            ["curl", "-sL", "--max-time", "10",
             "https://trade.500.com/jczq/?playid=270&g=2",
             "-H", "User-Agent: Mozilla/5.0"],
            capture_output=True, timeout=15
        )
        html = result.stdout.decode("gbk", errors="ignore")
    except Exception:
        return {}

    result = {}
    matches = re.findall(r'<tr[^>]*>.*?周[一二三四五六日]\d{3}.*?</tr>', html, re.DOTALL)
    for tr in matches:
        parts = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.DOTALL)
        if len(parts) < 5:
            continue
        # 找 code td (不固定位置)
        code_m = None
        odds_idx = None
        for i, p in enumerate(parts):
            cm = re.search(r'周[一二三四五六日]\d{3}', p)
            if cm:
                code_m = cm
                # 赔率 td 在 code 后 3 位
                if i + 4 < len(parts):
                    odds_idx = i + 4
                break
        if not code_m or odds_idx is None:
            continue
        code = code_m.group(0)
        odds_text = re.sub(r'<[^>]+>', ' ', parts[odds_idx])
        odds_text = re.sub(r'\s+', ' ', odds_text).strip()
        odds = re.findall(r'(\d+\.\d+)', odds_text)
        if len(odds) >= 8:
            result[code] = [float(o) for o in odds[:8]]  # [0球, 1球, 2球, 3球, 4球, 5球, 6球, 7+球]
    return result


# ── 5 玩法预测函数 ──

def predict_hda(p):
    """1. 90 分钟胜平负 (HDA)"""
    h, d, a = p['fin_h'], p['fin_d'], p['fin_a']
    return {
        'H': h, 'D': d, 'A': a,
        'pick': max([('H', h), ('D', d), ('A', a)], key=lambda x: x[1])[0]
    }


def predict_handicap(p, handicap):
    """2. 竞彩让球 (-1, -2 等): 推算让球后 HDA 概率

    让球 (对主队而言) = handicap. 让球后主队进球 - 让球数 vs 客队进球.
    P(H_q) = P(home - handicap > away) = sum over h,a of P(h-a > handicap)
    """
    lam_h, lam_a = p['lam_h'], p['lam_a']
    from scipy.stats import poisson as _poisson
    probs_H = probs_D = probs_A = 0.0
    for h in range(8):
        for a in range(8):
            p_score = _poisson.pmf(h, lam_h) * _poisson.pmf(a, lam_a)
            adj_h = h - handicap  # 让球后主队等效进球
            if adj_h > a: probs_H += p_score
            elif adj_h == a: probs_D += p_score
            else: probs_A += p_score
    pick = max([('H', probs_H), ('D', probs_D), ('A', probs_A)], key=lambda x: x[1])[0]
    return {
        'handicap': handicap,
        'H': round(probs_H * 100, 1),
        'D': round(probs_D * 100, 1),
        'A': round(probs_A * 100, 1),
        'pick': pick
    }


def predict_half_full(p, max_goals_ht=4, max_goals_ft=7):
    """3. 半全场 (HT/FT) 9 组合

    半全场 = 上半场比分 + 全场比分. 假设 HT 进球率约 45% (经验值).
    """
    lam_h, lam_a = p['lam_h'], p['lam_a']
    # 半场 λ 缩放
    HALF_RATIO = 0.45
    lam_h_ht = lam_h * HALF_RATIO
    lam_a_ht = lam_a * HALF_RATIO
    # 半场剩余 λ
    lam_h_2h = lam_h - lam_h_ht
    lam_a_2h = lam_a - lam_a_ht

    from scipy.stats import poisson as _poisson
    # 9 个 (HT, FT) 结果: 先算 HT 比分, 再算 FT 比分
    # FT 比分 = HT 比分 + 下半场比分
    # P(HT=score1, FT=score2) = P(HT=score1) * P(下半场比分=score2-score1)
    # 半全场标签: H/H, H/D, H/A, D/H, D/D, D/A, A/H, A/D, A/A
    # 但更简单: P(半全场 = (X, Y)) = P(半场胜平负=X) * P(全场胜平负=Y | 半场=X)
    # 这里用直接枚举方式

    hf_probs = {'HH': 0, 'HD': 0, 'HA': 0, 'DH': 0, 'DD': 0, 'DA': 0, 'AH': 0, 'AD': 0, 'AA': 0}

    for h1 in range(max_goals_ht + 1):
        for a1 in range(max_goals_ht + 1):
            p_ht_score = _poisson.pmf(h1, lam_h_ht) * _poisson.pmf(a1, lam_a_ht)
            if p_ht_score < 1e-8:
                continue
            # 半场结果
            if h1 > a1: ht = 'H'
            elif h1 == a1: ht = 'D'
            else: ht = 'A'
            # 下半场比分 (h2, a2) 加到 (h1, a1) = FT 比分
            for h2 in range(max_goals_ft + 1):
                for a2 in range(max_goals_ft + 1):
                    p_2h_score = _poisson.pmf(h2, lam_h_2h) * _poisson.pmf(a2, lam_a_2h)
                    p_total = p_ht_score * p_2h_score
                    if p_total < 1e-10:
                        continue
                    h_ft = h1 + h2
                    a_ft = a1 + a2
                    if h_ft > a_ft: ft = 'H'
                    elif h_ft == a_ft: ft = 'D'
                    else: ft = 'A'
                    key = ht + ft
                    hf_probs[key] += p_total

    # 归一化
    total = sum(hf_probs.values())
    if total > 0:
        hf_probs = {k: round(v / total * 100, 1) for k, v in hf_probs.items()}

    sorted_hf = sorted(hf_probs.items(), key=lambda x: -x[1])
    return {
        'matrix': hf_probs,
        'top3': sorted_hf[:3],
        'pick': sorted_hf[0][0]
    }


def predict_score(p, top_n=6):
    """4. 比分 (Top N)"""
    scores = mc_score_dist(p['lam_h'], p['lam_a'], n=100000)
    return {
        'top': scores[:top_n],
        'most_likely': scores[0][0] if scores else None
    }


def predict_total_goals(p, form_gap=False, market_odds=None):
    """5. 总进球数 (大/小 2.5 + 完整分布 + 投哪个数字 + 赔率 EV)

    market_odds: [0球赔率, 1球赔率, ..., 7+球赔率] 共 8 个 (from 500.com)
    """
    tg = _build_total_goals_recommendation(p['lam_h'], p['lam_a'], p['match_type'], form_gap)
    from scipy.stats import poisson as _poisson
    lam_total = p['lam_h'] + p['lam_a']
    dist = {}
    for g in range(8):
        pmf = _poisson.pmf(g, lam_total)
        dist[f"{g}球"] = round(pmf * 100, 1)
    dist["7+球"] = round(sum(_poisson.pmf(g, lam_total) for g in range(8, 15)) * 100, 1)

    pick_num = max([(k, v) for k, v in dist.items() if k != '7+球'], key=lambda x: x[1])[0]
    pick_num_prob = dist[pick_num]

    result = {
        **tg,
        'p_over_2_5_pct': tg['p_over_2_5_pct'],
        'p_under_2_5_pct': tg['p_under_2_5_pct'],
        'lam_total': tg['lam_total'],
        'distribution': dist,
        'pick': tg['pick'],
        'pick_num': pick_num,
        'pick_num_prob': pick_num_prob,
    }

    # 赔率 EV 计算 (如果有 market_odds)
    if market_odds and len(market_odds) >= 8:
        ev_list = []
        for i, (label, prob) in enumerate(dist.items()):
            if i < len(market_odds):
                odds = market_odds[i]
                prob_dec = prob / 100
                ev = prob_dec * odds - 1  # Expected Value (净收益率)
                # Kelly 简化: 用 EV 标记正/负
                ev_list.append({
                    'label': label,
                    'prob_pct': prob,
                    'odds': odds,
                    'ev': round(ev * 100, 1),  # 转 % 形式
                    'fair_odds': round(1 / prob_dec, 2) if prob_dec > 0 else 999,
                })
        # 找最大 EV (正 EV 才"值得投")
        positive_ev = [e for e in ev_list if e['ev'] > 0]
        if positive_ev:
            best_ev = max(positive_ev, key=lambda x: x['ev'])
        else:
            best_ev = max(ev_list, key=lambda x: x['ev'])  # 退而求其次: 负 EV 中最大
        result['ev_list'] = ev_list
        result['best_ev'] = best_ev
        result['has_market'] = True
    else:
        result['ev_list'] = []
        result['has_market'] = False

    return result


def get_form_gap(home_en, away_en):
    fs = _load_form_state()
    home_has = home_en in fs and len(fs[home_en]) >= 1
    away_has = away_en in fs and len(fs[away_en]) >= 1
    return (not home_has) or (not away_has)


# ── 主流程 ──


# ── 投票数据融合函数 ──

def fuse_vote_data(model_probs, vote_data, max_alpha=0.3, scale=100000):
    """
    将模型概率与投票数据融合。
    
    Args:
        model_probs: 模型输出概率 [home, draw, away] (百分比形式，如 [67.3, 17.9, 14.8])
        vote_data: 投票数据 {'home': float, 'draw': float, 'away': float, 'total': int}
        max_alpha: 最大融合权重 (默认 0.3)
        scale: 投票人数缩放因子 (默认 100000)
    
    Returns:
        fused_probs: 融合后的概率 [home, draw, away] (百分比形式)
        alpha: 实际融合权重
    """
    if not vote_data or vote_data.get('total', 0) == 0:
        return model_probs, 0.0
    
    # 将模型概率转换为小数形式
    model_probs_decimal = np.array(model_probs) / 100
    
    # 提取投票概率 (已经是小数形式)
    vote_probs = np.array([
        vote_data['home'] / 100,
        vote_data['draw'] / 100,
        vote_data['away'] / 100
    ])
    
    # 计算融合权重 (投票人数越多，权重越高)
    vote_count = vote_data['total']
    alpha = min(max_alpha, vote_count / scale)
    
    # 融合
    fused_probs_decimal = (1 - alpha) * model_probs_decimal + alpha * vote_probs
    
    # 归一化 (确保概率和为 1)
    fused_probs_decimal = fused_probs_decimal / fused_probs_decimal.sum()
    
    # 转换回百分比形式
    fused_probs = fused_probs_decimal * 100
    
    return fused_probs, alpha

def predict_one(m5, total_odds_map=None, scores365_data=None):
    """预测单场 5 玩法"""
    home_cn = m5['home_cn']
    away_cn = m5['away_cn']
    home_en = safe_normalize(home_cn) or home_cn
    away_en = safe_normalize(away_cn) or away_cn
    handicap = m5.get('handicap', 0)
    code = m5.get('code', '')

    p = predict_match(home_en, away_en, match_type='friendly')
    if not isinstance(p, dict):
        return None

    form_gap = get_form_gap(home_en, away_en)
    market_odds = total_odds_map.get(code) if total_odds_map else None

    # 365scores 数据匹配
    scores365_match = None
    if scores365_data:
        # 改进的模糊匹配算法
        home_lower = home_en.lower()
        away_lower = away_en.lower()
        home_words = set(word for word in home_lower.split() if len(word) > 2)
        away_words = set(word for word in away_lower.split() if len(word) > 2)
        
        best_score = 0
        best_match = None
        
        for g in scores365_data:
            g_home = g['home'].lower()
            g_away = g['away'].lower()
            
            # 计算匹配分数
            score = 0
            
            # 1. 完全匹配
            if home_lower == g_home and away_lower == g_away:
                score = 100
            # 2. 包含匹配
            elif home_lower in g_home or g_home in home_lower:
                score += 50
                if away_lower in g_away or g_away in away_lower:
                    score += 50
            # 3. 关键词匹配
            else:
                # 主队关键词匹配
                home_matches = sum(1 for word in home_words if word in g_home)
                if home_matches > 0:
                    score += home_matches * 20
                
                # 客队关键词匹配
                away_matches = sum(1 for word in away_words if word in g_away)
                if away_matches > 0:
                    score += away_matches * 20
            
            # 更新最佳匹配
            if score > best_score:
                best_score = score
                best_match = g
        
        # 只有分数足够高才认为匹配成功
        if best_score >= 40:
            scores365_match = best_match

    # 投票数据融合
    vote_data = scores365_match.get('votes') if scores365_match else None
    if vote_data:
        # 融合投票数据到 HDA 概率
        hda_probs = [p['fin_h'], p['fin_d'], p['fin_a']]
        fused_probs, alpha = fuse_vote_data(hda_probs, vote_data)
        p['fin_h'] = fused_probs[0]
        p['fin_d'] = fused_probs[1]
        p['fin_a'] = fused_probs[2]
        # 记录融合权重
        p['vote_fusion_alpha'] = alpha
    else:
        p['vote_fusion_alpha'] = 0.0
    

    # 人气排名融合
    pop_rank_home = scores365_match.get('pop_rank_home') if scores365_match else None
    pop_rank_away = scores365_match.get('pop_rank_away') if scores365_match else None
    
    if pop_rank_home and pop_rank_away:
        # 计算人气排名差异 (正值：主队人气更高，负值：客队人气更高)
        pop_rank_diff = pop_rank_away - pop_rank_home
        
        # 将人气排名差异作为额外信息存储
        p['pop_rank_diff'] = pop_rank_diff
        p['pop_rank_home'] = pop_rank_home
        p['pop_rank_away'] = pop_rank_away
    else:
        p['pop_rank_diff'] = 0.0
        p['pop_rank_home'] = None
        p['pop_rank_away'] = None

    # 趋势数据融合
    trend_home = scores365_match.get('trend_home', []) if scores365_match else []
    trend_away = scores365_match.get('trend_away', []) if scores365_match else []
    
    if trend_home and trend_away and len(trend_home) >= 3 and len(trend_away) >= 3:
        # 计算胜率
        wins_home = trend_home[0]
        draws_home = trend_home[1]
        losses_home = trend_home[2]
        total_home = wins_home + draws_home + losses_home
        
        wins_away = trend_away[0]
        draws_away = trend_away[1]
        losses_away = trend_away[2]
        total_away = wins_away + draws_away + losses_away
        
        if total_home > 0 and total_away > 0:
            win_rate_home = wins_home / total_home
            win_rate_away = wins_away / total_away
            
            # 计算胜率差异
            win_rate_diff = win_rate_home - win_rate_away
            
            # 存储趋势数据
            p['trend_win_rate_diff'] = win_rate_diff
            p['trend_win_rate_home'] = win_rate_home
            p['trend_win_rate_away'] = win_rate_away
            p['trend_home'] = trend_home
            p['trend_away'] = trend_away
        else:
            p['trend_win_rate_diff'] = 0.0
            p['trend_win_rate_home'] = 0.0
            p['trend_win_rate_away'] = 0.0
            p['trend_home'] = []
            p['trend_away'] = []
    else:
        p['trend_win_rate_diff'] = 0.0
        p['trend_win_rate_home'] = 0.0
        p['trend_win_rate_away'] = 0.0
        p['trend_home'] = []
        p['trend_away'] = []
    hda = predict_hda(p)
    rq = predict_handicap(p, handicap) if handicap != 0 else None
    hf = predict_half_full(p)
    sc = predict_score(p)
    tg = predict_total_goals(p, form_gap=form_gap, market_odds=market_odds)

    return {
        'code': m5['code'],
        'time': m5['time'],
        'status': m5.get('status', 'unknown'),
        'home_cn': home_cn,
        'away_cn': away_cn,
        'home_en': home_en,
        'away_en': away_en,
        'model': {
            'lam_h': p['lam_h'],
            'lam_a': p['lam_a'],
            'fin_h': p['fin_h'],
            'fin_d': p['fin_d'],
            'fin_a': p['fin_a'],
            'elo_h': p['elo_h'],
            'elo_a': p['elo_a'],
            'vote_fusion_alpha': p.get('vote_fusion_alpha', 0.0),
            'pop_rank_diff': p.get('pop_rank_diff', 0.0),
            'pop_rank_home': p.get('pop_rank_home'),
            'pop_rank_away': p.get('pop_rank_away'),
            'trend_win_rate_diff': p.get('trend_win_rate_diff', 0.0),
            'trend_win_rate_home': p.get('trend_win_rate_home', 0.0),
            'trend_win_rate_away': p.get('trend_win_rate_away', 0.0),
            'trend_home': p.get('trend_home', []),
            'trend_away': p.get('trend_away', []),
        },
        'hda': hda,
        'handicap': rq,
        'half_full': hf,
        'score': sc,
        'total_goals': tg,
        'bet_hda': p['bet_recommendation'],
        'scores365': scores365_match,
    }


def format_one_prediction(r):
    """格式单场完整预测 (5 玩法)"""
    if r is None:
        return "❌ 模型未收敛"
    lines = []
    m = r['model']
    lines.append(f"⚽ {r['code']}  {r['time']}")
    lines.append(f"   {r['home_cn']:12s}  vs  {r['away_cn']:12s}")
    lines.append(f"   EN: {r['home_en']:18s} vs {r['away_en']:18s}")
    lines.append(f"   Elo: {m['elo_h']:.0f} vs {m['elo_a']:.0f} (差 {m['elo_h']-m['elo_a']:+.0f})")
    
    # 365scores 数据
    if r.get('scores365'):
        s = r['scores365']
        lines.append(f"   📊 365scores: {s['competition']}")
        if s.get('votes'):
            v = s['votes']
            lines.append(f"   投票: 主{v['home']}% / 平{v['draw']}% / 客{v['away']}% ({v['total']}人)")
        lines.append(f"   主队趋势: {s['trend_home_desc']}")
        lines.append(f"   客队趋势: {s['trend_away_desc']}")
        
        # 融合权重
        alpha = r.get('model', {}).get('vote_fusion_alpha', 0)
        if alpha > 0:
            lines.append(f"   投票融合权重: {alpha:.1%} (投票人数越多权重越高，最高30%)")
        
        # 人气排名
        pop_rank_diff = r.get('model', {}).get('pop_rank_diff', 0)
        pop_rank_home = r.get('model', {}).get('pop_rank_home')
        pop_rank_away = r.get('model', {}).get('pop_rank_away')
        if pop_rank_home and pop_rank_away:
            # 人气排名差异解释
            if pop_rank_diff > 0:
                pop_desc = f"主队人气更高 (+{pop_rank_diff})"
            elif pop_rank_diff < 0:
                pop_desc = f"客队人气更高 ({pop_rank_diff})"
            else:
                pop_desc = "人气持平"
            lines.append(f"   人气排名: {pop_rank_home} vs {pop_rank_away} ({pop_desc})")
        
        # 趋势数据
        trend_win_rate_diff = r.get('model', {}).get('trend_win_rate_diff', 0)
        trend_win_rate_home = r.get('model', {}).get('trend_win_rate_home', 0)
        trend_win_rate_away = r.get('model', {}).get('trend_win_rate_away', 0)
        trend_home = r.get('model', {}).get('trend_home', [])
        trend_away = r.get('model', {}).get('trend_away', [])
        
        if trend_home and trend_away:
            # 趋势数据解释
            if trend_win_rate_diff > 0:
                trend_desc = f"主队近期状态更好 (+{trend_win_rate_diff:.1%})"
            elif trend_win_rate_diff < 0:
                trend_desc = f"客队近期状态更好 ({trend_win_rate_diff:.1%})"
            else:
                trend_desc = "近期状态持平"
            lines.append(f"   趋势胜率: 主{trend_win_rate_home:.1%} vs 客{trend_win_rate_away:.1%} ({trend_desc})")
        
        # 额外数据
        extras = []
        if s.get('venue'):
            extras.append(f"场地: {s['venue']}")
        if s.get('attendance'):
            extras.append(f"观众: {s['attendance']}")
        if s.get('avg_age_home') and s.get('avg_age_away'):
            extras.append(f"平均年龄: {s['avg_age_home']} vs {s['avg_age_away']}")
        if extras:
            lines.append(f"   附加: {' | '.join(extras)}")
    else:
        lines.append(f"   📊 365scores: 未匹配")
    
    lines.append(f"   λ:  {m['lam_h']} vs {m['lam_a']} (总 {m['lam_h']+m['lam_a']:.2f})")
    lines.append("")

    # 1. 90分钟胜平负
    hda = r['hda']
    pick_hda = hda['pick']
    icon_pick = {'H': '🔥主', 'D': '⚖️平', 'A': '🔥客'}.get(pick_hda, pick_hda)
    lines.append(f"   【1】90分钟胜平负: 主 {hda['H']:.1f}% | 平 {hda['D']:.1f}% | 客 {hda['A']:.1f}%")
    lines.append(f"       首选: {icon_pick}")
    # 门控
    br = r['bet_hda']
    if br['action'] == 'BET':
        lines.append(f"       ✅ 门控建议: 投 HDA ({br['best_pick']}, 边际 {br['margin_pp']}pp)")
    elif br['action'] == 'SKIP':
        lines.append(f"       ⛔ 门控建议: 跳过 (边际 {br['margin_pp']}pp < 10pp 弱信号)")
    elif br['action'] == 'SKIP_DATA':
        lines.append(f"       ⚠️ 门控建议: 跳过 (form 数据缺失)")
    lines.append("")

    # 2. 让球
    if r['handicap']:
        rq = r['handicap']
        icon_pick = {'H': '🔥让主', 'D': '⚖️让平', 'A': '🔥让客'}.get(rq['pick'], rq['pick'])
        lines.append(f"   【2】竞彩让球 ({rq['handicap']}): 让主 {rq['H']:.1f}% | 让平 {rq['D']:.1f}% | 让客 {rq['A']:.1f}%")
        lines.append(f"       首选: {icon_pick}")
    else:
        lines.append(f"   【2】竞彩让球: 无 (handicap=0)")
    lines.append("")

    # 3. 半全场
    hf = r['half_full']
    lines.append(f"   【3】半全场 (HT/FT) 9 组合:")
    for label in ['HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA']:
        prob = hf['matrix'].get(label, 0)
        marker = ' ← 首选' if label == hf['pick'] else ''
        bar = '█' * int(prob)
        lines.append(f"       {label}: {prob:5.1f}%  {bar}{marker}")
    lines.append("")

    # 4. 比分
    sc = r['score']
    lines.append(f"   【4】比分概率 (Top 6):")
    for s, prob in sc['top']:
        bar = '█' * int(prob)
        lines.append(f"       {s:>5s}  {prob:5.1f}%  {bar}")
    lines.append("")

    # 5. 总进球
    tg = r['total_goals']
    icon_tg = '🔼' if tg['action'] == 'BET_OVER' else '🔽' if tg['action'] == 'BET_UNDER' else '⏭️'
    lines.append(f"   【5】总进球数 (500.com 玩法, 8 选项):")
    lines.append(f"       λ_total={tg['lam_total']}  |  大2.5={tg['p_over_2_5_pct']:.1f}%  小2.5={tg['p_under_2_5_pct']:.1f}%  | {icon_tg} {tg['action']} ({tg['pick']}, |Δ|={tg['confidence_pp']}pp)")
    # 8 选项分布
    dist_items = list(tg['distribution'].items())
    pick_num = tg.get('pick_num', '?')
    for g, pct in dist_items:
        if pct < 0.1:
            continue
        bar = '█' * int(pct)
        marker = ''
        if g == pick_num:
            marker = f' ← 投这个 ({pct:.1f}%)'
        lines.append(f"       {g:>4s}  {pct:5.1f}%  {bar}{marker}")
    # 赔率 EV (如果有)
    if tg.get('has_market') and tg.get('ev_list'):
        lines.append("       赔率反推 (model_prob × odds - 1):")
        for ev in tg['ev_list']:
            icon = '🟢' if ev['ev'] > 0 else ('⚪' if ev['ev'] > -5 else '🔴')
            marker = ''
            if tg.get('best_ev') and ev['label'] == tg['best_ev']['label'] and ev['ev'] > 0:
                marker = ' ← 最佳 EV'
            lines.append(f"         {icon} {ev['label']:>4s}  赔率={ev['odds']:5.2f}  模型={ev['prob_pct']:5.1f}%  EV={ev['ev']:+5.1f}%  公平赔率={ev['fair_odds']:.2f}{marker}")
    lines.append("")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='今日 5 玩法完整预测')
    ap.add_argument('--json', action='store_true', help='JSON 输出')
    args = ap.parse_args()

    print("📡 抓取 500.com 今日赛程...")
    m5 = scrape_500_odds_today()
    print(f"✓ 找到 {len(m5)} 场国际赛")
    print("📡 抓取 500.com 总进球赔率 (playid=270)...")
    total_odds_map = scrape_500_total_goals()
    print(f"✓ 总进球赔率: {len(total_odds_map)} 场")
    
    # 抓取 365scores 数据
    print("📡 抓取 365scores 数据 (投票+趋势)...")
    scores365_raw = fetch_365scores_data()
    scores365_games = extract_games(scores365_raw) if scores365_raw else []
    print(f"✓ 365scores: {len(scores365_games)} 场比赛")
    print("=" * 80)

    results = []
    for m in m5:
        r = predict_one(m, total_odds_map=total_odds_map, scores365_data=scores365_games)
        if r is None:
            print(f"  ⚠ {m['code']} {m['home_cn']} vs {m['away_cn']} — 模型未收敛")
            continue
        results.append(r)
        if not args.json:
            print(format_one_prediction(r))
            print('─' * 80)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    # 汇总
    if not args.json and results:
        print("\n📋 投注汇总:")
        for r in results:
            br = r['bet_hda']
            tg = r['total_goals']
            rq = r['handicap']
            decisions = []
            if br['action'] == 'BET':
                decisions.append(f"HDA投{br['best_pick']}")
            if rq:
                decisions.append(f"让球({rq['handicap']})投{rq['pick']}")
            if tg['action'] in ('BET_OVER', 'BET_UNDER'):
                decisions.append(f"总进球投{tg['pick']}")
            hf_top = r['half_full']['top3'][0] if r['half_full']['top3'] else None
            if hf_top:
                decisions.append(f"半全场{hf_top[0]}首选")
            sc_top = r['score']['most_likely']
            if sc_top:
                decisions.append(f"最可能比分{sc_top}")
            print(f"  {r['code']} {r['home_cn']} vs {r['away_cn']}: {' | '.join(decisions) if decisions else '(全部弱信号, 不投)'}")


if __name__ == '__main__':
    main()
