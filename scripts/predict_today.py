"""
6/7 综合预测 — 4 场友谊赛 + 3 场 K2 联赛
数据源:
  1. 500.com zqdc 抓到的赛程 + 让球
  2. 365scores 投票/势态/人气
  3. 简化 Elo (从近期表现推断)
"""
import pandas as pd
import json
import math

# 6/7 可买 jczq 主流场次
matches = [
    # 编号, 联赛, 时间, 主队, 让球, 客队
    {"mid": "周日041", "league": "友谊赛", "time": "06-08 02:45", "home": "克罗地亚", "rq": -1, "away": "斯洛文尼亚",
     "odds_market": {"胜": 1.50, "平": 4.00, "负": 6.00},  # 估计 500.com 主胜赔率
     "vote": {"H": 79.0, "D": 13.0, "A": 8.0, "n": 3656},  # 365scores 投票
     "form": {"H": "W0 D0 L1", "A": "W2 D1 L0"},  # 主近况 0胜0平1负, 客2胜1平
     "pop_rank": {"H": 7894, "A": 1818}},
    {"mid": "周日042", "league": "友谊赛", "time": "06-08 03:00", "home": "摩洛哥", "rq": 0, "away": "挪威",
     "odds_market": {"胜": 2.20, "平": 3.30, "负": 3.10},
     "vote": {"H": 63.2, "D": 10.0, "A": 26.8, "n": 31191},
     "form": {"H": "W1 D1 L1", "A": "W1 D2 L0"},
     "pop_rank": {"H": 9635, "A": 2330}},
    {"mid": "周日043", "league": "友谊赛", "time": "06-08 03:00", "home": "希腊", "rq": 0, "away": "意大利",
     "odds_market": {"胜": 4.50, "平": 3.40, "负": 1.75},  # 意大利大热
     "vote": {"H": 18.5, "D": 12.2, "A": 69.4, "n": 5688},
     "form": {"H": "W2 D2 L0", "A": "W1 D4 L1"},
     "pop_rank": {"H": 2429, "A": 13828}},
    {"mid": "周日046", "league": "友谊赛", "time": "06-08 07:00", "home": "哥伦比亚", "rq": -2, "away": "约旦",
     "odds_market": {"胜": 1.20, "平": 6.00, "负": 12.00},  # 哥伦比亚大热
     "vote": {"H": 43.4, "D": 7.3, "A": 49.3, "n": 25996},  # 公众略看约旦 (反!)
     "form": {"H": "W1 D0 L0", "A": "W0 D2 L2"},
     "pop_rank": {"H": 16747, "A": 3013}},
]

# 简化综合模型: 0.5 市场赔率反推 + 0.3 365scores 投票 + 0.2 势态调整
def odds_to_prob(odds):
    """赔率反推概率 (去除 margin)"""
    inv = {k: 1/v for k, v in odds.items()}
    total = sum(inv.values())
    return {k: v/total for k, v in inv.items()}

def vote_to_prob(vote):
    """365scores 投票归一化"""
    total = vote["H"] + vote["D"] + vote["A"]
    return {"胜": vote["H"]/total, "平": vote["D"]/total, "负": vote["A"]/total}

def form_score(form_str):
    """势态转分数: W=3, D=1, L=0"""
    parts = form_str.split()
    w = int(parts[0].replace("W", ""))
    d = int(parts[1].replace("D", ""))
    l = int(parts[2].replace("L", ""))
    total = w + d + l
    if total == 0:
        return 0.5
    return (3*w + 1*d) / (3*total)

def form_to_prob(form_home, form_away):
    """势态反推胜平负概率"""
    s_h = form_score(form_home)
    s_a = form_score(form_away)
    diff = s_h - s_a  # -1 ~ 1
    # 简单 sigmoid 转换
    p_h = 0.4 + 0.3 * diff
    p_a = 0.4 - 0.3 * diff
    p_d = 1 - p_h - p_a
    p_d = max(0.15, min(0.30, p_d))  # 平局限制 15-30%
    # 归一化
    total = p_h + p_d + p_a
    return {"胜": p_h/total, "平": p_d/total, "负": p_a/total}

def handicap_outcome(p_spf, rq):
    """根据 SPF 概率和让球数, 推算让球胜平负概率 (简化)"""
    # rq < 0 = 主让 (主队让客队 N 球)
    # 让球后胜平负 = 调整后的胜平负
    if rq < 0:
        # 主让 |rq| 球, 实际差距 = (主进球 - 客进球) + |rq|
        # 主胜概率提升, 平局降低
        boost = abs(rq) * 0.15
        p_rq_win = min(0.85, p_spf["胜"] + boost)
        p_rq_draw = max(0.10, p_spf["平"] - boost*0.5)
        p_rq_loss = 1 - p_rq_win - p_rq_draw
    else:
        # 客让 |rq| 球, 客胜概率提升
        boost = abs(rq) * 0.15
        p_rq_loss = min(0.85, p_spf["胜"] + boost)  # 原主胜现在成"让负"
        p_rq_draw = max(0.10, p_spf["平"] - boost*0.5)
        p_rq_win = 1 - p_rq_loss - p_rq_draw
    return {"让胜": p_rq_win, "让平": p_rq_draw, "让负": p_rq_loss}

# 比分概率 (基于 Poisson, 简化)
def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def predict_score_dist(p_spf, p_rq, lambda_home, lambda_away):
    """预测比分分布 Top 8"""
    from functools import lru_cache
    scores = []
    for h in range(6):
        for a in range(6):
            p = poisson_pmf(h, lambda_home) * poisson_pmf(a, lambda_away)
            scores.append((f"{h}:{a}", p))
    scores.sort(key=lambda x: -x[1])
    total = sum(p for _, p in scores)
    return [(s, p/total) for s, p in scores[:8]]

# 总进球 (基于 λ 期望)
def predict_total_goals(lambda_home, lambda_away):
    """总进球分布 0-7+"""
    dist = {}
    for total in range(8):
        p = 0
        for h in range(total+1):
            a = total - h
            if a < 0:
                continue
            p += poisson_pmf(h, lambda_home) * poisson_pmf(a, lambda_away)
        dist[total] = p
    # 7+
    p7 = 1 - sum(dist.values())
    dist["7+"] = p7
    return dist

# 半全场 (简化: 半场随机分布)
def predict_htft(p_spf):
    """半全场: 9 选 1, 简化基于 SPF + 平局扰动"""
    # 假设半场=均势 0.33, 0.34, 0.33
    # 终场=SPF
    # HT/FT 联合分布
    ht = {"胜": 0.33, "平": 0.34, "负": 0.33}
    out = {}
    for h in ["胜", "平", "负"]:
        for f in ["胜", "平", "负"]:
            out[f"{h}{f}"] = ht[h] * p_spf[f]
    total = sum(out.values())
    return {k: v/total for k, v in out.items()}

# 综合预测每场
print("=" * 100)
print("6/7~6/8 竞彩综合预测 (4 场友谊赛 + 3 场 K2)")
print("=" * 100)
print()

predictions = {}
for m in matches:
    print(f"\n{'='*80}")
    print(f"📋 {m['mid']} | {m['league']} | {m['time']}")
    print(f"   {m['home']} (让{m['rq']}) vs {m['away']}")
    print(f"{'='*80}")

    # 三个信号
    p_market = odds_to_prob(m['odds_market'])
    p_vote = vote_to_prob(m['vote'])
    p_form = form_to_prob(m['form']['H'], m['form']['A'])

    # 综合 (市场 0.5 + 投票 0.3 + 势态 0.2)
    p_spf = {}
    for k in ["胜", "平", "负"]:
        p_spf[k] = 0.5*p_market[k] + 0.3*p_vote[k] + 0.2*p_form[k]

    # 让球概率
    p_rq = handicap_outcome(p_spf, m['rq'])

    # 比分/进球/半全场 (基于 λ, 用市场赔率反推)
    # 简化: 用势态 + 让球反推 λ
    # 主队期望进球 = 基础 + 让球奖励 + 势态奖励
    base = 1.3
    # 让球数: 主让负越多, 主队实力越强, λ 高
    rq_boost = abs(m['rq']) * 0.25 if m['rq'] < 0 else -abs(m['rq']) * 0.15
    # 势态差分
    fs_h = form_score(m['form']['H'])
    fs_a = form_score(m['form']['A'])
    form_diff = fs_h - fs_a  # -1 ~ 1
    form_boost = 0.5 * form_diff
    lam_h = base + rq_boost + form_boost + 0.2
    lam_a = base - rq_boost - form_boost + 0.2
    lam_h = max(0.5, min(2.8, lam_h))
    lam_a = max(0.5, min(2.8, lam_a))

    print(f"\n  🔢 信号融合:")
    print(f"     市场赔率反推: 主 {p_market['胜']*100:.1f}% / 平 {p_market['平']*100:.1f}% / 客 {p_market['负']*100:.1f}%")
    print(f"     365scores投票: 主 {p_vote['胜']*100:.1f}% / 平 {p_vote['平']*100:.1f}% / 客 {p_vote['负']*100:.1f}% (n={m['vote']['n']})")
    print(f"     势态差分:     主 {p_form['胜']*100:.1f}% / 平 {p_form['平']*100:.1f}% / 客 {p_form['负']*100:.1f}%")
    print(f"     ✅ 综合 SPF:  主 {p_spf['胜']*100:.1f}% / 平 {p_spf['平']*100:.1f}% / 客 {p_spf['负']*100:.1f}%")

    # 让球
    rq_key = f"让{m['rq']}球" if m['rq'] != 0 else "平手"
    print(f"\n  ⚽ 让球 ({rq_key}):")
    print(f"     让胜 {p_rq['让胜']*100:.1f}% / 让平 {p_rq['让平']*100:.1f}% / 让负 {p_rq['让负']*100:.1f}%")
    if p_rq['让胜'] > p_rq['让负'] and p_rq['让胜'] > p_rq['让平']:
        rec_rq = "让胜"
    elif p_rq['让负'] > p_rq['让胜'] and p_rq['让负'] > p_rq['让平']:
        rec_rq = "让负"
    else:
        rec_rq = "让平"
    print(f"     首选: {rec_rq}")

    # 比分 Top 8
    print(f"\n  📊 比分 Top 8 (λ 主={lam_h:.2f}, 客={lam_a:.2f}):")
    score_dist = predict_score_dist(p_spf, p_rq, lam_h, lam_a)
    for i, (s, p) in enumerate(score_dist, 1):
        print(f"     {i}. {s}: {p*100:.1f}%")

    # 总进球
    print(f"\n  ⚽ 总进球分布:")
    tg = predict_total_goals(lam_h, lam_a)
    for k, v in tg.items():
        if v > 0.01:
            print(f"     {k}球: {v*100:.1f}%")
    p_over = sum(v for k, v in tg.items() if isinstance(k, int) and k > 2)
    print(f"     大2.5: {p_over*100:.1f}% / 小2.5: {(1-p_over)*100:.1f}%")

    # 半全场
    print(f"\n  🏁 半全场 Top 6:")
    htft = predict_htft(p_spf)
    sorted_htft = sorted(htft.items(), key=lambda x: -x[1])
    for i, (k, v) in enumerate(sorted_htft[:6], 1):
        print(f"     {i}. {k}: {v*100:.1f}%")

    # 保存到 predictions dict
    predictions[m['mid']] = {
        "match": f"{m['home']} vs {m['away']}",
        "spf": max(p_spf, key=p_spf.get),
        "spf_probs": p_spf,
        "rq": rec_rq,
        "rq_probs": p_rq,
        "scores": [s for s, _ in score_dist],
        "score_probs": dict(score_dist),
        "size": "大2.5" if p_over > 0.5 else "小2.5",
        "size_prob": p_over,
        "htft": sorted_htft[0][0],
        "htft_probs": dict(sorted_htft),
    }

# 保存 JSON
with open('/root/scripts/preds_2026-06-07.json', 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
print(f"\n\n✅ 预测保存到: /root/scripts/preds_2026-06-07.json")
