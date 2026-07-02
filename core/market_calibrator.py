#!/usr/bin/env python3
"""
Market Calibrator v2 — 500.com 市场赔率校准层
===========================================
仅使用 SPF + 半全场赔率做校准，让球数据只做参考输出。
设计原则: 市场是昂贵的信号，但竞彩市场不完美且可能矛盾，
所以权重要保守 (0.12~0.20)，且遇到矛盾时降权。

Usage:
    from market_calibrator import calibrate
    cal = calibrate(model_hda, spf_odds, rq_odds, hf_9_odds, handicap=-1)
"""
import math
from typing import List, Tuple, Optional, Dict

def devig(odds: List[float]) -> List[float]:
    inv = [1.0 / max(o, 1.01) for o in odds]
    total = sum(inv)
    return [i / total for i in inv] if total > 0 else [1/len(odds)] * len(odds)

def margin(odds: List[float]) -> float:
    return sum(1.0 / max(o, 1.01) for o in odds) - 1.0

def spf_devig(spf_h: float, spf_d: float, spf_a: float) -> Tuple[float, float, float]:
    p = devig([spf_h, spf_d, spf_a])
    return (p[0], p[1], p[2])

def hf_devig(hf_9: List[float]) -> Dict[str, float]:
    labels = ['HH','HD','HA','DH','DD','DA','AH','AD','AA']
    probs = devig(list(hf_9))
    return {l: p for l, p in zip(labels, probs)}

def hf_to_ft(hf_dict: Dict[str, float]) -> Tuple[float, float, float]:
    h = hf_dict.get('HH',0)+hf_dict.get('DH',0)+hf_dict.get('AH',0)
    d = hf_dict.get('HD',0)+hf_dict.get('DD',0)+hf_dict.get('AD',0)
    a = hf_dict.get('HA',0)+hf_dict.get('DA',0)+hf_dict.get('AA',0)
    t = h+d+a
    return (h/t, d/t, a/t) if t > 0 else (1/3, 1/3, 1/3)

def consistency(m1: Tuple[float,...], m2: Tuple[float,...]) -> float:
    """Bhattacharyya 系数"""
    bc = sum(math.sqrt(a*b) for a, b in zip(m1, m2))
    return min(1.0, bc)

def calibrate(
    model_hda: Tuple[float, float, float],
    spf_odds: Optional[Tuple[float, float, float]] = None,
    rq_odds: Optional[Tuple[float, float, float]] = None,
    hf_9_odds: Optional[List[float]] = None,
    handicap: int = 0,
    base_weight: float = 0.15,
) -> Dict:
    """
    市场校准主函数

    方法:
    1. SPF 去水 → 市场胜平负概率 (market_spf)
    2. 半全场去水 → 独立推导的FT概率 (hf_ft)
    3. 一致性检查 → 动态权重
    4. 混合: (1-w) × 模型 + w × 市场SPF
    5. 半全场微调 (如果一致性高)

    返回:
        calibrated: [p_h, p_d, p_a]
        market_spf: 市场去水概率
        hf_ft_probs: 半全场推导概率
        market_weight: 实际市场权重
        margin_pct: 市场水位
        direction_change: 方向是否变化
    """
    p_h, p_d, p_a = model_hda
    t = p_h + p_d + p_a
    p_h /= t; p_d /= t; p_a /= t

    orig_dir = 'H' if p_h > p_d and p_h > p_a else ('D' if p_d > p_h and p_d > p_a else 'A')

    result = {
        'market_used': False,
        'market_weight': 0.0,
        'margin_pct': 0.0,
        'market_spf': None,
        'hf_ft_probs': None,
        'direction_change': False,
        'calibrated': (round(p_h,4), round(p_d,4), round(p_a,4)),
    }

    if not spf_odds or any(o <= 1.01 for o in spf_odds):
        return result

    # Step 1: SPF去水
    m_spf = spf_devig(spf_odds[0], spf_odds[1], spf_odds[2])
    marg = margin(list(spf_odds))
    result['market_spf'] = (round(m_spf[0],4), round(m_spf[1],4), round(m_spf[2],4))
    result['margin_pct'] = round(marg*100, 1)

    # Step 2: 半全场去水 & 一致性
    hf_ft = None
    w = base_weight
    if hf_9_odds and len(hf_9_odds) == 9:
        try:
            hf_dict = hf_devig(list(hf_9_odds))
            hf_ft = hf_to_ft(hf_dict)
            result['hf_ft_probs'] = (round(hf_ft[0],4), round(hf_ft[1],4), round(hf_ft[2],4))

            # 一致性: SPF vs 半全场
            cs = consistency(m_spf, hf_ft)
            # 与模型的一致性
            cs_model_vs_spf = consistency((p_h, p_d, p_a), m_spf)
            cs_model_vs_hf = consistency((p_h, p_d, p_a), hf_ft)

            # 调权规则:
            # SPF和半全场都指向同一方向 + 模型也接近 → 高置信度 → w提升
            # SPF和半全场矛盾 → 市场总体不可信 → w降低
            # 模型远离两个市场 → 市场可信 → w提升
            if cs > 0.85:
                w = min(w + 0.08, 0.30)  # 两个市场一致 → 信市场
            elif cs < 0.5:
                w = max(w - 0.06, 0.0)   # SPF和半全场矛盾 → 降权

            # 如果模型和市场严重对立 (都在50%以上但方向相反), 提升权重再平衡
            dir_spf = 'H' if m_spf[0] > m_spf[1] and m_spf[0] > m_spf[2] else ('D' if m_spf[1] > m_spf[0] and m_spf[1] > m_spf[2] else 'A')
            if orig_dir != dir_spf and max(p_h, p_d, p_a) > 0.45 and max(m_spf) > 0.35:
                # 模型和市场分歧较大 → 折中权重重些
                w = max(w, 0.22)

            # 水位越底越可信
            if marg < 0.06:
                w = min(w * 1.3, 0.30)
            elif marg > 0.15:
                w = w * 0.7

        except:
            pass

    # Step 3: 混合
    cal_h = p_h * (1-w) + m_spf[0] * w
    cal_d = p_d * (1-w) + m_spf[1] * w
    cal_a = p_a * (1-w) + m_spf[2] * w

    # Step 4: 如果有半全场且一致性高, 额外做一次 Dirichlet 平滑
    if hf_ft and consistency(m_spf, hf_ft) > 0.80:
        # 非常轻的平滑: 如果两个市场指向同一方向, 稍微boost
        cal_h2 = cal_h * 0.85 + hf_ft[0] * 0.15
        cal_d2 = cal_d * 0.85 + hf_ft[1] * 0.15
        cal_a2 = cal_a * 0.85 + hf_ft[2] * 0.15
        cal_h, cal_d, cal_a = cal_h2, cal_d2, cal_a2

    # normalize
    t = cal_h + cal_d + cal_a
    cal_h /= t; cal_d /= t; cal_a /= t

    cal_dir = 'H' if cal_h > cal_d and cal_h > cal_a else ('D' if cal_d > cal_h and cal_d > cal_a else 'A')

    result['calibrated'] = (round(cal_h,4), round(cal_d,4), round(cal_a,4))
    result['market_used'] = True
    result['market_weight'] = round(w, 3)
    result['direction_change'] = orig_dir != cal_dir

    return result


def test():
    """Run calibration on today's 4 matches"""
    matches = [
        {
            'name': '丹麦 vs 刚果(金)',
            'model': (0.515, 0.327, 0.157),
            'spf': (2.34, 3.11, 2.63),
            'rq': (1.37, 3.90, 7.00),
            'hf9': [2.10, 17.00, 55.00, 3.75, 5.75, 13.50, 27.00, 17.00, 11.00],
            'handicap': -1,
        },
        {
            'name': '荷兰 vs 阿尔及利亚',
            'model': (0.492, 0.317, 0.191),
            'spf': (1.69, 3.70, 3.75),
            'rq': (1.18, 5.25, 11.00),
            'hf9': None,  # TODO: fetch from 500.com
            'handicap': -1,
        },
        {
            'name': '波兰 vs 尼日利亚',
            'model': (0.371, 0.302, 0.327),
            'spf': (4.50, 3.45, 1.62),
            'rq': (1.98, 2.92, 3.56),
            'hf9': None,
            'handicap': -1,
        },
        {
            'name': '卢森堡 vs 意大利',
            'model': (0.080, 0.194, 0.725),
            'spf': (2.92, 3.10, 2.16),
            'rq': (8.00, 4.25, 1.30),
            'hf9': None,
            'handicap': +1,
        },
    ]

    print(f"\n{'='*75}")
    print(f"  Market Calibrator v2 — 4场验证")
    print(f"{'='*75}")

    for m in matches:
        r = calibrate(m['model'], m['spf'], m['rq'], m['hf9'], m['handicap'])
        md = m['model']
        ms = r['market_spf']
        mc = r['calibrated']

        # EV计算
        ev_orig_h = md[0]*m['spf'][0]-1
        ev_orig_d = md[1]*m['spf'][1]-1
        ev_orig_a = md[2]*m['spf'][2]-1
        ev_cal_h = mc[0]*m['spf'][0]-1
        ev_cal_d = mc[1]*m['spf'][1]-1
        ev_cal_a = mc[2]*m['spf'][2]-1

        print(f"\n  ── {m['name']} ──")
        print(f"  {'':>12s} {'主胜':>8s} {'平':>8s} {'客胜':>8s} {'水/权':>10s}")
        print(f"  {'模型':>10s}  {md[0]*100:>7.1f}% {md[1]*100:>7.1f}% {md[2]*100:>7.1f}%")
        print(f"  {'市场SPF':>10s}  {ms[0]*100:>7.1f}% {ms[1]*100:>7.1f}% {ms[2]*100:>7.1f}%  {r['margin_pct']:.0f}% / {r['market_weight']*100:.0f}%")
        if r.get('hf_ft_probs'):
            hf = r['hf_ft_probs']
            print(f"  {'半全场→FT':>10s}  {hf[0]*100:>7.1f}% {hf[1]*100:>7.1f}% {hf[2]*100:>7.1f}%")
        print(f"  {'校准后':>10s}  {mc[0]*100:>7.1f}% {mc[1]*100:>7.1f}% {mc[2]*100:>7.1f}%")
        if r['direction_change']:
            print(f"  ⚠️ 方向变化!")
        print(f"  {'EV模型':>10s}  {ev_orig_h*100:+>7.1f}% {ev_orig_d*100:+>7.1f}% {ev_orig_a*100:+>7.1f}%")
        print(f"  {'EV校准':>10s}  {ev_cal_h*100:+>7.1f}% {ev_cal_d*100:+>7.1f}% {ev_cal_a*100:+>7.1f}%")

if __name__ == '__main__':
    test()
