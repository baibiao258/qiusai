#!/usr/bin/env python3
"""
单独跑一场比赛的预测 - 用于调试和验证
用法: python3 run_single_match.py "葡萄牙" "尼日利亚"

从 predictions_log.csv 读取已有数据, 重新运行模型推理并输出完整预测。
用途: 调试单场预测、验证模型输出、重新生成预测结果。

依赖: daily_jczq.py 中的函数 (需在 /root 目录运行)
"""
import sys
import os
sys.path.insert(0, '/root')
os.chdir('/root')

from daily_jczq import (
    predict_match_wrapper, build_prediction_bundle, print_match_bundle,
    scrape_500_odds_today, load_365scores_today, build_365_map,
    compute_bet_action, HTFT_SHORT_MAP
)

def main():
    if len(sys.argv) < 3:
        print("用法: python3 run_single_match.py <主队> <客队>")
        print("示例: python3 run_single_match.py 葡萄牙 尼日利亚")
        sys.exit(1)
    
    home = sys.argv[1]
    away = sys.argv[2]
    
    print(f"\n{'='*70}")
    print(f"  单独预测: {home} vs {away}")
    print(f"{'='*70}\n")
    
    # 获取500.com赔率数据
    print("📡 获取500.com赔率...")
    _500_odds = scrape_500_odds_today()
    _500_map = {}
    for m5 in (_500_odds or []):
        _500_map[(m5['home_cn'], m5['away_cn'])] = m5
    
    # 查找匹配的比赛
    market_row = _500_map.get((home, away))
    if not market_row:
        # 尝试模糊匹配
        for (h, a), m5 in _500_map.items():
            if home in h or away in a or h in home or a in away:
                market_row = m5
                print(f"  ⚠️ 模糊匹配: {h} vs {a}")
                break
    
    if market_row:
        print(f"  ✅ 找到500.com赔率: {market_row.get('code', 'N/A')}")
        print(f"     赔率: H={market_row.get('odds_h', 0)} D={market_row.get('odds_d', 0)} A={market_row.get('odds_a', 0)}")
    else:
        print(f"  ⚠️ 未找到{home} vs {away}的500.com赔率")
    
    # 获取365scores数据
    print("\n📡 获取365scores数据...")
    score365_games = load_365scores_today()
    score365_map = build_365_map(score365_games)
    score_meta = score365_map.get((home, away))
    if score_meta:
        print(f"  ✅ 找到365scores数据")
    else:
        print(f"  ⚠️ 未找到365scores数据")
    
    # 运行预测模型
    print(f"\n🧠 运行预测模型...")
    p = predict_match_wrapper(home, away)
    
    if not p:
        print(f"❌ 预测失败: 模型无法处理 {home} vs {away}")
        sys.exit(1)
    
    print(f"  ✅ 模型: {p.get('model', 'unknown')}")
    print(f"  λ: home={p['lambda_ft']['home']:.3f}, away={p['lambda_ft']['away']:.3f}")
    print(f"  概率: H={p['probs']['H']:.4f} D={p['probs']['D']:.4f} A={p['probs']['A']:.4f}")
    
    # 构建预测bundle
    code = market_row['code'] if market_row else f"MANUAL-{home}-{away}"
    utc = market_row.get('time', '00:00') if market_row else '00:00'
    league = '友谊赛'
    
    bundle = build_prediction_bundle(
        code, home, away, utc, league, p, market_row, score_meta
    )
    
    # 计算bet_action (注意: 需要6个参数, 不是bundle)
    bundle['bet_action'] = compute_bet_action(
        league=league,
        model_type=p.get('model', ''),
        bet_analysis=bundle.get('bet_analysis'),
        htft_top6=bundle.get('htft_top6', []),
        handicap=bundle.get('handicap', 0),
        rq_probs={
            '让胜': bundle.get('pred_rq_win', 0) / 100,
            '让平': bundle.get('pred_rq_draw', 0) / 100,
            '让负': bundle.get('pred_rq_loss', 0) / 100,
        }
    )
    
    # 打印完整预测
    print_match_bundle(bundle)
    
    return bundle

if __name__ == '__main__':
    main()
