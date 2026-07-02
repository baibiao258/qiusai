#!/usr/bin/env python3
"""
system_analysis_report.py — 预测系统全面分析报告
================================================
分析当前系统的架构、数据流、模型状态和优化方向
"""

import os
import json
import pandas as pd
import joblib
from datetime import date

DATA_DIR = '/root/data'
WC_DIR = '/root/wc_2026_upgrade'

def load_system_state():
    """加载系统状态"""
    state = {}
    
    # 检查关键文件
    key_files = {
        'xgb_model_nat.pkl': '国家队XGBoost模型',
        'xgb_model_29.pkl': '旧30维模型',
        'dc_model.pkl': '国家队DC模型',
        'dc_club.pkl': '俱乐部DC模型',
        'elo_ratings.pkl': 'Elo评分',
        'training_data_with_odds.json': '训练数据',
        'wc_final_predictions.json': '世界杯预测',
        '500_odds_complete_20260614.json': '500.com完整赔率',
    }
    
    for filename, desc in key_files.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            state[filename] = {
                'exists': True,
                'size': size,
                'desc': desc,
            }
        else:
            state[filename] = {
                'exists': False,
                'desc': desc,
            }
    
    return state


def analyze_training_data():
    """分析训练数据"""
    filepath = os.path.join(DATA_DIR, 'training_data_with_odds.json')
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计
    total = len(data)
    
    # 按来源统计
    sources = {}
    for m in data:
        src = m.get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    
    # 按日期统计
    dates = {}
    for m in data:
        d = m.get('date', 'unknown')[:7]  # YYYY-MM
        dates[d] = dates.get(d, 0) + 1
    
    # 检查标签类型
    spf_types = {}
    for m in data:
        result = m.get('spf_result')
        t = type(result).__name__
        spf_types[t] = spf_types.get(t, 0) + 1
    
    return {
        'total': total,
        'sources': sources,
        'dates': dates,
        'spf_types': spf_types,
    }


def analyze_model_performance():
    """分析模型性能"""
    results = {}
    
    # 检查nat模型
    nat_model_path = os.path.join(DATA_DIR, 'xgb_model_nat.pkl')
    if os.path.exists(nat_model_path):
        try:
            model = joblib.load(nat_model_path)
            results['nat_model'] = {
                'exists': True,
                'n_features': model.n_features_in_,
                'n_estimators': model.n_estimators,
            }
        except:
            results['nat_model'] = {'exists': True, 'error': '加载失败'}
    
    return results


def analyze_odds_coverage():
    """分析赔率覆盖情况"""
    filepath = os.path.join(DATA_DIR, '500_odds_complete_20260614.json')
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计各玩法覆盖率
    play_types = ['spf', 'nspf', 'jqs', 'bf', 'bqc']
    coverage = {}
    for pt in play_types:
        count = sum(1 for m in data if m.get('odds', {}).get(pt))
        coverage[pt] = {
            'count': count,
            'total': len(data),
            'rate': count / len(data) * 100 if data else 0,
        }
    
    return {
        'total_matches': len(data),
        'coverage': coverage,
    }


def generate_report():
    """生成完整报告"""
    print('=' * 80)
    print('足球预测系统全面分析报告')
    print(f'生成时间: {date.today().strftime("%Y-%m-%d")}')
    print('=' * 80)
    
    # 1. 系统状态
    print('\n【1. 系统状态】')
    state = load_system_state()
    for filename, info in state.items():
        status = '✅' if info['exists'] else '❌'
        size_info = f' ({info["size"]:,} bytes)' if info.get('size') else ''
        print(f'{status} {info["desc"]}: {filename}{size_info}')
    
    # 2. 训练数据
    print('\n【2. 训练数据分析】')
    training = analyze_training_data()
    if training:
        print(f'总样本数: {training["total"]}')
        print(f'数据来源: {training["sources"]}')
        print(f'标签类型: {training["spf_types"]}')
        print(f'时间分布:')
        for d, count in sorted(training['dates'].items()):
            print(f'  {d}: {count} 场')
    else:
        print('❌ 无法加载训练数据')
    
    # 3. 模型性能
    print('\n【3. 模型性能】')
    model_perf = analyze_model_performance()
    for model_name, info in model_perf.items():
        if info.get('exists'):
            print(f'{model_name}:')
            for k, v in info.items():
                if k != 'exists':
                    print(f'  {k}: {v}')
    
    # 4. 赔率覆盖
    print('\n【4. 500.com赔率覆盖】')
    odds = analyze_odds_coverage()
    if odds:
        print(f'总场次: {odds["total_matches"]}')
        for pt, info in odds['coverage'].items():
            print(f'{pt}: {info["count"]}/{info["total"]} ({info["rate"]:.1f}%)')
    
    # 5. 系统架构
    print('\n【5. 系统架构】')
    print('''
┌─────────────────────────────────────────────────────────────────┐
│                    数据层 (Data Layer)                          │
├─────────────────────────────────────────────────────────────────┤
│  500.com ─────┐                                                 │
│  (HTML抓取)    │                                                │
│               ▼                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │ historical_kaijiang │    │ 500_odds_complete   │            │
│  │ (3,248场历史数据)   │    │ (当前赔率)           │            │
│  └─────────────────────┘    └─────────────────────┘            │
│               │                       │                         │
│               └───────────┬───────────┘                         │
│                           ▼                                     │
│              ┌─────────────────────────┐                       │
│              │ training_data_with_odds │                       │
│              │ (491条训练样本)          │                       │
│              └─────────────────────────┘                       │
├─────────────────────────────────────────────────────────────────┤
│                    模型层 (Model Layer)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ xgb_model_nat   │  │ dc_model.pkl    │  │ elo_ratings.pkl │ │
│  │ (11维XGBoost)   │  │ (国家队DC)      │  │ (11,045队Elo)   │ │
│  │ 75.4%准确率     │  │ (226队)         │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                           │                                     │
│                           ▼                                     │
│              ┌─────────────────────────┐                       │
│              │ calibrated_predictor.py │                       │
│              │ (加权融合预测器)         │                       │
│              └─────────────────────────┘                       │
├─────────────────────────────────────────────────────────────────┤
│                    输出层 (Output Layer)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ wc_final_predictions.json (64场完整预测)                │   │
│  │ - SPF胜平负: H/D/A概率 + 推荐                          │   │
│  │ - 让球: 让球值 + 让球后概率                              │   │
│  │ - 半全场: 9种组合概率                                    │   │
│  │ - 比分: Top15比分概率                                   │   │
│  │ - 总进球: 13档概率                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
''')
    
    # 6. 优化方向
    print('\n【6. 优化方向】')
    print('''
┌─────────────────────────────────────────────────────────────────┐
│ P0级 (必须修复)                                                 │
├─────────────────────────────────────────────────────────────────┤
│ 1. 训练样本不足: 491条 → 目标800+条                           │
│    - 方案: The Odds API赛后累积 + 500.com历史回捞              │
│                                                                │
│ 2. 标签类型混乱: int/str混型                                   │
│    - 状态: 已修复 (str())                                      │
│    - 验证: 准确率64.4% → 75.4%                                │
│                                                                │
│ 3. 平局灭绝问题                                                │
│    - 状态: 已修复 (Elo+Market估算)                             │
│    - 验证: 32/64场有合理平局概率                               │
├─────────────────────────────────────────────────────────────────┤
│ P1级 (重要优化)                                                 │
├─────────────────────────────────────────────────────────────────┤
│ 4. 500.com赔率集成                                             │
│    - 状态: ✅ 已完成抓取脚本                                   │
│    - 下一步: 集成到训练数据扩充                                 │
│                                                                │
│ 5. 双管线统一                                                   │
│    - 状态: 已统一用nat模型                                     │
│    - 验证: 29/30维模型已退役                                   │
│                                                                │
│ 6. 每日自动管线                                                 │
│    - 状态: ✅ daily_wc_pipeline.py + cron job                  │
│    - 下一步: 集成500.com赔率                                   │
├─────────────────────────────────────────────────────────────────┤
│ P2级 (长期优化)                                                 │
├─────────────────────────────────────────────────────────────────┤
│ 7. 俱乐部DC模型                                                 │
│    - 状态: dc_club.pkl (2,174队)                               │
│    - 问题: γ=0 (无主场优势), 豪门覆盖不足                      │
│                                                                │
│ 8. 特征工程                                                     │
│    - 当前: 11维 (Elo/DC/Market)                                │
│    - 优化: 添加伤病/近期状态/历史交锋                         │
│                                                                │
│ 9. 模型验证                                                     │
│    - 当前: 时间序列holdout 75.4%                               │
│    - 优化: 添加Brier/LogLoss/ROI回测                          │
└─────────────────────────────────────────────────────────────────┘
''')
    
    # 7. 关键文件清单
    print('\n【7. 关键文件清单】')
    print('''
核心模型:
  /root/data/xgb_model_nat.pkl          - 国家队XGBoost (11维, 75.4%)
  /root/data/dc_model.pkl               - 国家队DC (226队)
  /root/data/dc_club.pkl                - 俱乐部DC (2,174队)
  /root/data/elo_ratings.pkl            - Elo评分 (11,045队)

训练数据:
  /root/data/training_data_with_odds.json - 统一训练集 (491条)
  /root/data/historical_kaijiang.csv      - 历史赛果 (3,248场)
  /root/data/500_odds_complete_20260614.json - 500.com完整赔率

预测脚本:
  /root/wc_2026_upgrade/daily_wc_pipeline.py - 每日自动管线
  /root/wc_2026_upgrade/calibrated_predictor.py - 最终预测器
  /root/wc_2026_upgrade/wc_detail_pred.py - 详细5玩法预测
  /root/wc_2026_upgrade/fetch_500_complete.py - 500.com完整抓取
  /root/wc_2026_upgrade/integrate_500_odds.py - 赔率集成工具

配置文件:
  /root/data/team_name_mapping.json     - 中英队名映射
  /root/data/wc_final_predictions.json  - 64场预测结果
''')
    
    print('=' * 80)
    print('报告生成完成')
    print('=' * 80)


if __name__ == '__main__':
    generate_report()
