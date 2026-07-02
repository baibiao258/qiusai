#!/usr/bin/env python3
"""
365scores 模型重训练方案
=======================
设计如何将 365scores 特征添加到 XGBoost 模型

用法:
  python3 model_retrain_plan.py
"""

import sys
sys.path.insert(0, '/root')

from predict_match import _XGB_DIM

def main():
    """主函数"""
    print("=== 365scores 模型重训练方案 ===")
    print()
    
    print(f"当前 XGBoost 特征维度: {_XGB_DIM}")
    print()
    
    print("=== 新增特征 (13个) ===")
    print()
    
    new_features = [
        # 人气排名特征 (4个)
        ("pop_rank_home", "主队人气排名", "数值越小越强"),
        ("pop_rank_away", "客队人气排名", "数值越小越强"),
        ("pop_rank_diff", "人气排名差异", "正值=主队更强"),
        ("pop_rank_log_diff", "人气排名对数差异", "归一化差异"),
        
        # 趋势数据特征 (6个)
        ("trend_win_rate_home", "主队近期胜率", "0-1"),
        ("trend_win_rate_away", "客队近期胜率", "0-1"),
        ("trend_win_rate_diff", "胜率差异", "正值=主队更好"),
        ("trend_points_home", "主队近期积分", "场均积分"),
        ("trend_points_away", "客队近期积分", "场均积分"),
        ("trend_points_diff", "积分差异", "正值=主队更好"),
        
        # 投票数据特征 (3个)
        ("vote_home", "主队投票比例", "0-1"),
        ("vote_draw", "平局投票比例", "0-1"),
        ("vote_away", "客队投票比例", "0-1"),
    ]
    
    for i, (name, desc, note) in enumerate(new_features, 1):
        print(f"  {i:2d}. {name:25s} - {desc} ({note})")
    
    print()
    print(f"新增特征总数: {len(new_features)}")
    print(f"重训练后特征维度: {_XGB_DIM + len(new_features)}")
    print()
    
    print("=== 重训练步骤 ===")
    print()
    print("1. 数据准备:")
    print("   - 收集历史 365scores 数据 (至少 100 场)")
    print("   - 提取新特征")
    print("   - 合并到现有训练数据")
    print()
    print("2. 特征工程:")
    print("   - 处理缺失值 (用中位数或 0 填充)")
    print("   - 标准化/归一化 (可选)")
    print("   - 特征选择 (可选)")
    print()
    print("3. 模型训练:")
    print("   - 使用 XGBoost 训练新模型")
    print("   - 交叉验证")
    print("   - 超参数调优")
    print()
    print("4. 模型评估:")
    print("   - 对比新旧模型准确率")
    print("   - 分析特征重要性")
    print("   - 验证融合效果")
    print()
    print("5. 部署:")
    print("   - 保存新模型")
    print("   - 更新预测脚本")
    print("   - 监控效果")
    print()
    
    print("=== 时间估算 ===")
    print()
    print("1. 数据收集: 1-2 周 (每日自动收集)")
    print("2. 特征工程: 1-2 天")
    print("3. 模型训练: 1-2 天")
    print("4. 模型评估: 1 天")
    print("5. 部署验证: 1 天")
    print()
    print("总计: 2-3 周")
    print()
    
    print("=== 风险与挑战 ===")
    print()
    print("1. 数据质量:")
    print("   - 365scores 数据可能不完整")
    print("   - 投票数据可能有偏差")
    print("   - 需要数据清洗")
    print()
    print("2. 特征相关性:")
    print("   - 新特征可能与现有特征高度相关")
    print("   - 需要特征选择")
    print("   - 可能需要降维")
    print()
    print("3. 模型过拟合:")
    print("   - 特征增加可能导致过拟合")
    print("   - 需要正则化")
    print("   - 需要交叉验证")
    print()
    print("4. 部署风险:")
    print("   - 新模型可能不如旧模型")
    print("   - 需要 A/B 测试")
    print("   - 需要回滚机制")

if __name__ == "__main__":
    main()
