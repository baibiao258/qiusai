#!/usr/bin/env python3
"""
train_club_model.py — 俱乐部比赛预测器
======================================
俱乐部训练数据太少(96条),无法训XGBoost.
使用 dc_club + Elo + Market 简单融合.
"""
import json, os, sys, math, numpy as np, joblib
sys.path.insert(0, '/root')
DATA_DIR = '/root/data'

def predict_club(home, away, market_implied=0.0):
    """俱乐部比赛预测: dc_club + Elo + Market 加权融合"""
    dc = joblib.load(f'{DATA_DIR}/dc_club.pkl')
    elo = joblib.load(f'{DATA_DIR}/elo_ratings.pkl')
    
    eh = elo.get(home, 1500)
    ea = elo.get(away, 1500)
    
    # 1. dc_club 预测
    dc_probs = np.array([1/3, 1/3, 1/3])
    try:
        p = dc.predict_proba(home, away, neutral=False)
        dc_probs = np.clip(p, 0.01, 0.99)
    except:
        pass
    
    # 2. Elo 隐含概率
    op_h = 1 / (1 + 10 ** ((ea - eh) / 400))
    elo_probs = np.array([op_h, 0, 1-op_h])  # 简化: Elo只给主客
    
    # 3. 市场隐含概率 (去水)
    market_arr = np.array([market_implied, 0, 1-market_implied])
    
    # 加权融合: dc_club 权重最高
    weights = [0.5, 0.2, 0.3]  # dc_club, elo, market
    probs = weights[0] * dc_probs + weights[1] * elo_probs + weights[2] * market_arr
    probs /= probs.sum()
    
    return probs
