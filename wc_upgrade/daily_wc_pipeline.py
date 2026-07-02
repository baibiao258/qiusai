#!/usr/bin/env python3
"""
daily_wc_pipeline.py — 世界杯每日预测管线
=========================================
步骤:
  1. 从 The Odds API 拉今日赔率
  2. 加载模型预测
  3. 输出预测结果
  4. 检查已完成比赛→累积训练数据
"""
import json, os, sys, math, urllib.request, time
from datetime import date, timedelta
sys.path.insert(0, '/root')
DATA_DIR = '/root/data'
API_KEY = '425a7cb6604fe89fcbd46a524ac08a11'

def log(msg):
    print(f'[{date.today()}] {msg}', flush=True)

def fetch_odds():
    """拉世界杯赔率"""
    url = f'https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/?apiKey={API_KEY}&regions=uk&markets=h2h'
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    remaining = resp.headers.get('x-requests-remaining', '?')
    log(f'The Odds API: {len(data)} 场, 剩余配额={remaining}')
    return data

def fetch_scores():
    """拉已完成比赛赛果"""
    url = f'https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/scores/?apiKey={API_KEY}&daysFrom=3'
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    completed = [e for e in data if e.get('completed')]
    log(f'已完成比赛: {len(completed)}/{len(data)}')
    return completed

def odds_to_training(odds_data):
    """赔率→训练样本 (比赛完成后调用)"""
    # 等比赛完成后才调用
    pass

def main():
    log('🚀 每日世界杯预测管线启动')
    
    # Step 1: 拉赔率
    odds = fetch_odds()
    
    # Step 2: 检查已完成比赛
    scores = fetch_scores()
    for s in scores:
        if s.get('completed'):
            home = s['home_team']
            away = s['away_team']
            # 找比赛结果
            for score in s.get('scores', []):
                pass  # TODO: 累积训练数据
    
    # Step 3: 格式化赔率为预测输入
    matches = []
    for m in odds:
        home, away = m['home_team'], m['away_team']
        commence = m.get('commence_time', '')[:10]
        
        odds_h = odds_d = odds_a = 0
        for bm in m.get('bookmakers', []):
            for market in bm.get('markets', []):
                if market['key'] == 'h2h':
                    for o in market['outcomes']:
                        if o['name'] == home: odds_h = o['price']
                        elif o['name'] == away: odds_a = o['price']
                        elif o['name'] == 'Draw': odds_d = o['price']
                    break
            if odds_h > 0: break
        
        market_h = 0.0
        if all(o > 0 for o in [odds_h, odds_d, odds_a]):
            imp = [1.0/o for o in [odds_h, odds_d, odds_a]]
            market_h = imp[0] / sum(imp)
        
        matches.append({
            'date': commence,
            'home': home, 'away': away,
            'odds_h': odds_h, 'odds_d': odds_d, 'odds_a': odds_a,
            'market_h': market_h,
        })
    
    # 保存今日赔率
    path = f'{DATA_DIR}/wc_odds_{date.today().isoformat()}.json'
    json.dump(matches, open(path, 'w'), indent=2)
    log(f'保存赔率: {path}')
    
    # Step 4: 运行预测
    try:
        from calibrated_predictor import predict
        results = []
        for m in matches:
            hy, pipe = predict(m['home'], m['away'], m['market_h'])
            results.append({**m, 'h_pred': round(hy[2],1), 'd_pred': round(hy[1],1), 'a_pred': round(hy[0],1), 'pipe': pipe})
        
        results.sort(key=lambda r: -r['h_pred'])
        pred_path = f'{DATA_DIR}/wc_pred_{date.today().isoformat()}.json'
        json.dump(results, open(pred_path, 'w'), indent=2)
        log(f'预测完成: {pred_path}')
        
        # 输出摘要
        print(f'\n📊 {date.today()} 预测摘要:')
        print(f'{"主队":>22} {"客队":>22} {"H":>6} {"D":>6} {"A":>6} {"赔H":>6}')
        print('-' * 68)
        for r in results:
            print(f'{r["home"][:20]:>22} {r["away"][:20]:<22} '
                  f'{r["h_pred"]:>5.1f}% {r["d_pred"]:>5.1f}% {r["a_pred"]:>5.1f}% '
                  f'{r["odds_h"]:>6.2f}')
        
    except ImportError:
        log('⚠️ calibrated_predictor 未就绪, 跳过预测')
    
    log('✅ 管线完成')

if __name__ == '__main__':
    main()
