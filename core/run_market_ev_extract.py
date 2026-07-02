import pandas as pd
import json

with open('/root/data/predict_today.json', 'r') as f:
    data = json.load(f)

for item in data.get('matches', []):
    home = item['home_cn']
    away = item['away_cn']
    print(f"Match: {home} vs {away}")
    print(f"Prob: {item.get('fin_h')}% {item.get('fin_d')}% {item.get('fin_a')}%")
    print(f"EV: H={item.get('ev_h')} D={item.get('ev_d')} A={item.get('ev_a')}")
    print("---")
