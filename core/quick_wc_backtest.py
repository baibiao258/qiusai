# quick WC backtest
import requests, sys, numpy as np, os, joblib
sys.path.insert(0, '/root/wc_2026_upgrade'); os.chdir('/root')

# Load models manually
from daily_jczq import _load_poisson_elo_prior, _load_shared_models, _try_hybrid_predict
_load_poisson_elo_prior()
_load_shared_models()

# 12 WC matches
matches = [
    ("Sweden","Tunisia",5,1),("Côte d'Ivoire","Ecuador",1,0),
    ("Netherlands","Japan",2,2),("Germany","Curaçao",7,1),
    ("Australia","Türkiye",2,0),("Haiti","Scotland",0,1),
    ("Brazil","Morocco",1,1),("Qatar","Switzerland",1,1),
    ("USA","Paraguay",4,1),("Canada","Bosnia & Herzegovina",1,1),
    ("South Korea","Czechia",2,1),("Mexico","South Africa",2,0),
]

print(f"{'比赛':<30s} {'比分':>5s} {'实':>3s} {'H':>6s} {'D':>6s} {'A':>6s} {'预':>3s} {'结':>3s}")
print("─"*65)

hit=n=0; bs=0.0
for h, a, hg, ag in matches:
    actual = 'H' if hg>ag else ('D' if hg==ag else 'A')
    aid = 0 if actual=='H' else (1 if actual=='D' else 2)
    r = _try_hybrid_predict(h, a, '世界杯', None)
    if r and r.get('probs'):
        p=r['probs']
        pred='H' if p['H']>p['D'] and p['H']>p['A'] else ('D' if p['D']>p['H'] and p['D']>p['A'] else 'A')
        ok='✅' if pred==actual else '❌'
        pp=[p['H'],p['D'],p['A']]
        oh=[1.0 if c==aid else 0.0 for c in range(3)]
        br=sum((pp[c]-oh[c])**2 for c in range(3))/3.0
        if pred==actual: hit+=1; n+=1; bs+=br
        print(f"{h:<15s} vs {a:<15s} {hg:>2d}-{ag:<2d} {actual:>3s} {pp[0]:>5.0%} {pp[1]:>5.0%} {pp[2]:>5.0%} {pred:>3s} {ok:>3s}")
    else:
        print(f"{h:<15s} vs {a:<15s} {hg:>2d}-{ag:<2d} {actual:>3s} {'N/A':>19s}")

print("─"*65)
print(f"DC+Pinnacle: {hit}/{n} = {hit/n*100:.1f}%  Brier={bs/n:.4f}" if n>0 else "无数据")
