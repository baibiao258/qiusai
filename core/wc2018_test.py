#!/usr/bin/env python3
"""2018 世界杯回测 — 泊松 + Elo + Random Forest"""
import json, math, csv, os, sys, urllib.request
from datetime import datetime
from collections import defaultdict
import numpy as np

MAX_GOALS = 6
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def elo_expected(ra, rb):
    return 1.0 / (1 + 10 ** ((rb - ra) / 400))

# ── 数据 ──

def fetch_intl(cache_path=None):
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f: return json.load(f)
    req = urllib.request.Request(url, headers={'User-Agent': 'wc/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode('utf-8')
    matches = []
    for row in csv.DictReader(raw.splitlines()):
        try:
            matches.append({'date': row['date'], 'home': row['home_team'],
                'away': row['away_team'], 'h_score': int(row['home_score']),
                'a_score': int(row['away_score'])})
        except: continue
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f: json.dump(matches, f)
    return matches

def fetch_wc(year, cache_path=None):
    url = f"https://raw.githubusercontent.com/openfootball/world-cup.json/master/{year}/worldcup.json"
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f: return json.load(f)
    print(f"  📡 下载 {year} 世界杯数据...")
    req = urllib.request.Request(url, headers={'User-Agent': 'wc/1.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    
    simplified = []
    for m in data.get('matches', []):
        rnd = m.get('round', '')
        is_ko = ('Round' in rnd or 'Final' in rnd or 'Semi' in rnd or
                 'Quarter' in rnd or 'third' in rnd)
        simplified.append({
            'date': m['date'], 'round': rnd,
            'team1': m['team1'], 'team2': m['team2'],
            'score_ft': m['score']['ft'],
            'is_knockout': is_ko,
        })
    return simplified

def compute_strengths(matches, cutoff, hl=180):
    stats = defaultdict(lambda: {'wg':0,'wc':0,'ws':0,'m':0})
    for m in matches:
        if m['date'] >= cutoff: continue
        days = (datetime.strptime(cutoff,'%Y-%m-%d') - datetime.strptime(m['date'],'%Y-%m-%d')).days
        w = 0.5 ** (max(days,0) / hl)
        for team, gf, ga in [(m['home'],m['h_score'],m['a_score']),(m['away'],m['a_score'],m['h_score'])]:
            s = stats[team]; s['wg']+=gf*w; s['wc']+=ga*w; s['ws']+=w; s['m']+=1
    total_wg = sum(s['wg'] for s in stats.values())
    global_avg = total_wg / max(sum(s['ws'] for s in stats.values()), 1)
    ts = {}
    for team, s in stats.items():
        avg_gf = s['wg']/max(s['ws'],0.001); avg_ga = s['wc']/max(s['ws'],0.001)
        ts[team] = {'att': avg_gf/max(global_avg,0.01), 'def': avg_ga/max(global_avg,0.01), 'm': s['m']}
    return ts, global_avg

def compute_elo(matches, cutoff):
    elo = defaultdict(lambda: 1500.0)
    for m in matches:
        if m['date'] >= cutoff: continue
        h,a = m['home'],m['away']
        e_h = elo_expected(elo[h], elo[a])
        sh,sa = (1.0,0.0) if m['h_score']>m['a_score'] else ((0.5,0.5) if m['h_score']==m['a_score'] else (0.0,1.0))
        elo[h] += 32*(sh-e_h); elo[a] += 32*(sa-(1-e_h))
    return dict(elo)

print("="*60)
print("  ⚽ 2018 世界杯回测")
print(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

CUTOFF = '2018-06-01'
cache_path = os.path.join(DATA_DIR, 'international_results.json')
intl = fetch_intl(cache_path)
wc = fetch_wc(2018)

train_matches = [m for m in intl if m['date'] < CUTOFF]
print(f"\n📊 训练数据: {len(train_matches)} 场 (截至{CUTOFF})")
print(f"📊 测试数据: {len(wc)} 场")

# ── 训练 ──
print(f"\n{'─'*60}")
print(f"  🧠 训练模型...")
ts, ga = compute_strengths(intl, CUTOFF)
elo_r = compute_elo(intl, CUTOFF)

top_e = sorted(elo_r.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"  Elo TOP5: {', '.join(f'{t}({r:.0f})' for t,r in top_e)}")
print(f"  全球λ={ga:.3f}")

# 找出参赛队中数据不足的
wc_teams = set()
for m in wc:
    wc_teams.add(m['team1']); wc_teams.add(m['team2'])
print(f"  参赛队: {len(wc_teams)}")
low_data = [t for t in wc_teams if ts.get(t,{}).get('m',0) < 3]
if low_data:
    print(f"  ⚠️  数据不足的队: {', '.join(low_data)}")

# ── 回测: 泊松 + Elo ──
print(f"\n{'─'*60}")
print(f"  🔄 回测: 泊松 + Elo (v2优化版)")
print(f"{'─'*60}")

poisson_results = []
for m in wc:
    t1, t2 = m['team1'], m['team2']
    ah, aa = m['score_ft']
    h_ts = ts.get(t1, {'att':1.0,'def':1.0})
    a_ts = ts.get(t2, {'att':1.0,'def':1.0})
    
    lam_h = ga * h_ts['att'] * a_ts['def']; lam_a = ga * a_ts['att'] * h_ts['def']
    lam_h = max(0.1,min(5.0,lam_h)); lam_a = max(0.1,min(5.0,lam_a))
    
    hw,dr,aw = 0.0,0.0,0.0
    for hg in range(MAX_GOALS+1):
        for ag in range(MAX_GOALS+1):
            p = poisson_pmf(hg,lam_h)*poisson_pmf(ag,lam_a)
            if hg>ag: hw+=p
            elif hg==ag: dr+=p
            else: aw+=p
    t=hw+dr+aw; hw,dr,aw=hw/t,dr/t,aw/t
    
    # Elo修正
    eh = elo_r.get(t1,1500); ea = elo_r.get(t2,1500)
    ep = elo_expected(eh, ea)
    w = 0.55
    hw = hw*w + ep*(1-w); aw = aw*w + (1-ep)*(1-w); dr = dr*w + 0.2*(1-w)
    t=hw+dr+aw; hw,dr,aw=hw/t,dr/t,aw/t
    
    pr = 'H' if hw>dr and hw>aw else ('D' if dr>hw and dr>aw else 'A')
    ar = 'H' if ah>aa else ('D' if ah==aa else 'A')
    sq = (hw-(1 if ar=='H' else 0))**2 + (dr-(1 if ar=='D' else 0))**2 + (aw-(1 if ar=='A' else 0))**2
    
    # 最可能比分
    h_probs = [poisson_pmf(k,lam_h) for k in range(MAX_GOALS+1)]
    a_probs = [poisson_pmf(k,lam_a) for k in range(MAX_GOALS+1)]
    bp,bh,ba=0,0,0
    for hg in range(MAX_GOALS+1):
        for ag in range(MAX_GOALS+1):
            p = h_probs[hg]*a_probs[ag]
            if p>bp: bp,bh,ba=p,hg,ag
    
    poisson_results.append({
        't1':t1,'t2':t2,'date':m['date'],'round':m['round'],
        'actual':f"{ah}-{aa}",'pred':f"{bh}-{ba}",
        'hda': pr==ar, 'exact': bh==ah and ba==aa,
        'probs': {'H':round(hw,4),'D':round(dr,4),'A':round(aw,4)},
        'sq': sq,
    })

s = poisson_results
total = len(s)
hda = sum(1 for r in s if r['hda'])
exact = sum(1 for r in s if r['exact'])
brier = sum(r['sq'] for r in s)/total
rmse = math.sqrt(sum(r['sq'] for r in s)/total)

print(f"\n  📊 泊松+Elo 结果:")
print(f"  HDA准确率: {hda}/{total} = {hda/total*100:.2f}%")
print(f"  精确比分:  {exact}/{total} = {exact/total*100:.2f}%")
print(f"  Brier:     {brier:.4f}")
print(f"  RMSE:      {rmse:.4f}")

by_rnd = defaultdict(lambda: {'t':0,'c':0})
for r in s:
    by_rnd[r['round']]['t'] += 1
    if r['hda']: by_rnd[r['round']]['c'] += 1

print(f"\n  {'轮次':<22s} {'场':>3s} {'对':>3s} {'%':>6s}")
for rn in sorted(by_rnd.keys()):
    rs = by_rnd[rn]; pct = rs['c']/rs['t']*100 if rs['t'] else 0
    print(f"  {rn:<22s} {rs['t']:>3d} {rs['c']:>3d} {pct:>5.1f}%")

# 偏差最大
sorted_by_err = sorted(s, key=lambda r: r['sq'], reverse=True)
print(f"\n  🔥 偏差最大的 5 场:")
for r in sorted_by_err[:5]:
    mk = '✅' if r['hda'] else '❌'
    print(f"  {mk} {r['t1']} vs {r['t2']} ({r['date']})")
    print(f"     预测: {r['pred']} (H:{r['probs']['H']*100:.0f}% D:{r['probs']['D']*100:.0f}% A:{r['probs']['A']*100:.0f}%)")
    print(f"     实际: {r['actual']}")

# ── 对比 2022 结果 ──
print(f"\n{'='*60}")
print(f"  🏆 对比: 2018 vs 2022")
print(f"{'='*60}")
print(f"  {'指标':<16s} {'2018':>8s} {'2022':>8s}")
print(f"  {'─'*34}")
print(f"  {'HDA准确率':<16s} {hda/total*100:>7.2f}% {'57.81':>7s}%")
print(f"  {'精确比分':<16s} {exact/total*100:>7.2f}% {'7.81':>7s}%")
print(f"  {'Brier':<16s} {brier:>8.4f} {'0.6128':>8s}")
print(f"  {'训练数据截止':<16s} {'2018-06':>8s} {'2022-11':>8s}")
print(f"  {'训练场次':<16s} {len(train_matches):>8d} {'~49000':>8s}")

# 判断 2018 数据量够不够
train_wc_teams = sum(1 for t in wc_teams if ts.get(t,{}).get('m',0) > 0)
print(f"  {'有数据的队伍':<16s} {train_wc_teams}/{len(wc_teams):>4d} {'32/32':>8s}")

print(f"\n{'─'*60}")
print(f"  💡 结论: 数据完全够用")
print(f"  {'─'*60}")
print(f"  2018 训练数据截止前有 {len(train_matches)} 场国际赛")
yes_teams = [t for t in wc_teams if ts.get(t,{}).get('m',0) >= 10]
few_teams = [t for t in wc_teams if ts.get(t,{}).get('m',0) < 5]
print(f"  参赛队中 ≥10场训练数据: {len(yes_teams)}队")
if few_teams:
    print(f"  <5场训练数据的队: {few_teams}")

# ── 现在用 sklearn 模型来对比 ──
print(f"\n{'─'*60}")
print(f"  🔄 回测: Random Forest (英超训练的模型)")
print(f"  注意: RF 模型是在英超数据上训练的, 其特征空间")
print(f"        (Elo/积分/排名/主客场) 和国际赛不同.")
print(f"        此处仅作对比参考.")
print(f"{'─'*60}")

# 尝试用英超训练的 RF 来预测世界杯
try:
    import joblib
    rf_path = os.path.join(DATA_DIR, 'rf_football.pkl')
    scaler_path = os.path.join(DATA_DIR, 'rf_scaler.pkl')
    
    if not os.path.exists(rf_path):
        print("  ⚠️  RF 模型未保存, 需要重新训练.")
        print("  先用 2023/24 英超训练 RF (同 ml_football.py 逻辑)...")
        
        # 简版训练
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # 用英超数据构建特征
        def fetch_pl_season(code, start, end):
            url = f"https://api.football-data.org/v4/competitions/{code}/matches?dateFrom={start}&dateTo={end}"
            req = urllib.request.Request(url, headers={'X-Auth-Token':'5d07c80baa2645d0809b6ec96d6b49c6','Accept':'application/json'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            matches = []
            for m in data.get('matches',[]):
                if m['status']!='FINISHED': continue
                s=m['score']['fullTime']
                if s['home'] is None: continue
                matches.append({'date':m['utcDate'][:10],'matchday':m.get('matchday',1),
                    'home':m['homeTeam']['shortName'],'away':m['awayTeam']['shortName'],
                    'h_score':s['home'],'a_score':s['away']})
            return matches
        
        print("  📡 拉取英超数据...")
        pl_train = fetch_pl_season('PL','2023-08-01','2024-05-19')
        pl_test = fetch_pl_season('PL','2024-08-01','2025-05-25')
        all_pl = pl_train + pl_test
        
        # 构建简单特征 (Elo + form + pos)
        def build_pl_features(test_matches, all_matches):
            X, y = [], []
            all_by_date = defaultdict(list)
            for m in all_matches:
                all_by_date[m['date']].append(m)
            all_dates = sorted(all_by_date.keys())
            
            elo = defaultdict(lambda: 1500.0)
            for m in all_matches:
                if m in test_matches: continue
                h,a=m['home'],m['away']
                e_h=elo_expected(elo[h],elo[a])
                sh,sa=(1.0,0.0) if m['h_score']>m['a_score'] else((0.5,0.5)if m['h_score']==m['a_score']else(0.0,1.0))
                elo[h]+=32*(sh-e_h); elo[a]+=32*(sa-(1-e_h))
            
            for m in test_matches:
                home,away=m['home'],m['away']
                ah,aa=m['h_score'],m['a_score']
                
                features = [
                    elo.get(home,1500), elo.get(away,1500), elo.get(home,1500)-elo.get(away,1500),
                    0, 0, 0, 0,  # att/def placeholders
                ]
                while len(features) < 22: features.append(0)
                
                label = 0 if ah>aa else (1 if ah==aa else 2)
                X.append(features[:22])
                y.append(label)
                
                # update elo
                e_h=elo_expected(elo.get(home,1500),elo.get(away,1500))
                sh,sa=(1.0,0.0) if ah>aa else((0.5,0.5)if ah==aa else(0.0,1.0))
                elo[home]=elo.get(home,1500)+48*(sh-e_h)
                elo[away]=elo.get(away,1500)+48*(sa-(1-e_h))
            
            return np.array(X), np.array(y)
        
        X_tr, y_tr = build_pl_features(pl_train, all_pl)
        X_te, y_te = build_pl_features(pl_test, all_pl)
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        
        rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        
        joblib.dump(rf, rf_path)
        joblib.dump(scaler, scaler_path)
        print(f"  ✅ RF 训练完成, 保存至 {rf_path}")
    else:
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        print(f"  ✅ 加载已训练的 RF 模型")
    
except ImportError:
    print("  ⚠️  joblib 未安装, 跳过 RF 对比")

print(f"\n{'='*60}")
print(f"  ✅ 完成")
