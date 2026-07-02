#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_training_data.py
从 training_data_with_odds.json + form_state.json 生成训练特征表
"""
import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/root/data")
INPUT_JSON = DATA_DIR / "training_data_with_odds.json"
FORM_STATE = DATA_DIR / "form_state.json"
OUTPUT_CSV = DATA_DIR / "training_data.csv"

def load_form_state():
    with open(FORM_STATE, encoding="utf-8") as f:
        return json.load(f)

def get_form(team, fs, n=5):
    entries = fs.get(team, [])[-n:]
    if not entries:
        return [0.0, 0.0, 0.0]
    wins = sum(1 for e in entries if e[0] > e[1])
    gf = sum(e[0] for e in entries) / len(entries)
    ga = sum(e[1] for e in entries) / len(entries)
    return [wins / len(entries), gf, ga]

def build():
    with open(INPUT_JSON) as f:
        data = json.load(f)
    fs = load_form_state()
    
    rows = []
    for r in data:
        home_en = r.get('home_en', '')
        away_en = r.get('away_en', '')
        label = int(r['spf_result']) if r.get('spf_result') in ('0','1','3') else None
        if label is None:
            continue
        
        market_odds = float(r.get('market_odds', 0) or 0)
        if market_odds <= 1:
            # 用 spf_sp 兜底
            market_odds = float(r.get('spf_sp', 0) or 0)
        
        fh = get_form(home_en, fs)
        fa = get_form(away_en, fs)
        
        rows.append({
            'date': r['date'],
            'home_en': home_en,
            'away_en': away_en,
            'market_odds': market_odds,
            'form_home_win': fh[0],
            'form_home_gf': fh[1],
            'form_home_ga': fh[2],
            'form_away_win': fa[0],
            'form_away_gf': fa[1],
            'form_away_ga': fa[2],
            'label': label
        })
    
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ {OUTPUT_CSV}  {len(out)} 行")
    print(f"  label 分布: {out['label'].value_counts().sort_index().to_dict()}")
    print(f"  market_odds>0: {(out['market_odds']>0).sum()}")
    print(f"  特征列: {[c for c in out.columns if c != 'label']}")

if __name__ == "__main__":
    build()
