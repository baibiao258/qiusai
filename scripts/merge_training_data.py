#!/usr/bin/env python3
"""
merge_training_data.py — 合并 kaijiang 和 500.com 两种训练数据源
===============================================================
输出 unified training_data_with_odds.json 给 retrain_xgb_with_odds.py
"""
import json, os, csv

DATA_DIR = "/root/data"
OUTPUT   = f"{DATA_DIR}/training_data_with_odds.json"
KAITRAIN = f"{DATA_DIR}/training_data_with_odds.json"   # 刚由 prepare_training_data.py 生成
HIST_CSV = f"{DATA_DIR}/500_history_backfill.csv"

def load_kaijiang_training():
    """从 prepare_training_data.py 的输出加载 (360条, 2024式)"""
    path = f"{DATA_DIR}/training_data_with_odds.json"
    # 先备份当前
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # 判断格式: 含 spf_sp 的是 kaijiang 格式
        if data and 'spf_sp' in data[0]:
            print(f"📥 Kaijiang 格式: {len(data)} 条")
            return data
    return []

def load_500_training():
    """从 build_training_from_500.py 的输出加载 (150条, 2026式)"""
    path = f"{DATA_DIR}/training_data_500.json"
    if not os.path.exists(path):
        # 从当前 training_data_with_odds.json 判断
        with open(f"{DATA_DIR}/training_data_with_odds.json") as f:
            data = json.load(f)
        # 如果含 nspf_3 的是 500 格式
        if data and 'nspf_3' in data[0]:
            print(f"📥 500格式 (直接从文件): {len(data)} 条")
            return data
        return []
    
    with open(path) as f:
        data = json.load(f)
    print(f"📥 500格式 (从文件): {len(data)} 条")
    return data

def normalize_500_to_kaijiang_format(samples_500):
    """将 trade.500.com 的样本转换为 kaijiang 格式"""
    converted = []
    for s in samples_500:
        # market_implied_prob 从 nspf 计算
        nspf = [s.get('nspf_3', 0), s.get('nspf_1', 0), s.get('nspf_0', 0)]
        if all(o > 0 for o in nspf):
            imp = [1.0 / o for o in nspf]
            total = sum(imp)
            market_prob = imp[0] / total if total > 0 else 0.0
        else:
            market_prob = 0.0

        converted.append({
            'date': s['date'],
            'home_en': s['home_en'],
            'away_en': s['away_en'],
            'tournament': s.get('tournament', ''),
            'spf_result': s['spf_result'],
            'spf_sp': s.get('nspf_3', 0),   # 主胜赔率作为 SP
            'rqspf_sp': max(s.get('spf_3', 0), s.get('spf_1', 0), s.get('spf_0', 0)),
            'handicap': s.get('handicap', 0),
            'ft_h': s['ft_h'],
            'ft_a': s['ft_a'],
            'market_odds': s.get('nspf_3', 0),
            'market_implied_prob': market_prob,
            # stage features: 占位符
            'points_diff': 0.0,
            'rank_diff': 0.333,
            'is_knockout': 0.0,
            'round_num': 0.143,
        })
    return converted


def main():
    # 1. 加载 kaijiang 格式 (刚生成的 360 条)
    kaijiang = load_kaijiang_training()
    
    # 2. 如果当前文件是 kaijiang 格式, 需另找 500 数据
    #    先保存 kaijiang 数据到备份
    kaijiang_only = []
    if kaijiang:
        kaijiang_only = kaijiang
    
    # 3. 从 500.com trade 数据加载
    #    重新运行 build_training_from_500 输出到临时文件
    temp_path = f"{DATA_DIR}/training_data_500_from_trade.json"
    
    # 直接执行 build_training_from_500 并输出到独立文件
    print("🔄 重新从 trade.500.com 拉取 2026 赔率数据...")
    import subprocess
    r = subprocess.run([
        'python3', 'scripts/build_training_from_500.py',
        '--start', '2026-01-01', '--end', '2026-06-13', '--quick'
    ], capture_output=True, timeout=600, cwd='/root')
    print(r.stdout.decode())
    if r.returncode != 0:
        print(f"⚠️ 500 trade 拉取返回码 {r.returncode}")
    
    # 4. 从当前文件加载 500 格式 (build_training_from_500 已覆盖)
    #    但 build_training_from_500.py 覆盖了 training_data_with_odds.json
    #    所以需要重新读取
    with open(f"{DATA_DIR}/training_data_with_odds.json") as f:
        current = json.load(f)
    
    samples_500 = []
    samples_kaijiang = kaijiang_only
    
    # 判断当前文件是什么格式
    if current:
        if 'nspf_3' in current[0]:
            samples_500 = current
            print(f"📥 当前文件是 500 格式: {len(samples_500)} 条")
        elif 'spf_sp' in current[0]:
            samples_kaijiang = current
            print(f"📥 当前文件是 kaijiang 格式: {len(samples_kaijiang)} 条")
    
    # 5. 合并去重
    seen = set()
    merged = []
    
    # 先加 kaijiang
    for s in samples_kaijiang:
        key = (s['date'], s.get('home_en',''), s.get('away_en',''))
        if key not in seen:
            seen.add(key)
            merged.append(s)
    
    # 再加 500 (需转换格式)
    converted_500 = normalize_500_to_kaijiang_format(samples_500)
    for s in converted_500:
        key = (s['date'], s.get('home_en',''), s.get('away_en',''))
        if key not in seen:
            seen.add(key)
            merged.append(s)
    
    merged.sort(key=lambda x: x['date'])
    
    # 6. 写入
    if merged:
        # 备份旧的
        if os.path.exists(OUTPUT):
            os.rename(OUTPUT, OUTPUT + ".bak")
        
        with open(OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        
        # 统计
        years = {}
        for s in merged:
            y = s['date'][:4]
            years[y] = years.get(y, 0) + 1
        
        print(f"\n✅ 合并完成: {len(merged)} 条")
        print(f"   年份分布: {dict(sorted(years.items()))}")
        print(f"   Kaijiang来源: {len(samples_kaijiang)} 条")
        print(f"   500.com来源: {len(converted_500)} 条")
        print(f"   输出: {OUTPUT}")
    else:
        print("❌ 无数据")


if __name__ == '__main__':
    main()
