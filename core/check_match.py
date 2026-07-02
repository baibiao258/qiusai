import csv, json, os, glob

# Load all results
results = {}
for f in sorted(glob.glob('/root/data/results/2026-06-*.json')):
    date = os.path.basename(f).replace('.json', '')
    with open(f) as fh:
        data = json.load(fh)
    for m in data:
        code = m.get('code', '')
        if code and m.get('hda_result'):
            results[code] = {**m, 'result_date': date}

print(f'{len(results)} results loaded')

# Load predictions
with open('/root/data/predictions_log.csv', newline='') as f:
    reader = csv.DictReader(f)
    preds = list(reader)

print(f'{len(preds)} predictions loaded')
print()

# Cross-reference
matched = 0
for p in preds:
    code = p.get('code', '')
    if code in results:
        r = results[code]
        print(f"MATCH: {code} {p['home_cn']} vs {p['away_cn']} -> {r['score_full']} ({r['hda_result']}) pred={p.get('pred_spf_pick','?')} odds={p.get('odds_h','?')}/{p.get('odds_d','?')}/{p.get('odds_a','?')}")
        matched += 1

if matched == 0:
    print('NO CROSS-MATCHES: predictions_log codes:')
    for p in preds:
        print(f"  {p['code']} {p['home_cn']} vs {p['away_cn']}")
    print('Result codes:')
    for c in sorted(results.keys()):
        r = results[c]
        print(f"  {c} {r['home']} vs {r['away']}")

print(f'\nTotal matched: {matched}/{len(preds)}')