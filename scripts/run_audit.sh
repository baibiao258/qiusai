#!/usr/bin/env bash
set -Eeuo pipefail

TS="$(date +%F_%H%M%S)"
LOG="/root/data/run_wc2026_audit_${TS}.log"
BUNDLE="/root/data/wc2026_audit_bundle_${TS}.tar.gz"

echo "[INFO] TS=${TS}"
echo "[INFO] LOG=${LOG}"

echo "[STEP 1] Preflight files check"
for p in \
  /root/wc_2026_final.py \
  /root/run_group72_repro.py \
  /root/data/international_results.json \
  /root/data/theodds_api_data.json \
  /root/data/2026_groups.json
  do
    [[ -s "$p" ]] || { echo "[FAIL] missing or empty: $p"; exit 1; }
    stat --printf="[OK] %n size=%s mtime=%y\n" "$p"
  done


echo "[STEP 2] Train + backtest + champion simulation"
cd /root
python3 /root/wc_2026_final.py | tee "$LOG"

# keep canonical log path for downstream tools
cp -f "$LOG" /root/data/run_wc2026_audit.log

echo "[STEP 3] Gate checks from log"
grep -q "赛事过滤: 49257 → 4944" "$LOG" || { echo "[FAIL] filter count mismatch"; exit 1; }
grep -q "DC: ρ=0.2500" "$LOG" || { echo "[FAIL] rho mismatch"; exit 1; }
grep -q "验证: Acc=" "$LOG" || { echo "[FAIL] validation metrics missing"; exit 1; }
grep -q "Brier=" "$LOG" || { echo "[FAIL] brier missing"; exit 1; }
grep -q "2022 WC 回测" "$LOG" || { echo "[FAIL] wc2022 backtest missing"; exit 1; }
grep -q "保存: /root/data/final_results.json" "$LOG" || { echo "[FAIL] final_results not saved"; exit 1; }

python3 - <<'PY'
import re, sys
log_path='/root/data/run_wc2026_audit.log'
text=open(log_path,'r',encoding='utf-8').read()

m = re.search(r"验证: Acc=([0-9.]+)%\s+NLL=([0-9.]+)\s+Brier=([0-9.]+)", text)
if not m:
    print('[FAIL] cannot parse validation metrics')
    sys.exit(1)
acc=float(m.group(1)); nll=float(m.group(2)); brier=float(m.group(3))
print(f'[METRIC] val Acc={acc:.2f}% NLL={nll:.4f} Brier={brier:.4f}')
if brier > 0.465:
    print(f'[FAIL] val brier too high: {brier:.4f} > 0.4650')
    sys.exit(1)

m_dc = re.search(r"DC alone \|\s+[0-9.]+% Brier=([0-9.]+)", text)
m_hy = re.search(r"Hybrid \(20\+3\) \|\s+[0-9.]+% Brier=([0-9.]+)", text)
if not (m_dc and m_hy):
    print('[FAIL] cannot parse WC2022 brier metrics')
    sys.exit(1)
dc=float(m_dc.group(1)); hy=float(m_hy.group(1))
print(f'[METRIC] wc2022 DC={dc:.4f} Hybrid={hy:.4f}')
if hy > dc:
    print(f'[FAIL] hybrid brier worse than dc: {hy:.4f} > {dc:.4f}')
    sys.exit(1)

if 'Traceback' in text:
    print('[FAIL] traceback found in log')
    sys.exit(1)

print('[OK] log gates passed')
PY


echo "[STEP 4] Reproduce 72 group matches"
python3 /root/run_group72_repro.py | tee /tmp/run_group72_repro_${TS}.json

python3 - <<'PY'
import json, sys
p='/root/data/group_stage_predictions.json'
d=json.load(open(p,'r',encoding='utf-8'))
if d.get('total_matches') != 72:
    print(f"[FAIL] total_matches != 72: {d.get('total_matches')}")
    sys.exit(1)
for i,row in enumerate(d.get('predictions', []), 1):
    s=row['prob_home']+row['prob_draw']+row['prob_away']
    if abs(s-1.0) > 0.01:
        print(f"[FAIL] prob sum drift at row {i}: {s}")
        sys.exit(1)
print('[OK] 72-match file checks passed')
PY


echo "[STEP 5] Ensure scoreline Top5 exists"
[[ -s /root/data/group_stage_scoreline_top5.json ]] || { echo "[FAIL] missing /root/data/group_stage_scoreline_top5.json"; exit 1; }


echo "[STEP 6] Build audit bundle"
tar -czf "$BUNDLE" \
  /root/data/run_wc2026_audit.log \
  /root/data/final_results.json \
  /root/data/group_stage_predictions.json \
  /root/data/group_stage_predictions.txt \
  /root/data/group_stage_scoreline_top5.json \
  /root/data/2026_groups.json \
  /root/data/theodds_api_data.json

[[ -s "$BUNDLE" ]] || { echo "[FAIL] bundle not created"; exit 1; }
ls -lh "$BUNDLE"

echo "[DONE] audit pipeline success"
echo "[OUT] $BUNDLE"
