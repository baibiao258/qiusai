#!/usr/bin/env python3
"""
dashboard_generator.py — 每日回测可视化看板
==========================================
读取 backtest_results.json + predictions_log.csv,
生成静态 HTML 看板 dashboard.html.

核心图表:
  1. Brier Score / RPS 趋势折线图
  2. 概率校准曲线 (Reliability Diagram)
  3. 联赛/赛事维度准确率柱状图
  4. 模型对比 (俱乐部 vs 国际赛)

用法:
  python3 dashboard_generator.py
  → 生成 /root/data/dashboard.html
"""
import json
import csv
import os
from datetime import datetime
from collections import defaultdict

DATA_DIR = '/root/data'
OUTPUT = os.path.join(DATA_DIR, 'dashboard.html')


def load_backtest():
    path = os.path.join(DATA_DIR, 'backtest_results.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def load_predictions():
    path = os.path.join(DATA_DIR, 'predictions_log.csv')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_backtest_details():
    path = os.path.join(DATA_DIR, 'backtest_details.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def compute_calibration(preds):
    """计算校准度数据 (10 个桶)."""
    bins = defaultdict(lambda: {'pred': 0, 'actual': 0, 'count': 0})
    for p in preds:
        if p.get('checked') != '1':
            continue
        try:
            pred_h = float(p['pred_h']) / 100
            pred_d = float(p['pred_d']) / 100
            pred_a = float(p['pred_a']) / 100
            actual = p.get('actual_hda', '')
        except:
            continue

        for prob, outcome in [(pred_h, actual == 'H'), (pred_d, actual == 'D'), (pred_a, actual == 'A')]:
            bucket = int(prob * 10)  # 0-9
            if bucket >= 10:
                bucket = 9
            bins[bucket]['pred'] += prob
            bins[bucket]['actual'] += 1 if outcome else 0
            bins[bucket]['count'] += 1

    result = []
    for b in sorted(bins.keys()):
        if bins[b]['count'] > 0:
            result.append({
                'bin': b * 10,
                'pred_avg': bins[b]['pred'] / bins[b]['count'],
                'actual_rate': bins[b]['actual'] / bins[b]['count'],
                'count': bins[b]['count'],
            })
    return result


def compute_metrics_by_source(details):
    """按模型来源分组计算指标."""
    sources = defaultdict(list)
    for d in details:
        src = d.get('source', 'unknown')
        sources[src].append(d)
    return sources


def generate_html(backtest, predictions, details, calibration):
    """生成 HTML 看板."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # 准备图表数据
    brier_trend = json.dumps([r.get('brier', 0) for r in backtest[-20:]])
    rps_trend = json.dumps([r.get('rps', 0) for r in backtest[-20:]])
    labels = json.dumps([r.get('timestamp', '')[:10] for r in backtest[-20:]])

    cal_pred = json.dumps([c['pred_avg'] for c in calibration])
    cal_actual = json.dumps([c['actual_rate'] for c in calibration])
    cal_count = json.dumps([c['count'] for c in calibration])

    # 按来源分组
    sources = compute_metrics_by_source(details)
    source_labels = json.dumps(list(sources.keys()))
    source_brier = json.dumps([sum(d['brier'] for d in v) / max(len(v), 1) for v in sources.values()])
    source_acc = json.dumps([sum(d['accuracy'] for d in v) / max(len(v), 1) * 100 for v in sources.values()])

    total_preds = len(predictions)
    checked = sum(1 for p in predictions if p.get('checked') == '1')
    latest_brier = backtest[-1].get('brier', 'N/A') if backtest else 'N/A'
    latest_acc = backtest[-1].get('accuracy', 'N/A') if backtest else 'N/A'
    if isinstance(latest_acc, float):
        latest_acc = f"{latest_acc * 100:.1f}%"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚽ 预测模型 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; }}
  .header {{ text-align: center; padding: 30px 0; }}
  .header h1 {{ font-size: 2rem; color: #38bdf8; }}
  .header p {{ color: #94a3b8; margin-top: 8px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px; margin: 20px 0; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 20px;
           border: 1px solid #334155; }}
  .card .label {{ color: #94a3b8; font-size: 0.85rem; }}
  .card .value {{ font-size: 2rem; font-weight: 700; margin-top: 8px; }}
  .card .value.green {{ color: #22c55e; }}
  .card .value.blue {{ color: #38bdf8; }}
  .card .value.yellow {{ color: #facc15; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
                gap: 20px; margin: 20px 0; }}
  .chart-box {{ background: #1e293b; border-radius: 12px; padding: 20px;
               border: 1px solid #334155; }}
  .chart-box h3 {{ color: #cbd5e1; margin-bottom: 15px; font-size: 1rem; }}
  canvas {{ max-height: 300px; }}
</style>
</head>
<body>
<div class="header">
  <h1>⚽ 预测模型 Dashboard</h1>
  <p>更新时间: {now}</p>
</div>

<div class="cards">
  <div class="card">
    <div class="label">最新 Brier Score</div>
    <div class="value green">{latest_brier}</div>
  </div>
  <div class="card">
    <div class="label">最新准确率</div>
    <div class="value blue">{latest_acc}</div>
  </div>
  <div class="card">
    <div class="label">总预测数</div>
    <div class="value yellow">{total_preds}</div>
  </div>
  <div class="card">
    <div class="label">已核验</div>
    <div class="value green">{checked}</div>
  </div>
</div>

<div class="chart-grid">
  <div class="chart-box">
    <h3>📈 Brier Score 趋势</h3>
    <canvas id="brierChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>📈 RPS 趋势</h3>
    <canvas id="rpsChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>🎯 概率校准曲线</h3>
    <canvas id="calChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>🏟️ 模型来源对比</h3>
    <canvas id="sourceChart"></canvas>
  </div>
</div>

<script>
const chartDefaults = {{
  responsive: true,
  plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
    y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#334155' }} }}
  }}
}};

// Brier Trend
new Chart(document.getElementById('brierChart'), {{
  type: 'line',
  data: {{
    labels: {labels},
    datasets: [{{ label: 'Brier Score', data: {brier_trend},
      borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)',
      fill: true, tension: 0.3 }}]
  }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales,
    y: {{ ...chartDefaults.scales.y, min: 0, max: 1 }} }} }}
}});

// RPS Trend
new Chart(document.getElementById('rpsChart'), {{
  type: 'line',
  data: {{
    labels: {labels},
    datasets: [{{ label: 'RPS', data: {rps_trend},
      borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)',
      fill: true, tension: 0.3 }}]
  }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales,
    y: {{ ...chartDefaults.scales.y, min: 0, max: 0.5 }} }} }}
}});

// Calibration
new Chart(document.getElementById('calChart'), {{
  type: 'scatter',
  data: {{
    datasets: [
      {{ label: '模型', data: {cal_pred}.map((p,i) => ({{x: p, y: {cal_actual}[i]}})),
        backgroundColor: '#38bdf8', pointRadius: 6 }},
      {{ label: '完美校准', data: [{{x:0,y:0}},{{x:1,y:1}}],
        type: 'line', borderColor: '#475569', borderDash: [5,5], pointRadius: 0 }}
    ]
  }},
  options: {{ ...chartDefaults, scales: {{
    x: {{ ...chartDefaults.scales.x, title: {{ display: true, text: '预测概率', color: '#94a3b8' }} }},
    y: {{ ...chartDefaults.scales.y, title: {{ display: true, text: '实际频率', color: '#94a3b8' }}, min: 0, max: 1 }}
  }} }}
}});

// Source Comparison
new Chart(document.getElementById('sourceChart'), {{
  type: 'bar',
  data: {{
    labels: {source_labels},
    datasets: [
      {{ label: 'Brier (越低越好)', data: {source_brier}, backgroundColor: '#22c55e' }},
    ]
  }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales,
    y: {{ ...chartDefaults.scales.y, min: 0 }} }} }}
}});
</script>
</body>
</html>"""
    return html


def main():
    print("📊 生成 Dashboard...")
    backtest = load_backtest()
    predictions = load_predictions()
    details = load_backtest_details()
    calibration = compute_calibration(predictions)

    print(f"  回测记录: {len(backtest)} 条")
    print(f"  预测记录: {len(predictions)} 条 (已核验: {sum(1 for p in predictions if p.get('checked')=='1')})")
    print(f"  校准桶: {len(calibration)} 个")

    html = generate_html(backtest, predictions, details, calibration)
    with open(OUTPUT, 'w') as f:
        f.write(html)

    print(f"  ✅ 已生成 {OUTPUT}")
    print(f"  打开: file://{OUTPUT}")


if __name__ == '__main__':
    main()
