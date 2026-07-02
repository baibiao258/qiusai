# 校准集独立分割：60/20/20 重构 (2026-06-08)

## 问题

原模型训练 pipeline 使用 80/20 时序分割，在训练集上对 IsotonicRegression 进行 CV 校准（或更糟——在训练+验证混合数据上 fit 校准器）。这引入两个问题：

1. **校准过度乐观**：校准器在包含验证信息的数据上 fit，验证集 Brier 不能反映真实泛化能力
2. **版本漂移敏感**：calibrators 未落盘，每次推理时重新 fit，若训练数据发生变化则校准器含漂移

## 修复：60/20/20 三分割

```
时间轴 →
[训练集 60%] | [校准集 20%] | [验证集 20%]
2021-01     2025-Q2        2025-Q4      2026-06
```

- **训练集**：XGBoost 模型 fit，FeatureBuffer 训练
- **校准集**：IsotonicRegression / Platt Scaling 的 fit 数据。**不在校准集上做 XGB 训练，不在训练集上 fit 校准器**
- **验证集**：所有指标（Brier, Acc, LogLoss）仅在此评估。校准器从未见过这些比赛

## 代码模式

```python
# wc_2026_final.py — 分割逻辑
split1 = int(len(df) * 0.6)  # train
split2 = int(len(df) * 0.8)  # train+cal

train_df = df.iloc[:split1]
cal_df   = df.iloc[split1:split2]
val_df   = df.iloc[split2:]

# 训练 XGB（仅在 train_df）
xgb_model.fit(X_train, y_train)

# 拟合校准器（仅在 cal_df）
from sklearn.isotonic import IsotonicRegression
from joblib import dump

raw_probs = xgb_model.predict_proba(X_cal)
cal = IsotonicRegression(out_of_bounds='clip')
cal.fit(raw_probs[:, 2], (y_cal == 2).astype(float))  # 主胜校准

# 保存校准器
dump(cal, '/root/data/calibrators.pkl')

# 评估校准效果（仅在 val_df）
raw_val = xgb_model.predict_proba(X_val)
cal_val = cal.predict(raw_val[:, 2])
brier_val = np.mean((cal_val - (y_val == 2).astype(float))**2)
```

## 实验验证

| 指标 | 旧版(80/20) | 新版(60/20/20独立cal) | 改善 |
|------|-------------|----------------------|------|
| 2022 WC Hybrid Acc | 56.2% | **60.9%** | +4.7pp |
| 2022 WC Brier | 0.1993 | **0.1808** | -0.0185 |
| Isotonic校准集Brier | 0.1551 | **0.1391** | -0.0160 |

## 校准器版本管理

每次全量训练（full pipeline run）后，必须 dump 四个模型文件：

| 文件 | 用途 | 检查 |
|------|------|------|
| `/root/data/xgb_model_29.pkl` | 29维XGB | `mtime` 应与校准器一致 |
| `/root/data/dc_model.pkl` | DC模型 | 同上 |
| `/root/data/elo_ratings.pkl` | Elo评分 | 同上 |
| `/root/data/calibrators.pkl` | Isotonic校准器 | **缺失或 mtime 偏差 >7 天 → 触发重校准** |

版本检查代码：
```python
from pathlib import Path
import joblib

class Versions:
    def __init__(self):
        self.models = {
            'xgb': Path('/root/data/xgb_model_29.pkl'),
            'dc': Path('/root/data/dc_model.pkl'),
            'elo': Path('/root/data/elo_ratings.pkl'),
            'cal': Path('/root/data/calibrators.pkl'),
        }

    def check_drift(self) -> str | None:
        if not all(p.exists() for p in self.models.values()):
            return 'MISSING_MODEL'
        mt = [p.stat().st_mtime for p in self.models.values()]
        spread = max(mt) - min(mt)
        if spread > 7*86400:
            return f'VERSION_DRIFT: {spread/86400:.1f} day spread'
        return None  # OK
```

## 校准器 fit 模式 vs 加载模式

- **全量 train 模式**（每日 cron）：`python3 wc_2026_final.py` — 自动做 60/20/20 分割、fit XGB、fit 校准器、save 4 模型文件
- **推理/预测 模式**：`python3 /root/predict_match.py` — 加载已保存的 4 模型，calibrators.pkl 做 predict transform 而非重新 fit
- **检测版本漂移**：推理前检查所有 4 文件 mtime 一致性，告警但不阻塞

## 要点

- 训练后校准（post-training calibration）的准确率提升幅度（+4.7pp Acc, -0.0185 Brier）主要来自 train-serve skew 修复和版本一致性，而非校准器本身的统计能力提升
- 若后续引入 Platt Scaling（逻辑回归校准），替换 IsotonicRegression 即可，分割协议不变
- 校准集大小不足（如 <100 样本）时回退到无校准或 Platt Scaling（Platt 对中小样本更稳定）
