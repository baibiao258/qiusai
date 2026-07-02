# v6 Backtest: Phase One-Hot + Dynamic Elo K-Factor

## 背景
v1-v5 回测已验证：29维基线 (XGB29=52.2%, Stack=52.7%, 让球-1 DC=59.4%)。v3/v4/v5 尝试更复杂 stacking 全失败（小样本下复杂 meta 全反效果）。

v6 方向：**现有数据可用的结构化改进**，不引入外部赔率数据。

## v6 三个变体

| 变体 | 特征改动 | Elo K因子 |
|------|----------|-----------|
| v6_a | +5维 phase one-hot (R16/QF/SF/Final) | 固定 K=30 |
| v6_b | +5维 phase one-hot | 动态 K: group=30, knockout=60 |
| v6_c | +5维 phase one-hot | 动态 K: group=20, knockout=80 |

## 实现细节

### Phase One-Hot (5维)
```python
PHASE_MAP = {'group': 0, 'R16': 1, 'QF': 2, 'SF': 3, 'Final': 4}  # 必须模块级
# 训练/测试数据每场比赛都有 phase 字段 (来自 wc_historical_matches.json 的 round/group)
# 映射: group→0, Round of 16→1, Quarter-finals→2, Semi-finals→3, Final→4
# one-hot 编码: is_R16, is_QF, is_SF, is_Final (group 为基准全 0)
```

### 动态 K 因子 Elo
```python
def compute_elo_v6(matches, K_group=30, K_ko=60):
    for m in sorted(matches, key=lambda x: x['date']):
        K = K_ko if m.get('phase', 'group') != 'group' else K_group
        # 标准 Elo 更新
    return elo  # return 必须在 for 循环外！
```

**动机**：锦标赛内淘汰赛比赛信息量更大（单场决胜负），Elo 应更快适应。文献支持淘汰赛 K≈2×小组赛 K。

## 已知 Bug 修复记录 (2026-06-06)

首次运行产出全部 `nan%`。排查发现 3 个静默失败 bug：

### Bug 1: `dc.lambdas` 属性不存在
`DixonColes` 类无 `lambdas` dict 属性。获取 λ 的唯一方法是 `dc.predict_lambda(home, away, neutral=True)` → `(λ_home, λ_away)`。
```python
# ❌ 错
lh = dc.lambdas.get(h, 1.3)
# ✅ 对
lh, la = dc.predict_lambda(h, a, neutral=m.get('neutral', True))
```

### Bug 2: 辅助函数/变量作用域不可达
`elo_odds()` 定义在 `build_features_v6()` 内部（嵌套函数），但 `backtest_year()` 的测试预测循环也调用了它——无法访问 → `NameError`。
`phase_map` dict 也定义在 `build_features_v6()` 内部，但测试循环同样引用 → 同一问题。
```python
# ❌ 错: 嵌套函数在另一个函数中不可见
# ✅ 对: 搬到模块级
PHASE_MAP = {'group': 0, 'R16': 1, 'QF': 2, 'SF': 3, 'Final': 4}
def elo_odds(eh, ea): ...
```

### Bug 3: 异常处理垫片维度不匹配
测试循环的 `except` 分支用 `[0]*40` 填充失败样本，但实际特征维度是 42（32 base + 6 extra + 4 phase）→ XGBoost `predict()` 报 `ValueError: Feature shape mismatch, expected: 42, got 40`。
```python
# ❌ 错
test_X.append([0]*40)
# ✅ 对
test_X.append([0]*42)
```

### 根本原因
3 个异常全部被 `except Exception` 静默吃掉，导致：
- 训练阶段：`valid=False` → `X_train[valid]` 空 → `len(X_train) < 200` → `return None`
- 测试阶段：`test_X` 形状不匹配 → XGBoost 崩溃

**教训**：回测脚本中 `except Exception` 必须至少 `print(e)` 或 `traceback.print_exc()`，不能完全沉默。否则 bug 的表现为"全程无报错但 nan%"。

## 运行命令
```bash
python3 /root/wc_10edition_backtest_v6.py
# 输出: /root/data/v6_results.json
# 日志: /tmp/v6_fixed2_run.log
```

## 最终结果 (2026-06-07 结项)

**结论: 全部三个变体均低于 v3 基线，项目放弃。v6 相关文件已全部删除。**

| 版本 | XGB Acc | Stack Acc | 让球-1 DC | vs v3 baseline |
|------|---------|-----------|-----------|----------------|
| v3   | 52.2%   | 52.7%     | 59.4%     | baseline |
| v6_a | ~48-52% | ~49-53%   | ~57%      | ↓ (多届低于基线) |
| v6_b | worse   | worse     | worse     | ↓ 动态 K 无正向作用 |
| v6_c | worse   | worse     | worse     | ↓ |

### 失败原因

1. **Phase one-hot 信号被 Elo/DC 吸收**：进入淘汰赛的球队 Elo 普遍更高，phase 信息已在 Elo 评分中隐式编码，one-hot 无新增信号
2. **动态 K 因子在跨届回测中有害**：不同届的淘汰赛表现差异大（1990 弱队爆冷 vs 2018 强队统治），激进更新放大噪音
3. **早期届样本太少**：1986 仅 16 场淘汰赛，phase one-hot 过拟合

### 用户指令
"改回去吧，不要的删掉" — 保留 v2 基线 (`/root/wc_10edition_backtest.py`)，v3-v6 全部删除。

### 启示
结构化特征 (phase, dynamic K) 在小样本跨届场景下信号被 Elo/DC 吸收殆尽。未来方向：外部数据（真实赔率、阵容数据）而非内在特征工程，或接受 52-53% XGB 为模型上限。
