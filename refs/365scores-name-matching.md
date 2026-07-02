# 365scores 名称匹配工作流

## 问题背景

500.com 返回中文队名（葡萄牙、尼日利亚），365scores 返回英文队名（Portugal、Nigeria）。直接匹配会失败。

## 解决方案

### 1. 构建 map 时标准化

```python
def build_365_map(games):
    from team_name_normalizer import normalize_match_pair
    mapping = {}
    for g in games:
        h, a = normalize_match_pair(g.get('home', ''), g.get('away', ''))
        if h and a:
            mapping[(h, a)] = g
            mapping[(a, h)] = g  # 双向映射
    return mapping
```

### 2. 查找时标准化

```python
# ❌ 错误：直接用中文名查找
score_meta = score365_map.get((home, away))  # → None

# ✅ 正确：先标准化再查找
from team_name_normalizer import normalize_match_pair
h_norm, a_norm = normalize_match_pair(home, away)
score_meta = score365_map.get((h_norm, a_norm))  # → 找到数据
```

## team_name_normalizer 关键映射

```python
TEAM_ALIASES = {
    '葡萄牙': 'Portugal',
    '尼日利亚': 'Nigeria',
    '巴西': 'Brazil',
    '阿根廷': 'Argentina',
    # ... 完整列表见 /root/team_name_normalizer.py
}
```

## form_state.json 队名匹配断裂 (2026-06-13 新增)

同样的标准化问题也影响 form_state.json 查找:

```python
# form_state.json 用英文存储: {"Senegal": [...], "France": [...]}
# 500.com 返回中文: "塞内加尔" / "法国"
# normalize_match_pair() 对部分名返回中文原样:
#   法国 → France ✓
#   塞内加尔 → 塞内加尔 ✗ (form_state.json 中是 "Senegal")
```

**受影响队名**:
- 芬超: 雅罗、赫尔辛基、瓦萨、库奥皮奥、玛丽港、拉赫蒂、塞伊奈
- 非洲国家队: 塞内加尔、科特迪瓦、佛得角、埃及、库拉索

**验证**:
```python
python3 -c "
import json
with open('/root/data/form_state.json') as f:
    fs = json.load(f)
# 搜索 Senegal
for k in fs:
    if 'egal' in k.lower(): print(f'Found: {k}')
# → Found: Senegal (但 normalize_match_pair('塞内加尔') 返回 '塞内加尔')
"
```

**已修复的影响链**: form_state 查找失败 → `_try_hybrid_predict()` 返回 None → 影子模型也跳过 → 45% 场次无 pred30 数据. 修复后覆盖率提升至 86%。

**修复 (2026-06-13 已应用)**: `_resolve_name()` 函数 (daily_jczq.py line 38-44) 通过 `team_name_mapping.json` 做中→英映射。在 `_try_hybrid_predict` 的 form_state 查找和 DC/Elo 调用前应用。修复后覆盖率从 55% → 86%。

```python
# daily_jczq.py line 38-44
_TEAM_NAME_MAP = None
def _resolve_name(name):
    global _TEAM_NAME_MAP
    if _TEAM_NAME_MAP is None:
        with open('/root/data/team_name_mapping.json') as _f:
            _TEAM_NAME_MAP = json.load(_f)
    return _TEAM_NAME_MAP.get(name, name)

# 在 _try_hybrid_predict 中使用:
h, a = normalize_match_pair(home, away)
h, a = _resolve_name(h), _resolve_name(a)  # 中→英兜底
```

**仍会降级的场景**: 芬超球队 (雅罗/赫尔辛基/瓦萨等) 在 form_state.json 和 form_club.json 中均无数据, 正确降级到 market_fallback。

## score365_map 查找修复 (2026-06-13)

`score365_map` 的两个查找点也存在同样的中英文断裂:

```python
# ❌ 修复前 (use_500_only 分支 line ~2053)
score_meta = score365_map.get((h_norm, a_norm))  # 塞内加尔 ≠ Senegal → None

# ✅ 修复后
score_meta = score365_map.get((_resolve_name(h_norm), _resolve_name(a_norm))) or score365_map.get((h_norm, a_norm))
```

同样修复了联赛分支 (line ~2107)。修复前 365scores 匹配率: 1/6 (仅巴西/摩洛哥等 normalize 覆盖到的), 修复后: 3/6 (法国/塞内加尔等也能匹配)。

## 修复位置

1. `/root/daily_jczq.py` `_resolve_name()` 函数 (line 38-44)
2. `_try_hybrid_predict()` 中 form_state/DC/Elo 查找 (line ~364-368)
3. `main()` 中 `score365_map` 查找 (use_500_only 分支 line ~2053 + 联赛分支 line ~2107)
4. `/root/scripts/run_single_match.py` 查找逻辑 (待同步)

## 测试用例

- 葡萄牙 vs 尼日利亚: 365scores 返回 "Portugal vs Nigeria"
- 巴西 vs 摩洛哥: 365scores 返回 "Brazil vs Morocco"
- 塞内加尔 vs 法国: form_state.json 查找需要英文名 "Senegal"/"France"
- 任意中文队名 → 英文标准化 → map 查找
