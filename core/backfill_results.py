#!/usr/bin/env python3
"""
backfill_results.py — 多源赛果回填 + Brier Score 计算
=====================================================

数据源优先级 (默认):
  1. /root/data/results/YYYY-MM-DD.json  (500.com kaijiang, 每日cron生成)
  2. /root/data/historical_kaijiang.csv   (历史开奖CSV, 3248+场)
  3. football-data.org API                (9大联赛, 需API Key)

幂等设计:
  - 只更新 result_status=missing 的记录
  - 已有 actual_hda 的记录永不覆盖
  - 多源结果冲突时标记 result_status=conflict
  - checkpoint 记录最后成功处理的日期, 重启后跳过已处理范围

Brier Score (多分类):
  Brier = (1/r) * Σ_j (I_j - p_j)^2
  其中 r=3 (H/D/A), I_j=指示函数, p_j=模型概率

用法:
  python3 backfill_results.py                        # 回填所有缺失赛果
  python3 backfill_results.py --from-date 2026-06-01 # 从指定日期开始
  python3 backfill_results.py --to-date 2026-06-09   # 到指定日期截止
  python3 backfill_results.py --dry-run              # 只展示不修改
  python3 backfill_results.py --source results,kaijiang  # 指定数据源
  python3 backfill_results.py --stats                # 显示回填统计
"""

import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, date, timedelta
from pathlib import Path

# ─── 配置 ───
BASE_DIR = "/root/data"
PREDICTIONS_LOG = f"{BASE_DIR}/predictions_log.csv"
RESULTS_DIR = f"{BASE_DIR}/results"
KAIJIANG_CSV = f"{BASE_DIR}/historical_kaijiang.csv"
CHECKPOINT_FILE = f"{BASE_DIR}/backfill_checkpoint.json"
FOOTBALL_API_KEY = os.environ.get("FOOTBALL_API_KEY", "5d07c80baa2645d0809b6ec96d6b49c6")
FOOTBALL_API_BASE = "https://api.football-data.org/v4"

# ─── CSV 字段 (与 backtest_jczq.py 同步) ───
RESULT_FIELDS = [
    "actual_score", "actual_ht", "actual_hda", "actual_rq_result",
    "actual_goals", "actual_htft", "brier_spf", "brier_rq",
    "acc_score_top1", "acc_goals_top1", "goals_mae", "acc_htft_top1",
    "result_status",
    "settled_at", "backfill_source", "checked",
]


def load_checkpoint():
    """加载 checkpoint: {last_date: "2026-06-09", stats: {...}}"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"last_date": "", "stats": {"total_filled": 0, "total_conflict": 0}}


def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2, ensure_ascii=False)


def load_log():
    if not os.path.exists(PREDICTIONS_LOG):
        return [], []
    with open(PREDICTIONS_LOG, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
    return rows, fieldnames


def save_log(rows, fieldnames):
    with open(PREDICTIONS_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_fieldnames(fieldnames):
    """确保 fieldnames 包含所有 RESULT_FIELDS, 缺少的追加到末尾"""
    updated = False
    for f in RESULT_FIELDS:
        if f not in fieldnames:
            fieldnames.append(f)
            updated = True
    return fieldnames, updated


def ensure_match_key(row):
    """为缺少 match_key 的记录生成稳定主键"""
    if not row.get("match_key"):
        row["match_key"] = "|".join([
            row.get("date", ""),
            row.get("league", ""),
            row.get("home_cn", ""),
            row.get("away_cn", ""),
            row.get("time", ""),
        ])


# ─── Brier Score 计算 ───

def compute_brier_spf(row):
    """
    多分类 Brier Score: (1/r) * Σ_j (I_j - p_j)^2
    r=3 (H/D/A), p_j 来自 pred_h/pred_d/pred_a (百分比→小数)
    """
    hda = row.get("actual_hda", "")
    if hda not in ("H", "D", "A"):
        return ""

    try:
        pH = float(row.get("pred_h", 0)) / 100.0
        pD = float(row.get("pred_d", 0)) / 100.0
        pA = float(row.get("pred_a", 0)) / 100.0
    except (ValueError, TypeError):
        return ""

    # 指示向量
    iH = 1.0 if hda == "H" else 0.0
    iD = 1.0 if hda == "D" else 0.0
    iA = 1.0 if hda == "A" else 0.0

    brier = ((iH - pH) ** 2 + (iD - pD) ** 2 + (iA - pA) ** 2) / 3.0
    return f"{brier:.4f}"


def compute_brier_rq(row):
    """让球 3分类 Brier Score (让胜/让平/让负), 与 SPF 同构。"""
    rq = row.get("actual_rq_result", "")
    if rq not in ("让胜", "让平", "让负"):
        return ""
    try:
        pW = float(row.get("pred_rq_win", 0)) / 100.0
        pD = float(row.get("pred_rq_draw", 0)) / 100.0
        pL = float(row.get("pred_rq_loss", 0)) / 100.0
    except (ValueError, TypeError):
        return ""
    iW = 1.0 if rq == "让胜" else 0.0
    iD = 1.0 if rq == "让平" else 0.0
    iL = 1.0 if rq == "让负" else 0.0
    brier = ((iW - pW) ** 2 + (iD - pD) ** 2 + (iL - pL) ** 2) / 3.0
    return f"{brier:.4f}"


def check_score_accuracy(row):
    """比分 Top-1 准确率: pred_top_score == actual_score (去除横线格式差异)。"""
    pred = (row.get("pred_top_score") or "").strip()
    actual = (row.get("actual_score") or "").strip()
    if not pred or not actual:
        return ""
    # 统一分隔符 ":"
    pred_n = pred.replace("-", ":").replace("vs", ":")
    actual_n = actual.replace("-", ":").replace("vs", ":")
    return "1" if pred_n == actual_n else "0"


def check_goals_accuracy(row):
    """总进球 Top-1 准确率 + MAE。返回 (acc_top1, mae)。"""
    pred_s = (row.get("pred_top_goals") or "").strip()
    actual_s = (row.get("actual_goals") or "").strip()
    if not pred_s or not actual_s:
        return "", ""
    try:
        pred_g = int(pred_s)
        actual_g = int(actual_s)
    except ValueError:
        return "", ""
    acc = "1" if pred_g == actual_g else "0"
    mae = f"{abs(pred_g - actual_g):.1f}"
    return acc, mae


def check_htft_accuracy(row):
    """半全场 Top-1 准确率: pred_top_htft == actual_htft (带归一化)。"""
    pred = (row.get("pred_top_htft") or "").strip()
    actual = (row.get("actual_htft") or "").strip()
    if not pred or not actual:
        return ""
    # 统一格式: "HH" / "H/H" / "胜胜" 等
    def normalize_htft(s):
        s = s.replace("/", "").upper()
        s = s.replace("胜", "H").replace("平", "D").replace("负", "A")
        return s
    return "1" if normalize_htft(pred) == normalize_htft(actual) else "0"


# ─── 数据源 1: results JSON ───

def load_results_source():
    """加载 /root/data/results/*.json, 返回 {date: [{code, score_full, ...}]}"""
    source = {}
    if not os.path.isdir(RESULTS_DIR):
        return source
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if not fn.endswith(".json"):
            continue
        d = fn.replace(".json", "")
        path = os.path.join(RESULTS_DIR, fn)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                source[d] = data
        except (json.JSONDecodeError, IOError):
            continue
    return source


def match_from_results(row, results_source):
    """用 code (如 周二203) 从 results JSON 匹配"""
    pred_date = row.get("date", "")
    code = row.get("code", "")

    # 从 results 中找匹配日期的数据
    for d, matches in results_source.items():
        for m in matches:
            m_code = m.get("code", "")
            if m_code == code:
                return {
                    "score": m.get("score_full", ""),
                    "ht_score": m.get("score_ht", ""),
                    "hda": _normalize_hda(m.get("hda_result", "")),
                    "goals": m.get("goals", ""),
                    "htft": m.get("htft", ""),
                    "rq_result": _rq_from_result(m.get("rq_result", "")),
                    "source": f"results:{d}",
                }
    return None


# ─── 数据源 2: kaijiang CSV ───

def load_kaijiang_source():
    """加载 historical_kaijiang.csv, 返回 {code: row_dict}"""
    source = {}
    if not os.path.exists(KAIJIANG_CSV):
        return source
    with open(KAIJIANG_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r.get("code", "")
            if code:
                source[code] = r
    return source


def match_from_kaijiang(row, kaijiang_source):
    """用 code 从 kaijiang CSV 匹配"""
    code = row.get("code", "")
    kr = kaijiang_source.get(code)
    if not kr:
        return None

    ft_h = kr.get("ft_h", "")
    ft_a = kr.get("ft_a", "")
    if not ft_h or not ft_a:
        return None

    score = f"{ft_h}:{ft_a}"
    ht_h = kr.get("ht_h", "")
    ht_a = kr.get("ht_a", "")
    ht_score = f"{ht_h}:{ht_a}" if ht_h and ht_a else ""

    hda_raw = kr.get("spf_result", "")
    # kaijiang spf_result: 3=主胜, 1=平, 0=客胜
    hda_map = {"3": "H", "1": "D", "0": "A"}
    hda = hda_map.get(str(hda_raw), "")

    return {
        "score": score,
        "ht_score": ht_score,
        "hda": hda,
        "goals": str(int(ft_h) + int(ft_a)) if ft_h.isdigit() and ft_a.isdigit() else "",
        "htft": _compute_htft(ht_score, score),
        "rq_result": _compute_rq_result(ft_h, ft_a, row.get("rq", "0")),
        "source": "kaijiang",
    }


def match_from_365scores(row):
    """用 365scores current endpoint 兜底赛果回填。"""
    pred_date = row.get('date', '')
    home_raw = (row.get('home_cn') or '').replace(' ', '')
    away_raw = (row.get('away_cn') or '').replace(' ', '')
    try:
        from team_name_normalizer import normalize_team_name
        home = (normalize_team_name(home_raw) or home_raw).replace(' ', '')
        away = (normalize_team_name(away_raw) or away_raw).replace(' ', '')
    except Exception:
        home = home_raw
        away = away_raw

    url = "https://webws.365scores.com/web/games/current/?appTypeId=5&langId=27&timezoneName=Asia/Shanghai&games=1"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8', errors='replace'))
    except Exception:
        return None

    def iter_games(node):
        if isinstance(node, dict):
            if isinstance(node.get('games'), list):
                for g in node['games']:
                    yield g
            if isinstance(node.get('Games'), list):
                for g in node['Games']:
                    yield g
            for v in node.values():
                if isinstance(v, (dict, list)):
                    yield from iter_games(v)
        elif isinstance(node, list):
            for item in node:
                yield from iter_games(item)

    for g in iter_games(data):
        try:
            if int(g.get('statusGroup', -1)) != 4:
                continue
            hs = g.get('homeCompetitor', {}) or {}
            as_ = g.get('awayCompetitor', {}) or {}
            rh = (hs.get('name') or '').replace(' ', '')
            ra = (as_.get('name') or '').replace(' ', '')
            if rh != home or ra != away:
                continue
            sc_h = hs.get('score')
            sc_a = as_.get('score')
            if sc_h is None or sc_a is None:
                continue
            hs_i, as_i = int(sc_h), int(sc_a)
            score = f'{hs_i}-{as_i}'
            return {
                'score': score,
                'ht_score': '',
                'hda': _score_to_hda(f'{hs_i}:{as_i}'),
                'goals': str(hs_i + as_i),
                'htft': '',
                'rq_result': '',
                'source': f"365scores:{g.get('id') or g.get('gameId') or ''}",
            }
        except Exception:
            continue
    return None


# ─── 数据源 1.5: 365scores CSV fallback ───

def load_365scores_source():
    """加载 /root/data/365scores/*.csv, 返回 {date: [row_dict]}"""
    base = '/root/data/365scores'
    source = {}
    if not Path(base).exists():
        return source
    for fn in sorted(Path(base).glob('*.csv')):
        try:
            with fn.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    d = (r.get('time') or '')[:10]
                    if not d:
                        continue
                    source.setdefault(d, []).append(r)
        except Exception:
            continue
    return source


def match_from_365scores_csv(row, scores365_source):
    """优先从本地 365scores CSV 找完赛比分，先做队名标准化再匹配。"""
    pred_date = row.get('date', '')
    home_raw = (row.get('home_cn') or '').replace(' ', '')
    away_raw = (row.get('away_cn') or '').replace(' ', '')
    try:
        from team_name_normalizer import normalize_team_name
        home = (normalize_team_name(home_raw) or home_raw).replace(' ', '')
        away = (normalize_team_name(away_raw) or away_raw).replace(' ', '')
    except Exception:
        home = home_raw
        away = away_raw

    for r in scores365_source.get(pred_date, []):
        try:
            if str(r.get('status', '')).lower() != 'finished':
                continue
            rh = (r.get('home') or '').replace(' ', '')
            ra = (r.get('away') or '').replace(' ', '')
            if rh == home and ra == away:
                score = r.get('score') or ''
                if score and ':' in score:
                    hs, as_ = score.split(':', 1)
                    hs_i, as_i = int(hs), int(as_)
                    return {
                        'score': f'{hs_i}-{as_i}',
                        'ht_score': '',
                        'hda': _score_to_hda(f'{hs_i}:{as_i}'),
                        'goals': str(hs_i + as_i),
                        'htft': '',
                        'rq_result': '',
                        'source': f"365scores_csv:{r.get('id','')}" ,
                    }
        except Exception:
            continue
    return None


# ─── 辅助函数 ───

def _is_chinese(s):
    """判断字符串是否含中文。"""
    import re
    return bool(re.search(r'[\u4e00-\u9fff]', s))


def normalize_en_name(name):
    """统一英文队名格式: 小写 + &→and + 去变音符号 + 去多余空格。"""
    import unicodedata as _ud
    name = (name or '').strip()
    # 去变音符号: ü→u, é→e, í→i 等
    name = _ud.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    name = name.lower()
    name = name.replace(' & ', ' and ')
    name = name.replace('&', ' and ')
    name = name.replace('  ', ' ').strip()
    return name


def load_team_name_map():
    """加载 team_name_mapping.json, 返回 (cn_to_en, en_to_cn) 两个字典。

    支持双向映射:
    - 中文→英文:  {"捷克": "Czech Republic"}  (key含中文)
    - 英文→中文:  {"Czechia": "捷克"}          (value含中文)
    """
    cn_to_en = {}
    en_to_cn = {}
    path = '/root/data/team_name_mapping.json'
    if not os.path.exists(path):
        return cn_to_en, en_to_cn
    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)

        for k, v in raw.items():
            k_stripped = k.strip()
            v_stripped = v.strip()

            if _is_chinese(k_stripped):
                # 中文→英文 (标准方向)
                cn_to_en[k_stripped] = v_stripped
                en_clean = normalize_en_name(v_stripped).replace(' ', '')
                if en_clean not in en_to_cn:
                    en_to_cn[en_clean] = k_stripped
            else:
                # 英文→中文 (反向条目, 如 "Czechia": "捷克")
                en_key = normalize_en_name(k_stripped).replace(' ', '')
                cn_val = v_stripped
                if en_key not in en_to_cn:
                    en_to_cn[en_key] = cn_val
                # 也加入 cn_to_en 方便其他代码
                if cn_val not in cn_to_en:
                    cn_to_en[cn_val] = k_stripped

        return cn_to_en, en_to_cn
    except Exception:
        return cn_to_en, en_to_cn


# ─── 数据源 4: TheStatsAPI (终极兜底) ───

_THE_STATS_BASE = "https://api.thestatsapi.com/api"
_THE_STATS_KEY = os.environ.get('THE_KEY', '') or os.environ.get('THE_STATS_KEY', 'fapi_p14Z9YZeSwyXOMy1t9p0O1KBts5jXEww')
_THE_STATS_HDR = {"Authorization": f"Bearer {_THE_STATS_KEY}"}


# ─── TheStatsAPI 全局缓存 (带翻页) ───

_THE_STATS_CACHE = {"matches": None, "fetched": False}


def _fetch_all_thestats_matches():
    """翻页获取 TheStatsAPI 全部比赛数据，全局缓存。"""
    if _THE_STATS_CACHE["fetched"]:
        return _THE_STATS_CACHE["matches"]

    import requests as _req
    import time as _time

    all_matches = []
    _THE_STATS_CACHE["fetched"] = True  # 防止递归重入

    for page in range(1, 50):
        url = f"{_THE_STATS_BASE}/football/matches?per_page=100&page={page}"
        try:
            r = _req.get(url, headers=_THE_STATS_HDR, timeout=30)
            if r.status_code != 200:
                break
            data = r.json().get("data", [])
            if not data:
                break
            all_matches.extend(data)
        except Exception:
            break
        _time.sleep(0.3)  # 限速保护

    total = len(all_matches)
    finished = sum(1 for m in all_matches if str(m.get("statusGroup", "")) == "4" or str(m.get("status", "")).lower() == "finished")
    print(f"  📡 TheStatsAPI: 已缓存 {total} 场 (其中 {finished} 场完赛, 共 {page} 页)")
    _THE_STATS_CACHE["matches"] = all_matches
    return all_matches


def match_from_thestats(row, en_to_cn, _print_once=set()):
    """第4数据源: TheStatsAPI 全局缓存匹配。

    使用 _fetch_all_thestats_matches() 翻页获取全部比赛后匹配队名。
    不再依赖 pred_date (因为 API date 参数失效)。
    """
    import re as _re

    matches = _fetch_all_thestats_matches()
    if not matches:
        return None

    # 提取 row 中的队名 (去掉排名前缀和空格)
    def _clean(s):
        s = (s or '').replace(' ', '').replace('[', '').replace(']', '')
        s = _re.sub(r'^\[\d+\]', '', s).strip()
        return s

    home_val = _clean(row.get('home_cn', ''))
    away_val = _clean(row.get('away_cn', ''))

    def is_chinese(s):
        return bool(_re.search(r'[\u4e00-\u9fff]', s))

    home_is_cn = is_chinese(home_val)
    away_is_cn = is_chinese(away_val)

    for m in matches:
        try:
            # 筛选: 只匹配完赛比赛
            status_group = str(m.get("statusGroup", ""))
            status_str = str(m.get("status", "")).lower()
            if status_group != "4" and "finished" not in status_str:
                continue

            en_home = (m.get('home_team', {}) or {}).get('name', '')
            en_away = (m.get('away_team', {}) or {}).get('name', '')
            if not en_home or not en_away:
                continue

            # 使用 normalize_en_name 统一格式 (处理 &→and, ü→u 等)
            en_home_clean = normalize_en_name(en_home).replace(' ', '')
            en_away_clean = normalize_en_name(en_away).replace(' ', '')

            # 匹配策略:
            # 1. 如果 row 是中文名 → 用 en_to_cn 反向映射后对比中文
            # 2. 如果 row 是英文名 → 直接对比英文

            match_ok = False
            if home_is_cn:
                cn_found = en_to_cn.get(en_home_clean)
                if cn_found:
                    cn_clean = cn_found.replace(' ', '').replace('[', '').replace(']', '')
                    if cn_clean == home_val:
                        match_ok = True
                else:
                    key = f"home:{en_home}"
                    if key not in _print_once:
                        _print_once.add(key)
                        print(f"    ⚠️ [需补充字典] 找不到对应中文: {en_home}")
            else:
                if _clean(en_home.lower()) == home_val.lower():
                    match_ok = True

            if not match_ok:
                continue

            match_ok = False
            if away_is_cn:
                cn_found = en_to_cn.get(en_away_clean)
                if cn_found:
                    cn_clean = cn_found.replace(' ', '').replace('[', '').replace(']', '')
                    if cn_clean == away_val:
                        match_ok = True
                else:
                    key = f"away:{en_away}"
                    if key not in _print_once:
                        _print_once.add(key)
                        print(f"    ⚠️ [需补充字典] 找不到对应中文: {en_away}")
            else:
                if _clean(en_away.lower()) == away_val.lower():
                    match_ok = True

            if not match_ok:
                continue

            # 提取比分
            score = m.get('score', {}) or {}
            fs = score.get('final_score')
            if isinstance(fs, dict):
                hs = fs.get('home')
                ha = fs.get('away')
            else:
                hs = score.get('home')
                ha = score.get('away')

            if hs is None or ha is None:
                continue

            hs_i, ha_i = int(hs), int(ha)
            score_str = f"{hs_i}:{ha_i}"

            ht_score = score.get('ht_score', '')
            ht_str = ''
            if isinstance(ht_score, dict):
                hth = ht_score.get('home')
                hta = ht_score.get('away')
                if hth is not None and hta is not None:
                    ht_str = f"{int(hth)}:{int(hta)}"

            return {
                'score': score_str,
                'ht_score': ht_str,
                'hda': _score_to_hda(score_str),
                'goals': str(hs_i + ha_i),
                'htft': _compute_htft(ht_str, score_str),
                'rq_result': '',
                'source': f"thestats:{m.get('id', '')}",
            }
        except Exception:
            continue

    return None

def _normalize_hda(val):
    """将各种格式的 H/D/A 统一为 H/D/A"""
    val = str(val).strip()
    mapping = {
        "H": "H", "D": "D", "A": "A",
        "胜": "H", "平": "D", "负": "A",
        "主胜": "H", "平局": "D", "客胜": "A",
        "3": "H", "1": "D", "0": "A",
    }
    return mapping.get(val, val if val in ("H", "D", "A") else "")


def _rq_from_result(val):
    """将 rq_result (胜/平/负) 转为 让胜/让平/让负"""
    val = str(val).strip()
    if val in ("H", "D", "A"):
        return {"H": "让胜", "D": "让平", "A": "让负"}.get(val, "")
    if val in ("胜", "平", "负"):
        return "让" + val
    return ""


def _compute_htft(ht_score, ft_score):
    """从半场/全场比分计算半全场"""
    if not ht_score or not ft_score or ":" not in ht_score or ":" not in ft_score:
        return ""
    try:
        ht_h, ht_a = map(int, ht_score.split(":"))
        ft_h, ft_a = map(int, ft_score.split(":"))
    except ValueError:
        return ""

    def res(h, a):
        if h > a: return "H"
        elif h == a: return "D"
        else: return "A"

    return res(ht_h, ht_a) + res(ft_h, ft_a)


def _compute_rq_result(ft_h_str, ft_a_str, rq_str):
    """计算让球结果"""
    try:
        hg = int(ft_h_str)
        ag = int(ft_a_str)
        handicap = int(rq_str) if rq_str else 0
    except (ValueError, TypeError):
        return ""

    if handicap == 0:
        return ""

    adj_hg = hg + handicap
    if adj_hg > ag:
        return "让胜"
    elif adj_hg == ag:
        return "让平"
    else:
        return "让负"


def _score_to_hda(score):
    """从比分字符串推导 H/D/A"""
    if not score or ":" not in score:
        return ""
    try:
        hg, ag = map(int, score.split(":"))
    except ValueError:
        return ""
    if hg > ag:
        return "H"
    elif hg == ag:
        return "D"
    else:
        return "A"


# ─── 主回填逻辑 ───

def backfill(from_date=None, to_date=None, source_priority=None, dry_run=False):
    """主回填入口"""
    rows, fieldnames = load_log()
    if not rows:
        print("❌ 无 predictions_log.csv 或为空")
        return

    fieldnames, fn_updated = ensure_fieldnames(fieldnames)
    if fn_updated and not dry_run:
        save_log(rows, fieldnames)
        print(f"📝 CSV 字段已扩展 (新增回填字段)")

    # 加载数据源
    source_priority = source_priority or ["results", "kaijiang", "365scores_csv", "365scores", "thestats"]
    results_source = {}
    kaijiang_source = {}
    scores365_source = {}
    cn_to_en, en_to_cn = load_team_name_map()

    if "results" in source_priority:
        results_source = load_results_source()
        print(f"  📡 results JSON: {len(results_source)} 天有数据")
    if "kaijiang" in source_priority:
        kaijiang_source = load_kaijiang_source()
        print(f"  📡 kaijiang CSV: {len(kaijiang_source)} 场有数据")
    if "365scores_csv" in source_priority:
        scores365_source = load_365scores_source()
        print(f"  📡 365scores CSV: {sum(len(v) for v in scores365_source.values())} 场有数据")
    if "thestats" in source_priority:
        print(f"  📡 TheStatsAPI 兜底已启用")

    # 日期范围过滤
    today = date.today().isoformat()
    checkpoint = load_checkpoint()
    effective_from = from_date or checkpoint.get("last_date", "")
    if effective_from and not from_date:
        # checkpoint 后退1天, 处理可能的时区/延迟
        try:
            cp_date = date.fromisoformat(effective_from) - timedelta(days=1)
            effective_from = cp_date.isoformat()
        except ValueError:
            effective_from = ""

    # 筛选需要回填的记录
    needs_fill = []
    for row in rows:
        # 生成 match_key
        ensure_match_key(row)

        # 跳过已有赛果的
        if row.get("result_status") in ("filled", "conflict"):
            continue
        if row.get("actual_hda") and row["actual_hda"].strip():
            # 兼容旧数据: 有 actual_hda 但没有 result_status
            if not row.get("result_status"):
                row["result_status"] = "filled"
            continue

        pred_date = row.get("date", "")
        if not pred_date:
            continue

        # 日期范围过滤
        if effective_from and pred_date < effective_from:
            continue
        if to_date and pred_date > to_date:
            continue

        # 只处理已开赛的 (比赛日期 < 今天, 或比赛时间已过)
        # 简单判断: pred_date < today
        if pred_date >= today:
            continue

        needs_fill.append(row)

    if not needs_fill:
        print("✅ 无需回填 (所有记录已有赛果或日期未到)")
        return

    print(f"📋 找到 {len(needs_fill)} 条待回填记录")

    # 逐条回填
    filled = 0
    conflicted = 0
    errors = 0
    last_date = checkpoint.get("last_date", "")

    for row in needs_fill:
        code = row.get("code", "")
        home = row.get("home_cn", "")
        away = row.get("away_cn", "")
        pred_date = row.get("date", "")

        result = None
        for src_name in source_priority:
            if src_name == "results":
                result = match_from_results(row, results_source)
            elif src_name == "kaijiang":
                result = match_from_kaijiang(row, kaijiang_source)
            elif src_name == "365scores_csv":
                result = match_from_365scores_csv(row, scores365_source)
            elif src_name == "365scores":
                result = match_from_365scores(row)
            elif src_name == "thestats" and en_to_cn:
                result = match_from_thestats(row, en_to_cn)

            if result and result.get("score"):
                break

        if not result or not result.get("score"):
            print(f"  ⚠️ {code} {home} vs {away} — 所有源均无赛果")
            errors += 1
            continue

        # 冲突检测: 如果已有 actual_hda 且不一致
        existing_hda = row.get("actual_hda", "").strip()
        new_hda = result["hda"]
        if existing_hda and new_hda and existing_hda != new_hda:
            row["result_status"] = "conflict"
            conflicted += 1
            print(f"  ⚡ {code} {home} vs {away} — 冲突: 已有={existing_hda} 新={new_hda} (源:{result['source']})")
            continue

        # 填充
        score = result["score"]
        hda = new_hda or _score_to_hda(score)

        print(f"  ✅ {code} {home} vs {away} → {score} ({hda}) [{result['source']}]")

        if not dry_run:
            row["actual_score"] = score
            row["actual_hda"] = hda
            row["actual_ht"] = result.get("ht_score", "")
            row["actual_htft"] = result.get("htft", "")
            row["actual_goals"] = result.get("goals", "")
            row["actual_rq_result"] = result.get("rq_result", "")
            row["result_status"] = "filled"
            row["settled_at"] = datetime.now().isoformat()[:19]
            row["backfill_source"] = result["source"]
            row["checked"] = "1"

            # Brier Score (SPF)
            brier = compute_brier_spf(row)
            row["brier_spf"] = brier

            # Brier Score (让球 RQ)
            row["brier_rq"] = compute_brier_rq(row)

            # 比分 Top-1 准确率
            row["acc_score_top1"] = check_score_accuracy(row)

            # 总进球 Top-1 准确率 + MAE
            g_acc, g_mae = check_goals_accuracy(row)
            row["acc_goals_top1"] = g_acc
            row["goals_mae"] = g_mae

            # 半全场 Top-1 准确率
            row["acc_htft_top1"] = check_htft_accuracy(row)

        filled += 1

        # 更新 checkpoint
        if pred_date > last_date:
            last_date = pred_date

    # 写回 CSV
    if not dry_run and (filled > 0 or conflicted > 0):
        save_log(rows, fieldnames)

        # 更新 checkpoint
        checkpoint["last_date"] = last_date
        checkpoint["stats"]["total_filled"] = checkpoint["stats"].get("total_filled", 0) + filled
        checkpoint["stats"]["total_conflict"] = checkpoint["stats"].get("total_conflict", 0) + conflicted
        checkpoint["stats"]["last_run"] = datetime.now().isoformat()[:19]
        save_checkpoint(checkpoint)

    # 统计
    total = len(rows)
    has_result = sum(1 for r in rows if r.get("actual_hda") and r["actual_hda"].strip())
    has_brier = sum(1 for r in rows if r.get("brier_spf") and r["brier_spf"].strip())
    status_counts = {}
    for r in rows:
        s = r.get("result_status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\n{'='*50}")
    print(f"📊 回填结果: +{filled} 填充, {conflicted} 冲突, {errors} 无源")
    if dry_run:
        print(f"   (dry-run 模式, 未实际修改)")
    print(f"📊 总体状态:")
    print(f"   赛果覆盖: {has_result}/{total} ({has_result/total*100:.1f}%)")
    print(f"   Brier覆盖: {has_brier}/{total} ({has_brier/total*100:.1f}%)")
    for s, cnt in sorted(status_counts.items()):
        print(f"   {s}: {cnt}")

    # 计算已有Brier的平均值
    brier_vals = []
    for r in rows:
        b = r.get("brier_spf", "")
        if b:
            try:
                brier_vals.append(float(b))
            except ValueError:
                pass
    if brier_vals:
        avg_brier = sum(brier_vals) / len(brier_vals)
        print(f"   平均 Brier (SPF): {avg_brier:.4f} (n={len(brier_vals)})")

    # ── 5玩法独立校准报告 ──
    def _avg_float(rows, key):
        vals = []
        for r in rows:
            v = r.get(key, "")
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return vals

    def _pct(rows, key, val="1"):
        n = sum(1 for r in rows if r.get(key, "").strip() == val)
        return n, len(rows)

    print(f"\n  ── 5 玩法独立校准 ──")
    # 1. SPF (已有)
    spf_b = _avg_float(rows, "brier_spf")
    spf_acc = _pct(rows, "brier_spf") if spf_b else (0, 0)
    # 2. RQ
    rq_b = _avg_float(rows, "brier_rq")
    rq_acc_n, rq_acc_d = _pct(rows, "brier_rq")
    # 3. Score
    sc_acc_n, sc_acc_d = _pct(rows, "acc_score_top1")
    # 4. Goals
    gl_acc_n, gl_acc_d = _pct(rows, "acc_goals_top1")
    gl_mae = _avg_float(rows, "goals_mae")
    # 5. HTFT
    ht_acc_n, ht_acc_d = _pct(rows, "acc_htft_top1")

    if spf_b:
        print(f"   SPF Brier={sum(spf_b)/len(spf_b):.4f}  n={len(spf_b)}")
    if rq_b:
        print(f"   让球 RQ Brier={sum(rq_b)/len(rq_b):.4f}  n={len(rq_b)}")
    if sc_acc_d > 0:
        print(f"   比分 Score Acc={sc_acc_n}/{sc_acc_d}={sc_acc_n/sc_acc_d*100:.1f}%")
    if gl_acc_d > 0:
        print(f"   总进球 Goals Acc={gl_acc_n}/{gl_acc_d}={gl_acc_n/gl_acc_d*100:.1f}%  MAE={sum(gl_mae)/len(gl_mae):.1f}" if gl_mae else "")
    if ht_acc_d > 0:
        print(f"   半全场 HTFT Acc={ht_acc_n}/{ht_acc_d}={ht_acc_n/ht_acc_d*100:.1f}%")

    # ── 一次性迁移: 为已有赛果但缺少新校准列的记录补填 ──
    if not dry_run:
        backfill_missing_new_columns(rows, fieldnames)

    # ── 增量更新 Elo + Poisson λ (使用刚回填的赛果) ──
    if not dry_run and filled > 0:
        try:
            print(f"\n⚽ 增量更新 Elo + Poisson λ 先验...")
            ret = subprocess.run(
                [sys.executable, "/root/retrain_poisson_elo.py", "incremental"],
                capture_output=True, text=True, timeout=300,
            )
            for line in ret.stdout.strip().split("\n"):
                print(f"   {line}")
            if ret.returncode != 0:
                print(f"   ⚠️ 增量更新 stderr: {ret.stderr.strip()[:200]}")
        except Exception as e:
            print(f"   ⚠️ 增量更新异常: {e}")


def backfill_missing_new_columns(rows, fieldnames):
    """一次性迁移: 为已有赛果但缺少 brier_rq / acc_score_top1 等新校准列的行补填。
    幂等设计: 只更新 brier_rq/acc_score_top1/goals_mae/acc_htft_top1 为空的已回填行。
    """
    NEW_COLS = ['brier_rq', 'acc_score_top1', 'acc_goals_top1', 'goals_mae', 'acc_htft_top1']
    # 确保 fieldnames 包含新列
    for c in NEW_COLS:
        if c not in fieldnames:
            fieldnames.append(c)

    updated = 0
    for row in rows:
        # 只处理已有 actual_hda 的结果
        if not row.get('actual_hda', '').strip():
            continue
        # 检查是否有缺失的新校准列
        needs_update = False
        for c in NEW_COLS:
            if not row.get(c, '').strip():
                needs_update = True
                break
        if not needs_update:
            continue

        row['brier_rq'] = compute_brier_rq(row)
        row['acc_score_top1'] = check_score_accuracy(row)
        g_acc, g_mae = check_goals_accuracy(row)
        row['acc_goals_top1'] = g_acc
        row['goals_mae'] = g_mae
        row['acc_htft_top1'] = check_htft_accuracy(row)
        updated += 1

    if updated > 0:
        save_log(rows, fieldnames)
        print(f"   🔄 已补填 {updated} 条已有记录的缺失校准列")


def show_stats():
    """显示回填统计"""
    rows, _ = load_log()
    if not rows:
        print("❌ 无数据")
        return

    total = len(rows)
    has_result = sum(1 for r in rows if r.get("actual_hda") and r["actual_hda"].strip())
    has_brier = sum(1 for r in rows if r.get("brier_spf") and r["brier_spf"].strip())

    status_counts = {}
    source_counts = {}
    for r in rows:
        s = r.get("result_status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
        src = r.get("backfill_source", "")
        if src:
            source_counts[src] = source_counts.get(src, 0) + 1

    print(f"📊 predictions_log.csv 回填统计")
    print(f"{'='*50}")
    print(f"总记录: {total}")
    print(f"赛果覆盖: {has_result}/{total} ({has_result/total*100:.1f}%)")
    print(f"Brier覆盖: {has_brier}/{total} ({has_brier/total*100:.1f}%)")
    print(f"\nresult_status 分布:")
    for s, cnt in sorted(status_counts.items()):
        print(f"  {s}: {cnt}")
    if source_counts:
        print(f"\nbackfill_source 分布:")
        for s, cnt in sorted(source_counts.items()):
            print(f"  {s}: {cnt}")

    # Brier 统计
    brier_vals = []
    for r in rows:
        b = r.get("brier_spf", "")
        if b:
            try:
                brier_vals.append(float(b))
            except ValueError:
                pass
    if brier_vals:
        avg = sum(brier_vals) / len(brier_vals)
        mn = min(brier_vals)
        mx = max(brier_vals)
        print(f"\nBrier Score (SPF):")
        print(f"  平均: {avg:.4f}")
        print(f"  最小: {mn:.4f}")
        print(f"  最大: {mx:.4f}")
        print(f"  样本: {len(brier_vals)}")

    # 按 model_route 分组 Brier
    route_brier = {}
    for r in rows:
        b = r.get("brier_spf", "")
        route = r.get("model_route", "unknown")
        if b:
            try:
                route_brier.setdefault(route, []).append(float(b))
            except ValueError:
                pass
    if route_brier:
        print(f"\nBrier by model_route:")
        for route, vals in sorted(route_brier.items()):
            avg = sum(vals) / len(vals)
            print(f"  {route}: {avg:.4f} (n={len(vals)})")

    # 按 bet_action 分组命中率
    action_hit = {}
    for r in rows:
        hda = r.get("actual_hda", "")
        pick = r.get("pred_spf_pick", "")
        action = r.get("bet_action", "N/A")
        if hda and pick:
            # 转换 pick 到 H/D/A
            pick_map = {"主胜": "H", "平局": "D", "客胜": "A"}
            pick_hda = pick_map.get(pick, pick)
            hit = 1 if hda == pick_hda else 0
            action_hit.setdefault(action, []).append(hit)
    if action_hit:
        print(f"\n命中率 by bet_action:")
        for action, hits in sorted(action_hit.items()):
            rate = sum(hits) / len(hits) * 100
            print(f"  {action}: {rate:.1f}% ({sum(hits)}/{len(hits)})")


def show_trend_report():
    """每日趋势报告：Brier drift + 联赛分级 + 行动建议"""
    from collections import defaultdict

    rows, _ = load_log()
    if not rows:
        print("❌ 无数据")
        return

    daily = defaultdict(lambda: {"n": 0, "brier": [], "hit": [], "unfilled": 0})
    for r in rows:
        d = r.get("date", "")[:10]
        if not d:
            continue
        daily[d]["n"] += 1
        b = r.get("brier_spf", "")
        if b:
            try:
                daily[d]["brier"].append(float(b))
            except ValueError:
                pass
        hda = r.get("actual_hda", "")
        pick = r.get("pred_spf_pick", "")
        pmap = {"主胜": "H", "平局": "D", "客胜": "A"}
        pick_hda = pmap.get(pick, pick)
        if hda and pick_hda in ("H", "D", "A"):
            daily[d]["hit"].append(1 if hda == pick_hda else 0)
        if not hda:
            daily[d]["unfilled"] += 1

    print("\n" + "=" * 58)
    print(" 📊 每日性能趋势报告")
    print("=" * 58)
    print(f" {'日期':<12} {'覆盖':>4} {'Brier':>7} {'命中率':>6} {'空窗':>4} {'Drift':>8}")
    print("-" * 58)

    dates = sorted(daily.keys())
    prev_brier = None
    drift_warning = False
    drift_count = 0
    for d in dates:
        dd = daily[d]
        avg_b = sum(dd["brier"]) / len(dd["brier"]) if dd["brier"] else None
        hit_pct = sum(dd["hit"]) / len(dd["hit"]) * 100 if dd["hit"] else None

        drift_str = ""
        if avg_b is not None and prev_brier is not None:
            delta = avg_b - prev_brier
            if delta > 0.02:
                drift_str = "⚠️ +{:.2f}".format(delta)
                drift_count += 1
                if delta > 0.04:
                    drift_warning = True
            elif delta < -0.02:
                drift_str = "✅ {:.2f}".format(delta)
            else:
                drift_str = "   {:.2f}".format(delta)
        elif avg_b is not None:
            drift_str = "   (base)"

        b_str = f"{avg_b:.4f}" if avg_b is not None else "  N/A "
        h_str = f"{hit_pct:.0f}%" if hit_pct is not None else " N/A"
        print(f" {d:<12} {dd['n']:>4} {b_str:>7} {h_str:>6} {dd['unfilled']:>4} {drift_str:>8}")
        if avg_b is not None:
            prev_brier = avg_b

    print("-" * 58)

    # 联赛分级
    league_brier = defaultdict(list)
    league_hit = defaultdict(list)
    for r in rows:
        b = r.get("brier_spf", "")
        hda = r.get("actual_hda", "")
        if not hda or not b:
            continue
        try:
            league_brier[r.get("league", "?")].append(float(b))
        except ValueError:
            pass
        pick = r.get("pred_spf_pick", "")
        pmap = {"主胜": "H", "平局": "D", "客胜": "A"}
        pick_hda = pmap.get(pick, pick)
        if pick_hda in ("H", "D", "A"):
            league_hit[r.get("league", "?")].append(1 if hda == pick_hda else 0)

    print(f"\n🏆 联赛分级性能:")
    print(f" {'联赛':<20} {'Brier':>7} {'命中率':>6} {'样本':>5} {'评级':>4}")
    print("-" * 52)
    for league in sorted(league_brier, key=lambda l: -len(league_brier[l])):
        bvs = league_brier[league]
        hits = league_hit.get(league, [])
        avg_b = sum(bvs) / len(bvs)
        hit_r = sum(hits) / len(hits) * 100 if hits else 0
        n = len(bvs)
        # 评级
        if avg_b < 0.20:
            grade = "S"
        elif avg_b < 0.25:
            grade = "A"
        elif avg_b < 0.30:
            grade = "B"
        else:
            grade = "C"
        print(f" {league:<20} {avg_b:>7.4f} {hit_r:>5.0f}% {n:>5} {grade:>4}")

    # 行动建议
    print(f"\n💡 诊断摘要:")
    unfilled_total = sum(dd["unfilled"] for dd in daily.values())
    print(f"  待回填: {unfilled_total} 场")
    if drift_warning:
        print(f"  ⚠️ Brier 漂移警告: 连续 {drift_count} 天恶化 >0.04, 模型可能 drift")
    elif drift_count >= 3:
        print(f"  ⚠️ Brier 轻微上升趋势: {drift_count}/6 天恶化 >0.02, 建议关注")
    else:
        print(f"  ✅ Brier 稳定, 无显著 drift")

    worst_leagues = sorted(league_brier.items(), key=lambda x: -sum(x[1]) / len(x[1]))[:3]
    print(f"  最差联赛: " + ", ".join(f"{l}(Brier={sum(v)/len(v):.3f})" for l, v in worst_leagues if len(v) >= 3))

    print(f"  平均 Brier: {sum(float(r.get('brier_spf',0)) for r in rows if r.get('brier_spf')) / sum(1 for r in rows if r.get('brier_spf')):.4f}")


# ─── 主入口 ───

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多源赛果回填 + Brier Score")
    parser.add_argument("--from-date", help="从指定日期开始回填 (YYYY-MM-DD)")
    parser.add_argument("--to-date", help="到指定日期截止 (YYYY-MM-DD)")
    parser.add_argument("--source", help="数据源优先级, 逗号分隔 (results,kaijiang)")
    parser.add_argument("--dry-run", action="store_true", help="只展示不修改")
    parser.add_argument("--stats", action="store_true", help="显示回填统计")
    parser.add_argument("--report", action="store_true", help="每日趋势报告 (Brier drift + 联赛分级)")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.report:
        show_trend_report()
    else:
        source_priority = args.source.split(",") if args.source else None
        backfill(
            from_date=args.from_date,
            to_date=args.to_date,
            source_priority=source_priority,
            dry_run=args.dry_run,
        )
