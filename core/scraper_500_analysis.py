"""
500.com 比赛分析数据爬虫 v2 — 管线集成模块
=============================================
从 500.com 指数中心抓取:
  [核心赔率]
  - 历史比赛的欧赔 (平均欧指: H/D/A)
  - 历史比赛的亚盘 (盘口+水位)
  - 当前比赛的初盘/即时盘
  - matchid/fid 主键 (用于关联本地数据库)

  [赛程元数据]
  - 比赛时间、联赛类型、联赛ID
  - 未来赛事 (世界杯分组赛程, 用于疲劳度特征)

  [战绩交叉验证]
  - 近10场逐场明细 (赛事/日期/比分/盘口/半场/赛果/盘路/大小)
  - 主客场分别统计
  - 交战历史

  [辅助特征]
  - FIFA排名 + 积分变化
  - 预计首发阵容
  - 澳门心水推荐

供 daily_jczq.py 调用:
  from scraper_500_analysis import scrape_500_analysis, enrich_bundle_with_500
"""

import re
import json
import time
import os
from datetime import datetime, date
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': 'https://odds.500.com/',
}
BASE_URL = 'https://odds.500.com'
CACHE_PATH = '/root/data/500_analysis_cache.json'
CACHE_TTL = 3600  # 1小时缓存


def _fetch(url, timeout=12):
    """抓取500.com页面 (gb2312编码)"""
    req = Request(url, headers=HEADERS)
    try:
        resp = urlopen(req, timeout=timeout)
        raw = resp.read()
        for enc in ['gb2312', 'gbk', 'gb18030', 'utf-8']:
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return raw.decode('utf-8', errors='replace')
    except Exception:
        return None


def _extract_match_list(html):
    """
    从500.com比赛页提取同期所有比赛的 竞彩编号 和 shuju ID。
    返回: dict[竞彩code] -> {'id': shuju_id, 'home': str, 'away': str}
    """
    matches = {}
    pattern = (
        r'<span class="gray">(周[一二三四五六日]\d+)</span>.*?'
        r'href="/fenxi/shuju-(\d+)\.shtml"[^>]*>'
        r'.*?<em class="l">(.*?)</em>.*?<em class="r">(.*?)</em>'
    )
    for m in re.finditer(pattern, html, re.DOTALL):
        code = m.group(1)
        shuju_id = m.group(2)
        home = re.sub(r'<[^>]+>', '', m.group(3)).strip()
        away = re.sub(r'<[^>]+>', '', m.group(4)).strip()
        if home and away:
            matches[code] = {'id': shuju_id, 'home': home, 'away': away}
    return matches


def _parse_odds_row(row_html):
    """
    从单个 <tr> 行解析一场比赛的历史赔率数据。
    返回 dict 或 None。
    """
    # fid = 比赛ID (数据库关联主键)
    fid_m = re.search(r'fid="(\d+)"', row_html)
    fid = fid_m.group(1) if fid_m else ''

    # 跳过当前比赛 (bmatch, 尚未开赛)
    if 'bmatch' in row_html and 'display:none' in row_html:
        return {'fid': fid, 'is_current': True}

    # 联赛 (bgcolor + 链接里的联赛ID)
    league_m = re.search(r'zuqiu-(\d+)/', row_html)
    league_id = league_m.group(1) if league_m else ''
    league_name_m = re.search(r'title="([^"]*)"', row_html)
    league_name = league_name_m.group(1) if league_name_m else ''

    # 日期
    date_m = re.search(r'>(\d{2}-\d{2}-\d{2})<', row_html)
    match_date = date_m.group(1) if date_m else ''

    # 主客队名 + 比分
    home_m = re.search(r'class="dz-l[^"]*">(.*?)</span>', row_html)
    away_m = re.search(r'class="dz-r[^"]*">(.*?)</span>', row_html)
    home = re.sub(r'<[^>]+>', '', home_m.group(1)).strip() if home_m else ''
    away = re.sub(r'<[^>]+>', '', away_m.group(1)).strip() if away_m else ''

    # 比分 (在 <em> 标签内)
    score_m = re.search(r'<em>(.*?)</em>', row_html)
    score_raw = re.sub(r'<[^>]+>', '', score_m.group(1)).strip() if score_m else ''
    # 标准化: "0:1" -> "0:1", "VS" -> None
    score = score_raw if ':' in score_raw else None

    # 半场比分
    cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
    ht_score = None
    for cell in cells:
        clean = re.sub(r'<[^>]+>', '', cell).strip()
        if re.match(r'^\d+:\d+$', clean):
            ht_score = clean
            break

    # 赛果 (胜/平/负)
    result_m = re.search(r'class="(ying|ping|shu)">([胜负平])</span>', row_html)
    result = result_m.group(2) if result_m else ''

    # 盘路 (赢/输/走)
    panlu_m = re.findall(r'<td[^>]*><span class="(ying|ping|shu)">([赢输走])</span>', row_html)
    panlu = panlu_m[0][1] if panlu_m else ''

    # 大小 (大/小)
    daxiao_m = re.findall(r'<td[^>]*><span class="(ying|ping|shu)">([大小])</span>', row_html)
    daxiao = daxiao_m[0][1] if daxiao_m else ''

    # 亚盘 (盘口描述 + 水位)
    # 格式: <span>0.88</span><span class="table_pl_center">球半</span><span>0.90</span>
    ah_m = re.search(
        r'<span[^>]*>([\d.]+)</span>\s*<span class="table_pl_center">\s*([^<]+?)\s*</span>\s*<span[^>]*>([\d.]+)</span>',
        row_html
    )
    asian_handicap = {}
    if ah_m:
        asian_handicap = {
            'home_water': float(ah_m.group(1)),
            'line': ah_m.group(2).strip(),
            'away_water': float(ah_m.group(3)),
        }
    # 也提取数字盘口 (title="-1.5")
    ah_num_m = re.search(r'title="([^"]*)">\s*<span[^>]*>[\d.]+</span>\s*<span class="table_pl_center">', row_html)
    if ah_num_m:
        raw = ah_num_m.group(1).strip()
        try:
            asian_handicap['numeric'] = float(raw)
        except ValueError:
            pass

    # 欧赔 (H/D/A)
    # 格式: <p class="pub_table_pl"><span>1.30</span><span>5.20</span><span>8.66</span></p>
    odds_m = re.search(
        r'<p class="pub_table_pl"><span[^>]*>([\d.]+)</span><span[^>]*>([\d.]+)</span><span[^>]*>([\d.]+)</span></p>',
        row_html
    )
    euro_odds = {}
    if odds_m:
        euro_odds = {
            'home': float(odds_m.group(1)),
            'draw': float(odds_m.group(2)),
            'away': float(odds_m.group(3)),
        }

    return {
        'fid': fid,
        'is_current': False,
        'league_id': league_id,
        'league_name': league_name,
        'date': match_date,
        'home': home,
        'away': away,
        'score': score,
        'ht_score': ht_score,
        'result': result,
        'panlu': panlu,
        'daxiao': daxiao,
        'euro_odds': euro_odds,
        'asian_handicap': asian_handicap,
    }


def _parse_analysis(html, shuju_id):
    """解析单场比赛的分析页面，返回结构化数据"""
    data = {'shuju_id': shuju_id}

    # === matchid (隐藏字段, 数据库主键) ===
    mid_m = re.search(r'<input[^>]*name="matchid"[^>]*value="(\d+)"', html)
    if mid_m:
        data['matchid'] = mid_m.group(1)
    hash_m = re.search(r'<input[^>]*name="hash"[^>]*value="([^"]+)"', html)
    if hash_m:
        data['hash'] = hash_m.group(1)

    # === 比赛时间 + 联赛 ===
    tm = re.search(r'比赛时间([\d-]+ [\d:]+)', html)
    if tm:
        data['match_time'] = tm.group(1)
    # 联赛 (从页面头部提取)
    league_m = re.search(r'zuqiu-(\d+)/"[^>]*>([^<]+)<', html)
    if league_m:
        data['league_id'] = league_m.group(1)
        data['league_name'] = league_m.group(2).strip()

    # === FIFA排名 (含积分) ===
    fifa = {}
    # 匹配: <h3 class="lslayout1_stit">荷兰[世7]</h3>
    # 然后在同一 section 中找积分 (第二个表格行的积分列)
    for section_m in re.finditer(
        r'lslayout1_stit">(.*?)</h3>(.*?)(?=lslayout1_stit"|</div>\s*</div>\s*</div>)',
        html, re.DOTALL
    ):
        team_raw = re.sub(r'<[^>]+>', '', section_m.group(1)).strip()
        team = re.sub(r'\[世\d+\]', '', team_raw).strip()
        section = section_m.group(2)
        rank_m = re.search(r'<td class="td_sjpm">(\d+)</td>', section)
        # 积分在 <td>1757</td> 这样的格式中 (紧跟在排名变化之后)
        points_vals = re.findall(r'<td>(\d{3,5})</td>', section)
        rank = int(rank_m.group(1)) if rank_m else 0
        points = int(points_vals[0]) if points_vals else 0
        if team:
            fifa[team] = {'rank': rank, 'points': points}
    if fifa:
        data['fifa'] = fifa

    # === 当前比赛的欧赔 (从 bmatch 行提取, 可能 display:none) ===
    # 格式: <tr fid="1411007" sid="1" class="tr3 bmatch" style="display:none;">...<span>1.22</span><span>6.06</span><span>11.44</span>
    current_m = re.search(
        r'fid="%s"[^>]*>.*?<p class="pub_table_pl"><span>([\d.]+)</span><span>([\d.]+)</span><span>([\d.]+)</span></p>' % re.escape(shuju_id),
        html, re.DOTALL
    )
    if current_m:
        data['current_euro_odds'] = {
            'home': float(current_m.group(1)),
            'draw': float(current_m.group(2)),
            'away': float(current_m.group(3)),
        }
    # 当前亚盘 (紧跟在欧赔后面)
    current_ah_m = re.search(
        r'fid="%s"[^>]*>.*?<span[^>]*>([\d.]+)</span>\s*<span class="table_pl_center">\s*([^<]+?)\s*</span>\s*<span[^>]*>([\d.]+)</span>' % re.escape(shuju_id),
        html, re.DOTALL
    )
    if current_ah_m:
        data['current_asian_handicap'] = {
            'home_water': float(current_ah_m.group(1)),
            'line': current_ah_m.group(2).strip(),
            'away_water': float(current_ah_m.group(3)),
        }

    # === 历史战绩 (逐场明细, 含赔率) ===
    # 从 "数据分析" 表格提取 (team_zhanji1_1 和 team_zhanji1_0)
    for side, prefix in [(1, 'home'), (0, 'away')]:
        section_m = re.search(
            rf'id="team_zhanji1_{side}"(.*?)</form>',
            html, re.DOTALL
        )
        if not section_m:
            continue
        section = section_m.group(1)
        matches = []
        for row_m in re.finditer(r'<tr[^>]*>(.*?)</tr>', section, re.DOTALL):
            parsed = _parse_odds_row(row_m.group(1))
            if parsed and not parsed.get('is_current') and parsed.get('score'):
                matches.append(parsed)
        data[f'{prefix}_history'] = matches[:10]

    # === 历史战绩汇总 (从 bottom_info 提取) ===
    summary_pattern = (
        r'<p><strong>(.*?)</strong>近(\d+)场战绩'
        r'<span class="mar_left20"><span class="ying">(\d+)胜</span>'
        r'<span class="ping">(\d+)平</span><span class="shu">(\d+)负</span></span>'
        r'<span class="mar_left20">进<span class="ying">(\d+)球</span>'
        r'失<span class="shu">(\d+)球</span></span></p>'
    )
    summaries = re.findall(summary_pattern, html, re.DOTALL)
    for i, s in enumerate(summaries):
        key = 'home_form' if i == 0 else 'away_form'
        data[key] = {
            'team': s[0].strip(),
            'matches': int(s[1]),
            'wins': int(s[2]),
            'draws': int(s[3]),
            'losses': int(s[4]),
            'gf': int(s[5]),
            'ga': int(s[6]),
        }

    # === 赢盘率/大球率 (从 record_msg 提取) ===
    msgs = re.findall(r'record_msg">(.*?)</p>', html, re.DOTALL)
    for i, msg in enumerate(msgs):
        clean = re.sub(r'<[^>]+>', ' ', msg).strip()
        key = 'home_record' if i == 0 else 'away_record'
        wr = re.search(r'赢盘率\s*(\d+)%', clean)
        or_ = re.search(r'大球率\s*(\d+)%', clean)
        sr = re.search(r'胜率\s*(\d+)%', clean)
        data[key] = {
            'text': clean,
            'win_rate': int(sr.group(1)) if sr else 0,
            'cover_rate': int(wr.group(1)) if wr else 0,
            'over_rate': int(or_.group(1)) if or_ else 0,
        }

    # === 交战历史 ===
    h2h_section = re.search(r'交战历史(.*?)(?:近期战绩|$)', html, re.DOTALL)
    if h2h_section:
        h2h_text = h2h_section.group(1)
        if '暂无交战历史' in h2h_text:
            data['h2h'] = {'has_data': False, 'text': '无交战历史'}
        else:
            h2h_summary = re.search(r'his_info">(.*?)</span>', h2h_text, re.DOTALL)
            if h2h_summary:
                clean = re.sub(r'<[^>]+>', '', h2h_summary.group(1)).strip()
                data['h2h'] = {'has_data': True, 'text': clean}
            else:
                data['h2h'] = {'has_data': True, 'text': '有交战记录'}

    # === 澳门心水 ===
    tip = re.search(r'推介\s*-\s*<font[^>]*>(.*?)</font>', html)
    if tip:
        data['macau_tip'] = re.sub(r'<[^>]+>', '', tip.group(1)).strip()
    reason = re.search(r'td_no4">\s*(.*?)\s*</td>', html, re.DOTALL)
    if reason:
        data['macau_reason'] = re.sub(r'<[^>]+>', '', reason.group(1)).strip()

    # === 预计首发 ===
    lineup_section = re.search(r'预计阵容(.*?)澳门心水', html, re.DOTALL)
    if lineup_section:
        players = re.findall(
            r'<td class="td_one"><span class="td_sp3">(\d+)</span>(.*?)\((.*?)\)</td>',
            lineup_section.group(1)
        )
        home_starters = []
        for num, name, pos in players[:11]:
            home_starters.append({'number': int(num), 'name': name.strip(), 'position': pos.strip()})
        if home_starters:
            data['home_lineup'] = home_starters

    # === 未来赛事 (含世界杯分组) ===
    # 未来赛事section结束于下一个 <h4> 标签 (可能是 预计阵容/平均数据分析/澳门心水)
    future_section = re.search(r'<h4>未来赛事</h4>(.*?)(?=<h4>|<div class="odds_msg">)', html, re.DOTALL)
    if future_section:
        ft = future_section.group(1)
        all_future = re.findall(
            r'matchname"[^>]*>.*?>(.*?)</a>.*?>([\d-]+)</td>.*?class="dz-l"[^>]*>(.*?)</a>.*?class="dz-r"[^>]*>(.*?)</a>',
            ft, re.DOTALL
        )
        future = []
        for comp, fdate, fh, fa in all_future:
            future.append({
                'competition': re.sub(r'<[^>]+>', '', comp).strip(),
                'date': fdate.strip(),
                'home': re.sub(r'<[^>]+>', '', fh).strip(),
                'away': re.sub(r'<[^>]+>', '', fa).strip(),
            })
        if future:
            data['future_fixtures'] = future

    return data


def scrape_500_analysis(match_codes=None, delay=1.5):
    """
    从500.com批量抓取比赛分析数据。

    Args:
        match_codes: dict[竞彩code] -> {'id': shuju_id, 'home': str, 'away': str}
                     如果为 None, 则自动从已知页面提取全部竞彩比赛
        delay: 请求间隔秒数

    Returns:
        dict[竞彩code] -> analysis_data
    """
    cache = _load_cache()
    if cache is not None:
        print(f"  📦 500.com分析缓存命中 ({len(cache)}场)")
        return cache

    if match_codes is None:
        print("  📡 获取500.com竞彩比赛列表...")
        list_html = _fetch(f'{BASE_URL}/fenxi/shuju-1411007.shtml')
        if not list_html:
            print("  ⚠️ 无法获取500.com比赛列表")
            return {}
        match_codes = _extract_match_list(list_html)
        print(f"  📋 找到 {len(match_codes)} 场竞彩比赛")

    if not match_codes:
        return {}

    results = {}
    total = len(match_codes)

    for i, (code, info) in enumerate(match_codes.items()):
        shuju_id = info['id']
        home = info['home']
        away = info['away']

        print(f"  [{i+1}/{total}] {code} {home} vs {away} (ID:{shuju_id})")

        html = _fetch(f'{BASE_URL}/fenxi/shuju-{shuju_id}.shtml')
        if not html:
            continue

        analysis = _parse_analysis(html, shuju_id)
        analysis['code'] = code
        analysis['home'] = home
        analysis['away'] = away
        results[code] = analysis

        # 打印摘要
        hf = analysis.get('home_form', {})
        af = analysis.get('away_form', {})
        tip = analysis.get('macau_tip', '-')
        odds = analysis.get('current_euro_odds', {})
        n_hist = len(analysis.get('home_history', []))
        if hf:
            print(f"         {hf.get('team','')}: {hf.get('wins',0)}胜{hf.get('draws',0)}平{hf.get('losses',0)}负")
        if af:
            print(f"         {af.get('team','')}: {af.get('wins',0)}胜{af.get('draws',0)}平{af.get('losses',0)}负")
        if odds:
            print(f"         欧赔: {odds.get('home','-')} / {odds.get('draw','-')} / {odds.get('away','-')}")
        if n_hist:
            print(f"         历史赔率: {n_hist}场逐场数据含欧赔+亚盘")
        if tip:
            print(f"         澳门: {tip}")

        if i < total - 1:
            time.sleep(delay)

    _save_cache(results)
    return results


def _load_cache():
    """加载缓存 (1小时内有效)"""
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        ts = cache.get('_timestamp', 0)
        if time.time() - ts > CACHE_TTL:
            return None
        return cache.get('data', {})
    except Exception:
        return None


def _save_cache(data):
    """保存缓存"""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump({'_timestamp': time.time(), 'data': data}, f, ensure_ascii=False, indent=2)


def enrich_bundle_with_500(bundle, analysis):
    """
    将500.com分析数据注入到预测bundle中。
    不改变模型预测结果，只添加展示字段。
    """
    if not analysis:
        return bundle

    bundle['fifa'] = analysis.get('fifa', {})
    bundle['home_form_500'] = analysis.get('home_form', {})
    bundle['away_form_500'] = analysis.get('away_form', {})
    bundle['home_record_500'] = analysis.get('home_record', {})
    bundle['away_record_500'] = analysis.get('away_record', {})
    bundle['h2h_500'] = analysis.get('h2h', {})
    bundle['macau_tip'] = analysis.get('macau_tip', '')
    bundle['macau_reason'] = analysis.get('macau_reason', '')
    bundle['home_lineup_500'] = analysis.get('home_lineup', [])
    bundle['future_fixtures'] = analysis.get('future_fixtures', [])
    bundle['asian_handicap_desc'] = analysis.get('current_asian_handicap', {}).get('line', '')
    # 新增: 历史赔率数据
    bundle['home_history_500'] = analysis.get('home_history', [])
    bundle['away_history_500'] = analysis.get('away_history', [])
    bundle['matchid_500'] = analysis.get('matchid', '')
    bundle['current_euro_odds_500'] = analysis.get('current_euro_odds', {})
    bundle['current_asian_handicap_500'] = analysis.get('current_asian_handicap', {})
    bundle['league_id_500'] = analysis.get('league_id', '')

    return bundle


def format_500_analysis_lines(bundle):
    """
    格式化500.com分析数据为终端展示行。
    返回 list[str], 可直接 join 或 extend 到输出。
    """
    lines = []

    # FIFA排名
    fifa = bundle.get('fifa', {})
    if fifa:
        parts = []
        for k, v in fifa.items():
            rank = v.get('rank', '?') if isinstance(v, dict) else v
            pts = v.get('points', '') if isinstance(v, dict) else ''
            parts.append(f"{k}[{rank}]({pts}分)")
        lines.append(f"     🏆 FIFA: {' vs '.join(parts)}")

    # 近期战绩
    hf = bundle.get('home_form_500', {})
    af = bundle.get('away_form_500', {})
    if hf and af:
        h_str = f"{hf.get('team','')[:6]} {hf.get('wins',0)}胜{hf.get('draws',0)}平{hf.get('losses',0)}负(进{hf.get('gf',0)}失{hf.get('ga',0)})"
        a_str = f"{af.get('team','')[:6]} {af.get('wins',0)}胜{af.get('draws',0)}平{af.get('losses',0)}负(进{af.get('gf',0)}失{af.get('ga',0)})"
        lines.append(f"     📊 近10场: {h_str} | {a_str}")

    # 赢盘率/大球率
    hr = bundle.get('home_record_500', {})
    ar = bundle.get('away_record_500', {})
    if hr and ar:
        lines.append(
            f"     📈 赢盘率: {hr.get('cover_rate',0)}% vs {ar.get('cover_rate',0)}%"
            f" | 大球率: {hr.get('over_rate',0)}% vs {ar.get('over_rate',0)}%"
        )

    # 历史赔率摘要 (近3场含赔率的比赛)
    home_hist = bundle.get('home_history_500', [])
    if home_hist:
        hist_with_odds = [m for m in home_hist if m.get('euro_odds') and m.get('score')]
        if hist_with_odds:
            recent = hist_with_odds[:3]
            parts = []
            for m in recent:
                eo = m.get('euro_odds', {})
                ah = m.get('asian_handicap', {})
                parts.append(
                    f"{m.get('date','')[-5:]} {m.get('home','')[:4]}vs{m.get('away','')[:4]} "
                    f"{m.get('score','')} (欧{eo.get('home','-')}/{eo.get('draw','-')}/{eo.get('away','-')} "
                    f"盘{ah.get('line','-')})"
                )
            lines.append(f"     📜 近3场赔率: {' | '.join(parts)}")

    # 交战历史
    h2h = bundle.get('h2h_500', {})
    if h2h:
        if h2h.get('has_data'):
            lines.append(f"     ⚔️ 交战: {h2h.get('text', '')}")
        else:
            lines.append(f"     ⚔️ 交战: 无历史")

    # 澳门心水
    tip = bundle.get('macau_tip', '')
    reason = bundle.get('macau_reason', '')
    if tip:
        lines.append(f"     💬 澳门推介: {tip}")
        if reason and len(reason) < 80:
            lines.append(f"        {reason}")

    # 亚盘
    ah_desc = bundle.get('asian_handicap_desc', '')
    if ah_desc:
        lines.append(f"     🎯 亚盘: {ah_desc}")

    # 未来赛事 (仅展示世界杯)
    future = bundle.get('future_fixtures', [])
    wc = [f for f in future if '世界杯' in f.get('competition', '')]
    if wc:
        wc_str = ' | '.join([f"{w['date'][-5:]} {w['home']}vs{w['away']}" for w in wc[:3]])
        lines.append(f"     🏆 世界杯: {wc_str}")

    # 首发关键球员
    lineup = bundle.get('home_lineup_500', [])
    if lineup:
        names = ', '.join([p['name'] if isinstance(p, dict) else str(p) for p in lineup[:5]])
        lines.append(f"     👤 首发(前5): {names}...")

    return lines
