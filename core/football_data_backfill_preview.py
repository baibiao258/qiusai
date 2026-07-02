#!/usr/bin/env python3
import os, json, csv, urllib.request
from pathlib import Path
from difflib import SequenceMatcher
import sys
sys.path.insert(0, '/root')
from team_name_normalizer import normalize_team_name, TEAM_ALIASES

extra_aliases = {
    '南非': 'South Africa',
    '佛得角': 'Cape Verde',
    '埃及': 'Egypt',
    '塞内加尔': 'Senegal',
    '库拉索': 'Curaçao',
    'Curacao': 'Curaçao',
    '科特迪瓦': 'Ivory Coast',
    '民主刚果': 'DR Congo',
}
TEAM_ALIASES.update(extra_aliases)

LOG = Path('/root/data/predictions_log.csv')
OUT = Path('/root/data/predictions_log_football_data_backfill_preview.json')
API_KEY = os.environ.get('FOOTBALL_API_KEY', '5d07c80baa2645d0809b6ec96d6b49c6')
rows = list(csv.DictReader(LOG.open(encoding='utf-8')))

match_days = sorted(set('2026-' + r['time'].split(' ')[0] for r in rows if r.get('time')))
query_from = min(match_days)
query_to = max(match_days)
url = f"https://api.football-data.org/v4/matches?dateFrom={query_from}&dateTo={query_to}"
req = urllib.request.Request(url, headers={'X-Auth-Token': API_KEY, 'Accept': 'application/json'})
with urllib.request.urlopen(req, timeout=30) as r:
    payload = json.loads(r.read().decode('utf-8'))
matches = payload.get('matches', [])
finished = [m for m in matches if m.get('status') == 'FINISHED' and m.get('score', {}).get('fullTime', {}).get('home') is not None]

cands = []
for m in finished:
    home_raw = m['homeTeam']['name']
    away_raw = m['awayTeam']['name']
    home = normalize_team_name(home_raw)
    away = normalize_team_name(away_raw)
    utc = m.get('utcDate', '')
    cands.append({
        'id': m.get('id'),
        'utcDate': utc,
        'date': utc[:10],
        'home_raw': home_raw,
        'away_raw': away_raw,
        'home': home,
        'away': away,
        'score': m['score'],
        'competition': m.get('competition', {}).get('name', ''),
    })

def outcome(h,a):
    return 'H' if h>a else 'A' if h<a else 'D'

def rq_label(handicap, home_goals, away_goals):
    try:
        hcap = int(str(handicap).strip())
    except Exception:
        return ''
    x = home_goals + hcap
    return '让胜' if x > away_goals else '让平' if x == away_goals else '让负'

def sim(a,b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

updates = []
unmatched = []
for row in rows:
    if str(row.get('checked','')) == '1':
        continue
    log_home = normalize_team_name(row['home_cn'])
    log_away = normalize_team_name(row['away_cn'])
    target_date = '2026-' + row['time'].split(' ')[0]
    pool = [m for m in cands if m['date'] == target_date]
    exact = [m for m in pool if m['home'] == log_home and m['away'] == log_away]
    chosen = exact[0] if len(exact) == 1 else None
    if chosen is None:
        best = None
        best_score = -1
        for m in pool:
            score = 0.7*sim(log_home, m['home']) + 0.3*sim(log_away, m['away'])
            if score > best_score:
                best_score = score
                best = m
        if best is not None and best_score >= 0.88:
            chosen = best
    if chosen is None:
        unmatched.append({
            'code': row['code'], 'home_cn': row['home_cn'], 'away_cn': row['away_cn'],
            'norm_home': log_home, 'norm_away': log_away, 'target_date': target_date,
        })
        continue
    ft = chosen['score']['fullTime']
    ht = chosen['score'].get('halfTime') or {}
    hg, ag = int(ft['home']), int(ft['away'])
    hh, ah = ht.get('home'), ht.get('away')
    actual_hda = outcome(hg, ag)
    ht_out = outcome(hh, ah) if hh is not None and ah is not None else ''
    actual_htft = f'{ht_out}{actual_hda}' if ht_out else ''
    updates.append({
        'code': row['code'], 'home_cn': row['home_cn'], 'away_cn': row['away_cn'],
        'target_date': target_date,
        'api_home': chosen['home_raw'], 'api_away': chosen['away_raw'],
        'competition': chosen['competition'],
        'actual_score': f'{hg}:{ag}',
        'actual_ht': f'{hh}:{ah}' if hh is not None and ah is not None else '',
        'actual_hda': actual_hda,
        'actual_rq_result': rq_label(row.get('rq',''), hg, ag),
        'actual_goals': str(hg+ag) if hg+ag < 7 else '7+',
        'actual_htft': actual_htft,
    })

OUT.write_text(json.dumps({
    'query_from': query_from,
    'query_to': query_to,
    'fetched_matches': len(matches),
    'finished_matches': len(finished),
    'updates_found': len(updates),
    'sample_updates': updates[:20],
    'unmatched': unmatched,
}, ensure_ascii=False, indent=2), encoding='utf-8')
print(str(OUT))
print('updates_found=', len(updates))
print('unmatched=', len(unmatched))
