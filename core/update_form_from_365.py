#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_form_from_365.py
从 webws.365scores.com 拉取最近 N 天已完赛足球比赛，
更新 /root/data/form_state.json

格式: {"Team Name": [[home_goals, away_goals, "YYYY-MM-DD"], ...]}
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import requests

BASE_URL        = "https://webws.365scores.com/web"
HEADERS         = {
    "Accept":     "application/json",
    "Referer":    "https://www.365scores.com/",
    "User-Agent": "Mozilla/5.0 (compatible; FormUpdater/1.0)",
}
FORM_STATE_PATH = "/root/data/form_state.json"
LOG_PATH        = "/root/data/form_from_365.log"
MAX_HISTORY     = 25
SKIP_WORDS      = ("cancelled", "canceled", "postponed", "abandoned", "suspended")


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, file=sys.stderr)
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_form_state() -> dict:
    if not os.path.exists(FORM_STATE_PATH):
        return {}
    with open(FORM_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_form_state(data: dict) -> None:
    os.makedirs(os.path.dirname(FORM_STATE_PATH), exist_ok=True)
    with open(FORM_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_int(v) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def get_name(game: dict) -> Tuple[Optional[str], Optional[str]]:
    home = game.get("homeCompetitor", {}).get("name", "")
    away = game.get("awayCompetitor", {}).get("name", "")
    return home.strip() or None, away.strip() or None


def get_score(game: dict) -> Tuple[Optional[int], Optional[int]]:
    h = safe_int(game.get("homeCompetitor", {}).get("score"))
    a = safe_int(game.get("awayCompetitor", {}).get("score"))
    return h, a


def is_finished(game: dict) -> bool:
    return int(game.get("statusGroup", 0)) == 4


def should_skip(game: dict, home: str, away: str) -> bool:
    st = str(game.get("statusText", "")).lower()
    if any(w in st for w in SKIP_WORDS):
        return True
    if "(w)" in home.lower() or "(w)" in away.lower():
        return True
    return False


def fetch_games(date_str: str) -> list:
    r = requests.get(
        f"{BASE_URL}/games/current/",
        params={"sports": 1, "date": date_str, "games": 1,
                "startIndex": 0, "count": 200, "withTop": "true"},
        headers=HEADERS, timeout=20,
    )
    r.raise_for_status()
    return r.json().get("games", [])


def update_form_state(days: int = 2) -> dict:
    form_state = load_form_state()
    added = 0
    seen  = set()

    for i in range(days, 0, -1):
        date_str = (datetime.now(timezone.utc).date() - timedelta(days=i)).isoformat()
        try:
            games = fetch_games(date_str)
        except Exception as e:
            log(f"⚠️  {date_str} 拉取失败: {e}")
            continue

        log(f"📅 {date_str}: {len(games)} 场")
        day_added = 0

        for g in games:
            if not is_finished(g):
                continue
            home, away = get_name(g)
            if not home or not away:
                continue
            if should_skip(g, home, away):
                continue
            h, a = get_score(g)
            if h is None or a is None:
                continue

            key = (date_str, home, away, h, a)
            if key in seen:
                continue
            seen.add(key)

            for team, gh, ga in [(home, h, a), (away, a, h)]:
                form_state.setdefault(team, [])
                entry = [gh, ga, date_str]
                if not any(x[0]==gh and x[1]==ga and (len(x)<3 or x[2]==date_str)
                           for x in form_state[team]):
                    form_state[team].append(entry)
                    added += 1
                    day_added += 1

        log(f"   → 新增 {day_added} 条记录")
        time.sleep(0.5)

    for team in form_state:
        def _sort_key(x):
            if isinstance(x, (list, tuple)) and len(x) >= 3:
                return str(x[2])
            return '0000-00-00'
        form_state[team] = sorted(form_state[team], key=_sort_key)[-MAX_HISTORY:]

    save_form_state(form_state)
    return {
        "success": True,
        "days": days,
        "added_entries": added,
        "total_teams": len(form_state),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2)
    args = ap.parse_args()
    result = update_form_state(days=args.days)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
