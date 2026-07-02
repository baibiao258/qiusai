#!/usr/bin/env python3
"""
Odds API 刷新脚本 — 可独立运行或 import
用法: python3 scripts/refresh_odds.py

健壮性说明:
  - 使用 -D <file> + -o <file> 分离headers和body，避免 -D - 合并stdout解析脆弱
  - 2019 curl + subprocess text=True 环境下，\r\n\r\n 分割 header/body 失败率高
"""
import json, subprocess, sys, os, tempfile

API_KEY = "425a7cb6604fe89fcbd46a524ac08a11"
ODDS_FILE = "/root/data/theodds_api_data.json"


def refresh_winner_odds(api_key=API_KEY):
    """刷新夺冠赔率，保留现有的 H2H (upcoming) 赔率。"""
    url = f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup_winner/odds/?apiKey={api_key}&regions=uk,eu,us&oddsFormat=decimal"

    # 用临时文件分离 headers 和 body，避免 -D - 的 stdout 混合解析问题
    with tempfile.NamedTemporaryFile(prefix="odds_hdr_", suffix=".txt", delete=False) as hf, \
         tempfile.NamedTemporaryFile(prefix="odds_body_", suffix=".json", delete=False) as bf:
        header_path = hf.name
        body_path = bf.name

    try:
        result = subprocess.run(
            ["curl", "-s", "-D", header_path, "-o", body_path, url],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            print(f"ERROR: curl exit code {result.returncode}: {result.stderr}")
            return False

        if not os.path.getsize(body_path):
            # Try reading stdout as fallback
            if result.stdout:
                raw = json.loads(result.stdout)
            else:
                print("ERROR: empty response body")
                return False
        else:
            with open(body_path) as f:
                raw = json.load(f)

        # Parse remaining credits from header file
        remaining = 0
        with open(header_path) as f:
            for line in f:
                if line.lower().startswith("x-requests-remaining:"):
                    remaining = int(line.split(":")[1].strip())
                    break

        # Extract best (lowest) winner odds across all bookmakers
        winner_odds = {}
        for entry in raw:
            for bm in entry.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market["key"] == "outrights":
                        for outcome in market["outcomes"]:
                            t, p = outcome.get("name"), outcome.get("price")
                            if t and p and (t not in winner_odds or p < winner_odds[t]):
                                winner_odds[t] = p

        # Merge with existing H2H odds
        with open(ODDS_FILE) as f:
            old = json.load(f)

        result_data = {
            "winner_odds": winner_odds,
            "upcoming": old.get("upcoming", {}),
            "remaining_credits": remaining,
        }

        with open(ODDS_FILE, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"OK: {len(winner_odds)} teams, {remaining} credits remaining")
        return True

    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parse failed — {e}")
        # Debug info
        if os.path.exists(body_path):
            size = os.path.getsize(body_path)
            print(f"  body file size: {size}")
            with open(body_path) as f:
                print(f"  body preview: {f.read(500)}")
        return False
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return False
    finally:
        for p in [header_path, body_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    refresh_winner_odds()
