import requests, json, os, sys

KEY = os.environ.get('THE_KEY', '')
if not KEY:
    print("NO KEY")
    sys.exit(1)
    
HEADERS = {"Authorization": f"Bearer {KEY}"}
BASE = "https://api.thestatsapi.com/api/football"

# Check a match with 0 xG to see raw response
matches_to_check = [
    ("mt_979427670", "Indonesia vs Saudi Arabia (WCQ AFC) - no xG in CSV"),
    ("mt_153499065", "UAE vs Iraq (WCQ AFC) - no xG in CSV"),
    ("mt_209798753", "Sweden vs Tunisia (World Cup) - has xG in CSV"),
]

for mid, desc in matches_to_check:
    print(f"\n=== {desc} ===")
    print(f"Match: {mid}")
    sr = requests.get(f"{BASE}/matches/{mid}/stats", headers=HEADERS, timeout=30)
    print(f"Stats status: {sr.status_code}")
    if sr.status_code == 200:
        data = sr.json()
        # Print the full structure without the overview detail
        if isinstance(data, dict):
            print(f"Top keys: {list(data.keys())}")
            if "data" in data:
                d2 = data["data"]
                if isinstance(d2, dict):
                    print(f"  data keys: {list(d2.keys())}")
                    if "overview" in d2:
                        ov = d2["overview"]
                        if isinstance(ov, dict):
                            print(f"  overview keys: {list(ov.keys())}")
                            eg = ov.get("expected_goals", {})
                            if isinstance(eg, dict):
                                print(f"  expected_goals: {json.dumps(eg, indent=2)[:300]}")
                            else:
                                print(f"  expected_goals type: {type(eg)} = {eg}")
                        else:
                            print(f"  overview is not dict: {type(ov)}")
                    else:
                        print(f"  no 'overview' in data. Raw: {json.dumps(d2, indent=2)[:500]}")
    print(flush=True)
