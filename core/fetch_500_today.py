#!/usr/bin/env python3
"""500.com 今日竞彩赛程抓取 — 2026-06-02"""
import urllib.request, re, json

url = "https://trade.500.com/jczq/?playid=269&g=2&date=2026-06-02"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=15) as resp:
    raw = resp.read()
    try:
        content = raw.decode('gbk')
    except:
        content = raw.decode('utf-8', errors='replace')

print(f"Fetched {len(content)} chars")

# 500.com uses data-* attributes on tr elements for match data
# Try to extract match rows
# The actual match data is loaded via JS, but we can look for the structure

# Search for any script that contains match data
scripts = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)

for i, s in enumerate(scripts):
    s_clean = s.strip()
    if len(s_clean) > 100:
        print(f"\n--- Script {i+1} ({len(s_clean)} chars) ---")
        print(s_clean[:800])
        print("...")

# Also search for any JSON data
json_blocks = re.findall(r'=\s*(\{[^{}]+\})', content)
if json_blocks:
    print(f"\n--- JSON blocks ---")
    for jb in json_blocks[:5]:
        print(jb[:200])

# Check if there's an API endpoint in the page
api_urls = re.findall(r'(https?://[^\s"\'<>]+/api[^\s"\'<>]*)', content)
if api_urls:
    print(f"\n--- API URLs ---")
    for u in api_urls[:10]:
        print(u)
