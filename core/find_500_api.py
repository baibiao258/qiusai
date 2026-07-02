#!/usr/bin/env python3
"""Find 500.com AJAX API endpoint for match data"""
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

# Look for AJAX/fetch endpoints in JavaScript
# Common patterns in 500.com:
# 1. $.ajax({ url: ... })
# 2. fetch(...)
# 3. XMLHttpRequest
# 4. $.getJSON
# 5. Direct API URL patterns

# Also check for data-url or data-api attributes
ajax_patterns = [
    r'url\s*:\s*["\']([^"\']+)["\']',
    r'src\s*:\s*["\']([^"\']+)["\']',
    r'href\s*:\s*["\']([^"\']+)["\']',
    r'api\s*=\s*["\']([^"\']+)["\']',
    r'\.get\s*\(\s*["\']([^"\']+)["\']',
]

found_urls = set()
for pat in ajax_patterns:
    matches = re.findall(pat, content)
    found_urls.update(matches)

print("Potential API URLs found:")
for u in sorted(found_urls):
    if '500' in u or 'trade' in u or 'index' in u:
        print(f"  {u}")

# Look for the main JS file that handles match loading
js_files = re.findall(r'<script[^>]+src="([^"]+)"', content)
print(f"\nJS files ({len(js_files)}):")
for jf in js_files:
    print(f"  {jf}")

# The match loading script is likely one of the trade.500cache.com scripts
# Let's check specifically the jczq-specific script
for jf in js_files:
    if 'jczq' in jf or 'spf' in jf:
        print(f"\n*** Target JS: {jf}")
