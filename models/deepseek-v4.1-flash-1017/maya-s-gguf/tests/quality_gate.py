#!/usr/bin/env python3
"""Quality gate: fixed Q&A set, temp 0, exact/contains checks. Prints PASS/FAIL per Q + score."""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
QS = [
    ("What is 17 * 23? Answer with just the number.", "391"),
    ("What is the capital of France? One word.", "Paris"),
    ("What is 144 / 12? Just the number.", "12"),
    ("Name the chemical symbol for gold. Just the symbol.", "Au"),
    ("What is 15% of 200? Just the number.", "30"),
    ("Who wrote 'Romeo and Juliet'? Just the name.", "Shakespeare"),
    ("What is 2^10? Just the number.", "1024"),
    ("How many continents are there? Just the number.", "7"),
    ("What is the largest planet in our solar system? One word.", "Jupiter"),
    ("What is 9 * 9 - 9? Just the number.", "72"),
]
score = 0
for i, (q, expect) in enumerate(QS):
    body = json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": 48, "temperature": 0.0,
                       "messages": [{"role": "user", "content": q}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    try:
        out = json.loads(urllib.request.urlopen(req, timeout=300).read())
        text = (out["choices"][0]["message"].get("content")
                or out["choices"][0]["message"].get("reasoning_content") or "").strip()
        ok = expect.lower() in text.lower()
        score += ok
        print(f"Q{i+1} {'PASS' if ok else 'FAIL'} (want {expect!r}, got {text[:60]!r})")
    except Exception as e:
        print(f"Q{i+1} ERROR {e}")
print(f"SCORE {score}/{len(QS)}")
