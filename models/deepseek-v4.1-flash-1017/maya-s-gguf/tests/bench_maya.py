#!/usr/bin/env python3
"""Bench the maya-ds4 server: decode (256 tok, chat) + report tok/s."""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
P = "Write a vivid paragraph about a lighthouse keeper who discovers the sea has begun receding overnight, leaving boats stranded and fish glittering in the moonlight. Describe what he does next."

body = json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": N, "temperature": 1.0,
                   "messages": [{"role": "user", "content": P}]}).encode()
req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
t = time.time()
r = urllib.request.urlopen(req, timeout=900)
out = json.loads(r.read())
dt = time.time() - t
u = out.get("usage", {})
ct = u.get("completion_tokens", N)
msg = out["choices"][0]["message"]
text = (msg.get("content") or msg.get("reasoning_content") or "")[:200]
print(f"{ct} tok in {dt:.1f}s = {ct/dt:.2f} tok/s")
print("sample:", text.replace(chr(10), " ")[:180])
