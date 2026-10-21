#!/usr/bin/env python3
"""Cold prefill bench for the maya-ds4 server: unique prompt each run."""
import json
import random
import string
import sys
import time
import urllib.request

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
rnd = random.Random()
words = ["".join(rnd.choices(string.ascii_lowercase, k=rnd.randint(3, 9))) for _ in range(8192)]
words = words * 64
prompt = " ".join(words)[: n * 6]  # ~n tokens
body = json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": 1, "temperature": 1.0,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
t = time.time()
r = urllib.request.urlopen(req, timeout=1800)
u = json.loads(r.read())["usage"]
dt = time.time() - t
pt = u.get("prompt_tokens", n)
cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", "n/a")
print(f"COLD-PREFILL {pt} tok in {dt:.1f}s = {pt/dt:.0f} tok/s  (cached={cached})")
