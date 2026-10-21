#!/usr/bin/env bash
# Prefill bench with a UNIQUE prompt per run (defeats the prefix cache).
N=${1:-2048}
python3 - "$N" <<'EOF'
import json, random, string, sys, time, urllib.request
n = int(sys.argv[1])
rnd = random.Random()  # fresh seed from os entropy every run
# pseudo-words: same token-length distribution as text but never repeated
words = ["".join(rnd.choices(string.ascii_lowercase, k=rnd.randint(3, 9))) for _ in range(4096)]
words = words * 64
prompt = " ".join(words)[: n * 6]  # ~n tokens (short pseudo-words ≈ 3.5 tok/word)
body = json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": 1, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
t = time.time()
r = urllib.request.urlopen(req, timeout=1200)
u = json.loads(r.read())["usage"]
dt = time.time() - t
pt = u.get("prompt_tokens", n)
cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", "n/a")
print(f"COLD-PREFILL {pt} tok in {dt:.1f}s = {pt/dt:.0f} tok/s  (prefix_cache_hit={cached})")
EOF
