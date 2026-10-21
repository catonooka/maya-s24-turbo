#!/usr/bin/env bash
# Prefill bench: one ~2K-token prompt, reports prompt tok/s (time to first token).
N=${1:-2048}
URL=http://127.0.0.1:8000/v1/chat/completions
python3 - "$N" <<'EOF'
import json, sys, time, urllib.request
n = int(sys.argv[1])
words = ("the lighthouse keeper counted eleven gulls while the tide drew back over the sand "
         "and every boat in the harbour leaned farther until the moon rose twice ".split())
prompt = " ".join(words[i % len(words)] for i in range(n))
body = json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": 1, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
t = time.time()
r = urllib.request.urlopen(req, timeout=600)
u = json.loads(r.read())["usage"]
dt = time.time() - t
pt = u["prompt_tokens"] if "prompt_tokens" in u else u.get("prompt_eval_count", n)
print(f"prefill {pt} tok in {dt:.1f}s = {pt/dt:.0f} tok/s")
EOF
