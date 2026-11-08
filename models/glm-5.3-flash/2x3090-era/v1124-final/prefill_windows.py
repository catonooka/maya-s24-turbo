#!/usr/bin/env python3
"""Pure prefill at each window: max_tokens=1, so the request time ~= prefill.
Reports prefill t/s + needle at 8K/32K/64K/128K."""
import json
import random
import string
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm-5.3-flash"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

WORDS = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'river',
         'mountain', 'quiet', 'evening', 'ships', 'sail', 'distant', 'shores',
         'wind', 'carries', 'salt', 'stories', 'lanterns', 'harbor', 'gulls',
         'tide', 'marks', 'old', 'wooden', 'piers', 'sailors', 'tell']


def prefill(n_words, label):
    rng = random.Random(label)
    rnd = ''.join(random.choices(string.ascii_lowercase, k=10))
    doc = (f"[Doc {rnd}] Archive log:\n" + ' '.join(rng.choice(WORDS) for _ in range(n_words)) +
           f"\n\nMID-FACT: The lighthouse key opens the brass door at dawn. {rnd}\n\n" +
           ' '.join(rng.choice(WORDS) for _ in range(n_words)) + "\nEnd.")
    body = json.dumps({"model": MODEL, "max_tokens": 700, "temperature": 0.0,
                       "messages": [{"role": "user", "content": doc + "\nWhat does the lighthouse key open? One short line."}]}).encode()
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(
        urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=3600).read())
    dt = time.time() - t0
    u = r.get("usage", {})
    pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    msg = r["choices"][0]["message"]
    text = (msg.get("content") or msg.get("reasoning_content") or "")
    # pure prefill estimate: total time minus decode time at a reference 45 t/s
    decode_est = ct / 45.0
    pf_rate = pt / (dt - decode_est) if dt > decode_est else 0
    needle = "brass door" in text.lower()
    return pt, pf_rate, needle


print(f"== {MODEL} :{PORT} — PURE PREFILL (gen time subtracted) + needle ==")
for words, label in [(5900, "8K"), (23600, "32K"), (47000, "64K"), (95000, "128K")]:
    pt, rate, needle = prefill(words, label)
    print(f"{label:>5}: prompt {pt:>6} tok | prefill ~{rate:>5.0f} t/s | needle {'PASS' if needle else 'FAIL'}")
