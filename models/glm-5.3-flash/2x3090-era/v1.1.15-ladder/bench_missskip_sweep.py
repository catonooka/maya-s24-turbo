#!/usr/bin/env python3
"""Sweep STRATA_GLM_MISS_SKIP_W via per-request strata_tune (no restart).
Bench = 3 probes like bench.py (prose/facts/code), mean reported per setting."""
import json, time, urllib.request, sys

URL = 'http://127.0.0.1:8080/v1/chat/completions'
PROBES = [
    ("prose", "Write a 150-word pitch for a space tourism company called Nova.", 400),
    ("facts", "List 10 facts about Mars, one line each.", 400),
    ("code",  "Write Python: a complete binary search tree class with insert, delete, search. Code only.", 600),
]

def chat(prompt, max_tokens, tune, temp=0.6):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp, "stream": False}
    if tune: body["strata_tune"] = tune
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t0
    u = r.get('usage', {})
    ct = u.get('completion_tokens', 0)
    return ct / dt, dt

def bench(tune, label):
    rates = []
    for name, p, mt in PROBES:
        r, dt = chat(p, mt, tune)
        rates.append(r)
        print(f"  {name}: {r:.1f} t/s", flush=True)
    m = sum(rates) / len(rates)
    print(f"{label}: MEAN {m:.1f} t/s", flush=True)
    return m

if __name__ == '__main__':
    settings = [0.14, 0.16, 0.18, 0.20, 0.14]  # baseline, push, settle-back
    results = {}
    for w in settings:
        tune = {"miss_skip_w": w, "miss_skip_from": 23} if w else {}
        results[w] = bench(tune, f"miss_skip_w={w}")
    print("\nsummary:")
    for w, m in results.items():
        print(f"  w={w}: {m:.1f} t/s")
