#!/usr/bin/env python3
"""Bench GLM-5.3-flash on :8081 — decode prose/facts/code probes (like v1.1.15 campaign)."""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"
PROBES = [
    ("prose", "Write a 150-word pitch for a space tourism company called Nova.", 500),
    ("facts", "List 10 facts about Mars, one line each.", 500),
    ("code", "Write Python: a complete binary search tree class with insert, delete, search. Code only.", 700),
]


def chat(prompt, max_tokens):
    body = json.dumps({"model": "glm-5.3-flash", "max_tokens": max_tokens, "temperature": 0.6,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    dt = time.time() - t0
    ct = r.get("usage", {}).get("completion_tokens", 0)
    return ct / dt, dt


rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
for rd in range(rounds):
    rates = []
    for name, p, mt in PROBES:
        r, dt = chat(p, mt)
        rates.append(r)
        print(f"r{rd+1} {name}: {r:.1f} t/s ({dt:.1f}s)")
    print(f"r{rd+1} MEAN: {sum(rates)/len(rates):.1f} t/s")
