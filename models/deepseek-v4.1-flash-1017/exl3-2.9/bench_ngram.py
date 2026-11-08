#!/usr/bin/env python3
"""ngram-friendly bench: summarize a long document (output echoes prompt n-grams),
plus a fresh-prose control. Reports tok/s for each."""
import requests, time, random

URL = 'http://127.0.0.1:8000/v1/chat/completions'

random.seed(7)
doc_lines = []
for i in range(600):
    cat = random.choice(['geology', 'hydrology', 'climate', 'biology', 'history'])
    rid = random.randint(1000, 9999)
    doc_lines.append(f"Record {i:04d} [{cat}-{rid}]: sector {random.randint(1,9)} "
                     f"measures {random.random()*10:.2f} units at waypoint {random.randint(100,999)}.")
doc = "\n".join(doc_lines)

def bench(label, content, max_tokens):
    t0 = time.time()
    r = requests.post(URL, json={"model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=1200)
    dt = time.time() - t0
    u = r.json()['usage']
    ct, pt = u['completion_tokens'], u['prompt_tokens']
    print(f"{label}: {ct} tok out ({pt} in) in {dt:.1f}s = {ct/dt:.1f} tok/s decode, {pt/max(dt-ct/400,0.1):.0f} tok/s prefill-est")

bench("control-prose", "Write a long essay about mountain ecosystems.", 250)
bench("doc-copy", f"List the first 40 records from this document verbatim, one per line:\n\n{doc}", 400)
