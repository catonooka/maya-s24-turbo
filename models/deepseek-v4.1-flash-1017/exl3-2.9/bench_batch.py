#!/usr/bin/env python3
"""Concurrency scaling bench: N parallel streams, aggregate tok/s.
Usage: bench_batch.py [base_port] [n1,n2,n4,...]"""
import requests, time, threading, sys

PORT = sys.argv[1] if len(sys.argv) > 1 else '8001'
LEVELS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '1,2,4,8,16').split(',')]
URL = f'http://127.0.0.1:{PORT}/v1/chat/completions'

PROMPTS = [
    "Write a detailed essay about the history of maritime navigation.",
    "Explain how photosynthesis works at the molecular level.",
    "Write Python code for a red-black tree with insert and delete.",
    "Describe the political causes of World War I in detail.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the mathematics of Fourier transforms with examples.",
    "Describe the process of protein folding and why it matters.",
    "Write a business plan for an underwater hotel startup.",
    "Explain quantum entanglement for a curious teenager.",
    "List and explain the major architectural styles of the 20th century.",
    "Write a persuasive article about urban vertical farming.",
    "Explain the economics of container shipping.",
    "Describe the geology of the Grand Canyon layer by layer.",
    "Write a technical comparison of HTTP/2 and HTTP/3.",
    "Explain how vaccines train the immune system.",
    "Describe the life cycle of a star from nebula to remnant.",
]

def one_stream(idx, max_tok, out, lock):
    body = {"model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": PROMPTS[idx % len(PROMPTS)]}],
            "max_tokens": max_tok, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    r = requests.post(URL, json=body, timeout=1800)
    dt = time.time() - t0
    ct = r.json()['usage']['completion_tokens']
    with lock: out.append((ct, dt))

def run_level(n):
    out, lock = [], threading.Lock()
    ths = [threading.Thread(target=one_stream, args=(i, 220, out, lock)) for i in range(n)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    total = sum(ct for ct, _ in out)
    per = [ct/dt for ct, dt in out]
    print(f"N={n:2d}: {total} tok in {wall:.1f}s | aggregate {total/wall:.1f} tok/s | "
          f"per-stream min/mean/max {min(per):.1f}/{sum(per)/len(per):.1f}/{max(per):.1f}", flush=True)
    return total/wall

if __name__ == '__main__':
    one_stream(0, 20, [], threading.Lock())  # warmup
    print(f"== concurrency scaling on :{PORT} ==")
    for n in LEVELS:
        run_level(n)
