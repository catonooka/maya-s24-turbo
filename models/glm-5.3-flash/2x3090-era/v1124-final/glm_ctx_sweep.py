#!/usr/bin/env python3
"""Decode speed AT context size: build N-token context, then measure generation.
For each window: unique filler to N tokens, then a gen request that reuses that
context (prefix cache hit) so we measure pure decode at that depth."""
import json
import random
import string
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"


def make_ctx(n_words, tag):
    rnd = ''.join(random.choices(string.ascii_lowercase, k=10))
    filler = ' '.join(random.choices(
        ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
         'river', 'mountain', 'quiet', 'evening', 'ships', 'sail',
         'distant', 'shores', 'wind', 'carries', 'salt', 'stories'], k=n_words))
    return f"[Session {rnd}-{tag}] Story notes:\n{filler}\n\nEnd of notes."


def gen(ctx, max_tokens=220):
    body = json.dumps({
        "model": "glm-5.3-flash", "max_tokens": max_tokens, "temperature": 0.6,
        "messages": [
            {"role": "user", "content": ctx + "\nContinue the story with one vivid paragraph."},
            {"role": "assistant", "content": "Here is the continuation:"},
            {"role": "user", "content": "Another paragraph, different mood."},
        ]}).encode()
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(
        urllib.request.Request(URL, body, {"Content-Type": "application/json"}),
        timeout=1800).read())
    dt = time.time() - t0
    u = r.get("usage", {})
    ct = u.get("completion_tokens", 0)
    pt = u.get("prompt_tokens", 0)
    return pt, ct / dt


print(f"{'window':>10} {'prompt_tok':>10} {'decode t/s':>10}")
for words, label in [(60, "~100"), (700, "~1K"), (1400, "~2K"),
                     (2900, "~4K"), (5900, "~8K"), (11800, "~16K"),
                     (23600, "~32K"), (47000, "~64K"), (95000, "~128K")]:
    ctx = make_ctx(words, label)
    pt, rate = gen(ctx)   # first call = prefill+decode
    pt2, rate2 = gen(ctx) # second call = cached prefill, pure decode at depth
    print(f"{label:>10} {pt:>10} {rate2:>10.1f}   (1st run incl prefill: {rate:.1f})")
