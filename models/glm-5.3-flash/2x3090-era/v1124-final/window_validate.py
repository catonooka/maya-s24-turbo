#!/usr/bin/env python3
"""Full-window validation: prefill + decode + needle (KV divergence check) at 8K/32K/64K/128K.
Model-agnostic: works for glm-5.3-flash (:8081) and deepseek-v4.1-flash (:8080)."""
import base64
import json
import random
import string
import struct
import sys
import time
import urllib.request
import zlib

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm-5.3-flash"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

WORDS = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'river',
         'mountain', 'quiet', 'evening', 'ships', 'sail', 'distant', 'shores',
         'wind', 'carries', 'salt', 'stories', 'lanterns', 'harbor', 'gulls',
         'tide', 'marks', 'old', 'wooden', 'piers', 'sailors', 'tell']


def filler(n_words, seed):
    rng = random.Random(seed)
    return ' '.join(rng.choice(WORDS) for _ in range(n_words))


def prefill_and_decode(n_words, label):
    rnd = ''.join(random.choices(string.ascii_lowercase, k=10))
    doc = (f"[Doc {rnd}] Archive log:\n{filler(n_words, label)}\nEnd log.\n\n"
           f"MID-FACT: The lighthouse key opens the brass door at dawn. {rnd}\n\n"
           f"More notes:\n{filler(n_words, label + 'b')}\nEnd.")
    body = json.dumps({"model": MODEL, "max_tokens": 200, "temperature": 0.0,
                       "messages": [{"role": "user", "content": doc + "\n\nWhat does the lighthouse key open? Answer in one short line, then continue with two sentences about the harbor."}]}).encode()
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(
        urllib.request.Request(URL, body, {"Content-Type": "application/json"}),
        timeout=3600).read())
    dt = time.time() - t0
    u = r.get("usage", {})
    pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    msg = r["choices"][0]["message"]
    text = (msg.get("content") or msg.get("reasoning_content") or "")
    # effective prefill rate: (dt - decode_time) where decode = ct / expected_rate; simpler: report both raw
    needle = "brass door" in text.lower()
    return pt, ct / dt, needle, text[:90].replace("\n", " ")


print(f"== {MODEL} on :{PORT} — window sweep (prefill-inclusive + needle) ==")
for words, label in [(5900, "8K"), (23600, "32K"), (47000, "64K"), (95000, "128K")]:
    pt, rate, needle, sample = prefill_and_decode(words, label)
    print(f"{label:>5}: prompt {pt:>6} tok | gen+prefill {rate:>5.1f} t/s | needle {'PASS' if needle else 'FAIL'} | {sample[:70]}")


# pure decode at depth (same doc, follow-up turn = cached prefix)
def decode_at_depth(n_words, label):
    rnd = ''.join(random.choices(string.ascii_lowercase, k=10))
    doc = f"[Doc {rnd}] {filler(n_words, label)}"
    msgs = [{"role": "user", "content": doc + "\nReply with: understood."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Write one vivid paragraph about the night sea."}]
    body = json.dumps({"model": MODEL, "max_tokens": 220, "temperature": 0.6, "messages": msgs}).encode()
    # first call caches the big doc
    urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=3600).read()
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(
        urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=3600).read())
    dt = time.time() - t0
    ct = r.get("usage", {}).get("completion_tokens", 0)
    return ct / dt


print(f"\n== pure decode at depth (cached prefix) ==")
for words, label in [(5900, "8K"), (23600, "32K"), (47000, "64K"), (95000, "128K")]:
    print(f"{label:>5}: {decode_at_depth(words, label):.1f} t/s")
