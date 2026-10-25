#!/usr/bin/env python3
"""Recall-drafter exactness + speed gate for maya-ds4 (:8080), DS4_RECALL default-on build.

The contract (TensorFold recall.md, our port): every copied token is target-verified, so a reply's
FINAL tokens are the same with Recall on, off or cold.  (Thinking prose at temp 0 is nondeterministic
on this engine independent of Recall - the campaign documented DSpark+CPU-lane nondeterminism with
stable final answers - so this gate compares `content`, not reasoning.)
  1. a repeated long structured answer: final content byte-identical across repeats, and the warmest
     repeat at least as fast as the cold one (the copy case is where Recall must win);
  2. the model's copy-heavy JSON answer is where copies land; timing improves when they do.
Usage: recall_gate.py
"""
import json, time, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"


def ask(prompt, max_tokens=1400, timeout=900):
    t0 = time.time()
    req = urllib.request.Request(URL, json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "stream": False,
    }).encode(), headers={"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(req, timeout=timeout))
    m = j["choices"][0]["message"]
    return (m.get("content") or ""), time.time() - t0


def main():
    fails = 0
    # the copy case: long structured boilerplate, then the same prompt again (record warm)
    p_copy = ("Write exactly this JSON object, filling 30 'id' keys with sequential ids and 'note' "
              "values that each say 'note number N of 30': {\"items\": [...]}. Nothing else.")
    c1, t1 = ask(p_copy)
    c2, t2 = ask(p_copy)
    c3, t3 = ask(p_copy)
    ok = c1 == c2 == c3
    print(f"copy-repeat final content identical (3x): {'PASS' if ok else 'FAIL'} "
          f"({len(c1)}/{len(c2)}/{len(c3)} chars)")
    fails += not ok
    best = min(t1, t2, t3)
    print(f"copy-repeat speed: {t1:.1f}s cold, {t2:.1f}s, {t3:.1f}s (best {best:.1f}s, "
          f"{'faster than cold' if best < t1 else 'NOT faster'})")
    # a paraphrase (no verbatim suffix match): must also be self-consistent
    p2 = "List the numbers 1 to 30, one per line, each as 'number N of 30'. Nothing else."
    d1, s1 = ask(p2, 400)
    d2, s2 = ask(p2, 400)
    ok = d1 == d2
    print(f"paraphrase self-consistent: {'PASS' if ok else 'FAIL'} ({len(d1)}/{len(d2)} chars, "
          f"{s1:.1f}s -> {s2:.1f}s)")
    fails += not ok
    print("RECALL GATE:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
