#!/usr/bin/env python3
"""quant_ab.py - 2.9bpw vs 2.7bpw pack A/B on the live engine (:8080).

Both packs serve through maya-ds4; this script runs the SAME panel against
whichever pack is live and appends to a JSONL. Run: boot pack A, run; boot
pack B, run; compare. Panel = the quality-gate probes + a 24-probe greedy
agreement set (same prompts, compare answers pack-to-pack) + timing.

Usage: quant_ab.py --tag <2.9|2.7> [--out file]
"""
import argparse, json, time, requests

URL = "http://127.0.0.1:8080/v1/chat/completions"

# agreement probes: short factual/reasoning/code asks where greedy output should be
# stable; differences between packs here = quality delta surface
PROBES = [
    "What is 17 times 19? Reply with just the number.",
    "What is (17*19)+(23*8)-145? Reply with just the number.",
    "Name the capital of Australia, one word.",
    "Is 97 a prime number? Answer yes or no.",
    "Write a Python one-liner to reverse a string s.",
    "What's the chemical symbol for tungsten? One word.",
    "Sort these words alphabetically: banana, apple, cherry, date. Comma separated, no commentary.",
    "How many days in February 2028? Just the number.",
    "Convert 100 degrees Celsius to Fahrenheit. Just the number.",
    "What year did the Berlin Wall fall? Just the year.",
    "def fibonacci(n): complete this function in Python with docstring.",
    "Name three primary colors, comma separated.",
    "What is the square root of 1764? Just the number.",
    "Translate 'good morning' to French, Italian, German. Comma separated.",
    "List the first 5 elements of the periodic table, comma separated.",
    "What is 15% of 240? Just the number.",
    "Write a bash command to count lines in a file named x.txt.",
    "Which planet is closest to the sun? One word.",
    "What is the largest ocean on Earth? One word.",
    "Name the author of '1984'. One word.",
    "What's 2 to the power of 16? Just the number.",
    "Write a SQL query selecting all columns from users where active is true.",
    "What is the boiling point of water at sea level in Celsius? Just the number.",
    "Explain in exactly one sentence why the sky is blue.",
]

BENCH = [
    ("essay", "Write a 200-word essay on why the sky is blue at noon and red at sunset.", 256),
    ("code", "Write a Python function implementing binary search with docstring and 3 test asserts.", 256),
    ("facts", "List 10 capital cities with their countries, one per line, no commentary.", 256),
]

def ask(prompt, max_tokens, timeout=600):
    t0 = time.time()
    r = requests.post(URL, json={"model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0}, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    j = r.json()
    c = j["choices"][0]["message"].get("content") or ""
    return c.strip(), dt, j.get("usage", {}).get("completion_tokens", 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/home/aisever/maya-s24-turbo/dsv41-3090/deepseek-campaign-1024/quant-ab.jsonl")
    a = ap.parse_args()
    # warm
    ask("Say 'ready'.", 8)
    answers = {}
    for i, p in enumerate(PROBES):
        ans, dt, ct = ask(p, 220)
        answers[i] = ans
        print(f"[{a.tag} probe {i+1:02d}] {ct} tok {dt:.1f}s: {ans[:60]!r}")
    bench = {}
    for name, p, mt in BENCH:
        tps_runs = []
        for _ in range(2):
            ans, dt, ct = ask(p, mt)
            tps_runs.append(round(ct/dt, 2))
        bench[name] = tps_runs
        print(f"[{a.tag} {name}] {tps_runs} t/s")
    with open(a.out, "a") as f:
        f.write(json.dumps({"tag": a.tag, "probes": answers, "bench": bench}) + "\n")
    print(f"saved {len(answers)} probes + bench, tag {a.tag}")

if __name__ == "__main__":
    main()
