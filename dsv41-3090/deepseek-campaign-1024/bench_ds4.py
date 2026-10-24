#!/usr/bin/env python3
"""Fixed decode/prefill bench for maya-ds4 (:8080). Usage: bench_ds4.py [--runs N] [--tag NAME]
Probes: essay (prose floor), facts, code (hot experts), 25k prefill, short-prompt TTFT.
Temp 0, unique nonce per run for prefill; decode = 256 new tokens each. Appends JSONL to results file."""
import argparse, json, time, sys, requests

URL = "http://127.0.0.1:8080/v1/chat/completions"

PROMPTS = {
    "essay": "Write a 200-word essay on why the sky is blue at noon and red at sunset.",
    "facts": "List 10 capital cities with their countries, one per line, no commentary.",
    "code": "Write a Python function implementing binary search with docstring and 3 test asserts.",
}

def chat(prompt, max_tokens, temperature=0.0, timeout=600):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "top_p": 1.0,
        "stream": False,
    }, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    j = r.json()
    usage = j.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    pt = usage.get("prompt_tokens", 0)
    content = j["choices"][0]["message"].get("content") or ""
    return pt, ct, dt, content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="/home/aisever/maya-ds4-main/bench/results/ds4-bench.jsonl")
    args = ap.parse_args()

    # warmup (cold expert fill) - discarded
    chat("Say 'ready' and nothing else.", 8)
    time.sleep(1)

    results = []
    for run in range(args.runs):
        for name, p in PROMPTS.items():
            pt, ct, dt, content = chat(p, 256)
            tps = ct / dt if dt else 0
            results.append({"tag": args.tag, "run": run + 1, "probe": name,
                            "prompt_tokens": pt, "completion_tokens": ct,
                            "wall_s": round(dt, 2), "tps": round(tps, 2)})
            print(f"[{args.tag} r{run+1}] {name}: {pt}p {ct}t in {dt:.1f}s = {tps:.2f} t/s", flush=True)

    # prefill probe: ~25k tokens, HIGH-ENTROPY (repeated text compresses to 1/4 its token count)
    import random
    words = ("harbor lantern sediment glacier magnet orbit quiver thicket cobalt ember raster palindrome "
             "viaduct monsoon kestrel azimuth trellis fathom bramble zephyr anvil cascade dovetail ").split()
    rng = random.Random(4242 + args.runs)
    filler = " ".join(rng.choice(words) for _ in range(20000))
    needle_q = filler + "\n\nQuestion: reply with only the word 'done'."
    pt, ct, dt, content = chat(needle_q, 8)
    prefill_tps = pt / dt if dt else 0
    results.append({"tag": args.tag, "probe": "prefill25k", "prompt_tokens": pt,
                    "wall_s": round(dt, 2), "tps": round(prefill_tps, 2)})
    print(f"[{args.tag}] prefill25k: {pt} tokens in {dt:.1f}s = {prefill_tps:.1f} t/s", flush=True)

    with open(args.out, "a") as f:
        for r_ in results:
            f.write(json.dumps(r_) + "\n")
    # summary
    for probe in ("essay", "facts", "code"):
        vals = [r["tps"] for r in results if r["probe"] == probe]
        print(f"== {probe}: mean {sum(vals)/len(vals):.2f} t/s (runs: {vals})", flush=True)

if __name__ == "__main__":
    main()
