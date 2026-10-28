#!/usr/bin/env python3
"""Prefill probe set for the mixed build: 1.6k / 25k / 100k word-soup prompts (unique per run),
engine-line honest (reads 'prompt N tokens = ... read in X ms' lines). Usage: bench_prefill.py [tag]"""
import json, random, re, sys, time, requests

U = "http://127.0.0.1:8080/v1/chat/completions"
LOG = "/home/aisever/maya-ds4-main/ds4-mixed.log"
WORDS = ("harbor lantern sediment glacier magnet orbit quiver thicket cobalt ember raster palindrome "
         "viaduct monsoon kestrel azimuth trellis fathom bramble zephyr anvil cascade dovetail "
         "grebe sorrel tine wicket larch probe dune fulcrum midden spar vane").split()

def soup(n, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n))

def ask(prompt, mt=8):
    t0 = time.time()
    r = requests.post(U, json={"model": "deepseek-v4.1-flash",
                               "messages": [{"role": "user", "content": prompt + "\n\nReply with only: done"}],
                               "max_tokens": mt, "temperature": 0, "stream": False}, timeout=3600)
    dt = time.time() - t0
    r.raise_for_status()
    return r.json()["usage"]["prompt_tokens"], dt

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    seed = int(time.time())
    out = []
    for name, n in (("1.6k", 1000), ("25k", 16000), ("100k", 64000)):
        pt, dt = ask(soup(n, seed + n))
        out.append({"tag": tag, "probe": name, "prompt_tokens": pt, "wall_s": round(dt, 2),
                    "client_tps": round(pt / dt, 1)})
        print(f"[{tag}] {name}: {pt} tok in {dt:.1f}s = {pt/dt:.0f} tok/s (client)", flush=True)
    time.sleep(1)
    with open(LOG, "r", errors="replace") as f:
        tail = f.read()[-20000:]
    eng = re.findall(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in (\d+) ms", tail)
    for pt, ms in eng[-3:]:
        print(f"  engine: {pt} tok in {int(ms)/1000:.1f}s = {int(pt)/(int(ms)/1000):.0f} tok/s")
    with open("/home/aisever/maya-s24-turbo/dsv41-3090/deepseek-campaign-1024/prefill-probes.jsonl", "a") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")

if __name__ == "__main__":
    main()
