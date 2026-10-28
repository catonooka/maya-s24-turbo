#!/usr/bin/env python3
"""Sweep driver: restarts the maya-ds4 unit per config, waits ready, benches unseen prose.
Usage: sweep_decode.py <tag> <pool_idx>   (env overrides already set by caller)
Prompt pools: deterministic, disjoint across pool_idx (no text repeats between boots).
"""
import itertools, json, random, re, subprocess, sys, time, requests

URL = "http://127.0.0.1:8080/v1/chat/completions"
SUBJ = ["a clockmaker", "a marine biologist", "a subway driver", "an archivist", "a stonemason", "a beekeeper",
        "a glider pilot", "a night nurse", "a translator", "a dockmaster", "a cartographer", "a violin teacher",
        "a volcanologist", "a librarian", "a ferry captain", "a clock tower keeper", "a radio astronomer", "a locksmith",
        "a pastry chef", "a mountain guide", "a tapestry weaver", "a harbor pilot", "a watch repairer", "a salt farmer"]
PLACE = ["in a fog-bound port town", "on a slowly sinking island", "at a border station in the mountains",
         "in a city built on bridges", "beside a dried-up salt lake", "under the last standing lighthouse",
         "in a village moving on rails", "at the edge of an abandoned quarry", "in a library that floods each spring",
         "on a ship that never docks", "in a town where radios only receive", "at a monastery above the treeline"]
EVENT = ["discovers the tide has begun running backward", "receives a letter posted fifty years late",
         "finds every mirror in the building fogged from the inside", "hears the same song from two directions",
         "notices the maps have all quietly changed overnight", "meets a stranger who knows their handwriting",
         "watches the migratory birds arrive a season early", "inherits a keyring with no doors left",
         "is asked to inventory things that no longer exist", "sees the harbor lights answer in code",
         "loses the ability to dream in color", "finds the well water tasting of iron and rainbows"]
MODE = ["Write the scene.", "Tell it as a journal entry.", "Narrate it in close third person.",
        "Write it as a letter home.", "Describe only what can be sensed, no names.",
        "Tell it as a story for children.", "Write it as a report that keeps failing to stay formal.",
        "Narrate it twice: once calm, once urgent."]

def pool(i, n=6, seed=4242):
    rng = random.Random(seed + i * 1000)
    seen, out = set(), []
    while len(out) < n:
        p = (rng.choice(SUBJ), rng.choice(PLACE), rng.choice(EVENT), rng.choice(MODE))
        if p in seen: continue
        seen.add(p)
        out.append(f"{p[0]} {p[1]} {p[2]} {p[3]}")
    return out

def wait_ready(timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get("http://127.0.0.1:8080/v1/models", timeout=3).status_code == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(4)
    raise SystemExit("server never became ready")

def chat(prompt, max_tokens, temperature=0.7, timeout=900):
    t0 = time.time()
    r = requests.post(URL, json={"model": "deepseek-v4.1-flash",
                                 "messages": [{"role": "user", "content": prompt}],
                                 "max_tokens": max_tokens, "temperature": temperature, "top_p": 1.0,
                                 "stream": False}, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    u = r.json().get("usage", {})
    return u.get("completion_tokens", 0), dt

def main():
    tag, pool_idx = sys.argv[1], int(sys.argv[2])
    mt_list = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else "300,300,300,800,800".split(","))]
    boot_s = wait_ready()
    prompts = pool(pool_idx)
    # warmup
    chat("Say 'ready' and nothing else.", 8)
    log_pos = 0
    try:
        with open("/home/aisever/maya-ds4-main/ds4-mixed.log", "rb") as f:
            f.seek(0, 2); log_pos = f.tell()
    except FileNotFoundError: pass
    for p, mt in zip(itertools.cycle(prompts), mt_list):
        ct, dt = chat(p, mt)
    time.sleep(1.0)
    with open("/home/aisever/maya-ds4-main/ds4-mixed.log", "rb") as f:
        f.seek(log_pos); tail = f.read().decode(errors="replace")
    done = [float(m.group(1)) for m in re.finditer(r"\(([0-9.]+) tok/s\)", tail)]
    lane = re.findall(r"CPU lane ([0-9.]+)% of the experts", tail)
    hit = re.findall(r"expert cache ([0-9.]+)% hit", tail)
    rec = {"tag": tag, "pool": pool_idx, "boot_s": round(boot_s, 1), "done_tps": done,
           "mean800": round(sum(done[-2:]) / max(1, len(done[-2:])), 2),
           "lane_pct": lane[-1] if lane else None, "cache_hit": hit[-1] if hit else None}
    print(json.dumps(rec))
    with open("/home/aisever/maya-s24-turbo/dsv41-3090/deepseek-campaign-1024/sweep-decode.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
