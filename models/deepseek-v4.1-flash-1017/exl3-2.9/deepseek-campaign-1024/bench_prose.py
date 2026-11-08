#!/usr/bin/env python3
"""Held-out prose decode bench for the mixed-lane build (:8080).
Fresh, never-seen prose prompts (composed 2026-10-28, no prior traffic), two lengths.
Honest numbers = engine 'done:' lines from the serve log (client wall-clock shown too).
Usage: bench_prose.py [--log PATH] [--tag NAME]
"""
import argparse, json, re, time, requests

URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPTS = [
    "Write a scene where a night-shift train dispatcher in a coastal town realizes the freight schedule on her screen no longer matches the trains actually passing her window, and she has to decide whom to call first.",
    "Describe, in flowing prose, the journey of a single raindrop from a storm cloud above terraced rice fields all the way to a village cistern, including everything it passes and carries with it.",
    "Tell the story of an elderly luthier who receives an anonymous commission to build a violin from wood that arrives already smelling faintly of salt, and what he discovers when he finally strings it.",
    "A cartographer maps a city that rearranges its alleys every decade. Write a passage from her field journal on the evening she accepts that the map must be a song instead.",
    "Write a reflective essay-fragment on why harbor towns keep their best bakeries two streets inland from the fish market, following the smell of bread through the fog.",
    "An amateur radio operator in the mountains keeps hearing the same eleven notes at 3 a.m. Write the night she finally decodes them and drives down toward the valley.",
]

def chat(prompt, max_tokens, temperature=0.7, timeout=900):
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
    u = j.get("usage", {})
    return u.get("completion_tokens", 0), dt, (j["choices"][0]["message"].get("content") or "")

def engine_done(log_path, after_line=0):
    """Parse engine 'done:' lines after a byte offset; return list of (tok_s, text)."""
    try:
        with open(log_path, "r", errors="replace") as f:
            f.seek(after_line)
            txt = f.read()
    except FileNotFoundError:
        return [], after_line
    out = []
    for m in re.finditer(r"done:\s*([0-9.]+)\s*tok/s", txt):
        out.append(float(m.group(1)))
    return out, 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/home/aisever/maya-ds4-main/ds4-mixed.log")
    ap.add_argument("--tag", default="prose")
    args = ap.parse_args()
    try:
        off = 0
        with open(args.log, "rb") as f:
            f.seek(0, 2); off = f.tell()
    except FileNotFoundError:
        pass
    # warmup one short gen (discarded)
    chat("Say 'ready' and nothing else.", 8)
    time.sleep(0.5)
    off2 = 0
    with open(args.log, "rb") as f:
        f.seek(0, 2); off2 = f.tell()
    results = []
    for mt in (300, 800):
        for i, p in enumerate(PROMPTS):
            ct, dt, content = chat(p, mt)
            line = f"[{args.tag} {mt}tok p{i+1}] client {ct/dt:.2f} t/s ({ct} tok in {dt:.1f}s)"
            results.append({"tag": args.tag, "tokens": mt, "prompt": i + 1,
                            "completion_tokens": ct, "wall_s": round(dt, 2), "client_tps": round(ct / dt, 2)})
            print(line, flush=True)
    time.sleep(1.0)
    with open(args.log, "rb") as f:
        f.seek(off2); tail = f.read().decode(errors="replace")
    done = [float(m.group(1)) for m in re.finditer(r"done:\s*([0-9.]+)\s*tok/s", tail)]
    swaps = re.findall(r"(\d+) swaps", tail)
    print(f"\nengine done: lines: {['%.1f' % d for d in done]}")
    if done:
        core = done[len(done)//2:] if len(done) > 4 else done   # later half = converged cache
        print(f"engine mean(all)={sum(done)/len(done):.2f}  mean(later-half)={sum(core)/len(core):.2f} t/s")
    json.dump(results, open(f"/tmp/bench_prose_{args.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
