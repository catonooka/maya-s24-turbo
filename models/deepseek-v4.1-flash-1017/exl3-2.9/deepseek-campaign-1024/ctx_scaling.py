#!/usr/bin/env python3
"""ctx_scaling.py v2 - decode t/s vs context length (:8080), engine-line honest.

v1 lesson: growing one conversation re-prefilled history each probe (0 reused,
~410 t/s) = O(n^2). v2: ONE fresh prompt per depth (single 947 t/s prefill),
then a 128-token decode ask appended in the same request turn; engine 'done:'
line is the honest decode rate. Needle planted mid-prompt, asked in the same
request's final user turn.

Usage: ctx_scaling.py [--depths ...] [--tag NAME]
Appends JSONL to /home/aisever/maya-ds4-main/bench/results/ds4-ctx.jsonl
"""
import argparse, json, time, requests

URL = "http://127.0.0.1:8080/v1/chat/completions"

TOPICS = ["the history of bridges", "how coffee is grown", "the rules of chess",
          "ocean currents", "how pianos work", "the life of ants", "wind turbines",
          "the printing press", "honey production", "train signaling",
          "lighthouse keepers", "the invention of paper", "mountain weather",
          "salt mining", "early telegraph", "canal locks", "vineyards", "clock towers",
          "glassblowing", "the silk road", "harbor tides", "bookbinding"]

NEEDLES = ["BANANA-7741", "ZEPHYR-9042", "ORCHID-3319", "CANYON-6125", "MARLIN-2280", "JUNIPER-5533"]

def para(i, words):
    t = TOPICS[i % len(TOPICS)]
    return f"Passage {i+1} (about {t}): " + (
        f"{t.capitalize()} interests historians and engineers alike. " * 1
        + f"We discuss {t} in depth here. " * 1
        + (f"The study of {t} reveals patterns repeated across centuries. "
           f"Practitioners of {t} pass down methods orally and in writing. "
           f"Modern analysis of {t} combines measurement with narrative. ") * (words // 30)
    )[: words * 6 + 40]

def build_prompt(depth, needle):
    # fill to ~depth tokens (4.2 chars/token): passages + needle mid-way
    chars_target = int(depth * 4.2)
    parts, i, total = [], 0, 0
    while total < chars_target:
        p = para(i, 120)
        parts.append(p); total += len(p); i += 1
        if needle and total > chars_target // 2 and not any(needle in x for x in parts):
            parts.append(f"NOTE: the access code for section {i} is {needle}. Remember it.")
            total += 60
    return "\n\n".join(parts)

def ask(prompt, max_tokens, timeout=1800):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
    }, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    j = r.json()
    return (j.get("usage", {}).get("prompt_tokens", 0),
            j.get("usage", {}).get("completion_tokens", 0),
            dt, j["choices"][0]["message"].get("content") or "")

def engine_last():
    try:
        with open("/home/aisever/maya-ds4-main/ds4-baseline.log") as f:
            lines = f.readlines()[-60:]
        for l in reversed(lines):
            if "[maya] done:" in l and " 0 tokens" not in l:
                # "256 tokens in 12 s (24.1 tok/s)"
                s = l.split("(length")[0].split("(stop")[0]
                tok = int(s.split("tokens in")[0].split("done:")[1].strip())
                tps = float(s.split("tok/s)")[0].split("(")[-1])
                return tok, tps
    except Exception:
        pass
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="2000,6000,12000,24000,48000,96000")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="/home/aisever/maya-ds4-main/bench/results/ds4-ctx.jsonl")
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]

    # warm
    ask("Say 'ready'.", 8)
    for d in depths:
        needle = NEEDLES[depths.index(d) % len(NEEDLES)]
        prompt = build_prompt(d, needle)
        q = (f"\n\nQuestion ONE (answer in one short line): what is the access code mentioned "
             f"above?\nQuestion TWO: then continue writing passages in the same style for 8 more lines.")
        pt, ct, dt, content = ask(prompt + q, 384, timeout=2400)
        time.sleep(2)   # let the engine line flush
        etok, etps = engine_last()
        hit = needle in content or needle.split("-")[0] in content
        row = {"tag": args.tag, "depth": d, "prompt_tokens": pt, "completion_tokens": ct,
               "wall_s": round(dt, 2), "client_tps": round(ct / dt, 2),
               "engine_tok": etok, "engine_tps": etps, "needle": needle, "needle_hit": hit}
        print(f"ctx~{d:>6} (pt={pt:>6}): engine {etps} t/s ({etok} tok) | needle {needle} hit={hit}")
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    main()
