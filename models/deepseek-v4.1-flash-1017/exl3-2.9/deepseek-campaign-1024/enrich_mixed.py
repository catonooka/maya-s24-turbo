#!/usr/bin/env python3
"""Mixed-corpus profile enrichment for the 2.7 mixed-K pack (port 8080).

24 prompts across domains, ~450 tok each, temp 0.8 -> ~10k completions of
routing signal for the boot-time waterline. Disjoint from bench prose pools.
"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPTS = [
    # code - several languages and tasks
    "Write a Python function that merges overlapping intervals, with unit tests.",
    "Write a JavaScript async function that retries a fetch with exponential backoff, and explain it.",
    "Write a Rust function that counts word frequencies in a string using a HashMap, with comments.",
    "Write a SQL query joining orders, customers and products, showing top 10 by revenue last month.",
    "Write a bash script that finds the 10 largest files under a directory, excluding symlinks.",
    "Review this plan: store per-user feature flags in Redis with a 5-minute local cache. Risks?",
    "Explain what this regex does step by step: ^(?=.*[a-z])(?=.*\\d)[a-zA-Z\\d]{8,}$",
    "Write a C function that reverses a linked list iteratively, then convert it to recursive.",
    # prose - creative and formal
    "Write a short story about a cartographer mapping an island that changes shape overnight.",
    "Write a paragraph of literary criticism about the opening of Moby-Dick.",
    "Draft a polite email declining a meeting but proposing an async alternative, 120 words.",
    "Write a vivid descriptive paragraph about a harbor at dawn, fog lifting off the water.",
    "Summarize the plot of Hamlet in exactly five sentences, no quotes.",
    "Write a news-style report, 150 words, about a city installing tidal turbines under a bridge.",
    # reasoning / math
    "Reason step by step: a train leaves at 3pm at 80km/h, another at 4pm at 120km/h the same direction. When does it catch up?",
    "If a fair coin lands heads 7 times in a row, what is the probability the 8th is heads? Explain the reasoning carefully.",
    "Explain the difference between tensor parallelism and pipeline parallelism in LLM inference.",
    "Describe the architecture of a modern CPU memory hierarchy in detail, from registers to NUMA.",
    "Explain why DDR4 bandwidth, not PCIe, can bottleneck a CPU-offloaded MoE inference lane.",
    # lists / practical
    "List 8 practical ways to reduce PCIe bandwidth usage during MoE inference, one line each.",
    "List 12 common Linux performance tools and what each is for, one line each.",
    # translation / multilingual
    "Translate to French: The lighthouse keeper counted eleven gulls while the tide drew back.",
    "Translate to Vietnamese: Every morning the ferry crosses the river before the fog lifts.",
]

def main():
    t0 = time.time(); tot = 0
    for i, p in enumerate(PROMPTS):
        body = json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": 450, "temperature": 0.8,
                           "messages": [{"role": "user", "content": p}]}).encode()
        req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
        t = time.time()
        try:
            r = urllib.request.urlopen(req, timeout=900)
            u = json.loads(r.read())["usage"]; tot += u["completion_tokens"]
            print(f"[{i+1}/{len(PROMPTS)}] {u['completion_tokens']} tok in {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{i+1}] FAILED: {e}", flush=True); sys.exit(1)
    print(f"ENRICHED {tot} tok in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
