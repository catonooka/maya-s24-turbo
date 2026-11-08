#!/usr/bin/env python3
"""Diverse decode traffic to enrich the EC profile (Exp 6 pre-step).

Runs several varied prompts (code, prose, reasoning, list) so ec_profile.json
captures a broader expert-demand sample than the single bench prompt.
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPTS = [
    "Write a Python function that merges overlapping intervals in a list, with tests.",
    "Explain the difference between tensor parallelism and pipeline parallelism in LLM inference.",
    "Draft a short story about a cartographer mapping an island that changes shape overnight.",
    "List 8 practical ways to reduce PCIe bandwidth usage during MoE inference.",
    "Translate to French: The lighthouse keeper counted eleven gulls while the tide drew back.",
    "Reason step by step: if a train leaves at 3pm at 80km/h and another at 4pm at 120km/h, when do they meet?",
    "Write a SQL query joining orders and customers, showing top 10 by revenue last month.",
    "Describe the architecture of a modern CPU memory hierarchy in detail.",
]
for i, p in enumerate(PROMPTS):
    body = json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": 220, "temperature": 0.8,
                       "messages": [{"role": "user", "content": p}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t = time.time()
    r = urllib.request.urlopen(req, timeout=600)
    u = json.loads(r.read())["usage"]
    print(f"[{i+1}/{len(PROMPTS)}] {u['completion_tokens']} tok in {time.time()-t:.0f}s")
print("profile enriched")
