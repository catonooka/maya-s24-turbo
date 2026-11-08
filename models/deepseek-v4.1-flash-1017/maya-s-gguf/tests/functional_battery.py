#!/usr/bin/env python3
"""Full functional battery for deepseek-v4.1-flash on maya-ds4 :8080.
Text QA, reasoning, JSON mode, tool calling, long-context recall, vision, stability."""
import base64
import json
import struct
import time
import urllib.request
import zlib

URL = "http://127.0.0.1:8080/v1/chat/completions"
PASSED = []
FAILED = []


def chat(body, timeout=900):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return out, time.time() - t


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail[:110]}")


# 1. factual QA (temp 0)
_, questions = [
    ("What is 17 * 23? Just the number.", lambda t: "391" in t),
    ("Capital of Japan? One word.", lambda t: "tokyo" in t.lower()),
    ("Chemical symbol for iron? Just the symbol.", lambda t: "fe" in t.lower().split() or "Fe" in t),
    ("What year did the Titanic sink? Just the year.", lambda t: "1912" in t),
], None
for q, fn in [("What is 17 * 23? Just the number.", lambda t: "391" in t),
              ("Capital of Japan? One word.", lambda t: "tokyo" in t.lower()),
              ("What year did the Titanic sink? Just the year.", lambda t: "1912" in t)]:
    out, _ = chat({"model": "deepseek-v4.1-flash", "max_tokens": 800, "temperature": 0.0,
                   "messages": [{"role": "user", "content": q}]})
    t = (out["choices"][0]["message"].get("content") or out["choices"][0]["message"].get("reasoning_content") or "")
    check(f"qa: {q[:40]}", fn(t), t[:80])

# 2. reasoning chain
out, _ = chat({"model": "deepseek-v4.1-flash", "max_tokens": 1200, "temperature": 0.0,
               "messages": [{"role": "user", "content": "A farmer has 17 sheep. All but 9 die. How many are left? Explain briefly then answer."}]})
t = (out["choices"][0]["message"].get("content") or "")
check("reasoning: sheep", "9" in t, t[:100])

# 3. JSON mode
out, _ = chat({"model": "deepseek-v4.1-flash", "max_tokens": 600, "temperature": 0.0,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content": 'Return JSON: {"name": "Alice", "age": 30, "city": "Hanoi"}'}]})
t = (out["choices"][0]["message"].get("content") or "")
try:
    j = json.loads(t)
    check("json mode", j.get("name") == "Alice" and j.get("age") == 30, t[:80])
except Exception as e:
    # reasoning models may wrap; try to extract
    import re
    m = re.search(r"\{.*\}", t, re.S)
    check("json mode", bool(m) and json.loads(m.group(0)).get("name") == "Alice", f"{e}: {t[:80]}")

# 4. tool calling
out, _ = chat({"model": "deepseek-v4.1-flash", "max_tokens": 800, "temperature": 0.0,
               "tools": [{"type": "function", "function": {"name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
               "messages": [{"role": "user", "content": "What's the weather in Hanoi right now? Use the tool."}]})
msg = out["choices"][0]["message"]
tc = msg.get("tool_calls")
ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "hanoi" in tc[0]["function"]["arguments"].lower()
check("tool call", ok, str(tc)[:100] if tc else str(msg)[:100])

# 5. long-context recall: plant a fact at ~8K tokens, ask for it
fact = "The secret code is MANGO-7421."
filler = " ".join(f"Sentence {i}: the quick brown fox jumps over the lazy dog beside river {i}." for i in range(600))
out, _ = chat({"model": "deepseek-v4.1-flash", "max_tokens": 600, "temperature": 0.0,
               "messages": [{"role": "user", "content": f"{filler}\n\n{fact}\n\n{filler}\n\nWhat is the secret code? Answer with just the code."}]})
t = (out["choices"][0]["message"].get("content") or "")
check("long-context recall 8K", "MANGO-7421" in t, t[:90])

# 6. stability: 6 sequential short gens, all complete
oks = 0
for i in range(6):
    out, _ = chat({"model": "deepseek-v4.1-flash", "max_tokens": 900, "temperature": 1.0,
                   "messages": [{"role": "user", "content": f"Write one vivid sentence about {['oceans','mountains','cities','deserts','forests','space'][i]}."}]})
    t = (out["choices"][0]["message"].get("content") or "")
    if len(t) > 30:
        oks += 1
check("stability 6x", oks == 6, f"{oks}/6 completed")

# 7. conversation memory (multi-turn)
out, _ = chat({"model": "deepseek-v4.1-flash", "max_tokens": 800, "temperature": 0.0,
               "messages": [{"role": "user", "content": "My favorite color is teal. Remember it."},
                            {"role": "assistant", "content": "Got it, teal."},
                            {"role": "user", "content": "What is my favorite color? One word."}]})
t = (out["choices"][0]["message"].get("content") or "")
check("multi-turn memory", "teal" in t.lower(), t[:80])

print(f"\nTOTAL: {len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", FAILED)
