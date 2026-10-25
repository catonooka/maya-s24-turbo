#!/usr/bin/env python3
"""probe_stream_gap.py - measure the longest silent gap on the SSE wire during a forced
tool call with a large string parameter.

This is the pi-ai killer: the client's streamIdleTimeoutMs (300000 ms) fires when the
server sends no bytes for 5 minutes. A tool call whose string parameter buffers server-side
(the DSML rewriter held the whole value until the closing tag) is byte-silent for the
entire argument write - minutes at ~27 tok/s.

Usage: python3 probe_stream_gap.py [url] [stanzas]
Prints wall time, recv count, the largest inter-recv gaps, and an arguments-integrity
check. RED (unpatched): max gap ~= the whole argument write, few arg recvs.
GREEN (streamed args): max gap a few seconds, many arg recvs.
"""
import http.client
import json
import sys
import time
import urllib.parse

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/v1/chat/completions"
STANZAS = int(sys.argv[2]) if len(sys.argv) > 2 else 150

body = {
    "model": "deepseek-v4.1-flash",
    "stream": True,
    "stream_options": {"include_usage": True},
    "reasoning_effort": "none",
    "max_tokens": 12288,
    "temperature": 0,
    "tools": [{
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}},
                "required": ["path", "content"]}}}],
    "tool_choice": {"type": "function", "function": {"name": "write_file"}},
    "messages": [{
        "role": "user",
        "content": f"Write a file poems.txt containing exactly {STANZAS} different four-line "
                    "stanzas about the sea. Only the tool call, no commentary."}],
}

u = urllib.parse.urlparse(URL)
conn = http.client.HTTPConnection(u.hostname or "127.0.0.1", u.port or 80, timeout=900)
conn.request("POST", u.path, body=json.dumps(body),
             headers={"Content-Type": "application/json"})
resp = conn.getresponse()
if resp.status != 200:
    print(f"HTTP {resp.status}: {resp.read(400)!r}")
    sys.exit(1)

t0 = time.monotonic()
last = t0
recvs = 0
gaps = []
max_gap = (0.0, 0.0)
args_recvs = []          # (t, cumulative arg chars)
args_buf = ""
first_arg_t = None
buf = b""
while True:
    chunk = resp.read(512)     # returns as soon as bytes are available (chunked SSE)
    if not chunk:
        break
    now = time.monotonic()
    gap = now - last
    if gap > max_gap[0]:
        max_gap = (gap, now - t0)
    gaps.append(gap)
    last = now
    recvs += 1
    buf += chunk
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        line = line.strip()
        if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
            continue
        try:
            obj = json.loads(line[6:])
        except ValueError:
            continue
        d = (obj.get("choices") or [{}])[0].get("delta") or {}
        for tc in d.get("tool_calls") or []:
            fn = tc.get("function") or {}
            piece = fn.get("arguments") or ""
            if piece:
                args_buf += piece
                if first_arg_t is None:
                    first_arg_t = now - t0
                args_recvs.append((round(now - t0, 1), len(args_buf)))

wall = time.monotonic() - t0
gaps.sort(reverse=True)
print(f"wall {wall:.1f}s  recvs {recvs}  arg-delta recvs {len(args_recvs)}")
print(f"max silent gap {max_gap[0]:.1f}s at t+{max_gap[1]:.1f}s   top gaps: {[round(g, 1) for g in gaps[:6]]}")
print(f"first arg byte at t+{first_arg_t}s" if first_arg_t is not None else "NO ARG BYTES")
if args_recvs:
    print(f"last  arg byte at t+{args_recvs[-1][0]}s, {args_recvs[-1][1]} arg chars")
try:
    parsed = json.loads(args_buf)
    content = parsed.get("content") or ""
    print(f"arguments parse OK: path={parsed.get('path')!r} content={len(content)} chars, "
          f"~{len(content.splitlines())} lines")
except ValueError as e:
    print(f"ARGUMENTS PARSE FAIL: {e} ({len(args_buf)} chars)")
conn.close()
