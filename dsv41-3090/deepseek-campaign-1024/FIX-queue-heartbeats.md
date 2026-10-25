# Fix: queued requests were byte-silent → client stream idle timeouts

**Symptom:** pi-ai clients (deepseek-harness / dsh) failed turns with
`pi-ai stream idle timeout after 300000ms` and retried 5/5.

**Root cause (measured):** the maya serve frontend runs one conversation at a
time; other requests queue for a FIFO turn. A queued request sent **zero bytes**
while waiting — no SSE events, no keep-alives — for however long the running
turn took (reproduced: 103 s of total silence behind a 2000-word generation).
Any client with a stream idle timeout below that wait kills the turn. The
engine already had heartbeats for prompt reading and fair-slice re-queues; the
*initial* queue wait was the only silent phase.

**Fix (serve/server.py, `Service.run`):** inline the initial FIFO wait so a
queued request yields `("ping", None)` every 50 ms, which the SSE layer writes
as `: keep-alive` comment lines — exactly the existing fair-slice re-queue
behavior, now applied before the first turn. The patch lives in
`patches/queue-heartbeats.patch` (git format-patch, applies to serve/server.py).

**Verified after the patch:**

| path | result |
|---|---|
| queued OpenAI stream | 2104 keep-alives over a 105 s queue wait, max silent gap 2.2 s, answer correct |
| queued Anthropic stream | 1263 keep-alives, max gap 1.9 s, answer correct |
| solo stream / non-stream / thinking path | unregressed, 0 errors in log |

**Client-side hardening (optional):** dsh/pi-ai defaults
`streamIdleTimeoutMs` to 300000; raise it on the *provider route* (not top
level) in settings.yaml, then open a new session:

```yaml
llm-pi-ai:
  providers:
    <your-route>:
      streamIdleTimeoutMs: 1200000
```

**Not pursued:** `--parallel N` (true concurrent conversations) is implemented
in the GLM engine main (src/program/generate.cpp) only; the DS4 engine main
(ds4_main.cpp) has no such flag in any built binary (pr344, v1.1.24, v1.1.25).
Porting it to the DS4 engine is real engine work, tracked as a future item.
