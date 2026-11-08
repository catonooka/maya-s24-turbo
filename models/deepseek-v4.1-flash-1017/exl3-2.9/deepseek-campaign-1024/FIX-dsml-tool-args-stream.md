# Fix: DSML tool-call arguments were byte-silent → client stream idle timeouts (part 2)

**Symptom:** pi-ai clients (deepseek-harness / dsh) still failed long turns with
`pi-ai stream idle timeout after 300000ms` even after the queue-heartbeats fix —
but only on turns where the model wrote a **large tool call** (a `write` with
9-16k tokens of file content). Engine log signature: generations ending
`(cancel, cancel=True)` at 311-588 s, all mid `writing a tool call`, engine
healthy at 27-33 t/s throughout. Every cancelled request died exactly ~300 s
after its last streamed byte; every survivor finished under 300 s total.

**Root cause (measured, not guessed):** DeepSeek-V4.1 writes tool calls in DSML.
The frontend's `DsmlRewriter` buffered the **entire parameter value** —
`_plain()` collected the text into `parameter_text` and returned `""` — until the
parameter's closing tag arrived. A 12k-token `write` was byte-silent for its
whole duration. Socket-level probe (forced write, timestamps every read):
**max silent gap 367.6 s**, 6 TCP reads, all argument bytes in one burst at the
end. The `tool_args` delta plumbing existed in all three routes (OpenAI chat,
Responses, Anthropic) — the rewriter just never released anything.

The test suite missed it because the DSML tests asserted only the **sum** of the
argument deltas (`json.loads("".join(...))`), never their timing — a fully
buffered implementation passed every test.

**Fix (serve/frontend.py, + a small server.py cleanup):**
- `DsmlRewriter` streams string-parameter values (`string="true"`, the file-write
  case) as the model writes them: the JSON-escaped body goes out incrementally,
  holding back only a suffix that could still grow into the closing tag
  (longest-closer-prefix hold, same technique the parser already uses).
- The value's open marker is a new internal `<arg_stre>` token; the scanner gets
  a verbatim `stre` state (no re-escaping — the body's `<` is already `\u003c`,
  which is why no structural tag can hide in it). Raw (`string="false"`) values
  keep the whole-value path: they are short by construction and must arrive as
  one normalized JSON scalar.
- Closing-tag semantics match the reference decoder exactly, verified against
  `ds41_encoding.py`: only the **complete** `</｜DSML｜ parameter>` token is
  structural; embedded opening tags, partial closer prefixes and lone `<` are
  argument data. (This also fixes a pre-existing rewriter bug: a value ending in
  a partial closer prefix used to glue onto the real closing tag and corrupt the
  parse.)
- `parse_tool_call` (whole-parse path) decodes the streamed form identically.
- Also fixes the queue-heartbeats counter leak from the previous patch: a client
  leaving while queued threw GeneratorExit at the ping yield placed *before*
  the try — leaking the FIFO place and `status["queued"]`
  (`test_server.GoneClients` was failing since that commit; now wrapped inside
  the try, both given back in the finally via a `queue_counted` flag).

The patch lives in `patches/dsml-tool-args-stream.patch` (git format-patch,
applies to `serve/frontend.py` + `serve/server.py`). Probe tool:
`probe_stream_gap.py` (timestamps every SSE read; prints max silent gap).

**Verified after the patch** (same forced 12k-token write, live server):

| metric | before | after |
|---|---|---|
| max silent gap | **367.6 s** (kills at 300 s) | **11.3 s** (= the prefill, before the first token) |
| argument deltas on the wire | 5 (one end burst) | **12,269** |
| arguments integrity | 33 chars, no content | 51,830 chars, parse OK, 1822 lines |
| decode throughput | 33.3 t/s | 33.4 t/s (unchanged) |

Functional battery on the live server: temp-0 determinism (byte-identical),
streaming with `usage`, tool call over HTTP (args intact), thinking deltas
streamed, bat-and-ball trap answered correctly. Parser/DSML suites 71/71 green
including **3 new timing tests** (argument bytes must arrive *before* the final
`tool_call` event — index-compared, not concat-and-compared; closer-prefix
splits never leak; raw params stay whole). Full repo suite: 343 tests, 13
failures = exactly the pre-existing set on the unpatched tree (14 there; one of
which — GoneClients — this patch fixes).

**Client-side note:** a session that already exhausted its 5/5 retries is
poisoned client-side; start a fresh session. New turns stream normally.
