# benchkit — Benchmark + Health Toolkit for a Local LLM Server

Version: 1.0 (Stage 1 spec)
Target server: OpenAI-compatible LLM server at http://127.0.0.1:8080
Model name: glm-5.3-flash
Context window: 262,144 tokens (262k)

## 1. Global Facts & Conventions

- **Base URL**: `http://127.0.0.1:8080` (configurable via `--base-url` / env `BENCHKIT_BASE_URL`).
- **Model**: `glm-5-3-flash` is addressed as `glm-5.3-flash` in all API calls
  (`/v1/chat/completions`, `/v1/completions`).
- **Status endpoint**: `GET /v1/status` exists and returns JSON containing at least:
  - `context` — context window size (262000-range value)
  - `uptime_s` — server uptime in seconds
  Other fields may be present; monitor/report must tolerate extra keys.
- **Engine log**: `/home/aisever/project-maya/maya-maya-s24.log` contains health lines
  matching the prefix `glm stat:`. These lines are the authoritative source for engine-side
  health metrics (e.g., throughput, queue state, KV cache pressure as emitted by the engine).
- **Token counting rule (HARD REQUIREMENT)**: All token counts reported by benchkit MUST be
  taken from the API response field `usage.completion_tokens`. Never estimate tokens by
  dividing characters by 4, never use `len(tokens)` of a decoded stream, never count
  chunks. If `usage` is absent (e.g., some streaming setups), the field is recorded as
  `null` and the run is flagged `usage_missing` — it is NOT silently substituted.
- **Units**: latency in milliseconds, throughput in tokens/second (tok/s = completion_tokens
  / wall seconds of the request), timestamps ISO-8601 UTC.
- **Common CLI flags** (accepted by every module):
  - `--base-url URL` (default `http://127.0.0.1:8080`)
  - `--model NAME` (default `glm-5.3-flash`)
  - `--timeout S` (default 120)
  - `--json` (machine-readable output; default output is human-readable text)
- **Shared library**: `benchkit/common.py` provides `http_post`, `http_get`, `now_iso`,
  `tail_log(path, n)`, and a `Result` dataclass. All modules import from it; no module
  re-implements HTTP or log parsing.

## 2. Module: bench.py

**Purpose**: Measure raw generation performance of the server with fixed synthetic prompts.
Produces per-run and aggregate latency/throughput numbers suitable for before/after
comparison (e.g., after a server config change).

**CLI**
