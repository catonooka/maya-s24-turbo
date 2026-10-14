# benchkit — built by GLM-5.3-Flash itself (2026-10-14)

Benchmark + health toolkit for the Maya GLM-5.3-Flash server (:8080). Authored by the local
model in a 6-stage agent session (35.5k tokens of context exercised at 262k window), with
validator gates, a self-repair turn, and human review of every fix.

## Modules
- `bench.py` — probes (prose/facts/code, run-2 steady state) + concurrency ladder C1/C2/C4.
  Tokens from `usage.completion_tokens` only.
- `functest.py` — 8-check battery: chat, forced tool, zero-arg tool, multiturn+tool-result,
  SSE, greedy determinism, 15k needle, ASCII sanity. Exit code = fail count.
- `monitor.py` — /v1/status + engine-log `glm stat` sampler, JSONL out, thresholds
  (decode<20 alert, server_down). Carries last-known stat between engine emissions
  (engine logs `glm stat` only on long requests — verified).
- `report.py` — results JSON → self-contained HTML (3 tables, no deps).
- `cli.py` — `python -m benchkit {quick|full|monitor|report}`.

## Verified results (this session)
- Single-stream mean 58.78 t/s (baseline 59.2 ✓), functest 8/8, greedy determinism PASS,
  15k needle exact recall, zero errors across 21 requests.
- C2/C4: per-stream collapse (C4 min 5.3 t/s) — confirms concurrency is a lose on this
  2-GPU layer-split rig (matches 2026-10-13 --parallel measurement).
- Vision + thinking matrix (2026-10-14, vision_thinking_test.py + thinking_levels_test.py):
  vision 6/6 (OpenAI image_url, Anthropic image block, 640x480 large, vision×thinking on/off);
  thinking levels scale correctly low(0) < high(58) < max(155) chars on a trap question,
  same via Anthropic budgets (1k/5k/12k); thinking-off clean via reasoning_effort=none,
  enable_thinking=false, and Anthropic thinking disabled; streaming emits reasoning_content
  deltas (516 deltas/1765 chars) with usage. NOTE: on trivial prompts GLM-5.3-Flash thinks
  near-zero chars even at max effort, and pre-injected <think> can leave parsed
  reasoning_content empty on some turns — judge levels on a trap question, not trivial math.

## Usage
```bash
cd ~/maya-longtask
python3 -m benchkit quick                          # 1 probe C1
python3 -m benchkit full --bench-out results/full_results.json --functest-out results/functest_results.json
python3 -m benchkit monitor --duration 60 --interval 5 --out results/mon.jsonl
python3 -m benchkit report --bench results/full_results.json --functest results/functest_results.json --out results/report.html
```

## Session artifacts
- `PLAN.md` (project plan), `longtask_driver.py` (agent driver, resumable),
  `session.jsonl` (20 msgs), `driver.log`, `stages_done.json` (S1-S6 all PASS).
- Model repair turn: monitor.py Event-shutdown crash + stat parser — fixed by the model
  from error feedback in one shot (5.2k tokens).
- Human (Hermes) fixes: driver session-append duplication bug (400s), driver cwd bug,
  monitor backfill for between-emission samples, S6 re-run with real data.
