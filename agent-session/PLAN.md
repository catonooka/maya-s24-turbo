# PROJECT PLAN — benchkit (built BY the local model, long-task agent session)

Owner-agent: Hermes (orchestrator/monitor). Worker: GLM-5.3-Flash @ http://127.0.0.1:8080 (262k ctx).

## Objective
GLM-5.3-Flash authors, in one long multi-turn session, a working Python toolkit `~/maya-longtask/benchkit/`:
1. `bench.py` — single-stream probe bench (prose/facts/code) + concurrency ladder (C1/C2/C4), token
   counts from `usage.completion_tokens` only (never chunk counts), 2-run steady-state means.
2. `functest.py` — functional battery: chat, forced tool call (OpenAI tools API), SSE stream,
   greedy determinism (same prompt twice → byte-identical), long-ctx needle (~15k), vision-free.
3. `monitor.py` — health sampler: polls /v1/status + engine log `glm stat` lines, appends JSONL,
   threshold alerts (decode < 20 t/s, vram hit < 60%, server down).
4. `report.py` — renders results JSON → single self-contained HTML report (tables, no JS deps).
5. `cli.py` — `python -m benchkit {quick|full|monitor|report}` tying it together.

## Stages (each = one model turn; validator gate before next)
- S1 SPEC: model writes SPEC.md from this plan + fed context (server facts, our bench history).
  GATE: file >400 chars, names all 5 modules.
- S2 CORE: benchkit/bench.py. GATE: ast.parse OK; smoke 1-probe run vs live server returns t/s > 5.
- S3 BATTERY: benchkit/functest.py. GATE: ast.parse OK; live run ≥ 6/8 checks pass.
- S4 MONITOR: benchkit/monitor.py. GATE: ast.parse OK; 20s sample run writes ≥2 JSONL records.
- S5 REPORT+CLI: benchkit/report.py + cli.py + __init__.py + __main__.py.
  GATE: ast.parse all; `python -m benchkit quick` end-to-end writes report.html.
- S6 FULL RUN + ANALYSIS: driver runs `python -m benchkit full`; model reads results JSON and
  writes REPORT.md: numbers, regressions vs our known baselines (59.2 mean thinking-on), verdicts.
  GATE: REPORT.md has real measured numbers matching results JSON.

## Agent loop protocol (driver: longtask_driver.py)
- Conversation persisted to session.jsonl; resume-capable (stages_done.json).
- Model must emit files as fenced blocks tagged ```python:benchkit/<name>.py — driver extracts+writes.
- Validator failure → stderr/stdout fed back as next user message; up to 3 retries per stage.
- temperature 0.3 (code), reasoning_effort default (high), max_tokens 12000/turn.
- Context growth is a FEATURE (long-task test): full history kept until 230k, then oldest stage
  exchanges compacted to summaries (keeps recent code verbatim).

## Failure playbook (inference side — Hermes fixes)
- Server down / conn refused → check `systemctl is-active maya`, restart via
  `sudo systemd-run --uid=1000 --gid=1000 --unit=maya ... run-maya-s24-tuned.sh`, driver auto-resumes.
- Load killed by oomd → must be system slice (launcher already does this); verify journalctl.
- HTTP 5xx / stall → engine watchdog (STRATA_WATCHDOG_S=180) auto-recovers; driver retries w/ backoff.
- Context overflow → driver compaction pass.
- Degenerate loop → engine loop-guard closes it; driver detects empty answer, re-prompts once.

## Monitoring (Hermes)
- Poll driver log + `glm stat` engine lines + nvidia-smi every ~3 min.
- Record per-stage wall time, out-tokens, t/s; alert if decode < 20 t/s sustained or errors > 2.

## Success criteria
All 6 stage gates green; benchkit runs end-to-end; REPORT.md numbers within noise of our measured
baselines; total session ≥ 30k context tokens exercised; zero unfixed inference failures.
