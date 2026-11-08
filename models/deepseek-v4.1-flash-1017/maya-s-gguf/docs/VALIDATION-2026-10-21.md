# Validation Report — deepseek-v4.1-flash on maya-ds4 PR344 (2026-10-21)

GLM-unit-style full validation before locking the stack.

## Config under test
exe maya-ds4-pr344, --tp 2 --threads on --dspark on --dspark-conf 0.5
--prefill-chunk 8192; env DSPARK_CUM=0.5 STREAM=0 CACHE_THREADS=4
PROMPT_SKIP=2 (answers may shift slightly on >=1024-row prompts;
set 1 for exact — KL 1.9e-5..5.4e-7 per PR#343)

## Results
| Suite | Score | Notes |
|---|---|---|
| Quick QA gate (temp 0) | 10/10 | exact answers |
| Functional battery | 9/9 | QA, reasoning, JSON mode, tool calling (get_weather correctly called w/ Hanoi arg), 8K needle, 6x stability, multi-turn memory |
| Vision gate | 5/5 | solid red/blue ID, 3 and 5 black-square counting, left-half color |
| Long-doc 21.8K tok | PASS both needles | start-fact + end-fact recalled, 24.5s total (prompt+answer) |
| Anthropic API /v1/messages | PASS | thinking+text blocks, correct answer |
| OpenAI vision | PASS | 64x64 image → "Blue", 10.0s |

DS4_INDEX_TC=1: decode 29-31 (no gain), quality 10/10 — dropped (keep exact).

## Fixes made during validation
- serve-ds4-dual.sh: added numpy to pip installs (vision path needs
  Pillow+numpy inside the container; was 400 "pictures need Pillow and numpy")
- tests/vision_gate.py: 700-token budget (thinking models burn short budgets
  in reasoning; 64 was a test artifact), canvas-fit fix for 5 squares
- tests/functional_battery.py: stability test 120→900 tokens (same artifact)
- tests/quality_gate.py: 10-Q temp-0 gate (new)

## Conclusion
Model serves CORRECTLY: text, reasoning, JSON, tools, vision, long context,
both API dialects, stable across 6 sequential temp-1.0 gens. The "failures"
seen were all test-budget artifacts, not model/server defects.

Speeds at validation: decode ~28-31 tp2 / 33-35 tp1; prefill 400-442 @6K,
783 @25K (PR344 multi-GPU prompt skip engaged).
