# DeepSeek-V4.1-Flash-1017 on 2× RTX 3090 — the full campaign record

**748B-parameter MoE served on two 24 GB cards + 247 GB DDR4: 10.4 → 33-35 t/s
decode (3.4×) and 144 → 783 t/s cold prefill (5.4×), fully validated for
quality and vision.** Local files only — zero model downloads throughout.

This is the complete history (29+ commits, newest-first docs inside `docs/`)
of the DSV41 campaign on the same rig as the GLM-5.3-Flash work in this repo.
Engine: [mw00/project-maya](https://github.com/mw00/project-maya) `maya-ds4`
(GLM side of that repo is the `tuning/`+`exl3/`+`v1.1.15-ladder/` eras here).

## Final result (2026-10-21)

| Config | Decode | Prefill @6K | Prefill @25K |
|---|---:|---:|---:|
| vLLM custom stack (L1b, GPU0) — where we started | 10.4 | 305 | 166 |
| maya-ds4 v1.1.20, tp1 layer-split | 33-35 | 95 (windowed) | 144 |
| **maya-ds4 PR#344, tp2 (final)** | ~30 | 400-442 | **783** |
| maya-ds4 PR#344, tp1 (chat mode) | 33-35 | 314 | 516 |

Flip `--tp 1|2` in the config for chat-mode vs long-doc mode.

## Validation (GLM-unit standard, 2026-10-21)
9/9 functional (QA, reasoning, JSON mode, tool calling, 8K needle, stability,
multi-turn) · 5/5 vision (color/count/spatial) · 10/10 quick QA · 21.8K-token
document recalled both needles · Anthropic + OpenAI APIs both pass.
`tests/quality_gate.py`, `tests/vision_gate.py`, `tests/functional_battery.py`.

## The levers that mattered (each measured, each committed)
1. **Engine switch vLLM→maya-ds4** — tiered experts (VRAM hot set / CPU lane /
   bulk chunk DMA) vs per-token PCIe streaming: 10.4 → 17.5 t/s
2. **tp1 layer-split** (whole experts per card) — both cards packed, 3071
   experts in VRAM: → 33-35 t/s decode
3. **`--threads on`** — host thread per GPU unlocks chunked prefill on the
   split layout: 144 → 518 t/s (was 16-token windows)
4. **PR#344 build** (multi-GPU prompt skip + all-reduce off SMs): 518 → 783
5. DSpark conf 0.5, `MAYA_DS4_DSPARK_CUM=0.5`, `MAYA_DS4_STREAM=0`,
   `DS4_CACHE_THREADS=4`, chunk 8192 — each swept, numbers in the docs

## Measured and rejected (so nobody re-tests blind)
Route bias ×5 configs · tier depth/admission · EC decay/hysteresis · prefill
chunk/share/DMA/HYB · `--cpu-threads` (disables lane auto-calibration) ·
`--reserve-mib 100` (starves KV, context=0) · PROMPT_SKIP on split layouts
(v1.1.20 gate) · chunk 16384 · EP2/PP2/TP2-in-vLLM (all-to-all latency,
cross-layer KV breaks PP) · DS4_INDEX_TC (neutral, drops exact math).
Root causes with source line references in `docs/EXP10-BLOCKER-MAP-2026-10-21.md`.

## Honest blockers (documented, not hidden)
- Decode 50 t/s: needs a smaller quant (S24-class, 94.7 GB download — blocked
  by the no-download rule) or an EXL3 stream-lane patch (engine surgery)
- Prefill 1000: 783 reached; rest is upstream (PR#344-class work at tp8)
- The 44-53 t/s community rows are GLM on small quants, **not** DeepSeek

## Layout
```
docs/      full experiment history (RESTORE, EP2, MAYA-DS4-PATH, EXP1-11,
           TUNING, VALIDATION, FINAL-FINDINGS, SESSION capstones)
tests/     bench_decode/prefill (honest: unique prompts, cached=0), quality
           + vision + functional gates, p2p test, run logs
src/       the vLLM-era custom code (ct_vllm chain, EC profile, route bias)
deploy-*/  final serve script + config snapshot (container pattern)
```

## Credits
**@mw00** + project-maya contributors (the maya-ds4 engine, v1.1.18-20,
PR#343/#344, #228); PR#344 author (ds4-perf); issue reporters #329 (STREAM=0),
#302 (CPU-lane sweep), #280 (tp2 chunk datapoint); **0xSero/sybil**
(dsv41-flash-offload image + pack tooling); **05yuki** (dual-EPYC recipe);
**pfeifferj** (GSQ quant line); **Mal/Indras-Mirror** (Xolotl engine + docs);
papers: SpecMoE / HOBBIT / CacheMoE / AdaEDL / AdaptiveSpec / MoESD / ProMoE.
