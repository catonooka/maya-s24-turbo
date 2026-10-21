# v1.1.15 three-quant ladder — 2026-10-18 (EXL3-3.05bpw crosses 50 t/s)

The 2026-10-15 EXL3 era ended at 43-45 t/s (speed tier). This round upgrades the
engine to upstream **v1.1.15** and tunes all three GLM-5.3-Flash quants on the
same rig (2× RTX 3090, Threadripper PRO 3955WX Zen 2, 128 GB — later 247 GB),
all at 262K context:

| Quant | Decode mean | Engine stat | Prefill @27K | Notes |
|---|---:|---:|---:|---|
| **Maya-S24** (GGUF) | **63.6 t/s** | 70.3 | 1258 t/s | speed king |
| **Maya-S v2 IQ2_XXS** (2bpw) | **55.2 t/s** | 61.5 | — | smallest pack |
| **EXL3-3.05bpw** | **50.0 t/s** | 52.7 | ~990 t/s | quality tier, 50 t/s goal hit |

Journey: S24 59.2 → 63.6 · 2bpw ~45 → 55.2 · EXL3 43.8 → 50.0.

## What landed
- **Upstream v1.1.15** (33 commits): official Zen-2 AVX2 CPU-lane tier (our #194
  backport superseded), tensor-core decode attention (`MLA_MMA`, sm_80+), route
  warp top-k, LRU expert eviction, zai's current chat template.
- **`STRATA_GLM_MISS_SKIP_W`** (upstream #228): drop non-resident experts carrying
  less than W of the route weight, from layer 23. Swept per-request via
  `strata_tune`, no restart (`bench_missskip_sweep.py`). Knee per quant:
  **0.14 S24/EXL3, 0.12 2bpw**. Payoff: S24 58.2→63.6, 2bpw 52.1→56.2,
  EXL3 44.5→50.4 (+9-13%).
- **FETCH_WIDE kernel** (this fork, EXL3 only): 32-byte loads + expert-parallel
  grid on the RAM→VRAM copies — kept from the 2026-10-15 campaign
  (`glm_fast.cu`).

## Validation (per quant, on the shipped config)
- `functest.py` 8/8 (chat, tools, multiturn+tool result, SSE, vision, 17K
  needle, greedy determinism) — all three quants pass.
- Thinking suite (EXL3) 9/10; the miss (marbles at thinking-off) reproduces
  byte-identically with trims OFF — pre-existing EXL3-3.05bpw quant trait.
- Arithmetic A/B (temp 0): EXL3 byte-identical between quality and speed tiers;
  GGUF within OFF-tier variance (CPU-lane placement makes temp-0
  non-deterministic there).

## Files
- `TUNING.md` — full newest-first history (as of the round)
- `EXL3-50Tps-RESEARCH.md`, `EXL3-CURRENT-FINDINGS.md` — the 50 t/s research base
- `bench.py`, `functest.py`, `vision_probe.py` — the bench/validation tools
- `bench_missskip_sweep.py`, `bench_tune_sweep.py` — per-request lever sweeps
- `glm_fast.cu` — FETCH_WIDE kernel (drop-in for `src/kernels/cuda/glm_fast.cu`)
- `config-exl3-3.05bpw.json` — EXL3 serve config snapshot
- `expert-usage-glm.txt` — per-layer expert route counts (profile analysis)

## Honest ceiling
70 t/s class needs engine work (2-GPU multi-draft verify windows, SpecFit-class
trees, batched DMA) — not more flags. MISS_SKIP changes answers by design
(measured KL small; see TUNING.md) — gate before adopting on quality tiers.

## Credits
Engine + levers: **@mw00** (project-maya v1.1.15, #228 MISS_SKIP_W, #194 AVX2
lane, #188 split-K). Quant authors: Maya-S24/S/IQ2_XXS (@mw00), EXL3-3.05bpw
(turboderp). FETCH_WIDE kernel + sweeps + validation: this repo.
