# llamAmpere research (JakeATX/llamAmpere) — what transfers to maya GLM stacks
2026-11-07, after the S24 87.8 t/s push.

## What llamAmpere is
llama.cpp fork for RTX 3090/3090 Ti (sm_86), Qwen3.8 family. 121-125 t/s decode +
2,900 t/s prefill @100k on ONE 3090 + big RAM, Qwen3.8-Flash-Next 80GB AI3 file.
MIT. Successor of llama-cpp-qwen-ampere; based on TheTom/llama-cpp-turboquant.

## Finding 1: maya ALREADY absorbed this lineage
- maya's vendored ggml (third_party/llama.cpp) carries `mmq-config-ampere.cuh` —
  IDENTICAL to llamAmpere's for all standard types (IQ2_XXS/Q4_K/...). The only
  diff: llamAmpere adds their private types (PQ2_0, PTQ1_0, AI4, AIT, EXL3).
- maya upstream commit abe3b45d offered the community GSQ-RCO 3.5-bit GLM quant,
  later REPLACED by maya's own quants (Maya-S/S24/M/XS) — S24 IS the evolved
  product of that cross-pollination.
- => No drop-in kernel patch left to steal for sm86 MMQ. The kernel deltas
  (mmq.cu/vecdotq/dequantize, 150-450 lines each) are their new-type code plus a
  different llama.cpp base.

## Finding 2: our walls are BYTES, their wins are COMPUTE-regime
- Our EXL3 tier (3.05bpw, 125G): wall = bytes/token through 72G VRAM (97.5% hit,
  power 143-170W = bandwidth-bound). llamAmpere's EXL3 GEMV (exl3-gemv.cu,
  mma.m16n8k16, weight-major, coop launch) is the same design class as maya's
  src/kernels/cuda/exl3.cu (mma fragments, 128-out blocks, k-split). A kernel port
  does not move a bandwidth wall. REJECT for EXL3.
- S24 tier: wall = CPU lane (3.6-5.1 ms/tok) + ram fetches (4-7.4/tok) + machinery.
  MMQ tweaks move GPU compute that's hidden under those stalls. Low ceiling.
- llamAmpere's headline regime: 80GB model, ONE card, model mostly fits VRAM+RAM
  with their compact expert formats → their compute tuning pays. Ours is a
  3-GPU DRAM-offload regime — different physics.

## Finding 3: the ONE transferable idea = their AI+ overlay concept
"Each expert smaller than its standard copy, so the card holds ~1,300 more of them"
= attack bytes/expert. In maya's world that is the PACK LADDER, not code:
Maya-XS 73.4 GB < S24 94.7 < S 96.5 < M 116 < L 156.3.
Maya-XS on this rig (72 GB VRAM + 247 GB RAM) should go ~fully VRAM-resident
→ near-zero ram fetches + fewer CPU-lane experts → next rung toward 100.
RISK: catalog lists quality "-" (unmeasured) — gate before trusting it.

## Other llamAmpere items checked, not applicable
- SJ-KVaRN / TurboQuant KV types: maya's KV is its own latent int8 — different machinery.
- mb-MTP (keep more drafted words): maya analog = SPEC_SAMPLE + spec tune (spec=8 knee found).
- n-gram drafter +16.4%: maya has Engram already.
- 32K prompt passes / 4096 microbatch: maya --prefill 32768 (we run it) + PREFILL_SUB auto.
- fattn-mma-turbo: maya has its own tensor-core prefill attention (mla_attn, #10 lineage).
- Adaptive draft depth 3-4 (+1.5% over fixed 4): noise-level for us.

## Action taken
Bench Maya-XS + DFlash2 + --prefill 32768 on the 3-GPU rig (results appended below).

## XS bench result (the AI+ overlay idea applied via maya's pack ladder)
Maya-XS (IQ1_S experts, 73.4 GB) + DFlash2 + p32k on 3x3090: decode mean 50.0 t/s
(61.3/50.5/38.1) vs S24's 87.8. Prefill 819 @2k. REJECTED for this rig.
Law found: the bytes/token curve is U-SHAPED below ~2.3 bpw on Ampere+this engine —
IQ1_S dequant compute per byte eats the byte savings (and likely hurts draft
acceptance too). S24 (IQ2_XXS_S) is the knee. llamAmpere's own flagship file is
~3bpw-class, same lesson from their side.
Model kept at ~/Maya-data/models/Maya-XS-IQ1_S (69G) pending user decision; raw
download dirs and hf .cache leftovers removed.
