# DSV41 Campaign — Final Findings & Lessons (2026-10-20/21)

## Result
| Stage | Decode | Prefill @6K | Stack |
|---|---|---|---|
| Start | 10.4 t/s | 305 t/s | vLLM L1b (custom, single GPU) |
| End | **33-35 t/s** | 95 t/s (tp1) / 384 (tp2) | maya-ds4 v1.1.20, dual 3090 |
| Multiple | **3.4x** decode | — | — |

## The winning stack
- Engine: mw00/project-maya v1.1.20 (src/ds4, built sm_86 in ubuntu:24.04
  container for glibc 2.39 compat; binary `maya-ds4-container`)
- Pack: our local EXL3 2.9bpw + engram shards, packed by tools/ds4/pack.py
  (17M, no weights copied) — ZERO downloads throughout
- Topology: **tp1 layer-split** (CUDA0 L0-42, CUDA1 L19-39+head, ~3,071 whole
  experts in VRAM, both cards packed ~20.8G) — beats tp2 on decode
- tp2 tensor-parallel = prefill mode (384/516 t/s) — one flag to flip
- DSpark spec decode conf 0.5 (0.4/0.6 measured worse) + MAYA_DS4_DSPARK_CUM=0.5
- Expert profile persists in the pack; boot placement improves with use
- Serving: docker `maya-serve`, dashboard+API :8080, restart =
  `~/maya-ds4-main/serve-ds4-dual.sh`

## Root-cause lessons (why vLLM lost)
1. Host-offload vLLM is per-token PCIe-streaming-bound (~1.45 GB/token);
   maya-ds4 is tiered (VRAM hot experts / CPU lane overlap / bulk chunk DMA)
   — the win is architecture, not tuning.
2. Route bias (maya TRUNK_KEEP family) does NOT port to a churning EC mirror
   cache: bias->perturbed counts->EC churn->retarget loop (hit 47->30%,
   5 data points). Bias only pays between two CORRECT paths (maya's RAM-vs-
   VRAM), never into a PCIe miss.
3. Env space on sybil d141 image is EXHAUSTED (Exp 1-8 all measured): EC
   every/slots/margin/admit/decay/hyst, tier depth/limit, prefill chunk/share,
   DMA, HYB — baseline optimal.
4. EP/TP/PP in vLLM all dead on this rig (all-to-all latency; cross-layer KV
   breaks PP; svh layout breaks TP). The maya layer-split solves it natively
   via per-layer window-state hand-offs through host-mapped pinned memory —
   NO NVLink needed (ours is inactive since the GPU0 swap anyway, P2P 10.4GB/s).
5. memlock: bare-host runs die at NVRM map failures (8MB default limit);
   everything heavy runs in docker with --ulimit memlock=-1.
6. PROMPT_SKIP=2 (v1.1.19) does nothing on tp1 prefill (serialized split pass
   is the wall) and churns the cache; DSpark CUM filter is a keep.

## Infrastructure gotchas solved
- Mixed-K pack for sybil image: expand_native_dense + drop 43 bias_vl +
  true layer_bits (K2 L18-22 / K3 rest / K4 MTP) — RESTORE-2026-10-20.md
- maya-ds4 --ids is a comma list (no --ids-file, no --temp on one-shot);
  one-shot greedy hits early EOS on raw ids — use --serve for real benches
- Pack engram check needs shards 47/48 symlinked beside the EXL3 safetensors

## Watch-list (free wins later)
- TensorFold gate #827: would serve our exact EXL3 2.9bpw with grouped-GEMV
  CUDA path — check monthly
- ktransformers #2213: official 1017 support (needs >450G RAM today, no)
- maya updates: 3 releases in 72h; git pull + container rebuild ~15 min
- ONLY path meaningfully past 35 t/s: Maya-S24 quant 94.7G (fits VRAM+RAM,
  community 44-53 t/s on our hardware class) — blocked by no-download rule
  until user says otherwise

## Benchmarks (honest)
- bench_maya.py (decode, chat, temp 1.0), bench_maya_prefill.py (unique
  prompts, cached=0 verified) — prefix-cache-hit numbers are NOT speed
- vLLM fallback intact: serve-ec-ab.sh 4 (10.4 t/s) if maya ever breaks
