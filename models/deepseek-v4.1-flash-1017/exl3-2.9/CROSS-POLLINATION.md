# DSV41 -> GLM/maya cross-pollination — what the sybil stack taught us (2026-10-18)

Both rigs are the same physics problem: MoE decode bounded by expert bytes/token
crossing PCIe to a 24G card. sybil's dsv41-flash-offload solved parts of it differently
than project-maya. Transferable ideas, ranked:

## 1. RAM-tier CPU compute with AVX2 trellis kernels (both have it now)
- maya: upstream #194 (7.05 -> 0.51 ms/expert on Zen 2), #188 split-K
- sybil: ft_core.h AVX2 mul1 K=3 kernels, selftest max_rel_vs_gpu 0.0037
- Both engines converged on: CPU computes ~3-24 experts/token OVERLAPPED with PCIe
  fetches of the rest. Nothing new to transfer — v1.1.15 already ships it.

## 2. VRAM mirror cache with replan (sybil "EC") vs maya promotion pools
- sybil EC: 600 slots x 12.7 MiB, admits/evicts/replans per step, hit rate ~39% warm
- maya: expert-cache auto, LRU eviction (v1.1.15), PROMOTE churn is load-bearing
- Similar; maya's usage-file learning across boots is the edge (sybil has no
  persistence of expert demand).

## 3. DSpark/DSpark-class MoE drafting: KILLED on both
- dsv41: MoE draft streams its own experts -> 11 -> 5.3 t/s (this repo, bench log)
- maya: DFlash2 any drafter 9.4-11.5 t/s (calibration auto-reject)
- Law: on PCIe-bound offload rigs, MoE drafters multiply expert traffic. Only
  dense/lightweight drafters (MTP block) survive. MTP acceptance 62-97% is the lane.

## 4. sybil ideas maya does NOT have (candidate patches)
a) **Host-callback DMA inside CUDA graphs** (cudaLaunchHostFunc + pinned preadv
   staged in prepare_inputs): maya prefetches at fetch point but the sybil repo
   claims graph-safe host-initiated H2D. maya already runs FULL_DECODE_ONLY graphs;
   equivalent = maya's fetch overlap. Low delta.
b) **O_DIRECT row store with 264B/slot direct-mapped cache** (0xSero row_store.cpp):
   for disk-tier experts. We are RAM_RESIDENT (disk reads 0.00/tok) -> irrelevant
   unless we ever exceed 247G.
c) **Selftest-on-boot** (cpu tier parity check vs GPU, rel RMS): cheap, nice gate.
   Worth grafting into functest.py as a maya boot check. ~20 lines.

## 5. What GLM/maya has that DSV41 stack lacks
- 2-GPU layer split + MTP draft overlap (sybil stack is TP1 single-GPU)
- expert usage learning file persisted across boots
- TRUNK_KEEP / MISS_SKIP_W approximate trims with measured KL budget
- vision path, tool-call dialect, 262k streaming KV

## Verdict
The engines converged; the remaining wins are the R1/R2 plan in
project-maya/EXL3-50Tps-RESEARCH.md (AcceptMoE commitment-weighted verify pruning +
bigger MTP window with grouped union reads). sybil adds nothing structural beyond a
boot selftest nicety. GPU1 stays free for a dsv41 co-tenant if we want both models
simultaneously (maya needs both GPUs for the layer split though — pick one model at
a time, or re-split maya to GPU1 only and lose ~half the slots).
