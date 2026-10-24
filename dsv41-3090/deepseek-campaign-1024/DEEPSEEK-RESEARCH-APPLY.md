# DeepSeek inference research → maya-ds4 apply list (2026-10-24)

Sources: 3 parallel research agents (DeepGEMM repo+design, DeepSeek inference overview + V3/V3.2 papers, DeepEP + co-design paper arXiv 2505.09343). Full reports: ~/.hermes/cache/delegation/subagent-summary-0-*.txt, task-1/2 logs. Local copies: ~/ds_day6_inference_overview.md, ~/ds_v3_report.txt, ~/ds_v32_paper.txt, ~/ds_flashmla_fp8_deepdive.md, ~/deepinfer_readme.md.

## What DeepSeek does (verified, with transfer verdicts for 2x3090 TP2 + Zen2 + 247G RAM)

| # | Technique | DeepSeek mechanism | Our rig verdict |
|---|---|---|---|
| 1 | Prefill/decode phase split | different parallelism per phase (EP32 prefill vs EP144 decode), opposite bottlenecks (compute vs bytes) | MAYA already phase-splits (prefill TP2 compute path vs decode CPU-lane/PCIe). No action. |
| 2 | Two-microbatch overlap (prefill) | comm of mb A hidden behind compute of mb B | Engine work (pipeline overlap exists in layer-split). Config: none. Patch candidate later. |
| 3 | Pure-overlap hooks (DeepEP) | RDMA issued, SMs freed, hook = data-dependency point | Analog = copy-engine streams prefetching L+1 while computing L. Engine's PREFETCH/STREAM family is this. → unpin MAYA_DS4_STREAM=0 (v1.1.25 #329 judges whole decode steps) |
| 4 | Hot-expert redundancy | 32 duplicated hot experts, refresh ~10 min | Engine's expert-profile + VRAM/RAM tiers already do this (13248 ranked pairs, GLM line). DS4 pack has expert-profile.bin. No action. |
| 5 | MTP/DSpark spec decode + confidence scheduling | draft<=5, verify while cumulative confidence >= threshold | DSpark on + DSPARK_CUM=0.5 already = DeepInfer's scheme exactly. Sweep 0.3? small. |
| 6 | FP8/bf16 precision split | FP8 dispatch, bf16 attention/combine | SM86: no FP8 TC. DS4_AR_F16 (half allreduce bytes) default ON in v1.1.24+. Verify receipt. |
| 7 | Quantized transport over slow links | 1 byte/elem over wire | EXL3 2.9bpw already ~quantized transport. No action. |
| 8 | MLA KV smallness | 576 elem/tok/layer | Model-given (V4.1 has it). Engine handles. No action. |
| 9 | Sparse/dense attention crossover (L~3000) | masked-MHA below, DSA above | Model-given (CSA/HCA in V4.1). Engine handles. No action. |
| 10 | SWA bounded replay (DeepInfer) | re-prefill only 140-tok suffix on decode re-entry | = MAYA_DS4_PROMPT_SKIP lineage (layers 20-39 only last ~2700 tok). Ours pinned =2 but multi-GPU → INERT (docs: single-GPU only). Cleanup. |
| 11 | Grouped/masked GEMM one-launch MoE (DeepGEMM) | contiguous M-grouped persistent kernel, device-side masks | Engine already groups (win_group dedup, our prior design doc). No action. |
| 12 | CUDA graphs for decode | kill launch overhead | Engine has graphs where safe. No action. |
| 13 | cuBLASLt tiering + 16MiB workspace cap | route simple GEMMs to Lt | Engine-internal. No action. |
| 14 | Persistent kernels, unaligned blocks, L2 swizzle | scheduling | Engine-internal kernels. No action. |

Dead on SM86 (do not chase): FP8 tensor cores, TMA/clusters/DSMEM, UE8M0 hardware path, TMEM/tcgen05, PDL, setmaxnreg, remote mbarriers.

## APPLY LADDER (one lever per boot, fixed bench_ds4.py, 2-run means, ±2 t/s = tie)

Baseline v1.1.24 (this session): decode essay 21.6 / facts 20.4 / code 21.8; prefill 1166 t/s @33k; cache hit 58-77%.

- [ ] L1: v1.1.25 binary swap (build/maya-ds4-v1125) — ships #329 whole-step PCIe-lane judging, #393 calibrate fix, #394 slot checkpoints. Isolates engine delta.
- [ ] L2: unpin MAYA_DS4_STREAM=0 → default on (PCIe streaming lane; DeepEP-hook principle). Requires v1.1.25 for #329.
- [ ] L3: MAYA_DS4_Q8_BATCH=1 (#382 same-bits batched rounding).
- [ ] L4: DS4_AR_ZC=1 (P2P zero-copy allreduce over the 26 GB/s bridge + BAR1 32G; F16 already on).
- [ ] L5: --prefill-chunk 16384 A/B (with #369 MMA + P2P, re-test the 32k-chunks-worse finding).
- [ ] L6: MAYA_DS4_DSPARK_CUM 0.3 A/B (looser verify = more accepted drafts).
- [ ] L7 (ops): MAYA_DS4_SLOT_KEEP=1 — conversations survive restarts (agent-workload latency, no t/s claim).
- [ ] Cleanup: drop MAYA_DS4_PROMPT_SKIP=2 (inert on TP2).
- Quality gate at end: 17*19=323, greedy determinism x2, needle, plain-ASCII garbage check.

Engine-work candidates (NOT config; park for later): two-microbatch prefill overlap, AcceptMoE verify-window reranking (from earlier EXL3-50Tps research), grouped-fetch widening.
