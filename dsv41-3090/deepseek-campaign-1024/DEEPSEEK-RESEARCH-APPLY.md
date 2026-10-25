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

## Decode regression RCA (2026-10-24 evening, post-campaign)

"33-35 t/s" decomposed — NOT a v1.1.24/25 regression:
1. OLD 33-35 was REPEAT asks (same 65-tok prompt, 4th-5th repeat, expert cache 80%+ hot, CPU lane 16-20%). Fresh-ask steady state is 26-29 engine-side on ANY binary. First-ask after boot: 24-26.
2. L2 unpinned MAYA_DS4_STREAM=0 -> on TP1 boots the PCIe stream lane engages and LOSES on this RAM-heavy rig (~3-4 t/s). Pin restored.
3. Campaign config ran --tp 2 (prefill shape). TP2 costs ~4-6 t/s decode vs TP1 layer-split on this engine+pack (26-28 vs 31-35 warm).
4. v1.1.20-pr344 vs v1.1.24/25 binaries: measured ties at both TP1 and TP2.
5. Client-side bench t/s reads ~4-5 under engine t/s on fresh prompts (includes ~1-2 s prompt read) — engine lines are the honest number.

RESTORED production config: exe=maya-ds4-pr344, --tp 1, env = DSPARK_CUM 0.5 + CACHE_THREADS 4 + STREAM=0 + PROMPT_SKIP=2 (TP1: applies) + Q8_BATCH 1 (new, free) + SLOT_KEEP 1 (new).
Decode: 30.8-32.5 warm / 26-29 fresh engine-side. Tradeoff: TP1 prefill @33k = ~700 t/s vs TP2 1161 (flip --tp per workload, or engine rework for the nm>1 stream lane).
Housekeeping: maya.sh failed bootstrap rm -rf'd the serve .venv (rebuilt); DS4_PROF=1 works on v1125 binary only in serve mode via config log.

## TP2-for-prefill without losing decode (2026-10-24 night, researched)

Q: can we run TP2 prefill speed + TP1 decode speed together?
A: NO at runtime - TP is a boot-time WEIGHT LAYOUT (TP2 slices every layer's tensors across cards; TP1 assigns whole layer ranges). One process = one layout.

Measured matrix (pr344 binary, all cold-cache):
| shape | decode warm | decode fresh | prefill 33k | prefill 100k |
|---|---|---|---|---|
| TP1 layer-split | 30.8-32.5 | 26-29 | 737 | 835 |
| TP2 + PROMPT_SKIP=2 | 24.8-26.8 | ~22 | 1069-1071 | 1149 |

- PROMPT_SKIP (layer-20-39 skip) is ONE-GPU-ONLY per engine source+docs ("several GPUs read prompts as before") - inert on both our shapes; no free lunch there. Measured tie with/without on TP2 confirms.
- TP2 decode cost = allreduce round-trips per layer per token (cards P2P only via driver staging ~1.7ms).
- SLOT_KEEP=1 makes flips cheap: conversations survive restarts.

Shipped: flip-tp.sh (sudo bash flip-tp.sh 1|2) - one command, ~5 min boot. Default TP1 (decode); flip to TP2 for big-prompt sessions.
Engine-work path if we ever want both at once: port the prompt-skip to multi-GPU (cut at chunk boundary) OR port the PCIe stream lane to nm>1 - both are engine patches, tracked above.

## Multi-GPU prompt-skip: no patch needed (2026-10-24 late)

Investigated patching yoco_layer() for multi-GPU. READING THE SOURCE CLOSELY: the multi-GPU skip path is ALREADY IMPLEMENTED (cut at chunk boundary, rows go to groups whole); the guards only disable it when (a) flow chunks exist (STREAM lane) or (b) Engram layers sit at/after the cut (layer 20). Our pack: Engram at layers 1,14 (before cut), STREAM=0 => skip ALREADY ACTIVE on TP1 split.

Measured A/B @103k cold (TP1): skip on 769 vs DS4_YOCO=0 710 t/s => +8%, answers exact (mode 1).
Mode matrix: 0=off, 1=exact skip (DEFAULT, bit-stable), 2=DeepSeek 128-replay (answers shift a little). Production set to 1.

Determinism investigation (quality concern, resolved):
- Thinking-prose rewording run-to-run (0/4 identical on short probes) = DRAFT + CPU-lane expert-path variance, documented trait; FINAL answers converge identically.
- With dspark off: 4/4 identical on structured probes; colors probe still varies in reasoning wording only.
- Gate updated: chain-math expected value corrected (362 not 465 — operator error, model was right).
FINAL gate 7/7: chain-math, 17x19, needle 2k, needle 30k (exercises skip path over cut), ascii, lines4, determinism.

FINAL production (verified): pr344, TP1, dspark on, PROMPT_SKIP=1, prefill ~810 t/s @103k cold (737@33k), decode 24-32 t/s.
