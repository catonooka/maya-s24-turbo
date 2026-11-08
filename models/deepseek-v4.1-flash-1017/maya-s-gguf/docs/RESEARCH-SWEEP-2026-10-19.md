# Research sweep 2026-10-19 — solution per bottleneck (6 agents, ~115 queries)

## ⚡ KILLER FINDING: our exact rig already runs this model at 15.5-21.3 t/s
sybil-solutions/local-ai-registry PR#152 (0xSero dsv41-flash-offload, campaign D118):
same engine family as ours (our image d141-main is NEWER), 1x RTX 3090 + ~213 GiB pinned RAM,
decode C1 15.55 (D117, 8192 chunks) / **C1 21.34, C4 28.3 agg (D107, 1024-token chunks)**,
prefill 575-765 t/s. Baseline D030 (no CPU tier) = 4.10 — matches our 5.4 start.
Env deltas vs our current serving config:
- EXL3_HOST_DMA=1 + EXL3_DMA_GEMM=1 (staged-DMA prefill, FP16 GEMM for busy experts) ← WE NEVER SET
- VLLM_EXL3_FAT_THRESHOLD=32
- DSV41_EC_ELASTIC=1, margin 900
- KV 4.5 GiB (we run 1 GiB), --max-num-batched-tokens 16384
- Pack: Mia 3.0bpw + FP16 wo_a/compressor overlay (D010); ours is 2.9bpw local (smaller = fine)
- Their rig: EPYC 7443P 8ch DDR4-3200 (204 GB/s) ≈ our 3955WX 8ch DDR4 (204 GB/s)
URL: https://github.com/sybil-solutions/local-ai-registry/pull/152

## Per-bottleneck solutions (ranked, with sources)
- **B1 PCIe bytes**: (a) KT-kernel/Fiddler CPU-compute of cold experts (activations ship,
  not weights) — proven 20+ t/s DeepSeek-V4-Flash on 1x5090+RAM, ktransformers repo;
  (b) EXL3_HOST_COLD_PACK 2bpw cold experts = halve bytes; knee research below;
  (c) CacheMoE cache-aware routing (arXiv 2412.00099): training-free logit bias toward
  resident experts, ~free, slight quality tradeoff.
- **B2 UVA 22-25 vs copy 56 GB/s**: PR152's EXL3_HOST_DMA staged path already in our image
  (prefill side); graph-safe decode fetch = device-side copy kernels over pinned arena
  (maya pattern, IEEE 9845392 'Demand MemCpy'); conditional nodes (CUDA 12.3) to skip
  fetch when resident. Vectorized 32B copy kernels ≈ 12.5; CE-patched memcpy nodes for
  static prefix.
- **B3 no grouped kernel**: exllamav3 exl3_moe_kernel.cuh IS the blueprint (ticket
  scheduler, tiered M-tiles 16/32/64, dequant-B-once-per-tile); Marlin sm_86 matched
  kernel; CUTLASS grouped-GEMM device-descriptor pattern for graph safety. SpecMoE
  (arXiv 2604.10152) validates window-union fetch: 76.7% transfer cut, 4.3x ceiling.
- **B4 EC 47% hit**: power law b≈0.238 (flat!) → +100 slots ≈ +0.03pts; policy dead
  (J&M + SGLang SSD pack agree). Real plays: HOBBIT dual-precision (arXiv 2411.01433,
  9.93x): serve 2bpw copy instantly on miss while full streams — removes miss stall;
  2bpw hot-set = 608 slots (51%), 1.5bpw = 811 (55%).
- **B5 K2 layers**: nothing public; our own ft_core.h patch idea stands (BITS template).
- **B6 MTP layers outside CT**: SpecMoE pins draft experts (N>=topK) in VRAM — dspark
  head is small (128 exp × topk3), fits; eliminates our 5.3 t/s draft-streaming failure.
- **B7 spec dead**: papers give the exact breakeven: T=2 window = 1.71x ceiling (~19 t/s)
  with ngram on doc-workloads; MTP+grouped at T=2 = 1.2-1.4x (13-15 t/s) IF draft experts
  pinned. Grouped fetch must land FIRST. (SP-MoE 2510.10302, AcceptMoE 2608.02989,
  MoE-Spec 2602.16052, BigMoMo 2609.14643 all validate.)
- **B8/B9/B10 both-cards**: 05yuki SGLang+KT dual-EPYC recipe: 27.6-28 t/s decode,
  1M ctx prefill 406 t/s — needs 2-socket 226 GB/s; our single-socket 204 GB/s is close;
  repo = the 18-fix migration guide (github.com/05yuki/dsv41-flash-dual-epyc).
  diffbot 2xRTX-PRO-6000: 74-159 t/s (2x96G VRAM, not our class).
- **B11 CPU tier**: KT-kernel AMX kernels (AVX-512 class) don't run on Zen2; our AVX2
  tier is at silicon limit. PR152 hits 15.5-21 t/s with the same AVX2 tier → not the wall.
- **B12 engram OOM @N=4**: KV 4.5G config in PR152 implies margin exists with elastic EC;
  margin 900 + elastic release covers it.
- **B13 prefill 375**: PR152 D117 = 575-765 t/s with DMA_GEMM=1 → env-only fix.

## Quantization knee (agent 2, HF cards + papers)
- 3.0bpw uniform ≈ lossless (diffbot KL 0.0545 vs noise 0.0158; HumanEval 95.7)
- 2.7bpw MIXED-K (sudoingx: K2/3/4/5 sensitivity-allocated) beats uniform 2.9:
  KLD 0.0243, ppl 1.517, top-1 95.6% — the best sub-3 pack for THIS model
- uniform 2.0bpw = cliff (KL 0.36-0.44, top-1 77-80%) — never
- Engram tables FP8→MXINT-4 (satgeze): 189→97.6G, gates pass
- Rule: hot K4-5 VRAM / cold K2-3 host / never uniform <2.5

## Ordered action plan (post-research)
1. Apply PR152 env deltas (no code): DMA=1, DMA_GEMM=1, FAT=32, elastic, KV 4.5G, 16384
   → target 15.5 C1; then try 1024-chunk decode trick → target 21 C1
2. Pin dspark draft experts in VRAM + retry spec at T=2 (B6+B7) → 13-19 t/s
3. L3 grouped kernel (exl3_moe_kernel.cuh blueprint) → then ngram T=2 → ~19 t/s
4. HOBBIT 2bpw hot-set + instant-fallback (B4) if miss-stall still dominates
5. Consider 05yuki SGLang+KT migration if we want 27+ (big lift, 18 fixes documented)
