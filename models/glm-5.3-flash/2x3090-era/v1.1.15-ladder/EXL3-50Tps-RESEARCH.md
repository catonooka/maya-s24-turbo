# EXL3-3.05bpw @ 50 t/s decode + 1000 t/s prefill — research + ranked methods (2026-10-15)

Baseline: 35.5-36.4 t/s decode, 402 t/s prefill @2.7K (8K-protocol number TBD). Wall = PCIe bytes
(~520 MB/token after lane offload) + verify-window expert unions.

## GitHub finds (researched 2026-10-15)
- mw00/project-maya #204 (open, by zetlaw): STRATA_GLM_SPLIT_ROWS=1..8 — batched decode across
  layer-split GPUs; +27% AGGREGATE on 4× RTX 4000 Ada with 8 requests; single-request ±5%.
  → helps C2/C4 concurrency, NOT single-stream 50.
- #180 (merged in our v1.1.11): tensor-core EXL3 decode (mma.m16n8k16, up to 8 activations as B
  columns!) + prompt row-rotation. The 8-B-column layout is exactly the multi-token shape the
  verify-window path needs.
- alexvnesta/glm53-flash-cmp170hx-tp: SAME model, 2× CMP 170HX 64 GB (all 320B in HBM, NCCL P2P
  TP): LRU-reasoning decode 46.8 t/s, counting 129 t/s, cold prefill 1.49-1.53K t/s, PCIe Gen2.
  → proof: full residency + spec ≈ 47 t/s reasoning on this model class.
- BobClawblaw/glm53-flash-exl3-2x-dgx-sparks: EXL3 4bpw TP=2 on 2× DGX Spark (273 GB LPDDR5X,
  ~273 GB/s): greedy 46-66 t/s, cold prefill 962-1001 t/s @6K-114K prompts.
  → proof: ~1000 t/s prefill happens at ~4× our DRAM bw with bigger chunks.
- vcruz305/vllm-exl3: no CPU executor ("fail-closed eligibility plan" only), GB10-qualified.
  → NOT eligible for us.
- ktransformers: alive but no EXL3/trellis support. FreeToken (rust port exists): GGUF-class MoE,
  no EXL3. → full-system swaps rank last: no trellis kernels.

## arXiv finds
1. OSDI'26 2606.10493 (CPU-GPU hybrid SLP): stream-loading prefill = 3-thread loader/model/unloader
   pipeline at sub-layer granularity → 1200 t/s prefill, 1800 DSLP (2× 5090); AVX-512 FP8 GEMV 4-5×;
   dual-batch attention-MoE overlap +50% batched throughput; intra-node P-D disaggregation.
2. AcceptMoE 2608.02989: THE RERANKING METHOD — verifier-side expert-set selection for MoE spec
   decode under offload: rank experts by commitment-weighted router demand (draft-node accept
   probability × router score), residency-aware pruning of nonresident low-demand experts.
   2.06× under offloading, -73.6..77.1% H2D traffic, -0.27pp accuracy.
3. SpecMoE 2604.10152: self-assisted spec (draft = non-expert parts on GPU) — verify phase COALESCES
   expert fetches: multiple tokens per fetch; -76.7% transfers, up to 4.3×.
4. SpecMoEOff 2508.21706: spec decoding enlarges per-verify expert workload → hides I/O; 2.5× vs
   MoE-Lightning. Same principle: bigger windows = more tokens per byte moved.
5. SPICE 2608.21240: low-rank (LoRE) surrogates for low-confidence misses + CPU residual execution;
   3.12× TPOT. (Approximate = quality risk; EXL3 LoRE would need training.)
6. MoE-SpAc 2603.09983: SD verification as lookahead sensor (activation-frequency utility) → optimal
   GPU/CPU expert partition per layer per step; 42% over SD baseline.
7. ProMoE 2410.22134 / YALIS 2603.19289 / DuoServe 2509.07379: predictor/prefetch/phase-split
   variants — mostly already in our engine (PREFETCH_N=8, phase-split pools).

## Engine facts verified today (live log + source)
- Spec active: MTP acceptance 62.8-97.5% per bench → headroom for larger windows.
- Verify windows: x3::win_experts (WIN_X3, default ON) already GROUPS union reads (moe_win_group)
  for EXL3 — but NO commitment weighting, NO residency-aware pruning. moe_experts_multi is
  `!Ly.exl3`-gated (generic-quant path only).
- Prefill: --prefill auto → 8192-token chunks on the split (both cards read the same chunks;
  smallest card sets pace). Docs: 2× V100 got 709 t/s @8192 vs 551 @32768 — smaller chunks won
  there, but Sparks proves 1000-class needs bigger effective throughput. Sweep required.
- #180's mma layout supports 8 B-columns → grouped multi-token expert compute is kernel-ready.

## RANKING (fused, by expected value for OUR 2×3090 + Zen2 + 247 GB)
R1. AcceptMoE-style verify-window reranking (CUSTOM PATCH) — commitment-weighted expert demand +
    residency-aware pruning; pruned experts take the CPU lane (exact compute, no PCIe). No new
    training; reuse acceptance stats the engine already logs. Est fetches 53→~30-35/token.
    → +25-40% decode. Accuracy: -0.3pp class (gate with greedy probes + functest).
R2. Larger MTP window + WIN_X3 grouping (CONFIG + small patch) — SpecMoE/SpecMoEOff principle:
    coalesced verification = more tokens per byte. Try n=8-10 (acceptance 85-97% supports it).
    → +10-25% on top of R1 (compounding: fewer fetches × more tokens/fetch).
R3. Prefill 8K fair bench + chunk sweep (4096/8192/16384/32768) + prestage/LEND_PCT/PREFILL_MB
    (CONFIG ONLY) — SLP overlap is already the engine's shape; find our optimum. Est 402→600-800.
    1000 needs R4-class bw or 4bpw.
R4. #204 SPLIT_ROWS — concurrency play: C2/C4 aggregate 27-28 → maybe 35+ (pull PR, it's clean).
    Not single-stream 50.
R5. 3rd 3090 (~$600-700, x8-elec slot) — proven-class jump to ~45+; stack with R1-R2 → 55-60.
R6. NOT ranked in: vllm-exl3 (no CPU executor), ktransformers/FreeToken (no trellis), SPICE LoRE
    (training + quality risk), P-D disaggregation (single-request irrelevant).

## Fused plan (the "one" recommendation)
Decode-50 = R1 + R2 together: prune the verify union by commitment AND enlarge the window so the
remaining fetches amortize over more tokens. Bytes/token 520 → ~300 MB → ~45-50 t/s if acceptance
holds ≥85%. Gates: greedy determinism probes, functest 8/8, thinking suite, marbles/knaves traits.
Prefill-1000 = R3 first (free); if short, the honest answer is bandwidth: 4bpw pack (165 GB, still
KL 0.035-0.07 class) or 3rd GPU. Both S24 (59 t/s) and 4bpw-on-Sparks (46-66 t/s + 1000 t/s
prefill) show quant+residency is the other axis.

Sequence: R3 (config sweep, 1 hr) → R2 (window sweep, 1 hr) → R1 (patch, the real work) → R4 (PR
pull, optional). Every step re-gated.
