# DSA blockers — ranked, tested, and what actually matters (2026-10-26)

Question: which current blockers can DSA (the indexer/sparse-attention family) resolve, ranked, with custom tests.

## The measured foundation (tonight, live engine v1.1.28+recall)

`ctx_scaling.py` (shipped) — one fresh prefill per depth, engine-line honest numbers:

- **decode is FLAT 25.9–28.8 t/s from 2k → 190k ctx** (needles hit at every depth incl. 189k)
- prefill: 214 t/s @1.6k → 925 @17k → ~1200 @70k–145k (plateau), 1169 @189k
- engine map: decode reads only 128 dense window rows + ≤512 top-k selected compressed rows per query per layer — the DSA design is already fully in the engine, with a candidate pre-filter (blk=8 max-pool, top-2048 blocks) activating above ~32k tokens

**Consequence: attention/indexer is NOT the decode wall. The wall is CPU-lane expert streaming over PCIe.** No DSA algorithm fixes that; it's MoE weight traffic.

## Ranked blockers (DSA-resolvable only), with verdicts

**R1. Indexer scan growth at deep ctx — PARKED (not binding today).**
Mechanism today: all-pairs FP32 rescore of the whole compressed history every 16-token window (never cached), bounded by the candidate filter above 32k. Evidence it doesn't bite: flat decode curve to 190k. The fix if/when we chase 512K–1M ctx is HISA (arXiv 2603.28458): B=128 mean-pooled block keys (cached incrementally) → top-m=64 blocks → token-level refine only inside them; 2–4× kernel-level at 64K, 1.65× e2e / 1.70× TTFT in SGLang, training-free, near-DSA needle/LongBench. Our engine already has the two-stage SHAPE (cand_blocks); HISA's contribution = pooled B=128 stage-1 (128× fewer stage-1 keys than our blk=8) + always-on. Effort: moderate (new pooled-key cache + restrict_ rewrite). **Do it when we run 512K prompts, not before.**

**R2. `DS4_INDEX_TC=1` tensor-core indexer scoring — TESTED, REJECTED.**
The one config-level DSA lever found. Exact int8 mma scoring vs FP32 CUDA-core. A/B (2-run bench + 24k ctx probe): decode/facts/code all within noise of baseline; prefill 882 vs 925 t/s = no gain (indexer isn't the cost); determinism 4/6 flips vs baseline-fresh-record 0. The engine's own comment warns TC sums differ in last bits and greedy runs diverge. **Verdict: off (default). No speed, real quality risk.**

**R3. Short-prompt prefill overhead (214 t/s @1.6k) — REAL, but not DSA.**
~500ms fixed cost per prefill: window/chunk setup, cold expert misses, host round-trips. Indexer scoring at these lengths is trivial. Lever class = engine scheduling + expert preheat, not sparse attention. Park under engine work.

**R4. Cross-layer index reuse (IndexCache-style) — ALREADY OWNED.**
Pack runs index sources {2,8,14,20,24,28,32,36} shared across 61 layers, KV sources {2,8,14,20}: the engine natively does what IndexCache papers add to other stacks. No action.

**R5. Incremental top-k maintenance (rescore-every-N-steps, O(new+k) merge) — viable but low value now.**
Would kill the per-window rescore (16× fewer full scans). Only pays when R1 pays (512K+). Same parking condition.

**R6. SM86 reality checks (from research, saves future wasted effort):**
- No FP8 tensor cores / FP8 ALU on Ampere: FP8 indexer-GEMM "speedups" don't exist here. INT8 store + dequant into FP16 MMA is the working pattern (what INDEX_TC does — and see R2).
- Attention gather at our scale ≈ 0.65MB/layer × 61 ≈ 40MB/step ≈ <0.1ms at DRAM bw — the sparse gather kernel is NOT worth rewriting; FA2-Ampere (mma.m16n8k16 + cp.async double-buffer) is the reference if ever needed (FlashMLA Ampere port exists, cached in dsa_research/).
- SGLang's DSA knobs (SGLANG_NSA_FUSE_TOPK, KV_CACHE_STORE_FP8, USE_REAL_INDEXER) map to things we already have or rejected.

## New quality finding (root-caused tonight, not DSA)

**Determinism wobble on exact-repeat probes after the record warms: 'Red, blue, yellow' vs 'red,...' (3–4 of 8 trials).** Root cause chain: Recall drafts depend on RECORD STATE, which grows between trials → accepted drafts change 16-token window composition → FP4 group boundaries shift → last-bit logit ties flip. Fresh record → deterministic (all earlier 7/7 gates). DSpark alone is deterministic (model drafts, same every trial). Needle/chain-math/ascii unaffected — only exact-repeat tie tokens. Options: (a) accept + document (chosen for now: answers stay correct), (b) offer-floor bump for response-position matches, (c) per-conversation record scoping (kills the cross-request copy win). Recorded here; decision left explicit.

## Full research ranking (agent report, verbatim verdicts + engine-shape corrections)

The research pass (12 primary sources cached in `dsa_research/`) ranked all training-free DSA techniques. Items not already covered above:

- **#5 DeepSelect fused top-k kernel** — threshold+refine topk (no sort), topk=512 is literally our config; 2–20× vs torch.topk. Our engine already has a custom topk (block a row + segmented variant ≥32k rows), so the win here is porting DeepSelect's threshold-refine INSIDE our kernels. Low effort (2–4d), low priority (topk not measured binding).
- **#6 INT8 indexer-key storage** — halves key-cache traffic (dequant in-kernel, bf16 GEMM unchanged). Our keys are ALREADY FP4-packed (kKvFp4p, 0.5B/row × 128-d) — we're 4× past INT8. Already owned.
- **#8 MISA head-subset routing** — router picks 8 of H_I indexer heads per query; 1.47× e2e prefill at 256K, claims to beat HISA on avg quality. Our indexer: 32 heads × 128-d. Worth benching in the 512K era alongside HISA — the two compose (MISA cuts heads, HISA cuts tokens).
- **#9 JSSA low-rank projected score cache** — r=64 projection of the 2048-d score space, ~32× score-traffic cut, 6.43× decode vs full-KV (B200, GLM-4.7-Flash). Needs offline SVD calibration on captured q·k. Again a 512K-era item; pairs with #1.
- **#10 Batched score GEMMs** — layer-batch the 8 shared source indexers' scoring into one wide GEMM per window (scheduling only). Cheap, do if profiling ever shows the score stage.
- **#11 References** — llama.cpp PR #21149 is a complete hand-rolled CPU DSA (closest analog to our engine, incl. lightning indexer + lineage-bench validation method); vLLM issue #30644 confirms NO ready SM86 sparse-MLA kernel exists (FlashMLA sparse = SM90/100 only) — validates "keep our custom kernel" stance.
- **#3 dense/sparse crossover** — llama.cpp: "DSA works exactly like MLA up to 2048 tokens"; SGLang switches at index_topk threshold. Ours: nsel = min(512, (p+1)/R) naturally saturates at ~1k tokens (R=2) — the crossover is emergent in our engine, no dispatch needed.
- **#12 excluded (need training)**: NSA (natively trained, 943.7B tokens), HashAttention (learned hashes), Quest-as-is (page min/max, coarser than our per-token indexer), DSA-from-dense. Our model HAS the trained V4-lineage indexer — everything above exploits it without touching weights.

**Correction to the agent's derived numbers**: its "65GB/step indexer traffic at 262K ⇒ ~70ms/step" assumed V3.2's 2048-d per-layer keys × 61 layers. Our engine: 128-d FP4-packed keys, 8 shared source layers, ≤16-query windows, candidate-filtered above 32k — the real scan cost is orders below that, which is exactly what the flat decode curve (25.9–28.8 t/s to 190k) confirms empirically. Ranking logic survives; the traffic math does not apply to our shape.

## Bottom line

DSA machinery in this engine is already at/near the quality-optimal config for ≤256K ctx, and the decode wall is expert PCIe traffic, not attention. The two real levers remain: #476 single-GPU 2× decode path (untested, needs a one-GPU boot) and expert-side work. HISA-style indexer rework is the one DSA item worth doing — when 512K prompts become a workload.

Artifacts: `ctx_scaling.py` (v2), `dsa_research/` (12+ cached primary sources), `/tmp/dsa-report-clean.json`, bench JSONL tags `v1128-base`, `idxtc`, `idxtc-ctx`, `v1128-cap`.
