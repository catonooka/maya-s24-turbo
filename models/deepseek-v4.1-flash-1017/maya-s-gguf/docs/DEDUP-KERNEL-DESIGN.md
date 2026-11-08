# DSV41 expert-dedup research → kernel design (2026-10-19)

## Sources mined
1. **maya/strata engine source** (local, ~/project-maya/src) — full CUDA source of the
   50 t/s GLM rig. Key: `glm_fast.cu` moe_route_kernel (~L3365-3569) does inline slot promotion:
   device-side residency table tab[layer*n_exp+expert] -> VRAM ptr, read/updated by the ROUTE kernel
   itself — first token to miss claims a slot; all later tokens in the SAME STEP hit VRAM. Zero host
   round-trip, zero extra launch. Window kernels (`win_gu/win_down/win_combine`, exl3.cu:685-736,
   driver win_experts():1505) process the whole multi-draft window as one grouped launch, weights
   dequantized ONCE per expert per launch (block_gemv `nc` activation-reuse, kTok=8).
   Measured union multipliers: T=2/3/4 tokens → 1.75x/2.4x/3.05x fewer expert-reads than per-token.
   glm_window.cu:1141-1151 keeps sort/unique stats; report at :1729.
2. **J&M Labs GB300 GLM-5.3 campaign** (al-engr.com) — pro team, same problem: masked_row_copy =
   46.6% of decode; slot cache 33.8→43.1 t/s (+27%); +MTP(1) → 54.7; their "amortize miss bytes
   across accepted tokens" = maya's window idea. Measured dead: draft-routing prefetch (chance-level
   0.0316 vs 0.0312), launch coalescing (bytes-bound), K=2 MTP (quality gates + 44% more misses).
3. **Our dsv41 image** (subagent-mapped): decode = ONE stock `exl3_moe` launch/layer, UVA pointer
   tables (gate/up/down trellis ptrs per expert), all 384 experts/layer cold (host-plan = all).
   thor split branch (`xmoe_set_force_group`) ABSENT from this .so = dead code. EC replans every
   DSV41_EC_EVERY=16 steps off decayed scores; per-step batch-union counts ALREADY exist on device
   (`l._ct["counts"]`, atomicAdd in ft_tier_cu_v.cu:32).

## Blocker-by-blocker resolution map
| Blocker | Fix from research | Cost |
|---|---|---|
| EC admission latency (16-step staleness) | EC_EVERY env → then per-step admit patch in ct_vllm.py:309-333 | env / ~40 py lines |
| Cross-token expert re-stream in exl3_moe | maya win_experts port: one grouped launch per layer-step, dequant once per expert | C++ kernel, days |
| EC hit ceiling 37% (600 slots, VRAM full) | per-step admission makes each slot count ~2x more (J&M: hit-rate is the #1 lever) | same patch |
| Draft amortization | ngram verify window feeds the grouped launch (union of 4 draft tokens ≈ 2.4x fewer reads) — only pays AFTER grouping exists | later |
| Prefetch | DON'T (J&M measured chance-level; maya doesn't bother) | — |

## Ordered patch ladder (each gated by bench_decode.py + 17*19=323 + selftest)
1. EC_EVERY sweep 16→4→2→1 (env only)            [running now]
2. Per-step admission python patch (ct_vllm.py)   — admit prev-step's top misses each step
3. win_experts-style grouped MoE launch in the fused path: compute unique experts per step
   (already have counts), dequant each once into scratch, run all rows against it
4. ngram spec re-test on top of 3 (verify window now amortizes)

## Risk notes from maya source reading
- exl3_moe's pointer-table indirection (per-expert ptrs) is actually GOOD for us: a grouped
  kernel can keep the same ABI, just change WHO reads (one block per expert, rows gathered),
  like maya's route-kernel promotion table.
- CUDA-graph safety: ct counts/seq pattern (device counter) is the proven pattern; keep all
  per-step logic in device tensors written by the graph, host reads only in ec_step (outside).
- The J&M team could NOT beat slot-cache policy with fancier replacement (LRU/static/prefetch
  replay: <0.1% miss cut). Speed comes from ADMITTANCE LATENCY + amortization, not policy.
