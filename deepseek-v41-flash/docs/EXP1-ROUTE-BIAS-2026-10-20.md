# Exp 1 — Route bias toward EC residents (maya TRUNK_KEEP port) — NEGATIVE

Date: 2026-10-20. Idea from the maya/Xolotl family (their measured wins: bias
0.5/0.3 = 21.7 t/s vs 9.7 exact on the dsx 2-bit Xolotl build; maya ships
TRUNK_KEEP/MISS_SKIP approximate routing with a KL budget).

## What was built
- `src/ct_route_bias.py` — after every EC replan, adds `bonus * residency_mask`
  to each layer's `gate.e_score_correction_bias` IN PLACE (graph-safe: dsv4_topk
  reads the bias by address each launch, same property as the EC pointer tables;
  pristine base copy kept; rewrites skipped when EC residency is unchanged).
- Bias tensors found via one-shot gc walk at start() (S["layers"] holds the
  experts modules; the bias Parameter lives on the parent FFN).
- `src/ct_vllm.py` — chained loader (execs stock ct_vllm_stock.py into its own
  namespace, then layers experiments; env-driven, stock behavior when unset).
- `serve-rb.sh <mult>` — boots with DSV41_RB=mult (bonus = mult x mean std of
  the bias tensors).

## Measured (bench_decode.sh 256 tok, warm; hit_rate from ct stats)
| config | single t/s | EC hit |
|---|---|---|
| L1b baseline (no RB) | 10.65 | 47% |
| RB mult=0.15 (bonus 0.009) | 10.12-10.14 | 30.0% |
| RB mult=0.25 (bonus 0.015) | 10.15-10.22 | 32.8% |
| RB mult=0.50 (bonus 0.030) | 10.22-10.31 | 31.6% |

Quality gates pass at all three (17*19=323, Paris/42).

## Why it loses here (root cause)
1. **Feedback loop fights itself**: our bonus keys on CURRENT residency, and the
   EC's admission keys on OBSERVED counts — which the bias just perturbed. The
   routed distribution shifts, EC churn rises (admit/evict cycling), hit rate
   drops 47 -> 30-33%, and each replan re-biases toward the new set. Maya/Xolotl
   bias toward a STABLE property (their own VRAM trunk / a fixed top-M pool
   learned per model), not a churning mirror cache.
2. **Near-tie margin is thinner here**: DSV41-Flash sigmoid scores are flatter
   than the dsx 2-bit builds Xolotl tuned on; even 0.15x std swaps real picks
   (hit -17pts) without enough consolidation to pay for it.
3. The win in maya comes WITH RAM-resident experts (bias decides GPU-vs-CPU,
   both correct paths). Our misses go to PCIe streaming, so a swapped pick
   costs a full miss, not a lane choice.

## Verdict
CLOSED for this stack shape. Do not revisit bias-vs-EC without first making
the residency target stable (e.g. a fixed learned top-M set — Xolotl's
STRATA_RB_POOL=M — instead of the live EC set).

## Assets kept
- ct_route_bias.py (install(CT), env DSV41_RB) — reusable if we later ship a
  fixed expert pool.
- ct_vllm.py + ct_vllm_stock.py chain — the new experiment harness for Exp 2+.
- serve-rb.sh.

## Next (Exp 2)
CPU-tier split-K memory-order batching (maya #188 shape: read each expert in
memory order as split-K pieces, whole route as one job batch, AVX2 path) —
attacks B11 (cpu_per_call 3.37, gpu_wait 0.72 ms/call measured today).
