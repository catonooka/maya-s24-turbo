# Exp 6 + 7 — fixed-pool RB and EC decay/hysteresis — NEGATIVE / FLAT, CLOSED

## Exp 6: fixed-pool route bias (Xolotl STRATA_RB_POOL=M shape)
Fix for Exp 1's churn failure: bias toward a STABLE per-layer top-M set
learned from the (enriched, 8-task) EC profile, applied ONE-SHOT at start,
never rewritten (src/ct_rb_pool.py, DSV41_RBP_MULT / DSV41_RBP_M,
serve-rbp.sh). Profile enriched first via tests/enrich_profile.py.

| config | decode t/s | EC hit |
|---|---|---|
| baseline | 10.0-10.65 | 47% |
| RBP M=12 mult 0.5 | 10.0-10.2 | 33.8% |
| RBP M=24 mult 0.5 | 9.7-9.9 | 33.3% |

STILL NEGATIVE. Root cause now measured across the whole family (5 data
points incl. Exp 1's three): on this stack a swapped pick = a full PCIe
miss (~1.45 GB/token regime), and no top-M pool captures enough of DSV41's
routing mass to pay for the misses it causes. Maya/Xolotl win because their
bias chooses between TWO CORRECT paths (VRAM vs RAM-resident); ours chooses
between fast and slow. Route bias: CLOSED for vllm_exl3 + host-offload.

## Exp 7: EC decay / hysteresis (tuned for Mia pack, never ours)
| config | decode t/s | EC hit |
|---|---|---|
| baseline (decay 0.9, hyst 1.0) | 10.0-10.65 | 47% |
| decay 0.95 hyst 1.5 | 10.17-10.31 | 27.2% |
| decay 0.85 hyst 1.0 | 10.0 (sweep cut short) | ~30% |

FLAT-to-negative; 0.9/1.0 stays optimal. EC env space fully swept this
campaign (EVERY, SLOTS, MARGIN, ADMIT earlier; DECAY, HYST now).

## HONEST PREFILL NUMBERS (bench methodology fix)
tests/bench_prefill.sh reused the same prompt -> prefix-cache hits were
reported as "warm prefill" (1144-1164 t/s). New tests/bench_prefill_cold.sh
generates a UNIQUE prompt per run (verifies via cached_tokens when the API
exposes it):

| prompt | TRUE cold prefill |
|---|---|
| ~7.2K tok | 276-294 t/s |
| ~14.5K tok | 166 t/s |

Real prefill on novel text = ~290 t/s @7K (and it degrades at longer
contexts as KV grows). The 375 t/s@4K figure from the stock-image era is
consistent with this. Decode numbers were never affected (prefix cache
covers prompts only).

## Campaign final state
All env/config-level levers measured and closed. Keepers: L1b env (best)
+ EC profile persistence (neutral-to-positive infra). Remaining levers are
code-level only: L3 grouped/split-K kernel, maya-ds4 engine (built+packed
on disk).
