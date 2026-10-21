# Exp 8 — maya prefill ideas, env-level sweep on the HONEST (cold) bench — 2026-10-21

User call: real prefill numbers only (tests/bench_prefill_cold.sh, unique
prompt per run — the old bench's "1144 t/s" was prefix-cache hits).

## Maya prefill idea list (from repo study) vs what our stack exposes
| maya idea | our env surface | measured (cold, ~6.2K tok) |
|---|---|---|
| PREFILL_HELP: idle GPU streams a share over its own PCIe (#270) | none — needs engine work | n/a (code lever) |
| 16K-32K chunks (v1.1.18 ds4) | DSV41_MAX_NUM_BATCHED_TOKENS | 32768 = CUDA OOM at util 0.90; 16384 measured earlier = no gain on cold bench |
| row-rotation instead of weight re-decode (#180) | none — kernel change | n/a (code lever) |
| least-used experts to CPU during prompts (#188) | DSV41_CT_HYB_MAX | 197 first / 303 steady — no gain |
| staged DMA prefill | EXL3_HOST_DMA=1 | 152 first / 307 steady — no gain |

## Cold sweep results (~6.2K-token unique prompts)
| config | first-run | steady |
|---|---|---|
| L1b baseline | 181 | **306** |
| EXL3_HOST_DMA=1 | 152 | 307 |
| CT_HYB_MAX=2048 | 197 | 303 |
| DMA + HYB | 178 | 302 |
| chunk 32768 | OOM (4 GiB alloc > free at util 0.90) | — |

First-run variance (152-197) is page-cache state of the 197G pack, not config.
Steady cold prefill = **~305 t/s** on every config — env-invariant, exactly like
decode. D107's prefill wins do not transfer without their thor xmoe hooks.

## Conclusion
Env-level prefill work is exhausted too: ~305 t/s is this stack's cold-prefill
ceiling on our pack. The maya prefill gains live in engine code we'd have to
port (PREFILL_HELP dual-GPU share, row-rotation) — same bucket as the L3
kernel. Meanwhile the honest reference numbers for the repo:
- decode: 10.4-10.5 single / ~15 agg@4
- cold prefill: ~305 t/s @6K, 166 @14.5K
- prefix-cache hit (same prompt re-sent): ~1100 t/s (legit feature, not a bench)
