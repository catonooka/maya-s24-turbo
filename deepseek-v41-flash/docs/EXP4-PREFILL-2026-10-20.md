# Exp 4 — prefill chunk / decode-share sweep — BASELINE OPTIMAL, CLOSED

Maya/D107-inspired sweep of the prefill knobs never swept on OUR mixed-K pack.
DSV41_MAX_NUM_BATCHED_TOKENS (chunk) x DSV41_DECODE_SHARE, warm 2154-tok prefill
+ 256-tok decode:

| config | prefill t/s | decode t/s |
|---|---|---|
| chunk 8192 DS 0.25 (L1b default) | **1164** | **10.0** |
| chunk 4096 DS 0.25 | 908 | 9.89 |
| chunk 16384 DS 0.50 | 879 | 9.37 |
| chunk 16384 DS 0.25 | 863 | 9.53 |

L1b's defaults are the optimum of this sweep on our pack — smaller chunks lose
DMA batching, bigger chunks thrash the KV/graph budget; decode-share above 0.25
steals from decode. (D107's 1024-chunk win was on THEIR uniform Mia pack +
thor xmoe hooks we lack — not transferable.)

## Exp 5 — copy-engine fetch staging: CLOSED EARLIER by measurement
Already covered by campaign data (commits b5c6d35, 5ea3143): mode-1 decode DMA
= 1.4 t/s (catastrophic), mode-2 asserts without thor hooks, DMA_GEMM env =
+15% prefill but decode-negative. No new run needed.

## Campaign summary after Exps 1-5 (all one-by-one, each benched + committed)
| Exp | Idea (source) | Result |
|---|---|---|
| 1 | route bias to EC residents (maya TRUNK_KEEP) | NEGATIVE 10.1-10.3 vs 10.65; hit 47→30% (churn feedback) |
| 2 | tier depth/limit (maya #188 env-level) | FLAT 10.0-10.2 (K2 gate; C kernel is the real lever) |
| 3 | EC profile persistence (maya usage file) | mechanism VALIDATED, bench-neutral, kept as infra |
| 4 | prefill chunk/share sweep (D107) | baseline OPTIMAL (8192/0.25) |
| 5 | copy-engine staging | closed earlier (negative, commit b5c6d35) |

Env/config space on this stack is now EXHAUSTED top to bottom. The remaining
levers are all code: (a) L3 grouped-fetch / split-K C kernel (maya #188 proper,
the big one), (b) fixed-pool RB variant (Xolotl STRATA_RB_POOL), (c) maya-ds4
engine (built + packed on disk, dual-GPU native).
