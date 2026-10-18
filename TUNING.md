# 2x RTX 3090 tuning — three quants, one rig, everything measured

Branch: `tuned-exl3-2x3090` (based on upstream **Project Maya v1.1.15**).
Rig: 2x RTX 3090 24 GB (PCIe 4.0 x16, NVLink), Threadripper PRO 3955WX (Zen 2, no AVX-512),
128 GB DDR4. All numbers: 262K context, steady-state `bench.py` (3 probes, run-2), engine stat
lines quoted where useful. Full per-step history in [CURRENT-SETTINGS.md](CURRENT-SETTINGS.md)
and the research notes; this file reads **newest first**.

---

## 2026-10-18 — v1.1.15, all three quants tuned and validated

| Quant | Decode mean | Engine stat | Prefill (27K) | Notes |
| --- | ---: | ---: | ---: | --- |
| **Maya-S24** (GGUF, Q4_K attn) | **63.6 t/s** | 70.3 | 1258 t/s | speed king |
| **Maya-S v2 IQ2_XXS** (2bpw) | **55.2 t/s** | 61.5 | — | smallest pack |
| **EXL3-3.05bpw** (turboderp) | **50.0 t/s** | 52.7 | ~990 t/s | quality tier |

Journey on this rig (same hardware, same battery): S24 45 → 59.2 (v1.1.11 campaign) → **63.6**;
2bpw ~40-45 → **55.2**; EXL3-3.05 37.3 → 43.6 (TRUNK_KEEP) → **50.0**.

### What landed in this round
- **Upgrade to upstream v1.1.15** (33 commits): official Zen-2 AVX2 CPU-lane tier (our backport
  superseded), tensor-core decode attention (`MLA_MMA`, default on sm_80+), route warp top-k,
  LRU expert eviction, zai's current chat template. May be worth ~1 t/s by itself here — the
  big wins below are the levers on top.
- **`STRATA_GLM_MISS_SKIP_W`** (upstream #228, new): leave out non-resident experts carrying
  less than W of the route's weight, from layer 23. Swept per-request (`strata_tune`), no
  restart. Knee measured per quant: **0.14 on S24 and EXL3, 0.12 on 2bpw**. Payoff:
  S24 58.2→63.6, 2bpw 52.1→56.2, EXL3 44.5→50.4 (+9-13%).
- **FETCH_WIDE kernel** (this fork, EXL3 only): 32-byte wide loads + expert-parallel grid for
  the RAM→VRAM copies. Kept from the v1.1.11 campaign.

### Validation (per quant, on the shipped config)
- `functest.py` 8/8 (chat, forced + zero-arg tool calls, multiturn + tool result, SSE, vision,
  17K mid-context needle, greedy determinism) — S24, 2bpw, EXL3 all pass.
- Thinking suite (EXL3): 9/10; the one miss (a marbles question at thinking-off) reproduces
  byte-identically with both trim levers OFF — a pre-existing EXL3-3.05bpw trait (greedy
  chain-math settles on a consistent wrong value; thinking-on is correct).
- Arithmetic A/B (5 probes, temp 0): EXL3 byte-identical between quality and speed tiers.
  On the GGUF quants temp-0 is not bit-deterministic (CPU-lane expert placement flips between
  runs — upstream documents this); the lever's answers stay within the OFF tier's own
  run-to-run variance, so the gate there is functest + variance-band comparison.

### Speed vs quality tiers
`MISS_SKIP_W` and `TRUNK_KEEP` change answers by design (upstream-documented KL: 0.14 ≈ 0.052,
TRUNK_KEEP=6 ≈ 0.02). On EXL3 they only pay TOGETHER (measured: both off 44.5, MISS_SKIP alone
44.1 — a noise-band tie, TRUNK_KEEP alone 43.8, **both on 50.0**). Ships as:
- **Speed tier** (default in our configs): trims ON.
- **Quality tier**: delete `MISS_SKIP_W`/`TRUNK_KEEP` from the config env — EXL3 44.5, or use
  the EXL3 pack with 3x lower KL than the GGUF quants (turboderp's chart: 0.0681 vs 0.19-0.23).

### Per-quant configs (config jsons are local; env blocks recorded here)

Common: `--max-context 262144 --prefill auto --kv-streaming on --expert-cache auto`

```
S24  (GGUF):  RAM_RESIDENT=1 KV_INT8=1 PROMOTE_MIN=6 SPEC_SAMPLE=1 SPEC_CANCEL=1
              PCIE_SHARE=0.25 MISS_SKIP_W=0.14 MISS_SKIP_FROM=23
2bpw (GGUF):  the same minus PCIE_SHARE, MISS_SKIP_W=0.12
EXL3-3.05:    RAM_RESIDENT=1 KV_INT8=1 SPEC_CANCEL=1 SPLIT=22 TRUNK_KEEP=6
              PREFETCH_N=2 AT=fetch RANK=4 BLOCKS=8 FETCH_WIDE=1
              MISS_SKIP_W=0.14 MISS_SKIP_FROM=23
```
GGUF ≠ EXL3: `PROMOTE_MIN` pays on GGUF and is toxic on EXL3; EXL3 wants split balancing and
the prefetch lane instead. `--calibrate` (v1.1.15) confirmed engine defaults elsewhere.

### The honest ceiling
Config space ends here: 70 t/s on 2x3090 needs engine work, not configuration —
1) a multi-draft verify window on the 2-GPU layer split (per-position state save/restore;
upstream's pipelined window needs 3+ GPUs), 2) SpecFit-class cost-aware draft trees (shape the
verify subtree by the per-layer expert UNION — also why DFlash2 loses on byte-bound rigs), 
3) per-layer batched DMA (`cudaMemcpyBatchAsync`, CUDA 12.8+). Dead ends are recorded with
receipts in the research notes (DFlash2 verify doubling fetches, `--parallel` on layer-split,
manual CPU-lane overrides, power-limit raises, PROMOTE_MIN on EXL3, split 23/24).

---

## 2026-10-15 — v1.1.11 campaign (previous round)

TRUNK_KEEP=6 speed tier on EXL3-3.05 (43.0-43.6 t/s), SPEC_CANCEL + SPLIT=22 on quality tier
(37.3), 262K context validated with a 194K needle, sustained-bench receipts over 5 runs, and
the GGUF S24 ladder to 59.2 t/s (KV INT8 → PROMOTE_MIN → RAM_RESIDENT, `--calibrate`'s
PCIE_SHARE=0.25). Details: [CURRENT-SETTINGS.md](CURRENT-SETTINGS.md),
[EXL3-50Tps-RESEARCH.md](EXL3-50Tps-RESEARCH.md), [EXL3-CURRENT-FINDINGS.md](EXL3-CURRENT-FINDINGS.md).

---

## Credit

- **Engine: Project Maya** by Peasant Smith and contributors — everything here runs on their
  work; this fork adds one kernel (FETCH_WIDE) and tuning documentation.
- **Quants**: Maya's own (S24, S v2) and **turboderp's EXL3-3.05bpw**.
- The tuning uses the engine's own documented levers; no novel engine changes were needed.
