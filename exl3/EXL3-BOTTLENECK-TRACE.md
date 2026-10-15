# EXL3-3.05bpw bottleneck trace + custom patches — 2026-10-15 (v1.1.11, engine commit 8931339)

## Per-token budget (34 ms @ 29 t/s baseline, measured)

| component | cost | evidence |
|---|---|---|
| PCIe expert fetch | ~24-30 ms | 77-90 misses × 9.9 MB = 760-890 MB/token; both links gen4 x16; SM-copy kernel measured 11.5 GB/s vs 25 theoretical; implied util ~82% of 2×25 |
| EXL3 trellis decode kernels | ~8 ms | 42 MoE layers × ~190 µs (PR #180 class); partially overlaps fetch |
| MTP draft+verify | small | spec 84-96% accepted |
| KV/attention | ~0 | kv-streaming 99.6% VRAM hits |
| Disk | 0 | RAM-resident |
| Cross-GPU hop | negligible | 64 KB/token, decode pipelined (both links pull simultaneously) |

SM util 100% with VRAM-bw util only 16-29% = trellis kernels spin on data arrival. System is
PCIe-bound with an ~8 ms compute tail. Not power-bound (1965 MHz boost held, 250 W < 300 W cap).

## Bottleneck-point research + what we did

### Point 1: prefetch overlap (router prediction) — CUSTOM PATCH, WIN
- Engine's own stats (STRATA_GLM_TIMING): prediction names 5.0-5.6 of 8 experts one layer early;
  48-51% of NON-RESIDENT ones (92607/61551 misses measured). But stock prefetch capped at N=2, 2 spares.
- Our patch: N up to 8 (pf_ids/pf_src/pf_dst arrays already 8-wide — only the cap and the
  navail<2 guard needed lifting; with depth>2 the 3rd spare is used too).
- Tuning matters more than depth: AT=route (stock default) CONTENDS with the layer's own fetch on
  the same link → 26.4. AT=fetch + BLOCKS=8 (few SMs, post-fetch issue) → **30.3-30.6 t/s (+4-5%)**.
- Config: `PREFETCH_N=8 PREFETCH_AT=fetch PREFETCH_RANK=4 PREFETCH_BLOCKS=8`

### Point 2: fetch kernel efficiency — CUSTOM PATCH, WIN
- Stock moe_fetch_kernel: one grid-stride loop over ALL experts (i loop serial inside the kernel),
  16-byte loads, 11.5 GB/s achieved of 12.5 measured ceiling (memcpy 10.7).
- Our patch: moe_fetch_kernel_wide — 32-byte loads (ulonglong4), grid.y = expert index
  (expert-parallel), STRATA_GLM_FETCH_WIDE=1.
- Result: **31.3-31.7 t/s (+8-9% over 29.0-29.2 baseline)**. Decode 33.6→31.4 ms/tok.

### Point 3: split imbalance — RESEARCHED, already optimal
- The auto split search (glm_search_bounds) prices EVERY placement: [0,24)/[24,45) predicted 35.1 ms
  (32.7/35.1 per part); next candidate [0,26) predicted 66 ms. Placement is not the issue.
- Decode is PIPELINED across parts (draft block on last GPU; "a token costs the slower part"),
  so both links fetch simultaneously — budget ≈ 760MB/(2×11.5) — consistent with measurement.
- Levers exposed: STRATA_GLM_SPLIT=K1,K2 manual, STRATA_GLM_SPLIT_LAYER_MS cost override,
  STRATA_GLM_USAGE live usage file (enabled: data/expert-usage-glm.txt, written per request).

### Point 4: hardware paths (research, no purchase made)
- +1× RTX 3090 (24 GB): pool slots scale ~+50% (94-98 slots/layer → ~140+), fetch traffic ÷~1.4
  (three links), predicted decode ~40-45 t/s at same KL. Cost: ~$600-700 used, 3rd x16 slot +
  lane bifurcation check on TRX40 (ours: 2 x16 used; board has 3rd x16 electrical x8 — 5 GB/s
  class, still net positive), +350 W PSU headroom, +~20 W idle.
- AVX-512 host (Zen 4/5 or TR 7000): engine ships an AVX2+FMA EXL3 CPU tier (v1.1.11 changelog:
  "Zen 2/3 and Intel CPUs without AVX-512 ran it in scalar" — meaning AVX-512 CPUs got the fast
  tier all along). On Zen4 the CPU lane ~4× faster than our Zen2 7.05 ms → ~1.7 ms/expert; with
  PCIE_SHARE ~0.5 and the lane SYNC this still adds latency — needs the async lane design (below).
  Verdict: CPU upgrade alone does NOT fix it on this engine; a 3rd GPU does.
- Upstream async lane (filed #187): no movement yet on making the EXL3 CPU lane overlapped.

### Point 5: what upstream v1.1.10/11 changed under us (we rebased + re-measured)
- v1.1.10: DUAL_PREFILL (NVLink pair stages prompt experts through both links — prompt-side only);
  HDD warm order; AMD verify-window multi-expert batching (STRATA_GLM_WIN_MULTI — AMD only).
- v1.1.11: Monitor VRAM breakdown; --expert-cache/--vram-reserve-mib pool sizing.
- None touch the NVIDIA EXL3 decode fetch path; our 29.1 stock-on-v1.1.11 re-measure confirms.

## FINAL config (31.3-31.7 t/s, +8%, quality identical: 8/8 functest, same greedy outputs)
- env: RAM_RESIDENT=1, KV_INT8=1, USAGE=data/expert-usage-glm.txt,
  PREFETCH_N=8, PREFETCH_AT=fetch, PREFETCH_RANK=4, PREFETCH_BLOCKS=8, FETCH_WIDE=1
- args unchanged (131072 ctx, prefill auto, kv-streaming on, expert-profile)
- patches committed locally: aca22b2 (on v1.1.11)
- Concurrency still loses (C2 26.8 agg < 31.5 serial) — unchanged.
