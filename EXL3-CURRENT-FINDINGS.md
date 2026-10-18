# EXL3-3.05bpw — current findings & limits (2026-10-15, final state of the speed campaign)

## Rig
2x RTX 3090 (gen4 x16 each, NVLink), Threadripper PRO 3955WX 16c/32t Zen 2 (AVX2, no AVX-512),
247 GB DDR4-2400 8-channel (all populated, 1 NUMA node), NVMe. Both GPUs: 300 W cap, full boost held.

## Current speeds (verified live)
- **Decode: 35.5-36.4 t/s** thinking-on production bench (best run 36.4; prose 31.7 / facts 41.5 / code 36.0)
- **Prefill: 402 t/s @ 2.7k** (~360 at 12k chars probe)
- Prefill @8K repo protocol: ~1000 t/s class (different bench; see bench.py)
- functest 8/8; thinking 9/10 (chain-math needs effort on — now often correct even off via lane arithmetic)
- Quality: KL 0.068 vs FP (3.05bpw class) ≈ 3x closer than S24's 2.3bpw class

## The journey (decode t/s)
29.2 stock v1.1.11 → 31.7 (+our deep-prefetch & wide-fetch patches) → 35.9 (+#194 AVX2 CPU lane)
→ 35.5-36.4 (+#188 split-K merged; tie on decode, +prefill CPU share). Total +25%.

## Running stack (commit 27b5111, on v1.1.11 = 8931339)
- upstream #194: EXL3 CPU lane AVX2+FMA tier (Zen 2 was SCALAR before — 7.05 → 0.51 ms/expert, 14x)
- upstream #188: split-K CPU expert reads (memory-order streaming), no-barrier job batches, EXL3
  prompt CPU share. Merged by hand onto #194 + our patches; graft: AVX2 band-tier fallback for
  split-K rows + 4 stale exl3_cpu_vector() AVX-512 gates → tier==2 (fixed SIGILL boot crash on Zen 2)
- ours: deep prefetch N=8 @fetch/rank4/blocks8 (uses 3rd spare at depth>2; issue point AT=fetch is
  the critical part — AT=route CONTENDS: 26.4) + wide-fetch kernel (32B loads, expert-parallel grid)
- env: RAM_RESIDENT=1 KV_INT8=1 USAGE=data/expert-usage-glm.txt PF_N=8 PF_AT=fetch PF_RANK=4
  PF_BLOCKS=8 FETCH_WIDE=1 (lane at 8t/gpu default; 16t measured worse 34.2)

## Utilization (all measured, nothing idle)
- RAM: 8/8 channels, ~80 GB pinned experts + lane reads at memory-order (#188's design target);
  host sustained ~63 GB/s of ~134 theoretical = platform-typical efficiency
- PCIe: both links gen4 x16 simultaneous (pipelined 2-part decode); ~45% of former fetch traffic
  now computed on CPU instead (lane: 16-24 experts/token in 10-13 ms overlapped)
- VRAM: expert pools 98.5% full (4272/4336 slots), promotions live, zero disk reads
- SM: 100% during decode; VRAM-bw util only ~25% (expected — trellis kernels wait on data, not bw)
- Not power/clock/thermal bound (1965 MHz, 250 W < 300 W cap)

## The limit (why ~36 is the wall on this hardware)
Per token ~28 ms: PCIe still moves ~53 misses × 9.9 MB ≈ 520 MB (was 860) over 2 links while the
lane computes the rest. To go faster you must move FEWER bytes over PCIe:
1. +1 GPU (3rd 3090): ~$600-700 used, +24 GB slots + 3rd link → predicted ~45 t/s (traffic ÷1.4)
2. AVX-512 host (Zen 4/5): lane ~4x faster → CPU could take ~70% of routes → ~40+ t/s. But a
   platform swap is a bigger investment than a GPU for less certain gain on this engine
3. Smaller pack: S24 (2.3bpw) IS this — 59 t/s, KL 0.19-0.23 class (already on disk, one-command swap)
4. Upstream future: window verify batching on NVIDIA (STRATA_GLM_WIN_MULTI is AMD-only today);
   pipelined fetch queues. Watch #194/#188 merge + follow-ons.

## Measured-negative list (do not retry)
- DFlash2 any drafter on 2-GPU EXL3 (9.4-11.5 t/s: window doubles fetch traffic; Q2_K rejected
  by loader, bf16 crashes #173)
- GGUF env stack on EXL3 (PROMOTE_MIN → 17.8; promotions must churn)
- CPU lane BEFORE #194 (scalar = 20x loss); PCIE_SHARE manual (planner's auto-split is right)
- 16t lane/gpu (worse); --parallel concurrency (C2 27 / C4 28 agg < 36 serial)
- LRU evict (26.5); ctx 64k (tie); spec window 6 (tie); PROMOTE_FILL (neutral)

## Ops notes
- Launch: sudo systemd-run --uid=1000 --gid=1000 --unit=maya run-maya-exl3-tuned.sh (user slice
  oomd-kills the 91 GB pin). Re-diff config after any --setup/--rebuild (ctx/host resets).
- S24 rollback: run-maya-s24-tuned.sh (59 t/s). EXL3 rollback: revert to 95af329 (pre-#188).
- Upstream reports: #187 (ours, 3 datapoints), #194 (thanks + our numbers), #188 (stack test in
  our repo). Public repo: catonooka/maya-s24-turbo (exl3/ has full history).


## SPEED TIER ADDENDUM (2026-10-15 evening): TRUNK_KEEP=6 — 43 t/s
- STRATA_GLM_TRUNK_KEEP=6: route ranks 7-8 of RAM-tier experts left out of decode (skipped, not
  fetched). Decode 37.3 -> 43.0-43.6 t/s (+16%). RAM fetches 47->30/tok, lane 21->11/tok.
- QUALITY: NOT lossless (upstream: KL~0.018 class, changes answers). Our gates at parity: 8/8
  functest, 9/10 thinking (same T3 marbles trait as stock), 5-question probe suite answers
  byte-identical to baseline (31.25 / 5/8ths / 10 / 12 / 80), code/knowledge/translation clean.
- TRUNK_KEEP_MAP 8/7/6 by layer = 41.2 (early layers kept full costs 2 t/s, no measurable quality
  gain on our probes) — flat 6 chosen for the speed tier.
- Tier scheme: speed = TK6 (43), quality = drop TRUNK_KEEP (37.3). One env var apart.
