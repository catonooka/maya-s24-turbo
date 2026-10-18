# DSV41 Flash EXL3-2.9bpw — bottleneck audit & 50 t/s roadmap (2026-10-18)

## Measured bottlenecks (single GPU0, best config)
| # | Component | Measurement | Verdict |
|---|-----------|-------------|---------|
| 1 | PCIe link (host→GPU) | 17.5 GB/s burst, ~12.5 GB/s step-avg; ~9.8 GiB/token streamed | THE wall: ~1.45 GB/token → 85 ms step |
| 2 | CPU tier | 5.1 experts/call, gpu_wait 1.0-1.2 ms/call, 35/40 layers (K2 layers bail) | healthy, absorbs what it can |
| 3 | EC VRAM cache | 600 slots, 38.7% hit warm | working; misses go PCIe |
| 4 | Engram disk tier | 0.00 disk reads/tok, Rss 0 (pure mmap+memcpy gather) | at research parity, nothing left |
| 5 | Batch scaling | N=1: 8.6 → N=2: 11.1 → N=4: 10.9 tok/s aggregate | FLAT: no expert-union dedup in engine → batching is NOT the lever |
| 6 | Speculative | DSpark: 5.3 (MoE draft streams own experts). ngram: 6.5 @100% accept (verify re-streams union) | both dead on PCIe-bound rigs |
| 7 | GPU compute | SM 75-100% during step but at 180ms/step = streaming-bound | not compute-bound |

## Killed paths (do not retry)
- `--tensor-parallel-size 2`: engram assert patched (works) → dies `svh shard=0 dest (25600,) != loaded (12800,)` — EXL3 width-sharding of svh doesn't match pack layout (padded-geometry path also TP1-only, raises NotImplementedError)
- `--enable-expert-parallel`: expert sharding WORKS (384/192 per rank, host-plan full-width, no slicing error) → host OOM at layer 23: both ranks pin ~98G experts + BOTH mmap the full 189G engram tables (Rss cold but address space + page cache double-buffer)
- Dual single-GPU instances: 2×197G pinned > 247G RAM → host OOMKilled
- ngram / DSpark spec: see table
- CPU quota, NUMA pinning: already maximal (30 cpus, single node)

## The only road to 50 t/s: make EP=2 fit RAM
Patch plan (in /home/aisever/dsv41-patch/):
1. engram_disk.py: split table rows by rank (each rank mmaps only its half via offset+shape), or
   share the mmap via /dev/shm mapping between workers (same pages counted once).
2. Re-test EP2 boot: expect each rank ~98G pin + ~95G half-tables ≈ 193G total → fits 247G.
3. If boots: bench. EP dispatch rides NVLink NV4 (~78 GB/s peer) instead of PCIe for the
   cross-rank expert traffic — this is the DeepSeek-native shape for this rig.
4. Fallback if engram tables can't split: quantize tables EngramQ4-style (189G→95G) then split.

## Secondary wins if EP lands
- EC slots rebalance (margin tune) per GPU
- K2 layers (18-22) into CPU tier (ft_core.h BITS=3 template patch) — 5 more layers off PCIe
- EXL3_HOST_COLD_PACK 2bpw cold experts (halves bytes) with quality gate

## Current serving state
- `dsv41` container, GPU0, :8000, config: CT_LAYERS=35, EC_MARGIN_MB=300, no spec decode.
- Best measured: 11.75 t/s decode single stream, 375 t/s prefill @4k. Smoke 17*19=323 correct.
