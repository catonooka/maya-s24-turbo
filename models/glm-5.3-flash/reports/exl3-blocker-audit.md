# EXL3 full blocker audit — every layer, measured status (2026-11-07 night, post DEFER_CPU=2)

Engine: EXL3-3.05, DFlash2 Q4_K_M, 3x3090 split 15/15/15, 262k ctx, RAM_RESIDENT
(all 13,536 experts pinned), KV int8 streaming 32768, FETCH_WIDE, P2P, PF_CE,
DEFER_CPU=2, p32k, server-default spec=8. Decode stat windows: 65.9-80.7 t/s.

## Layer-by-layer status

| # | layer | measured state | verdict |
|---|---|---|---|
| 1 | Draft acceptance (creative prose) | code 74-75, prose 44-46; spec_min_p 0.3/0.5, spec_cum 0.9, spec_n_min ALL LOSE (61.7-67) | WALL. DFlash2's own gating optimal; needs a better drafter model |
| 2 | Verify window | spec=8 knee (12 flat, 16 backfires); now server-default | DONE |
| 3 | CPU lane threads | 14 (of 32 logical) OPTIMAL: 20/26/30 all worse (73.6→67.0; SMT/dequant contention) | DONE - do not touch |
| 4 | CPU lane overlap | DEFER_CPU=2 (+30%); stat shows lane 8-15ms/tok mostly hidden; each part's last layer still waits (structural) | DONE (structural tail remains) |
| 5 | Expert tier VRAM hit | 93-95.6% hit, 4.8-9.3 ram fetches/tok, promotions ~2/tok | NEAR-COMPLETE; fetches are cold-tail experts, bytes-bound |
| 6 | Tier adapt cadence | Strata #4870 2/192: flat at our hit rate | CLOSED |
| 7 | Light-miss trimming | TRUNK_KEEP_MAP alone +5%; ON DEFER: -6.4 (starves overlap) | CLOSED (keep off) |
| 8 | Drafter fidelity | Q6_K flat (68.9 vs 69.1) | CLOSED |
| 9 | CPU pool topology | CPU_SHARED=0 (per-GPU pools) worse 62.2; shared pool optimal | CLOSED |
| 10 | PCIe fetch kernel | FETCH_WIDE on (32B loads); PF_CE flat-kept | DONE |
| 11 | Split layout | auto-priced 15/15/15 under DFlash2; manual MAYA_GLM_SPLIT untested — re-pricing uses usage counts, already optimal-ish | LOW ceiling (~1-2%) |
| 12 | KV cache | int8 + streaming (window 32768); FP8 KV not in maya's GLM path | DONE; window sweep only matters at 200k+ ctx |
| 13 | Long-ctx decode | [PENDING - 250k probe running] old pre-DEFER collapse was 13.7 t/s | measuring |
| 14 | Attention kernels | DSA graphs, MLA_MMA, DSA_ALL default-on since v1.1.55 | DONE |
| 15 | Prefill | 432-449 @2k, ~445 @100k = copy-lane bytes at 3.05bpw | WALL (bytes); p32k already on |
| 16 | Host hops | 2x64KB/token boundary hops (split cost) | negligible |
| 17 | Power/clocks | 143-170W of 250W cap = bandwidth-bound, power headroom unused, no clock lever for BW | N/A |
| 18 | DRAM channels | 8ch DDR4 ~200GB/s; 28 experts x 9.9MB = 1.4ms theoretical floor; lane is dequant-compute-bound, 14 threads saturate 16 physical cores | DONE - channels fed |
| 19 | RAM pinning | all experts pinned (RAM_RESIDENT verified at boot) | DONE |
| 20 | Server wrapper | spec default injected; streaming chunk granularity coarseness affects TTFT feel only | optional polish |
| 21 | Quant size | U-curve measured: 3.05=69, S24-class=87.8 (different tier), XS=50 | WALL for THIS tier |

## Remaining actionable ideas (ranked, honest)
1. Long-ctx KV window (if #13 still collapses post-DEFER): MAYA_GLM_KV_RESIDENT sweep
   at 200k+ ctx. Boot-level, answers-safe.
2. Manual split nudge (11): MAYA_GLM_SPLIT per-card custom — expect <=2%.
3. Drafter replacement/fine-tune (1): the only path to 100 on prose. Model work,
   not config.
4. S24-for-prose + EXL3-for-final workflow split (zero new tech, uses both tiers).

## Bottom line
Every memory/CPU/PCIe/tiering/kernel layer is now measured-closed or hidden.
69.1 bench / 74-75 code / 46 prose client-side. The stack is at its physics:
bytes (prefill) + prose draft acceptance (decode).
