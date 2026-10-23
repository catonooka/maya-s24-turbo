# GLM under P2P fork driver — regression analysis (2026-10-22)

## What happened
After enabling the aikitoria fork driver (for DSV41's P2P prefill win),
GLM EXL3-3.05 regressed on the same boot:
- bench mean 45 -> 22.8-29.4 (oscillating)
- prefill @1.5K 569 -> 104-202, @6K 880 -> 511-578, @12K 838 -> 523
- decode engine windows: PEAKED 57-58 t/s (best ever) but oscillate
  down to 35 (VRAM hit 92-94% -> 87-91%)

## Root cause (source + logs)
The fork's changed host-memory path (BAR1 mapping + new
cudaHostRegister logic) slightly slows maya's pinned RAM-tier expert
fetch. The engine's boot calibration measured new lane costs
(CPU 0.467ms vs PCIe 0.364ms expert) and shifted MORE experts to the
CPU lane (0-6 of 8 vs 0-3 before) -> decode stalls when CPU lane
saturates; PREFILL_CALC's learned share also stale -> prefill down.
Forcing PCIE_SHARE=0.25 did not recover (still oscillates).

## Verdict
- Fork driver = WIN for DSV41 tp2 (prefill 907, allreduce over 26 GB/s)
- Fork driver = LOSS for GLM layer-split (pinning path regression)
- Not a bug in either project: two engines, two memory-layout
  sensitivities, one driver can't be optimal for both as-is.

## Options (ranked)
1. v1.1.21 rebuild first (HOST_FREE decode +9% class, leaner tiles
   +5-11% prompt) — may shift the balance; re-bench both after.
2. Per-model driver choice: stock 595 modules for GLM sessions, fork
   for DSV41 (depmod swap + reboot between; both module sets built).
3. Hugepage-align maya's pinned arena (fork's fast path for
   cudaHostRegister needs 2MiB-aligned whole-hugepage buffers) —
   engine tweak, medium effort.
4. Accept: GLM serves best on stock driver (45 t/s / 880 pf),
   DSV41 on fork (907 pf).

## Numbers table (same benches, three driver states)
| Config | GLM dec | GLM pf@6K | DSV41 pf@25K |
|---|---:|---:|---:|
| stock 595, v1.1.20 | 45 | 880 | 783 (host allreduce) |
| fork 615-p2p, v1.1.20 | 25 (osc) | 578 | 907 (P2P allreduce) |

## Context-window sweep addendum (pure decode at depth, cached prefix)
Two full sweeps (glm_ctx_sweep.py): decode 48-52 t/s short, 42-47 t/s at
128K — ~14% total droop over the whole window, NO cliff. The earlier
"25 t/s" bench regression was oscillating lane calibration during fresh
prompts; steady-state generation at depth is healthy under either driver.
KV streaming + INT8 KV doing their job.

| window | decode t/s |
|---|---|
| ~100 | 48-50 |
| ~2K | 46-52 |
| ~8K | 46-47 |
| ~32K | 45-47 |
| ~64K | 44-47 |
| ~128K | 42-43 |
