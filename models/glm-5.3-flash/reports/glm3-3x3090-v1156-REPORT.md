# GLM-5.3-Flash EXL3-3.05bpw on 3x RTX 3090 + 247G DRAM — maya v1.1.56 — 2026-11-07

## Rig
3x RTX 3090 (slots 1/5/7, PCIe gen4, no NVLink pairs active), TR PRO 3955WX (Zen2 AVX2),
247 GB DDR4 (8-ch), all GPUs capped **250 W** (nvidia-power-cap.service, boot-durable),
disk: pack on NVMe. Engine: **upstream v1.1.56 (7645ddd5) + FETCH_WIDE port (efe38c3c)**
in worktree `~/maya-glm3` (branch glm3-3090x3 off ~/project-maya origin/main).
Model re-downloaded fresh (turboderp/GLM-5.3-Flash-exl3 @ 3.05bpw, 118G) and
**re-packed with the v1.1.56 packer** (#861: expert selection bias kept F32 — quality bump
over the old pack for free). DFlash2 drafter Anbeeld/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf
(697 MB, sha256 f4158ffb…f017c verified against maya.py DFLASH pin).

## Config
`~/maya-glm3/maya-glm3-3gpu.json`, launcher `run-glm3.sh` (systemd unit glm3, system slice,
port 8080, host 0.0.0.0). gpu [0,1,2], ctx 262144, kv-streaming on, expert-profile glm.
Env: RAM_RESIDENT=1 KV_INT8=1 USAGE=expert-usage-glm.txt PF_N=2@fetch/rank4/blocks8
FETCH_WIDE=1 SPEC_CANCEL=1 TRUNK_KEEP=6 MISS_SKIP_W 0.14/23 P2P=1 PF_CE=1.
Boot ~2.5 min (parallel load). Split auto: [0,15)/[15,30)/[30,45); with DFlash2:
HEAD_SPLIT on (head cols split CUDA2/CUDA1), drafter 717 MB on CUDA2.

## Results (same-day, same-script 3-probe bench, 2 rounds each)
| boot | config | bench mean | engine windows |
|---|---|---:|---|
| 1 | 3-way split + MTP (v1.1.20 env carried) | 48.5 / 47.9 | 49.7-57.0 t/s |
| 2 | + DFlash2 drafter (pipelined window) | 53.6 / 52.2 | 59.3-70.0 t/s |
| 3 | + P2P=1 | 54.8 / 51.3 | 58.6-68.5 t/s |
| 4 | + PF_CE=1 (final) | 53.3 / 52.7 | 52.2-70.3 t/s |

- **DFlash2: +9.7% bench mean — THE keeper.** Prose acceptance 28-53%, facts/code 85-96%.
- P2P: FLAT (2x64KB hops/token were noise). Kept (same bytes, zero cost).
- PF_CE: FLAT on these probes. Kept (engaged on all 3 cards, zero cost).
- vs 2x3090 v1.1.20 era (bench 43-45): **+18-23%**; windows 57.2 max -> 70.3 max.
- 3rd card moved the wall: ram fetches 20/tok -> 2.6/tok (MTP boot), vram hit 92 -> 97%;
  under DFlash2's verify window fetches rise again to ~15-18/tok — that is the next wall
  alongside prose draft acceptance.
- Helper-GPU config (2-split + EXPERT_HELPER) REJECTED without boot: costs a split card
  (33% of layer capacity) to absorb misses that are already 94-96% covered; auto-split at
  97% hit is worth more. Upstream's +11% was on a FREE iGPU helper.
- Power under load: 143-170 W of the 250 W cap, clocks 1695-1950 MHz — bandwidth-bound,
  the cap costs nothing (same as the 2-card findings).
- Quality: functest 7/8 (vision tower not installed — the one FAIL is the 400 on
  /images; text/tools/stream/65K-ctx/greedy all PASS), 17x19=323 OK, prose coherent.

## FETCH_WIDE port notes (95af3292 -> v1.1.56)
- Deep-prefetch N>2 half: DROPPED (upstream caps at 2 spares since v1.1.15; v1120 ran N=2).
- Wide-fetch half: ported to both the plain `moe_fetch` (dedicated y-parallel kernel) and
  the fused `moe_wait_fetch` (32B loads branch, `wide` param). Getter honors
  MAYA_GLM_FETCH_WIDE then STRATA_GLM_FETCH_WIDE. Engine env echo line confirms it on.
- Conflict resolution took upstream wholesale for the prefetch/PF_CE machinery.

## Ops
- Restart: `bash ~/maya-glm3/run-glm3.sh` (stops, drain-waits, reset-failed, systemd-run).
- Bench: `python3 ~/maya-glm3/bench_glm.py 2`. Gate: `python3 ~/maya-glm3/functest.py`.
- Download dir: `~/Maya-data/models/EXL3-3.05bpw/{src,dflash2}`; pack at src/pack (in place).
- Old DSV41 pack and old Maya-data packs are GONE (dir was empty) — this is the only pack.
- Next levers, ranked: prose draft acceptance (DFlash2 window tuning, SPEC_WIN),
  ram-fetch overlap under the verify window, TRUNK_KEEP_MAP per-layer sweep (docs show
  +23% on a 3090+3060 Maya-M rig), MAYA_GLM_DEFER_CPU=2.
