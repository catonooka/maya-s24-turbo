# maya-s24-turbo

**GLM-5.3-Flash (Maya-S24 quant) on 2× RTX 3090: tuned to 59 t/s thinking-on at 262k context —
with a full measurement log, an agent-session where the model built its own benchmark toolkit,
and hard receipts for every lever accepted or rejected.**

This is a **tuning + verification record** on top of [mw00/project-maya](https://github.com/mw00/project-maya),
the engine that makes running GLM-5.3-Flash on consumer GPUs possible at all.
All credit for the engine, the Maya-S24 quant, the MTP drafting, KV streaming, and the
CPU/PCIe expert lanes belongs to its author (**@mw00**) and contributors — see [Credits](#credits).

## The result (2026-10-13/14)

| metric | stock v1.1.5 (this rig) | + this repo's tuning | repo README reference¹ |
|---|---|---|---|
| decode, official bench (greedy, thinking-off) | 49.2 t/s @ 262k | **52.2 t/s** @ 262k | 53.0 t/s @ 131k |
| decode, production (thinking-on, t=0.6) | ~41.9 (v1.0.24 stack) | **57–59 t/s** | — |
| prefill 8k | 1009 t/s | **1014 t/s** | 885 t/s |
| context served | 262k | **262k** (131k↔262k benches a tie) | 131k |
| functional battery | — | **8/8** (tools incl. GLM native dialect, SSE, 17k needle, greedy, vision) | — |
| thinking-level suite | — | **10/10** (off/low/high/max × OpenAI+Anthropic, streaming, determinism) | — |

¹ `@briepala`'s 2×3090 report ([project-maya #135](https://github.com/mw00/project-maya/issues/135), v1.1.2) —
the closest published same-class hardware. Their 53.0 vs our 52.2 on the official bench is inside the
±4–5% run-to-run noise band, at double the context window. Their run was stock settings; ours adds the
tuning stack below and vision enabled.

## Hardware

2× RTX 3090 24 GB (No Nvlink, 300 W cap) · Threadripper PRO 3955WX · 247 GB DDR4 · NVMe.
Everything here should reproduce on any 2×24 GB + ~128 GB RAM box.

## The tuning stack

Launcher (`tuning/run-maya-s24-tuned.sh`) + config (`tuning/config-snapshot.json`):

```bash
export STRATA_GLM_RAM_RESIDENT=1    # pin every non-VRAM expert in RAM — disk reads drop to 0.00/tok
export STRATA_GLM_PROMOTE_MIN=6     # stop VRAM slot churn on one-off experts
export STRATA_GLM_KV_INT8=1         # 544 B/token/layer latent cache (vs 1024)
export STRATA_GLM_MTP_KEEP=2        # draft keeps its 2 best-scored missing experts
export STRATA_GLM_SPEC_SAMPLE=1     # exact-distribution speculative sampling
export STRATA_GLM_SPEC_CANCEL=1     # cancel a wrong draft's pass early
# config "env": STRATA_GLM_PCIE_SHARE=0.25  (from ./maya.sh --calibrate; engine default 0.60 → 43.4
#                                          t/s stock, 0.25 → 48.8 on this rig)
# config "args": --kv-streaming on --expert-profile <repo>/data/expert-profile-glm.bin
```

Two operational rules that matter as much as the flags:

1. **Serve from the system slice** (`sudo systemd-run --uid=1000 --gid=1000 --unit=maya ...`).
   A pinned-RAM engine load from a user-slice terminal is killed by `systemd-oomd`
   (memory-pressure kill at 50%) — silent, empty logs, looks like a crash.
2. **Re-diff the config after touching `./maya.sh --setup`/`--calibrate`** — setup rewrites
   `--max-context` (to 32768) and `host` (to localhost-only) on every run.

Full lever-by-lever history with measurements: [`tuning/CURRENT-SETTINGS.md`](tuning/CURRENT-SETTINGS.md).

## Measured and rejected (negative results, so nobody re-tests them blind)

| lever | result | why |
|---|---|---|
| DFlash2 drafter | 29.1 vs 47.3 t/s (MTP) | verify passes multiply CPU-expert work on an offload-bound rig; the repo's own `--calibrate` measured and auto-rejected it |
| `--parallel 2/4` | C1 59→33 t/s; C4 aggregate 44 < serial 59 | conversation-parallel reorganization of the 2-GPU layer-split forfeits MTP draft overlap; concurrency pays on single-GPU rigs only |
| 131k vs 262k context | tie | kv-streaming + INT8 KV made context ~free here |
| manual CPU-lane threads, power-cap raises, PROMOTE_FILL | neutral or worse | auto-calibration already at parity; decode is expert-fetch-bound, not power-bound |

## benchkit — the model benchmarked itself

In a 6-stage agent session (driver: `agent-session/longtask_driver.py`, ~35.6k tokens of context
exercised at the full 262k window), the served GLM-5.3-Flash **wrote its own benchmark and health
toolkit** — spec → bench → functional battery → log-tailing monitor → HTML report → CLI — with
validator gates between stages and a self-repair turn (it fixed its own monitor crash from raw
error text in one shot). Every stage passed; a human reviewed and fixed the driver-side bugs
(session-history duplication, cwd, monitor backfill between engine stat emissions).

```bash
python3 -m benchkit quick                     # 1 probe, C1
python3 -m benchkit full --bench-out r.json --functest-out f.json
python3 -m benchkit monitor --duration 60 --interval 5 --out mon.jsonl
python3 -m benchkit thinking_test             # 10-check thinking-level suite
```

Its own measured verdict: 58.78 t/s single-stream (vs 59.2 human-bench baseline), 8/8 functional,
per-stream collapse under concurrency — independently re-derived from its own data.

## Repository layout

```
tuning/           lever history, launcher, config snapshot, official bench logs
benchkit/         self-benchmark toolkit (model-authored, human-reviewed)
results/          full_results.json, functest 8/8, monitor samples, HTML report
agent-session/    PLAN.md, the agent driver, session/stage logs
```

## Credits

- **[mw00/project-maya](https://github.com/mw00/project-maya)** — the entire engine: Strata-style
  tiered expert serving, the Maya-S24 quant of GLM-5.3-Flash, MTP-block speculative drafting,
  KV streaming, CPU/PCIe expert lanes, the speed calibration, and the benchmark used for the
  official numbers here. The 41.9→49.2 t/s jump in this record is simply *upgrading to their
  v1.1.x work*; this repo's contribution is the ~+6% tuned stack on top and the verification suites.
- **@briepala** — the 2×3090 speed report ([issue #135](https://github.com/mw00/project-maya/issues/135))
  used as the reference row, and the reminder that clean engine-line receipts make reports reproducible.
- **zai-org** — [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT).
- Contributions referenced during tuning: **@needmorevram** (KV streaming, CPU expert sharing),
  **@sociolog** (speculative sampling/cancel), the project-maya contributor community.

Engine: MIT (project-maya). This repo's scripts and notes: MIT. No model weights are included.

## Honest ceiling note

On 2×3090 + S24, decode is bounded by resident-expert count, not config: ~59 t/s thinking-on is
the measured ceiling of this quant on this hardware. 80 t/s needs more VRAM, an AVX-512 CPU,
or a faster upstream quant — not more flags. Watch project-maya's releases; the engine ships fast.
