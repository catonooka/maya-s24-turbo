# GLM EXL3-3.05bpw on maya v1.1.20 (latest) — 2026-10-21

## What changed from v1.1.15
- Engine: upstream main @ 35bb2b2 (v1.1.20, 49 commits past v1.1.15):
  PREFILL_STAGES instrumentation, exllamav3 1.6 half-bit K support,
  conversation-slot RAM parking (#236), CPU-expert deferral (#252),
  2-GPU no-NVLink prompt share (#270), split rows (#324), KDA 4-block
  prompt recurrence (#199), shared-expert sub-batches (#215), short-prompt
  batched path (#232), DSA indexer FP16 tensor cores (#233), MTP split
  state backup (#290), warm-prompt opt-in (#285)
- Binary: `build/maya` (v1.1.20 merged strata into the unified `maya` engine)
- Launch: **sudo systemd-run --slice=system.slice** (glm-serve2 unit) —
  user-slice loads die to systemd-oomd at RAM pinning (~940 s in, same
  signature as the 27B notes)

## Config (mayads json, port 8081)
Full v1.1.15 tuned env carried over (RAM_RESIDENT, KV_INT8, USAGE profile,
PREFETCH@fetch/rank4/blocks8, FETCH_WIDE, SPEC_CANCEL, SPLIT=22,
TRUNK_KEEP=6, MISS_SKIP_W 0.14 from L23) — all still recognized, engine
logs confirm each (tiers follow usage profile; light misses <0.140 left out;
chunks of 8192 borrowing 2194 MB; split [0,22)/[22,45) auto = our SPLIT).

## Results (same 3-probe bench as the v1.1.15 campaign)
| | v1.1.15 campaign | v1.1.20 now |
|---|---:|---:|
| Bench mean (3 probes, t0.6) | 50.0 | 43-45 |
| Engine decode windows | 52.7 | **50.3 mean / 57.2 max** |
| Prefill @1.5K / @6K | — | **437 / 770 t/s** |
| Quality gate | pass | **6/6** |

Honest read: per-window decode is at parity-or-better (57.2 max window is
the best we've logged on this quant), prefill is far ahead of the old ~990
@27K curve at short lengths. The bench-mean drop (50→43) is the prose
probe's draft acceptance (~56-65% vs 85-96% facts/code) + first-token
prompt overhead in short probes — not a decode regression.

DFlash2 drafter (+12% 2-GPU per docs): tried — the EXL3 gguf is not a
DFlash draft file; the real drafter is a separate download (blocked by the
no-download rule). Removed cleanly.

## Serving
- Port 8081 (DSV41 keeps 8080), systemd unit glm-serve2 (system slice)
- Restart: sudo systemd-run --uid=1000 --gid=1000 --unit=glm-serveN
  --slice=system.slice ~/project-maya/.venv/bin/python
  ~/glm-latest-tree/maya.py --config ~/glm-latest-tree/maya-exl3-latest.json
  --port 8081   (reset-failed if the unit name was used before)
- DSV41 maya-serve stopped for this run (needs both GPUs);
  restart = bash ~/maya-ds4-main/serve-ds4-dual.sh
