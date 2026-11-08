# Push to 50 decode / 1000 prefill — Exp 9: tp1 + --threads on (2026-10-21)

## Root causes found in engine source (v1.1.20)
1. **Prefill in windows vs chunks**: tp1 without --threads runs prompts in
   windows of 16 tokens (kWin) = the 89-95 t/s slow path. `prompt_par =
   par || tp > 1` — --threads on enables chunked prefill (a host thread a
   GPU) while KEEPING the layer-split layout for decode.
   Engine's own comment: Maya-S 7-GPU rig 8K/32K = 115/99 windows vs
   461/1010 in chunks of kVlRows=8192.
2. **PCIe streaming lane is MXFP4-only**: lane_layer_ok() requires
   fmt == kFmtMXFP4 — silently OFF for EXL3 packs. Decode overlap lane
   (stream experts to GPU during CPU-lane compute) unavailable to us
   without a GGUF/MXFP4 quant (no-download rule blocks).
3. **--cpu-threads override disables lane auto-calibration** (tune_lane
   bails) — 16 threads is the engine's MEASURED optimum, leave alone.

## Results (all unique-prompt cold prefill, cached=0)
| Config | Decode | Pf@6K | Pf@25K |
|---|---|---|---|
| tp1 (old default) | 33-35 | 95 | 144 |
| tp2 | 26-28 | 384 | 516 |
| **tp1 + threads on + chunk 8192** | **31-33** | **280** | **518** |
| tp1 + threads on + chunk 16384 | — | 278 | 461 |

## Standing config (mayads4.json)
--tp 1 --threads on --dspark on --dspark-conf 0.5 --prefill-chunk 8192
env: MAYA_DS4_DSPARK_CUM=0.5

## Remaining gap to targets
- Decode 33 vs 50: CPU lane computes 22-42% of experts on 16 threads,
  streaming lane off (EXL3). Levers left: DSpark >2.5 tok/round (conf is
  optimal), STRATA_DS4_RAM_TIER=1 (RAM tier for file experts — we're
  all-in-RAM already), engine mods (tc tiles for EXL3 lane?).
- Prefill 518 vs 1000: 6K prompts cap at 280 (chunk overhead amortizes
  over length). Bigger = faster but 16K regressed. MAY need prompt_par
  with both GPUs on disjoint chunk halves (engine work) or v1.1.21+.
