# Exp 10: the remaining blockers, mapped (2026-10-21)

## What was tested this round
| Change | Result | Verdict |
|---|---|---|
| PROMPT_SKIP=2 on chunked tp1 | 273/516 — no change | dead |
| PROMPT_SKIP=2 on tp2 | 366/492 — no change | dead |
| --reserve-mib 100 (more experts in VRAM) | context=0 boot fail (starves KV) | dead, 600 restored |
| --cpu-threads 32 | disables lane auto-calib; no gain | dead |
| tp2 re-verify | 366/492, decode 25-28 | tp1+threads stays |

## Root causes (source-verified, v1.1.20)
1. **yoco_layer() gate**: prompt-skip requires ALL trunk layers on ONE
   GPU with nm==1 (ds4_model.cu:8592). tp2 fails (nm=2), our split fails
   (dev changes at L19 vs yoco boundary L20). Single-GPU-only feature;
   also needs n-keep >= 1024 (kYocoMin). Works only for 1-GPU packs.
2. **CPU lane saturated**: 3955WX is 16C/32T (NOT 32C — earlier notes
   wrong). Lane uses all 16 physical cores. The #188 AVX2 rewrite gains
   (14.3->20.9 on 1x3090) are ALREADY in our v1.1.20 binary.
3. **Stream lane (decode overlap)**: lane_layer_ok() requires
   kFmtMXFP4 — EXL3 locked out. This is THE decode lever we can't reach
   without a GGUF/MXFP4 pack (download blocked).

## Where the targets stand
- Decode 33-35 (tp1) / 30-32 (tp1+threads) vs 50 target: remaining gap
  is structural — EXL3 stream lane + yoco skip both single-GPU or
  MXFP4-only. Community 44-53 t/s rows use S24 quant (94.7G, fits VRAM).
- Prefill 518 vs 1000: tp2+PROMPT_SKIP can't engage (yoco gate);
  layer-major chunks are an engine-mod project.

## Conclusion
Config space is EXHAUSTED on v1.1.20 for this pack. Next real steps
(need approval):
1. Engine patch: relax yoco gate for layer-split (L19 split vs L20
   yoco — off by ONE layer; DS4_SPLIT=20,40 would align them, but the
   dev-uniform check still fails -> needs code change)
2. Engine patch: extend stream lane to EXL3 (fmt check + staging copy)
3. Maya-S24 quant download (94.7G) — the proven 44-53 t/s path
