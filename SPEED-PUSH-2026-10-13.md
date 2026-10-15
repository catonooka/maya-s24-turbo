# Maya GLM-5.3-Flash speed push — 2026-10-13 (issue #135 replication + 80 t/s goal)

## Goal
Phase 1: v1.0.24 → main (a709524, past v1.1.5), reproduce issue #135's **53 t/s** (2×3090, Maya-S24,
131k ctx, greedy/thinking-off bench). Phase 2: our tuning stack + DFlash2/KV-streaming/CPU-MT toward **80 t/s**.
Constraint: tool use must stay stable (functional battery 8/8) — no quality-breaking levers.

## Baseline (v1.0.24, thinking-on, 262k ctx, our env stack)
~45 t/s warm mean (code peaks 50+). MTP acceptance 89.6% mean. 8/8 functional PASS.

## Issue #135 reference numbers (v1.1.2 stock, 131k ctx, greedy, thinking-off, 225W caps)
- 2×3090: **53.0 t/s** decode mean (51.7/52.5/55.0), MTP drafting, split [0,24)/[24,45), 133/126 slots/layer
- 1×3090: 26.0 t/s. Prefill 8K: 885 t/s. Spec: 243/254 drafts accepted (95.7%), 89.3% positions.
- Their rig: 3970X (AVX2, no AVX-512), 126G RAM, no NVLink, 225W power cap.

## Phase-1 checklist
- [x] backup config (maya-maya-s24.json.bak-v1.0.24)
- [x] git pull main (a709524) — local parser fix absorbed upstream (#146); no local patches to carry
- [ [ ] rebuild engine (running)
- [ ] boot v1.1.5 stock-equivalent: our env vars OFF first (match #135 as closely as possible),
      262k ctx (config unchanged — our production ctx; #135 used 131k)
- [ ] run ./maya.sh --bench (repo's own bench = what #135 reported) → expect ≥50 t/s
- [ ] re-add our env stack (RAM_RESIDENT=1 PROMOTE_MIN=6 KV_INT8=1 MTP_KEEP=2) → bench again
- [ ] functional battery 8/8 (functest.py)
- [ ] then Phase 2 ladder, one lever at a time: kv-streaming, CPU_MT, DFlash2 (needs --setup --draft dflash2,
      ~0.7GB, CC BY-NC — user license call), spec-sample/cancel, calibrate
