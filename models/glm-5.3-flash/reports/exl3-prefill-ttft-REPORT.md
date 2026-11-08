# EXL3 prefill / TTFT / long-context campaign (2026-11-08)

Goal: speed up prefill, TTFT, stable speed at long context. EXL3-3.05 tier, 3x3090, 250W.

## Law 1 — CKPT_EVERY=0 is the prefill lever (+70% fresh long reads)
Fresh boot, fresh ~99.6k-token read (never-seen nonce), wall for full request:
| config | run1 (fresh) | run2 (2nd conv) |
|---|---|---|
| exl3-final (CKPT 16384, KV 32768) | 107.1 s (~960 t/s) | ~220 s (est) |
| exl3-kv64 (CKPT 16384, KV 65536) | 105.7 s (~970 t/s) | 224.5 s |
| **exl3-ck0 (CKPT 0, KV 32768)** | **63.3 s (1637 t/s)** | **163.7 s** |
Engine lines authoritative: ck0 run1 = "99573 tokens in 60816.5 ms (1637 tok/s)".
- CKPT_EVERY=0 only kills PERIODIC prompt cuts. System-prompt + last-turn-start
  snapshots remain (SETTINGS.md "checkpoints" section) -> turn+1 restore barely
  degrades; PROMPT_MEMO still covers identical prefixes.
- Decode unaffected (63.7 t/s code @ temp 0.6, server-default spec=8).
- PROMOTE: exl3-ck0.json is the standing QUALITY-tier config (unit glm3-exl3-ck0).

## Law 2 — conversation-slot accumulation tax (structural, not config)
Second concurrent-context 100k read on the SAME engine runs 2.1-2.6x slower
regardless of config (163 vs 63; 224 vs 105; final-engine warm sequence 108-240 s).
Cause: slotting the previous ~100k conversation out (disk slot + pool re-lend).
Mitigation for real workloads: keep conversations continuing (checkpoints +
PROMPT_MEMO make turn 2+ nearly free); this tax hits only NEW long contexts after
a previous long conversation. Single-run wall probes are invalid on this rig:
always fresh-boot per arm, engine rate lines are truth.

## Law 3 — KV_RESIDENT window is neutral for prefill
65536 vs 32768: 105.7 vs 107.1 s — same. Decode effect not isolated; not worth a
boot. Keep 32768 default.

## Bugs fixed
- run-exl3.sh stop list didn't cover sibling glm3-exl3-* units -> ck0/kv64 boots
  DIED on "port 8080 in use" and "measurements" silently hit the old engine
  (fake 108.5 s "kv64 win"). Fixed with glob stops 'glm3-exl3-*.service'
  'glm3-s24-*.service'. Lesson: verify the serving unit is the one you booted
  (systemctl is-active + invocation id) BEFORE probing.

## TTFT toolkit (what exists, measured)
- PROMPT_MEMO: identical 100k prompt re-read = 1.2 s wall (vs 63-107 s fresh).
- Turn-boundary checkpoints: continuing conversation reads only new tokens.
- CKPT_EVERY=0: fresh long reads 1637 t/s (from ~960).
- Short-prompt prefill (official --bench 2k): 432-449 t/s unchanged.

## Latest-repo sweep (2026-11-08 evening, official --bench, same-day A/B)
- Upstream main still dry past v1.1.56 (1 cosmetic commit 7bf15ac6). Repo: mw00/project-maya.
- PR #904 (DS4_HELPER_DRAFT) + PR #867 (DS4_EXL3_LEND, mmap chunk 16384 +40%): DS4
  path only (mmap-experts + iGPU helper); NOT the GLM path. Chunk idea tested here:
  --prefill 16384 on ck0 = 443/718 vs 454/734 @2k/8k — no transfer, p32k stands.
- MAYA_GLM_PREFILL_CALC=1: 454/728 vs 454/734 — FLAT on 3-GPU RAM_RESIDENT
  (docs' +25-48% was a different lane setup). Closed.
- Issue #898: Strata-native Qwen path does 1900-2000 t/s prefill @128-154k on a
  weaker 2-GPU/32GB rig vs maya's GLM path — engine-path difference, no exposed
  knob (issue has no config comments). Our 1637 fresh @100k is near that class.
- pkill law: `pkill -f "build/maya"` MATCHES THE CALLING WRAPPER SHELL (its own
  cmdline contains the pattern) → wrapper dies -9/-15 with zero output. Use
  `pkill -f "[b]uild/maya"` char-class form.

## Community cross-check (2026-11-08 night: is anyone faster per card?)
Per-card decode: US EXL3 70.6/3 = 23.5/card; #414 8x3090 EXL3-4.05 117.4/8 = 14.7/card.
US S24 87.8/3 = 29.3/card; #349 4x3090 S24+MTP 100.8/4 = 25.2/card; #89 4x4090 118/4 =
29.5/card (sm89, 2x compute). Prefill @8k: US EXL3 734 / S24 1002; #414 1470 (184/card
vs our 334/card S24). We are at or above every 3090-class row per card.
- Chunk law on EXL3 (bench @2k/@8k): 1024 = 459/690, 16384 = 443/718, 32768 = 454/734
  → p32k optimal HERE. #414's --prefill 1024 win and #89's 2048 win are NVMe/mmap
  regimes (chunk borrow evicts experts to SSD); we are RAM_RESIDENT — refetch over
  PCIe is cheap, so small chunks only lose pipeline depth.
- #349 4x3090 detail: MTP>DFlash2 on 4 cards (100.8 vs 86.5) — we verified the
  OPPOSITE on 3 cards (DFlash2 85.6 > MTP 80.6); card-count economics, not config.
- Strata fresh issues: GLM-5.3 path is still a scope proposal (#5544); nothing
  portable to maya's GLM engine.

## Current standing numbers
- QUALITY tier: glm3-exl3-ck0 (DEFER_CPU=2 + CKPT 0 + p32k + STRATA_SPEC_DEFAULT=8)
  decode 69.1 t/s bench / 63.7 t/s code live; 100k fresh read 1637 t/s; 31 t/s
  decode @100k ctx (engine stat, unchanged).
- SPEED tier: s24-dflash-p32k 87.8 t/s bench; prefill 699-1009 t/s @2k-8k, ~926 @100k.
