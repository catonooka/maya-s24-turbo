# FIX: v1.1.28 upgrade — looted speedups, two upstream bugs fixed, one of ours

Date: 2026-10-26 · Engine: `~/maya-ds4-main` branch `recall-tensorfold` = **v1.1.28 + 6 commits** (patch: `patches/tensorfold-recall.patch`, 631 lines vs v1.1.28) · Live: `:8080` via systemd-run `maya-ds4` unit, `serve-v1124-baseline.sh`

## What we took from upstream v1.1.28 (base 0128fb3e ≈ v1.1.20)

- **Rebase**: dropped our 2 redundant commits (upstream absorbed both our queue-heartbeat and DSML-streaming fixes — theirs cites #403) and rebased only the Recall commit onto v1.1.28, then 5 more.
- **Conversation slots across restarts** — KV state persisted to `maya-ds4/slots/`, restored at boot ("2 conversations kept" verified). Our SLOT_KEEP env equivalent, now first-class.
- **512K/1M context support** (#442) — we run 262144, unchanged.
- **#476 ~2× decode on one GPU** — gated to Linux+NVIDIA+**one** GPU; we run 2×3090 layer-split so we keep the v1.1.27 path. `DS4_STREAM_WIN=1 DS4_LAND=1 DS4_STREAM_ZC=1` can force it on; **untested here** — candidate for a later single-GPU bench.
- **Prompt attention on tensor cores** (#369, `DS4_ATTN_MMA`, default ON sm_80+) — suspected of the chain-math flip, exonerated (see below).
- Misc: Windows HIP fixes, hipblaslt CMake, helper-GPU docs. 119 files, ~21k lines.

## Upstream bugs found & fixed (in our tree)

1. **Thinking-budget clamp crushes small requests** (#363, v1.1.24+): `think_budget = max(1, min(32768, max_tokens − reserve))` with `reserve = max(1024, max_tokens/4)` ⇒ for any `max_tokens ≤ 1024` the budget is forced to **1 token**. The think block closed after one word; the model answered with no reasoning (chain-math `(17×19)+(23×8)−145` = **323 instead of 362**, deterministic 3/3). Needles/ASCII/determinism all passed — only reasoning-needing probes broke. Our v2 fix: when `max_tokens ≤ reserve`, the request reserves a **quarter of its own tokens** for the answer (`max_tokens=200 → 150 think + 50 answer`). Proof: max_tokens=5000 → 362 correct, pre-fix. Post-fix gate: **chain-math PASS at max_tokens=200**.
2. (Observation) The engine-restarter in `server.py` can loop on a crashing engine (exit −6 bad_alloc); root cause below was ours, not upstream's.

## Our bug found & fixed (Recall)

- **4.6 TB bad_alloc at boot** (`exit code -6`, SIGABRT): a damaged/misaligned `record.bin` tail read as a chunk length sized `buf.assign(n)` — 0x43800001000 = two token IDs packed as int64. Every Recall-OFF boot was stable; every Recall-ON boot after the first abort crashed. Fix: loader caps chunk lengths at the capacity (`n > cap_tokens_` ends the load, good prefix kept) + regression test (damaged record: prefix kept, no abort). **14/14 unit tests** now.
- Corrupt `record.bin` (133,807 tokens held from earlier session) moved to `/tmp/record-corrupt-*.bin`; fresh record accumulating.

## Verdicts on upstream's changes (accept/reject)

- ACCEPT: slots, 512K/1M support, ATTN_MMA (exonerated; prefill gained), Windows/HIP/CMake fixes.
- NOT FOR US: #476 one-GPU path (2-GPU layer-split rig; revisit if we ever run single-GPU), MCDMA/RoCE etc. (hardware N/A).

## Evidence

- Quality gate **7/7 PASS** (chain-math 362, 17×19 323, determinism, needle2k BANANA-7741, needle30k ZEPHYR-9042, ascii, lines4) at small max_tokens — the honest setting.
- Recall gate PASS: copy-repeat 3× byte-identical (1193 chars), best 11.8s vs 15.3s cold; paraphrase 470 chars consistent.
- Warm bench v1.1.28 (2 runs): essay 21.2 / facts 23.4 / code 23.5 t/s; **prefill 947.6 t/s @25k** (old band 737–835 → **+14–28%**).
- Unit: recall_drafter_test **14/14** (incl. damaged-record regression); suffix_drafter_test ok; slot tests ok.
- Engine stable post-fix across reboots (zero bad_alloc in fresh log).

## Rollback

Old binary `build-ds4/maya-ds4-pr344` retained; flip `maya-ds4.json` "exe" back and restart the unit. `record.bin` corruption is self-healing with the loader cap.
