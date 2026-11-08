# GLM-5.3-Flash EXL3 3.05bpw — QUALITY tier (3x RTX 3090 + 247G DRAM)

Pack: `~/Maya-data/models/EXL3-3.05bpw/` (turboderp/GLM-5.3-Flash-exl3 rev 3.05bpw,
125G, re-packed with v1.1.56 packer). Engine: `~/maya-glm3` (v1.1.56 + FETCH_WIDE
efe38c3c + STRATA_SPEC_DEFAULT 6eed2be3, branch glm3-3090x3).

## SERVING config: `exl3-ck0.json`
`bash models/glm-5.3-flash/exl3-3.05bpw/run-exl3.sh exl3-ck0.json` (or from ~/maya-glm3).
Unit `glm3-exl3-ck0` on :8080. Sanity 17×19=323.

env: RAM_RESIDENT, KV_INT8, USAGE profile, PF_N=2, FETCH_WIDE, SPEC_CANCEL,
TRUNK_KEEP=6, MISS_SKIP_W=0.14/FROM=23, P2P, PF_CE, **DEFER_CPU=2**,
**CKPT_EVERY=0**, STRATA_SPEC_DEFAULT=8; args: `--dflash .../DFlash2-Q4_K_M.gguf
--prefill 32768 --kv-streaming on --expert-cache auto`.

## Measured (official --bench unless noted)
- decode 70.6 mean (code 69 / expository 65.7 / html 77.3); creative prose ~46-56
  (draft-acceptance wall, not engine)
- prefill 454 @2k / 734 @8k / **1637 t/s fresh 100k** (63s; was 960 before CKPT_EVERY=0)
- repeat prompt (PROMPT_MEMO): 1.2s · continuing turns: incremental (checkpoints)
- decode @100k ctx: 31 t/s · @250k: 26 t/s (KV/indexer state squeezes expert slots)
- gate: vision now ON (mmproj wired, reads text perfectly); thinking levels none/auto/low/medium/high/xhigh/max all live

## Knob ladder (one knob per boot, same-day A/B)
| knob | verdict |
|---|---|
| baseline | 53.3 |
| TRUNK_KEEP_MAP | 56.0 alone; **64.7 stacked on DEFER — never stack** |
| **DEFER_CPU=2** | **69.1 (+30%) KEEPER** |
| Q6_K drafter | 68.9 flat |
| CPU_SHARED=0 | 62.2 worse |
| adapt 2/192 (Strata #4870) | 68.4 flat @97.5% hit |
| spec sweep live | spec=8 knee; 16 + spec_n_min backfire |
| cpu_threads | 14 optimal (SMT contention above) |
| **CKPT_EVERY=0** | **+70% fresh long reads — KEEPER** |
| KV_RESIDENT 65536 | prefill-neutral (105.7 vs 107.1s) |
| PREFILL_CALC=1 | flat 454/728 |
| --prefill 16384 / 1024 | 443/718 · 459/690 — worse than 32768 |

## Laws
- Slot tax: 2nd concurrent long-context read on a warm engine = 2.1-2.6x slower,
  structural (slot-out write + pool re-lend). Fresh-boot probe1 = the only
  comparable arm.
- Launcher must glob-stop `glm3-exl3-*.service` `glm3-s24-*.service` + verify
  invocation id, or the new boot dies silently on :8080 (fake-boot trap).
- `pkill -f "build/maya"` kills the calling wrapper — use `pkill -f "[b]uild/maya"`.

Reports: ../reports/ (exl3-100push, exl3-blocker-audit, exl3-prefill-ttft).

## Thinking-level API parity (verified live vs z.ai GLM-5.3-Flash contract)
API set: thinking.type=enabled only (cannot disable), reasoning_effort low/high/max only.
Local (ck0 + serve patch): enabled ✓, low ✓, high ✓, max ✓ (max = deep reasoning ~6x
longer chain; low/high = brief), bare effort ✓, no-params default = thinking on + dynamic
depth (same as API default). Local is a SUPERSET (also accepts none/auto/medium/xhigh and
ignores thinking.type=disabled instead of erroring) — clients using only API-legal values
behave identically on both endpoints.

## 1.5M-pool serving config: exl3-ck0-m15m.json (2026-11-08 night)
- --max-context 1572864 + MAYA_GLM_KV_RESIDENT=262144: 256k KV window in VRAM
  (0.43-0.57 GB/card), full 1.5M latent cache pinned in DRAM (9.4 GB).
- STRATA_CONV_MAX=262144 (LOCAL serve patch): one conversation capped at 256k
  tokens; 315k prompt -> clean 400 in 0.2s. Pool serves many conversations.
- RAM SLOTS: MAYA_GLM_SLOT_DIR=/dev/shm/glm-slots (tmpfs), SLOT_GB=48, SLOTS=8.
  Old conversations' KV saved to RAM (1827-tok conv = 0.17 GB in 0.1s), restore
  = RAM->VRAM stream. NOTE slot bytes/token scale with max-context (~93KB/tok
  at 1.5M ctx vs ~8KB/tok at 262k): a full-256k conv slots at ~24 GB -> budget
  holds ~2 max-size or many normal chats.
- ENV LAW (learned hard): config "env" reaches the ENGINE only; server-side
  vars must be registered in SERVER_ENV (serve/server.py) — the spec-default
  patch sat dead until registered. /proc/PID/environ won't show in-process sets.
- Decode unchanged (66-70 t/s code plain-client). 1.5M > model's 1M training
  window: quality past 1M off-label.
