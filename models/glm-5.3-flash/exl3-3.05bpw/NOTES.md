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
- gate 7/8 (vision miss by design)

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
