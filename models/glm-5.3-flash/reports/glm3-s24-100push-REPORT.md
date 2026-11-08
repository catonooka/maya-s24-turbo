# GLM 100 t/s push — S24 stack, 3×3090, v1.1.56+FETCH_WIDE (2026-11-07 evening)

## Question: what layers were missing to hit 100 t/s?
Answer: the QUANT was the missing layer. EXL3-3.05bpw (125G pack) on 72G of VRAM is a
bytes/token wall — the 100 t/s club on 3090s runs ~2.3bpw-class packs (#349: 4×3090 S24+MTP
= 100.8; #414: 8×3090 EXL3-4.05 = 117). No flag set fixes that on 3 cards.

## Ladder (official --bench, greedy, 3 answers + prefill 2k/8k)
| config | decode mean | notes |
|---|---|---|
| EXL3-3.05 + DFlash2 (prev) | 53.3 | quality tier, keep for final answers |
| S24 + MTP (MTP_KEEP=2) | 80.6 | 79.8/81.1/80.8, acceptance 82-83% |
| S24 + DFlash2 | 85.6 | 88.6/76.5/91.7 — dflash wins on 3 cards |
| **S24 + DFlash2 + --prefill 32768** | **87.8** | 86.9/80.9/95.5 — KEEPER, serving now |
| S24 + DFlash2 + PROMOTE_MIN=8 | 83.8 | regression — promotion traffic > savings; 6 stands |

Prefill: 699-720 t/s @2k, 1002-1009 t/s @8k, **~926 t/s sustained at 100k-token prompts**
(v1.1.56 + 247G RAM beats the 4×3090 reference's 778/979).

Client-side (temp 0.6) with per-request tune sampling.strata_tune {"spec": 8}:
code 84.8 -> 93.9 (+11%), prose 67.0 -> 76.5 (+14%). spec=8 is the knee (12 flat, 16 and
spec_n_min backfire — verify cost overtakes acceptance).

## Serving stack
- unit glm3-s24-dflash-p32k.service (system.slice), http://0.0.0.0:8080, ctx 262144
- config ~/maya-glm3/s24-dflash-p32k.json; launcher run-s24.sh <cfg>
- env: RAM_RESIDENT, KV_INT8, PROMOTE_MIN=6, SPEC_SAMPLE, SPEC_CANCEL, PCIE_SHARE=0.25,
  MISS_SKIP 0.14/23, FETCH_WIDE=1, P2P=1, PF_CE=1
- drafter: DFlash2 Q4_K_M gguf (EXL3 folder), args --dflash + --prefill 32768 + --kv-streaming on
- model id glm-5.3-flash (same API name as EXL3 config)

## Remaining gap to a clean 100
- prose decode 77-81: DFlash2 acceptance on natural text (drafter quality), not memory.
  Levers left: window tune per workload via spec tune (done, client-side), a better
  drafter quant (Q5/Q6 dflash gguf), TRUNK_KEEP_MAP/MISS_SKIP_W sweeps (answers-changing).
- EXL3-3.05 stays installed as the quality tier (both packs coexist; boot other config).

## Ops lessons (this session)
- polkit now blocks user systemd-run transient units (D-Bus "Connection timed out" red
  herring; journal shows polkitd auth FAIL). Fix: sudo systemd-run (NOPASSWD works).
- Transient units start with cwd=/ — absolute paths + -p WorkingDirectory required.
- Failed transient units stay loaded: sudo systemctl reset-failed <unit> before re-run.
- maya.py --bench refuses while the server holds VRAM; stop the unit first (sudo).
- --setup --model glm-maya-s24 --yes --no-start: detects GPUs, builds pack (908M),
  writes config; needs vision/vocab ggufs from the repo's vision/ folder.
- python json edits: don't filter arg lists by value ('auto' collision) — rebuild by flag.
