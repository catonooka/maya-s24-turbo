# maya-s24-turbo

**GLM-5.3-Flash (Maya-S24 quant) on 2× RTX 3090: tuned to 59 t/s thinking-on at 262k context —
with a full measurement log, an agent-session where the model built its own benchmark toolkit,
and hard receipts for every lever accepted or rejected.**

Now also home to the **DeepSeek-V4.1-Flash-1017 campaign** ([`deepseek-v41-flash/`](deepseek-v41-flash/)):
the 748B MoE served on the same rig — 10.4 → 33-35 t/s decode, 144 → **783 t/s** cold prefill,
fully validated for quality and vision. See the History log below.

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

## History log

### 2026-10-20/21 — DeepSeek-V4.1-Flash-1017: 10.4 → 33-35 t/s decode, 144 → 783 t/s prefill

The same rig serves the 748B-parameter MoE (40×384 experts, Engram tables) from local files
only — no downloads. Full campaign in [`deepseek-v41-flash/`](deepseek-v41-flash/).

- **vLLM custom stack 10.4 t/s** → **maya-ds4 engine 17.5** — tiered experts (VRAM hot set /
  CPU lane / bulk chunk DMA) beat per-token PCIe streaming; every ported vLLM idea measured
  flat or negative across 8 experiments (route bias ×5, tiers, EC profile, chunks, DMA)
- **17.5 → 33-35** — tp1 layer-split (whole experts per card, both cards packed) + DSpark MTP;
  `--threads on` unlocks chunked prefill on the split: 144 → 518 t/s
- **518 → 783 t/s prefill @25K** — PR#344 build (multi-GPU prompt skip, all-reduce off SMs)
  with tp2 + chunk 8192 + `DSPARK_CUM/STREAM=0/CACHE_THREADS=4`
- **Validated**: 9/9 functional (JSON, tools, 8K needle, multi-turn) · 5/5 vision · 10/10 QA ·
  21.8K-doc both needles · OpenAI + Anthropic APIs
- Honest blockers: decode 50 t/s needs a smaller quant (S24-class download) or an EXL3
  stream-lane patch; the 44-53 t/s community rows are GLM, not DeepSeek

### 2026-10-18 — v1.1.15 three-quant ladder: EXL3-3.05 crosses 50 t/s

All three GLM quants tuned on one rig at 262K context ([`v1.1.15-ladder/`](v1.1.15-ladder/)):
**S24 63.6** (70.3 stat, 1258 t/s prefill) · **2bpw 55.2** · **EXL3-3.05 50.0** (52.7 stat).
New lever everywhere: `STRATA_GLM_MISS_SKIP_W` (upstream #228; knees 0.14 S24/EXL3, 0.12 2bpw,
+9-13%), swept per-request with no restart. FETCH_WIDE kernel kept from the EXL3 era.
Validation per quant: functest 8/8; EXL3 arithmetic A/B byte-identical between tiers.

### 2026-10-15 — EXL3-3.05bpw era: 29.2 → 44 t/s decode, 1016 t/s prefill (2×3090)

The quality tier moves to EXL3-3.05bpw (KL 0.068 vs FP; ~3× closer than S24's 2.3bpw class).
Full campaign in [`exl3/`](exl3/): trace, patches, and every lever measured.

- **29.2 → 31.7** — our engine patches: deep prefetch (`PREFETCH_N=8 @fetch/rank4/blocks8`) + wide-fetch
  kernel (32 B loads, expert-parallel grid; `FETCH_WIDE=1`)
- **31.7 → 35.9** — upstream [#194](https://github.com/mw00/project-maya/pull/194): the EXL3 CPU lane's
  AVX2+FMA tier (Zen 2 ran scalar before — 7.05 → 0.51 ms/expert). CPU now computes ~24 experts/token
  overlapped with PCIe fetches
- **35.9 → 36.4** — upstream [#188](https://github.com/mw00/project-maya/pull/188) merged on top
  (split-K CPU experts, prompt CPU share; needed an AVX2 graft — its rows path was AVX-512-only)
- **36.4 → 37.3** — `SPEC_CANCEL=1` (abort the head GPU's doomed pass on a wrong draft) +
  `SPLIT=22` (rebalanced layer split; the auto-search misprices the post-lane balance)
- **37.3 → 43-45 t/s** — `TRUNK_KEEP=6` speed tier: route ranks 7-8 of RAM-tier experts skipped.
  **Not lossless** (KL ~0.02 class) but gate-verified at parity: functest 8/8 ×4, thinking 9/10
  (T3 trait pre-dates it), arithmetic probe suite byte-identical to baseline, and the one miss
  (chain-math) reproduces identically with TK off — A/B-proven pre-existing, not a regression
- **Prefill**: 822 t/s @8K → 908 @20K → **1016 @33K** → 995 @65K (auto/8192 chunks; 32K chunks
  measured worse on 2×3090)
- Sustained check: 5 runs 43.0-44.8 MEAN, no drift; expert pools 98.5% full, zero disk reads
- Bottleneck trace + rejected levers: [`exl3/SETTINGS-EXL3.md`](exl3/SETTINGS-EXL3.md) and
  [`exl3/CURRENT-FINDINGS.md`](exl3/CURRENT-FINDINGS.md); research base for the next jump
  (multi-draft verify windows, AcceptMoE-class pruning): [`exl3/`](exl3/)

### 2026-10-13/14 — Maya-S24 era: 49.2 → 59 t/s @ 262k

The original campaign (table above): env stack + calibration + benchkit, 52.2 greedy / 59
thinking-on, functest 8/8, thinking 10/10. S24 stays on disk as the speed rollback tier.

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
