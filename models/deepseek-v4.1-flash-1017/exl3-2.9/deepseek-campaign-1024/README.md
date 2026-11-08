# DeepSeek-inference research → maya-ds4 speed campaign (2026-10-24)

Full day's work: DeepSeek paper/kernel research → apply ladder → decode-regression RCA → multi-GPU prompt-skip discovery → restored + verified production config.
Hardware: 2× RTX 3090, Threadripper PRO 3955WX (Zen2, no AVX-512), 247 GB RAM. Model: DeepSeek-V4.1-Flash EXL3-2.9bpw (320B MoE, 262k context).
Server: `~/maya-ds4-main`, port 8080. Everything reproducible from this folder.

## Result: speed growth through the day

| stage | decode (warm / fresh) | prefill @33k | prefill @103k |
|---|---|---|---|
| morning: campaign config (v1.1.24, TP2, STREAM unpinned) | ~21-23 / 20-22 t/s | 1166 t/s | — |
| evening: RCA'd + restored (pr344, TP1, old pins) | 30.8-32.5 / 26-29 t/s | 737 t/s | 835 t/s |
| **+ multi-GPU prompt-skip verified active (mode 1, exact)** | unchanged | unchanged | **+8% (~810-870)** |
| **FINAL, quality gate 7/7** | **30.8-32.5 / 26-29 t/s** | **737** | **~810-870** |

Decode recovered **+45-50%** vs the morning config; TP2 remains a one-command flip for prefill-heavy work (1145-1167 t/s @33k, 1149 @103k — the prefill king shape).

## FINAL production config (verified live, 2026-10-24 night)

```json
// ~/maya-ds4-main/maya-ds4.json  (launcher: bash serve-v1124-baseline.sh under sudo systemd-run --unit=maya-ds4)
"exe":   "~/maya-ds4-main/build-ds4/maya-ds4-pr344",
"args":  ["--pack","~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/maya-ds4",
          "--max-context","262144","--tp","1","--dspark","on",
          "--prefill-chunk","8192","--dspark-conf","0.5","--threads","on"],
"env":   {"MAYA_DS4_DSPARK_CUM":"0.5", "DS4_CACHE_THREADS":"4",
          "MAYA_DS4_STREAM":"0",        // TP1 WIN: the PCIe stream lane loses on this 176 GB-RAM rig
          "MAYA_DS4_PROMPT_SKIP":"1",   // exact skip — VERIFIED ACTIVE on multi-GPU (see discovery below)
          "MAYA_DS4_Q8_BATCH":"1",      // new this campaign: batched int8 rounding, same bits
          "MAYA_DS4_SLOT_KEEP":"1"}     // new this campaign: conversations survive restarts
```

**Measured on this config (engine-side lines, the honest number):**
- decode warm (cache 80%+): 30.8-32.5 t/s · fresh: 26-29 t/s
- prefill: 737 t/s @33k · ~810-870 t/s @103k (skip amortizes with length)
- quality gate 7/7: chain-math, 17×19, greedy determinism, needle @2k, **needle @30k (exercises the skip path over the cut)**, ASCII, exact line count

`flip-tp.sh 1|2` switches decode-shape ↔ prefill-shape in one command (~5 min boot; SLOT_KEEP keeps conversations alive across the flip).

## Timeline (what happened, in order)

1. **Research fan-out (3 agents)**: DeepGEMM repo+design notes; DeepSeek inference overview + V3/V3.2/FlashMLA/DeepInfer; DeepEP + co-design paper arXiv 2505.09343. Output: what's dead on SM86, what the engine already has, what's config-reachable. Full table in `DEEPSEEK-RESEARCH-APPLY.md`.
2. **Apply ladder L1-L7** (one lever per boot, fixed bench, 2-run means): v1.1.25 tie · STREAM unpin tie-on-TP2 · Q8_BATCH kept · CPU_LANE=smt **REJECTED (21.6→13.5)** · AHEAD=2 neutral · chunk 16384 **REJECTED** · DSPARK_CUM 0.3 **REJECTED**. SLOT_KEEP added.
3. **Decode regression RCA**: the remembered "33-35 t/s" decomposed into (a) cache-hot repeat asks, (b) my STREAM unpin costing ~3-4 t/s on TP1, (c) campaign running TP2. Binaries (pr344 vs v1.1.24/25) measured ties — no engine regression. Restored pr344 + TP1 + old pins → 30.8-32.5 warm verified.
4. **TP2-for-prefill research**: runtime TP flip is impossible (boot-time weight layout); PROMPT_SKIP measured inert-looking on TP2; shipped `flip-tp.sh` profile switcher instead.
5. **Engine-patch attempt → no patch needed**: reading `yoco_layer()` closely showed the multi-GPU skip path is already implemented (cut at chunk boundary); its guards only block flow chunks (STREAM lane) and Engram layers at/after the cut (layer 20). Our pack has Engram at layers 1/14 and STREAM=0 → **the skip was already active on our TP1 split**. A/B @103k: 769 (on) vs 710 (off) = **+8%, answers bit-exact**. Docs' "one GPU only" is the guard's condition, not a hard limit. Production moved to mode 1 (exact) over mode 2 (DeepSeek's 128-replay, shifts answers slightly).
6. **Determinism investigation** (quality): thinking-prose rewording between runs traced to DSpark drafting + CPU-lane expert-path variance (documented upstream trait); with dspark off, structured probes are 4/4 identical; final answers converge identically. Gate chain-math expected value was my arithmetic error (362, not 465 — the model was right). FINAL gate 7/7.

## What's dead / what's next

Dead on SM86 (never chase): FP8 tensor cores, TMA/clusters/DSMEM, UE8M0 HW path, TMEM/tcgen05, PDL, setmaxnreg.
Config space: exhausted (this README is the map of it). Next decode jump = engine work:
- CPU-lane wait is 45 ms of a 125 ms token (DS4_PROF) — port the decode PCIe stream lane to `nm > 1` + EXL3 packs (currently gated `nm == 1 && mx`)
- AcceptMoE-style verify-window expert reranking (from the earlier EXL3-50Tps research)
- Multi-GPU prompt-skip already active; TP2 prefill ceiling is the allreduce staging path (~1.7 ms/copy)

## Gotchas (learned the hard way)

- `maya.sh` failed-venv bootstrap does `rm -rf .venv` — it wiped the serve venv mid-day. Rebuild: `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements-serve.txt`.
- Bench probes: repeated-sentence prefill fillers compress ~4× in tokens (use high-entropy word soup); a needle with an arithmetic PATTERN invites compute-not-retrieve (use un-patterned magic strings); cap `max_tokens` or early-stop answers deflate t/s; client-side t/s reads ~4-5 under engine lines on fresh prompts (prompt-read included).
- Compare engine `done:` lines, never client wall-clock alone, when judging decode.

## Files

- `DEEPSEEK-RESEARCH-APPLY.md` — raw research table + ladder + RCA working notes
- `baseline-v1124.out … FINAL-prod` — every bench receipt, in run order
- `quality-gate*.out` — the 7/7 gate receipts (incl. the 30k needle)
- `ds4-bench.jsonl` — machine-readable per-probe results
- `bench_ds4.py`, `quality_gate.py`, `prefill100k.py` — the fixed bench + gate tools
- `maya-ds4.json`, `flip-tp.sh` — final config + shape switcher
- `FIX-queue-heartbeats.md` + `FIX-dsml-tool-args-stream.md` — the two idle-timeout
  fixes (queue wait; tool-arg buffering), root causes measured on the wire
- `probe_stream_gap.py` — socket probe: timestamps every SSE read, prints the max
  silent gap (the tool that found both bugs)
- `patches/` — `queue-heartbeats.patch`, `dsml-tool-args-stream.patch` (git
  format-patch, apply to `serve/server.py` / `serve/frontend.py`)

## 2026-10-27 (II) — S5: 44.2 t/s (12.1 -> 3.7x), the waterline unlock

The mixed placement was **layer-sequential**: layers 0-2 consumed GPU0's entire expert
room (384/384 each), layer 3 took the remainder (278), layers 4+ got **zero**. The
profile's top-60 coverage (89% of routing mass) could never land - 75% of picks still
hit the CPU lane.

**Fix**: one candidate pool per GPU group - every mixed layer's experts ranked by
share-of-layer-mass, filled until the room is exhausted (a global heat waterline).
Engine commit `20d7cab9` (atop S4 `046baad1`, S1 `42393c32`).

| config | t/s | lane ms/tok | lane share |
|---|---|---|---|
| S4 baseline | 12.1 | 63.5 | 74.8% |
| + waterline | 29.6 | 13.1 | 14.9% |
| + DSpark drafter | **44.2** | ~9 eff | 9.1% |

- Correctness re-verified on the waterline build: ppl 12.16 on the OOD README probe
  (was 12.19/12.08 pre-S5 - placement does not change logits), greedy "Paris is the
  capital of France.", counting run ends 318/319/320 identical to non-drafted.
- Drafter accept: 1: 99%, 2: 34%, 3: 67% (S4's FP8->MXFP4/BF16 conversion holds).
- Both commits on `mixed-lane`; expert-profile.bin = self-trained (counting runs).

**Next lever toward 50**: the profile is trained on counting traffic only - a mixed
corpus (chat + code + prose) should push top-60 coverage 89 -> 95%+, worth another
~15-20%. Also untested: DS4_DRAFT_KEEP sweep, prompt-window path on the waterline
placement (prompt-side speed).

## 2026-10-27 (III) — S5: the mixed swap cache; honest prose numbers

**Engine `eeeb9bf4` + `e93e2b11` (mixed-lane).** The static-arena profile could never serve
unseen text (its profile is another corpus's routing). The fix: mixed layers join the engine's
adaptive expert cache.

- `DS4_MIX_SWAP=1`: hottest CPU-lane expert per tick copied into the coldest same-class VRAM
  slot (no spare slots needed - tables switch between windows), evicted expert's region
  materialized for the lane. Direct table writes in the switch (tick_flush's pair budget
  drops pairs -> stale tables -> segfault; caught by the lane's loud check).
- `DS4_PROMPT_SWAPS=k`: after a prompt, queue its own hottest picks (a prompt's routing ~ its
  continuation's). Lane cost 41 -> 26 ms/token.

**Real held-out prose** (`bench-prose.ids`, no counting):

| config | t/s | swaps | lane ms |
|---|---|---|---|
| static arenas | 10.4 | 0 | 77 |
| + swap cache (300 tok) | 15.9 | 1537 | 41 |
| + swap cache (800 tok) | 21.1 | 2440 | ~30 |
| + prompt swaps | 21.1 steady | 1874/495 | 26 |

Degenerate-repeat traffic: 60.2 t/s at lane 4ms (accept 100% all depths) - the ceiling when
routing concentrates; reported as machinery proof, **not** a prose number.

Boot: 221 -> 162 s (`67706582`: branch-B parallel split reads, merged-extent prewarm,
`DS4_WARM_CAP_GB=90`).

**Where this leaves the 50 t/s goal:** seen/repetitive 44-60 t/s; unseen prose 21 t/s and
converging (per-layer serial swaps + the DDR4 lane floor bound it). Remaining levers: swap
batching per tick, faster repack, boot tail 85 s unexplained.

## 2026-10-28 (IV) — the OOM fix, the upstream ports, and S3: prefill 125 -> 588 t/s

Engine `f3854d67` (mixed-lane, pushed to fork). Config env: `DS4_MIX_SWAP=1`,
`DS4_PROMPT_SWAPS=8`, `DS4_CACHE_MARGIN=1.1`, `DS4_CACHE_INFLIGHT=12`,
`DS4_CACHE_THREADS=4` (6 REJECTED), launcher `serve-mixed.sh` via sudo systemd-run.

**The OOM root cause (found live)**: S5's swap cache materialized every evicted
expert's region into a per-layer buffer that was NEVER freed on swap-back-in
(~14MB a swap, thousands of swaps). A 40k window-mode prompt grew anon RSS to
251GB and the kernel took the engine mid-gate. Fix `4304a146`: unique_ptr
ownership, freed when the expert re-enters VRAM.

**Ports from the base repo (mw00/project-maya, 100 commits ahead)**: layer-split
parallel prompt reads (`f9fdb4e0`), expert-table swap race #550 (`32f3c8c6`),
prompt lookup #540/#543 (`23cd6c38`,`1bd23a1c`), TP-wedge fix + serve watchdog +
NUMA RAM copies (`94747fc6`). Skipped #551 (its condition never holds on a
swapping rig). 3 conflicts resolved keeping our Recall/slot code.

**S3 chunked prefill for mixed-K (the headline)**: design in
`S3-CHUNK-STAGING-DESIGN.md`; class tag (cls+1) rides the ring pointers' low
byte, `mix_cls` reads it first, `plan_streams` gains the mixed vs-scheme,
`run_set`/`run_vram` take the eptr path, `copy_ahead` ships a tagged pointer
table, `prompt_active` keeps S5 switches out of a run, `sw_inflight` counted
once (was double-decremented - the cap never bound).

| probe | windows | S3 chunks |
|---|---|---|
| prefill @25k | ~125 t/s | 491 client |
| prefill @100k | ~125 t/s | **588 t/s engine** |
| decode after 107k prompt | - | 26.3 t/s, CPU lane 0.0% |

**Decode knob sweep (6 boots, fresh prose pool each)**: fixbase 24.09 ->
ps8 25.02 / infl12 25.20 / marg11 25.39 / rskip02 24.89 / cth6 **21.02 REJECTED**
(6 cache threads thrash the lane's DDR4; stay at 4). rskip budget 2% = speed
wash, left off. Checkpoint was 21.1 -> now ~25 t/s unseen prose @800 (+19%).

**Left for the "test for real" pass**: 7/7 quality gate on `f3854d67` (ran
clean prose but the gate battery has not re-run), single-class 2.9 pack
regression bench (tag-0 path should be bit-identical), boot tail 85s still
unexplained (boots read 220-264s today), DS4_CACHE_BATCH verdict (A/B running),
mixed-corpus profile validation.

### Close-out (same day)

- **CACHE_BATCH A/B**: batch3 21.79 (lane 32.2%) vs control 23.89 -> REJECTED,
  same DDR4-thrash signature as cth6. The knob stays default 1; the burst-tracking
  fix inside the commit keeps its value.
- **Combo is not additive**: ps8+marg11+infl12 together read 23.9 (pool 8) vs
  marg11-alone 25.39 (pool 3) - pools vary +-1 t/s, so treat 24-25 as the plateau.
- **Final serve config** (commit `1e497755`, in serve-mixed.sh defaults):
  `DS4_PROMPT_SWAPS=8`, `DS4_CACHE_MARGIN=1.1`, everything else campaign base.
  Confirmation boot: 23.58 t/s @800 unseen (pool 9), 0 window-mode lines, chunks active.
- **Net vs the 21.1 t/s checkpoint**: decode +15-19% sustained, prefill 125.6 -> 588 t/s @100k (4.7x).
- Server left up on the final config (unit `maya-ds4`, :8080).
- Deferred to the test-for-real pass: 7/7 gate, 2.9-pack regression bench, S3
  logits-identity spot check, boot tail (220-268s today, was 162s at checkpoint).

## 2026-10-28 (V) — decode push: item 1 (profile retrain) FAILED, items 2-3

Goal: decode 24-25 -> 30+ (2.9-pack warm reference). NVLink found DOWN
(NVML all links inactive — bridge question for the user; TP2's 1166 t/s
prefill shape and peer staging are gated on it).

**Item 1 — expert-profile retrain: REJECTED, both variants.**
- Soft enrich (+9.4k mixed-corpus tok on the old profile): pool 10 = 24.41,
  lane 27.7, hit 73.1 — noise (+3%).
- Zero-and-retrain (19.9k tok fresh corpus, `enrich_mixed2.py`): pool 11 =
  **20.2, lane 35.9, hit 64.7** — clear regression. 20k diverse tokens rank
  the experts too weakly; the accumulated profile (max count 179k, counting +
  sweeps + today's pools) places better than a small clean corpus.
- Restored `expert-profile.enriched1.bak`; retrained kept as
  `expert-profile.retrain1.bin`. Verdict: profile is not the wall — the
  DDR4 lane is. A retrain would need ~100k+ tokens to beat the incumbent.
- Ops note: `sudo tee < /dev/zero` never EOFs (file hit 42.6GB before kill;
  disk fine, truncated). Zero root-owned files with `dd count=1`.

**Item 2 — DSpark coverage: REJECTED, keep CUM=0.5.**
cum75 pool 12 = 23.2; cum90 pool 13 = 24.78; the 0.5 plateau is 24-25.4.
Drafting more positions buys nothing on unseen prose (pos-1 accept 70-78%
caps the chain; fixing that needs a better draft model, not a knob).

**Item 3 — cache-heat knobs: NO CHANGE, defaults already optimal.**
decay95 pool 14 = 20.31 REJECT (stale residents, hit 66); decay80 = 24.20;
bonus15 = 23.99; recent16 = 23.59 — all inside the 23.5-25.4 pool-noise plateau
of the defaults (DECAY 0.9, BONUS 1.0, RECENT 2/2/8).

**Decode-push session verdict:** every knob/drafter/profile lever around this
operating point is exhausted; 24-25 t/s unseen-prose @800 is the plateau for
TP1 layer-split with ~73-76% hit and ~27-29% lane mass. The remaining decode
levers are structural: NVLink bridge (hardware, found down), TP2 mixed mode,
lane-to-GPU streaming experiment, or a stronger draft model. Server restored
to the final config (ps8+marg11, defaults elsewhere).
