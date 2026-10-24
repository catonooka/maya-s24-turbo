# DeepSeek-inference research → maya-ds4 speed campaign (2026-10-24)

Full day's work: DeepSeek paper/kernel research → apply ladder → decode-regression RCA → restored + verified production config.
Hardware: 2× RTX 3090, Threadripper PRO 3955WX (Zen2, no AVX-512), 247 GB RAM. Model: DeepSeek-V4.1-Flash EXL3-2.9bpw.
Server: `~/maya-ds4-main`, port 8080. Everything reproducible from this folder.

## FINAL production config (verified live, 2026-10-24 evening)

```json
// ~/maya-ds4-main/maya-ds4.json  (launcher: bash serve-v1124-baseline.sh under sudo systemd-run --unit=maya-ds4)
"exe":   "~/maya-ds4-main/build-ds4/maya-ds4-pr344",
"args":  ["--pack","~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/maya-ds4",
          "--max-context","262144","--tp","1","--dspark","on",
          "--prefill-chunk","8192","--dspark-conf","0.5","--threads","on"],
"env":   {"MAYA_DS4_DSPARK_CUM":"0.5", "DS4_CACHE_THREADS":"4",
          "MAYA_DS4_STREAM":"0",        // TP1 WIN: the PCIe stream lane loses on this 176 GB-RAM rig
          "MAYA_DS4_PROMPT_SKIP":"2",   // applies on single-engine (TP1) boots; inert on TP2
          "MAYA_DS4_Q8_BATCH":"1",      // new this campaign: batched int8 rounding, same bits
          "MAYA_DS4_SLOT_KEEP":"1"}     // new this campaign: conversations survive restarts
```

**Measured on this config (engine-side lines, the honest number):**

| workload | rate |
|---|---|
| decode, same-prompt repeats (cache 80%+ hot) | 30.8 – 32.5 t/s (ladder: 26.1→29.4→30.0→30.8→32.5) |
| decode, fresh asks (cache 60-70%) | 26 – 29 t/s |
| prefill @33k high-entropy prompt | ~700 t/s |

TP2 variant (flip `--tp` for prompt-heavy work): decode 23.5-27.2 t/s, prefill @33k 1145-1167 t/s.

## Decode "regression" RCA (why 33-35 read as 20-22)

Three stacked causes, **no engine regression** — pr344 vs v1.1.24/25 binaries measured ties at both TP1 and TP2:

1. **STREAM unpin (L2) was wrong on TP1.** Default-on streaming engages the PCIe lane on single-engine boots and loses ~3-4 t/s when the RAM tier holds every expert. Pin `MAYA_DS4_STREAM=0` restored.
2. **--tp 2 was the campaign shape** (chosen for prefill). TP2 costs 4-6 t/s decode vs TP1 layer-split on this engine+pack.
3. **Bench-method artifact.** The old "33-35" lines were the SAME 65-token prompt's 4th-5th repeat with the expert cache 80%+ hot and CPU lane at 16-20%. Fresh asks are 26-29 on ANY binary; first ask after boot 24-26. Client-side t/s also reads ~4-5 under engine lines on fresh prompts (it includes the ~1-2 s prompt read) — always compare engine lines.

## The apply ladder (one lever per boot, fixed bench, 2-run means)

Baseline v1.1.24 TP2: essay 21.6 / code 21.8 t/s client-side, prefill 1166 @33k.

| lever | verdict |
|---|---|
| L1 v1.1.24→v1.1.25 binary | tie — kept 1125 path available |
| L2 STREAM default-on (unpin) | tie on TP2; **REVERTED after RCA: TP1 loss ~3-4 t/s** |
| L3 Q8_BATCH=1 (#382) | tie (same-bits) — kept, free |
| L4 CPU_LANE=smt (32 threads) | **REJECTED 21.6→13.5 t/s** (SMT siblings belong to readers; docs were right) |
| L5 DS4_AHEAD=2 (+EXTRA 2) | tie (all-RAM, nothing to prefetch) |
| L6 prefill-chunk 16384 | **REJECTED** (8192 wins decode AND prefill) |
| L7 DSPARK_CUM 0.3 | **REJECTED** (acceptance 5/5→1/2 kept) |
| SLOT_KEEP=1 | kept (latency win for agent workloads, no t/s claim) |

## Research findings that mattered (3 parallel agents; full reports in ~/.hermes/cache/delegation/)

Dead on SM86 (never chase): FP8 tensor cores, TMA/clusters/DSMEM, UE8M0 HW path, TMEM/tcgen05, PDL, setmaxnreg.
Already in the engine (no action): grouped/masked MoE GEMM, hot-expert tiering, MTP+confidence scheduling (= DSpark+CUM exactly), prefix-cache (= SLOT_KEEP), CUDA graphs, cuBLASLt tiering.
Config-reachable and measured above; the remaining decode lever is engine work: DS4_PROF splits the 125 ms token into attention 20 / routing+shared 5 / gate-up 6 / **CPU-lane wait 45** / down+post 49 — and the decode PCIe stream lane is source-gated `nm == 1 && mx` (single-GPU + MXFP4), so TP2+EXL3 can't engage it. Port that gate or AcceptMoE verify-reranking = the next jump.

## Gotchas hit today

- `maya.sh` failed-venv bootstrap does `rm -rf .venv` — it wiped the serve venv mid-day. Rebuild: `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements-serve.txt`.
- Bench probes: repeated-sentence prefill fillers compress ~4× in tokens (use high-entropy word soup); a needle with an arithmetic PATTERN in the filler invites compute-not-retrieve (use un-patterned magic strings); cap `max_tokens` or early-stop answers deflate t/s.
- Official `--bench` needs the server STOPPED and the venv present; engine boot facts ("ds4:" lines) ride stderr to the config `log` path, per-request stats to the pump (see `[maya] done:` lines).

## Files

- `DEEPSEEK-RESEARCH-APPLY.md` — the raw research table + ladder as run (working notes)
- `baseline-v1124.out … RESTORED-tp1-oldenv.out` — every bench receipt, in run order
- `official-bench-tp1.out`, `maya-bench.log` — repo official bench receipts
- `quality-gate.out` — 4/4 gate (17×19=323, greedy determinism, needle BANANA-7741, ASCII)
- `ds4-bench.jsonl` — machine-readable per-probe results
- `bench_ds4.py` — the fixed bench (probes + prefill + repeats)
- `maya-ds4.json` — the final config as shipped
