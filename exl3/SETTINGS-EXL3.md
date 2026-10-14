# Maya GLM-5.3-Flash — current settings & findings (2026-10-14: EXL3-3.05bpw A/B added)

## EXL3 push to 45 — attempted 2026-10-14, measured verdict: NOT reachable lossless on this rig
- DFlash2 on EXL3: engine's own speed calibration = 9.4 vs 25.9 t/s MTP → auto-rejected (config
  stripped it). The batched window's expert-dedup doesn't rescue a 2-GPU PCIe-bound rig: drafter
  head + window costs land on the same serial PCIe path. (Upstream's DFlash2+EXL3 wins are 9-GPU rigs
  with CPU lanes / huge VRAM pools.)
- PROMOTE_FILL=64: neutral (29.1). VRAM_EVICT=lru: WORSE (26.5). ctx 131k→64k: neutral (kv-streaming
  keeps freed VRAM negligible). Spec acceptance is healthy (84-96%) — the wall is physics:
  ~87 RAM fetches/tok x ~9.9MB/expert ≈ ~860MB/token over two PCIe4 x16 links ≈ 18-20ms + verify.
  45 t/s lossless needs PCIe traffic halved → no config lever exists; needs CPU lane (AVX-512 host),
  3rd/4th GPU, or a smaller pack (S24 IS that: 59 t/s at KL 0.19 class).
- SETTLED EXL3 config: ctx 131072, env RAM_RESIDENT+KV_INT8 only, MTP drafting, 29.0-29.2 t/s.


- Engine: main @ 4c9f1b1 (v1.1.9, tensor-core EXL3 decode). Launcher run-maya-exl3-tuned.sh, unit maya :8080.
- Config maya-exl3-3.05bpw.json: ctx 131072, kv-streaming on, host 0.0.0.0.
- env: **ONLY** RAM_RESIDENT=1 + KV_INT8=1 (both neutral). 131k ctx, vision on.
- **29.1-29.2 t/s** thinking-on mean (24.8 prose / 32.8 facts / 29.6 code), 2-run stable.
- Functest 8/8; thinking 9/10 (T3: multi-step arithmetic needs thinking on this quant — wrong but
  STABLE at greedy off/low (60/50 vs 53), correct @max; knowledge/code/translation flawless —
  weights verified healthy, this is a behavioral trait of 3.05bpw chain-math without reasoning).

## CRITICAL EXL3 finding (why not 17.8): our S24 env stack is TOXIC on EXL3
- With PROMOTE_MIN=6 + SPEC_SAMPLE/CANCEL: 17.8 t/s, vram hit 40%, promotions 0.00.
- Stock env: 29.2 t/s, vram hit 74-84%, promotions 38-58/tok LIVE. PROMOTE_MIN=6 strangles the
  promotion churn EXL3 needs (bigger experts = fewer slots = must follow usage aggressively).
- RAM_RESIDENT/KV_INT8: neutral (keep for capacity). Rule: **GGUF env vars DO NOT transfer to EXL3 packs.**
- CPU lane measured OFF by engine: EXL3 expert on CPU 7.08ms vs 0.36ms PCIe (20x — Zen2 AVX2 trellis;
  matches our exllamav3 standalone research). All RAM-tier fetches over PCIe; that's the 29 ceiling.
- First 2-GPU EXL3 datapoint: split [0,24)/[24,45) clean, 4336 slots (vs 5565 GGUF S24), works fine.

## S24 (speed tier) — swap back with: config switch to maya-maya-s24.json in run-maya-s24-tuned.sh
- 59 t/s thinking-on mean, 8/8 + 10/10 every gate, KL ~0.19-0.23 class. Quant: 94.7GB, on disk.

## EXL3 vs S24 verdict (turboderp KL chart, HF model card)
- EXL3-3.05 KL 0.0681 vs S24-class ~0.19-0.23 → ~3x quality jump for 30 t/s vs 59 (half speed).
- 4.05bpw (KL 0.0346): rejected — 165GB would collapse slots further (2x closer to FP not worth it
  on 48GB VRAM; #142 measured 19-21 on 4x3090 with 96GB).
- SC_3.10 (KL 0.0574, best KL/GB): not a setup option; would need manual wiring.



## Stack
- **Engine**: Project Maya **main @ a709524** (v1.1.5+, past issue #135's v1.1.2) — rebuilt engine + vision
- **Model**: GLM-5.3-Flash Maya-S24 (94.7GB pack, ~/Maya-data) — unchanged
- **Serve**: systemd transient unit `maya` :8080 (LAN 0.0.0.0), OpenAI + Anthropic APIs, 262k ctx
- **Launcher**: `~/project-maya/run-maya-s24-tuned.sh`

## Tuning env vars (run-maya-s24-tuned.sh)
- `STRATA_GLM_RAM_RESIDENT=1` — RAM tier holds every non-VRAM expert (verified line at boot)
- `STRATA_GLM_PROMOTE_MIN=6` — churn killer
- `STRATA_GLM_KV_INT8=1` — 544B/tok/layer latent cache
- `STRATA_GLM_MTP_KEEP=2` — upstream default
- `STRATA_GLM_SPEC_SAMPLE=1` + `STRATA_GLM_SPEC_CANCEL=1` — exact-distribution spec sampling + wrong-draft cancel (v1.1.4+, free ~+1-2%)
- Config `env`: `STRATA_GLM_PCIE_SHARE=0.25` (from --calibrate: 0.25→48.8 vs default 0.60→43.4 stock tok/s)
- Config args: `--kv-streaming on --expert-profile <repo>/data/expert-profile-glm.bin` (setup-added, kept)

## Measured (2026-10-13, 2×3090, both boots same day)
| config | greedy/official bench (thinking-off, greedy) | production bench (thinking-on, t=0.6) |
|---|---|---|
| v1.0.24 old stack | — | 41.9 mean |
| v1.1.5 stock @262k | 49.2 (issue #135 ref: 53.0 @131k) | — |
| v1.1.5 + our env stack | 52.2 | 57.0 |
| + SPEC levers | — | 57.6 |
| + PCIE_SHARE 0.25 | 51.1 (tie/noise) | 58.4 |
| final (same, rerun) | — | **59.2** (51.5/61.5/64.6) |

Trend across runs 57.0→59.2 partly expert-profile learning (usage file warms across boots).
131k ctx = tie with 262k (kv-streaming+INT8 made ctx ~free). Spec window 6 = tie, keep default.

## Rejected (measured, do not retest without new hardware)
- **DFlash2 drafter: 29.1 vs 47.3 t/s MTP** (repo's own calibration, 2×3090 S24) — same physics as our
  EXL3 finding: drafter verify multiplies CPU-expert passes on an offload-bound rig. Calibration auto-rejected.
- CPU_LANE manual overrides (auto-calibration wins), PROMOTE_FILL raises (older stack), power-cap raises.
- **--parallel 2/4 (2026-10-13, from 0xSero research): C1 33 t/s, C2 agg 49.9, C4 agg 44.4 — all BELOW
  serial 58-59.** On the 2-GPU layer-split, parallel mode reorganizes to conversation-parallel and loses the
  MTP draft/verify overlap; SEQ_CTX=32768 didn't help (identical). Concurrency only pays on single-GPU rigs
  (0xSero: C4 33.98 vs C1 28.15; Maya docs V100 +21-41%). Do not enable on this rig.

## 0xSero/glm53-flash-offload findings (v4.6, read 2026-10-13)
- DRAM ladder (EXL3 3.05bpw, 1×3090 + EPYC DDR4): all-RAM 28.2 / 55G+NVMe 19.5 / 16G+NVMe 14.4 t/s C1.
  Confirms: resident-expert count is the whole game; we're already all-RAM (disk reads 0.00).
- Session drift ±17% on their host — only compare back-to-back runs (we already do 2-run means).
- B70 companion (Arc Pro B70 16GB as pure expert server, glm53-b70-expert-server): 19.5→26.5 t/s on their
  box. A "cheap extra card as expert store" is the hardware path if we ever chase 80 t/s.
- nvme-exact mode is bit-exact at 8.3 t/s — exactness tax is ~2.4x; their fast mode KL 0.005 is the sweet spot.

## Path to 80 t/s (none left in config space on this quant)
Ceiling = resident experts; S24 already the measured-best quant. 80 needs: bigger VRAM pool (hardware),
AVX-512 CPU, or a future faster quant/engine release (repo ships fast — recheck tags monthly).
Watch: --parallel N (concurrent serving), KV-streaming at >262k, STRATA_GLM_Q3/quant ladder if published.

## Gotchas (new this session)
- setup/--calibrate rewrites config: ctx→32768, host→None. ALWAYS re-diff after touching setup.
- Calibration + engine loads MUST run under `sudo systemd-run` (user-slice oomd kills pinned-RAM loads).
- functest.py vision probe now builds a valid PNG programmatically (old hand-hex PNG = false 400).
- `git stash -u` then `stash drop` loses untracked files — recover via `<stash>^3` commit.

## 0xSero-style RAM/CPU-lane push on EXL3 — measured 2026-10-14, NEGATIVE (architecture, not config)
- Host: 8-channel DDR4-2400 (8x32GB, 247GB) — same RAM class as 0xSero's EPYC rig. Their speed comes
  from computing ~152 experts/token ON THE CPU in parallel with GPU work (async elastic lane).
- Maya's EXL3 CPU lane is SYNCHRONOUS on this build: any CPU share adds its full latency to the token.
  Measured: PCIE_SHARE 0.5 -> 4.5 t/s (lane 43/tok in 361ms); 0.85+LANE24 (3.86ms/expert) -> 18.1 t/s;
  0.95 -> engine routes ~everything to PCIe = 29.0 (baseline). Best config remains: lane OFF (auto).
- Root: engine's lane planner prices per-expert LATENCY (7.0ms CPU vs 0.36ms PCIe = 20x) and with
  Zen2 AVX2 trellis that per-expert cost can't drop; 0xSero's win is THROUGHPUT overlap, which needs
  an async lane design. An overlap-capable EXL3 CPU lane (or PCIe fetch + CPU compute pipelined per
  layer) is the concrete upstream ask: at 50% overlap of ~87 fetch-experts' cost, decode 29 -> ~45.

## DFlash2 drafter sweep + prefetch overlap on EXL3 — 2026-10-14, both measured NEGATIVE
- DFlash2 drafters available (Anbeeld GGUF: Q2_K..Q8_0+bf16). Q2_K: engine rejects (selector tables
  unreadable). bf16: known crash (#173). Tested Q6_K (most precise supported): boots, 98.5% window
  acceptance, but 11.5 t/s — the window DOUBLES distinct-expert fetches per round (87→173/tok) and on
  serial PCIe that is pure loss. Q4_K_M was 9.4 via calibration. Verdict: no DFlash2 variant pays on
  a PCIe-bound 2-GPU EXL3 rig.
- Engine DOES have an overlap lane: STRATA_GLM_PREFETCH_N (router-predicted next-layer prefetch).
  pf1@fetch/rank2: 28.7 (vram hit 75→76.7, fetches 87→81 — helps residency, not bandwidth).
  pf2@route/rank4: 28.2 (copies contend with fetches on the link). Net: noise-band neutral.
- CONCLUSION: all three async/overlap surfaces (CPU lane, DFlash2 window dedup, prefetch) exist and
  engage, but none creates PCIe bandwidth. 29 t/s is the measured architectural ceiling for
  EXL3-3.05bpw on 2x3090; 45 t/s requires a 3rd card, AVX-512 host, or upstream pipelined fetch.
