# 0xSero 2.7bpw on 2×3090 — mixed-K unlock plan

Goal: DSV41-EXL3-2.7-0xSero on maya-ds4, dual 3090, decode 50 t/s with ngram (Recall) — all blockers fixed step by step.

## STATUS 2026-10-27: S1 DONE (commit 42393c32, branch mixed-lane)
- Model runs CORRECT: ppl 2.15 (prose), greedy "capital of France → Paris", counts 1→220+
- Both placements agree (all-CPU ppl 12.19 = mixed-VRAM 12.08 on OOD README text; prose 2.15)
- Decode 12.3 t/s @ 400 tok (counting prompt, cold-ish): wall = CPU lane 65 ms/token
  (~184 experts/token × 0.35 ms), head wait 77 ms overlapped, layers 4.1 ms
- The earlier 49–55 t/s was FAKE-FAST: p.n=0 made the lane skip trellis row advance (L1-cache loops)
- Root cause of 27M ppl: p.k/p.n from l.hl (never built for mixed layers) → ld=0 in rows_avx2_n

## Pack anatomy (measured from tensors.txt)
- 15,360 routed experts, 40 layers × 384, every layer MIXED:
  K2 6,278 (41%) / K3 7,447 (48.5%) / K4 1,600 (10%) / K5 35 (0.2%) — avg K 2.70
- All experts "esplit" (scales clustered at file start, trellis 84MB later) → RAM-staged at load
- MTP drafter native FP8/MXFP4 → --dspark none until S4
- Engram shards 47/48 symlinked (sha256-verified)

## Speed reality check (2026-10-27, post-S1 measurements)
- Steady-state: 12.4 t/s @ 1962 tokens (counting), no warmup effect; wall = CPU lane 63.5 ms/token
- VRAM density is MAX: expert blobs = all 3 parts (~14 MB avg K2.7); 42 GB VRAM holds 3518/15360 (23%);
  decode hit rate 25% VRAM / 75% lane (heat-ranked now; shipped profile's top-64 = 73% of route mass
  but domain mismatch vs live prompts)
- CPU lane per expert: 0.355 ms (AVX2 bands, correct math). Threads capped 16 (pool sizing; --cpu-threads
  ignored - pool built from node cores); memory-bound anyway
- S5 whole-expert PCIe streaming is DEAD at this quant: 179 experts x 14 MB = 2.5 GB/token => 210 ms on
  gen3. Only viable streaming subset: K2 experts (2.95 MB) -> ~20 t/s ceiling. Parked.
- exllamav3 #449's 46 t/s = 3 cards + mtp drafter; their streaming wins on 4x3090 gen3 rigs too, but
  our per-expert size (14 MB) is 2-3x their per-GPU share - the rig class limit
- PATH TO 50: speculation. S4 drafter (native FP8 already on disk!) at 2.5-3.5x accept: 31-43 t/s.
  + ngram tier-B on repeat-heavy workloads: 50 reachable on chat, not fresh prose. Report honestly.

## Remaining steps
- [ ] S2: gates (chain-math, needle BANANA-7741 @30k, greedy determinism, recall gate) + quant_ab panel
- [ ] S3: prefill throughput (windows-only for now; chunks need class-tagged staging = S5's tag)
- [ ] S4: MTP drafter quant (serve → capture Hessians via sudoingX p4_drafter.py → EXL3 drafter)
- [ ] S5: EXL3 PCIe stream lane (lane_layer_ok → EXL3; class tag in pointer low bits for staging slots)
      → the 65 ms CPU-lane wall is the target; exllamav3 #392/#449 precedent (46 t/s on 3 cards)
- [ ] S6: full bench matrix + README history + patches dir + ship

## Research findings (2026-10-26, agent deleg_58f12e0c)
- Quality vs native 1017 (199-conv KLD panel): 0xSero 2.7 = 0.0304 / 94.95% top-1 — TIED with
  LibertAIDAI 2.9 (0.0304 / 94.76%). Swap is free. sudoingX 1017 2.7 = 0.0243 / 95.47% (better calib).
- K plan: one K5 super expert per layer (34/40); L37-39 all-K3 (drafter reads); L35-36 near-all-K2.
  Plan summary: ~/sero_dsv41_exl3_plan_summary.json
- S5 precedent (exllamav3): #392 pinned-arena stream decode (3060: 11.2→19 t/s; 3090+5950X: 27.2→32.8
  @k=6); #449 DeepseekV41 itself, RAM experts + mtp draft: 46.4 t/s / 1397 t/s prefill@40k on 3 cards.
- Pitfalls: gate/up/down share K within an expert (0xSero does); KMix-by-value breaks 4KB params
  (Pascal only); stream-vs-CPU split rig-dependent → sweep k; CPU-fp32 vs GPU-fp16 ~1e-2 rel.

## Rules
- Lossless only: every draft target-verified; gates compare final content.
- Engine tree ~/maya-ds4-main branch mixed-lane (S1 @ 42393c32, off recall-tensorfold d2c81d64).
- Repo notes → dsv41-3090/deepseek-campaign-1024/.
- Placement tiers: K5 supers + L37-39 stay VRAM; L35-36 K2 → CPU lane first.
