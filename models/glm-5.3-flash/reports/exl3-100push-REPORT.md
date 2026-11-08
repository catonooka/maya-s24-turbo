# EXL3 100 t/s push — DEFER_CPU=2 breakthrough (2026-11-07 night)

## The missing layer: the CPU lane was SERIALIZING with GPU work
MAYA_GLM_DEFER_CPU=2 — a MoE layer's CPU-lane experts join the residual after the
NEXT layer's mixer, overlapping CPU compute with attention. On 3-GPU EXL3 (CPU lane
was 3.95 ms of ~19 ms/token): **53.3 → 69.1 t/s (+30%)**. Official --bench ladder:

| config | mean | code/prose-expo/html |
|---|---|---|
| baseline (DFlash2 Q4, today) | 53.3 | — |
| + TRUNK_KEEP_MAP (docs map) | 56.0 | +5%, not worth quality cost here |
| **+ DEFER_CPU=2** | **69.1** | 65.1 / 67.6 / 74.7 ← KEEPER |
| + TRUNK on top of DEFER | 64.7 | trunk starves the overlap — never stack |
| + DFlash2 Q6_K drafter | 68.9 | flat; drafter fidelity ≠ acceptance |
| + CPU_SHARED=0 (per-GPU pools) | 62.2 | shared pool wins on 3 GPUs |
| + adapt-every 2 / swaps 192 (Strata #4870) | 68.4 | flat at our 97.5% hit |

Client reality (temp 0.6, spec=8 tune): code 75.4, creative prose 43.0 (bench's
"prose" probe is expository; CREATIVE prose is the true acceptance wall).
Prefill: 432-449 @2k, ~445 t/s @100k (bytes-bound at 3.05bpw; p32k on).
Gate: 7/8 PASS (vision by design). KL ~0.038 documented upstream.

## Research sweep (repos)
- Strata (Niko1221/Strata, 41k stars) = maya's MIT engine ancestor (Qwen3.8-Flash-Next).
  Issue-mined: #4870 adaptive tier defaults cost 10% at 45% coverage (adapt 2/192 →
  flat for us at 97.5% hit); #2038 async-launch loss class of bugs; maya kept
  --adapt-every/--adapt-swaps in src/program/generate.cpp with Strata's slow defaults.
- TensorFold (ashhart, 1.7k stars): MiaAI GLM-5.3 EXL3 4bpw on 2× DGX Spark = 60.4
  prose / 114.7 structured single-stream, ~1950 t/s prefill, FP8 KV, 4-bit dense,
  DFlash2+copy drafts; aggregate 108.8 prose @4 streams. Same prose/structured
  acceptance wall we have. "Copy drafts" (n-gram + DFlash2 stacking) not available
  in maya's GLM path. --parallel exists for aggregate throughput if ever needed.
- llamAmpere: already absorbed by maya upstream (earlier research note).
- upstream maya: 1 commit since v1.1.56 (cosmetic) — nothing to pull.

## What remains between 69 and 100 on EXL3
1. Creative-prose draft acceptance (~43 t/s there): drafter-side, not memory.
   Q6 didn't move it; next would be a drafter fine-tune (out of scope tonight).
2. Prefill at 445 t/s @100k = pure bytes at 3.05bpw through the copy lane.
3. Physics: 125G pack / 72G VRAM; DEFER already hides the CPU lane.

## Serving
- unit glm3-exl3-final.service (system.slice), config ~/maya-glm3/exl3-final.json,
  launcher run-exl3.sh, http://0.0.0.0:8080, 262144 ctx, model glm-5.3-flash.
- Env adds vs old: MAYA_GLM_DEFER_CPU=2 only. Args add: --prefill 32768.
- S24 speed tier config on disk (s24-dflash-p32k.json, 87.8) — swap with
  bash run-s24.sh s24-dflash-p32k.json.

## Post-ladder: server-default spec window
serve/server.py patched (env-gated, local): STRATA_SPEC_DEFAULT=8 injects the
spec=8 verify window for requests that name no tune (explicit client tunes win).
exl3-final.json env carries it. Plain clients now: code 74.4 / prose 46.0
(vs 71.5/40.7 before). Serving invocation c82467f6.

## Verdict vs the 100 t/s goal
Missing layer found and fixed (CPU-lane serialization → DEFER_CPU=2, +30%).
Remaining gap is creative-prose draft acceptance (46 t/s there; code 74-75) —
drafter-model quality, not memory/CPU/kernels. 100 sustained needs a better
drafter or a workflow split (S24 tier for prose, EXL3 for final answers).
