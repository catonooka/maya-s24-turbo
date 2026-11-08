# EXL3 ecosystem for DeepSeek-V4.1-Flash-1017 — external research (2026-10-26)

Research agent sweep of GitHub + HF (full log: exl3-ecosystem-research.log). Context: our rig = 2×3090 + 128G RAM, LibertAIDAI EXL3-2.9bpw, custom maya-ds4 engine, 26-31 t/s decode.

## Where we stand vs the world

No published 24GB-GPU datapoint beats our 26-31 t/s on this model. Closest comparable:
- 0xSero D141, 1×3090 + 503GB RAM: 19.3/24.0/26.6 t/s (C1/C2/C4) — our rig with HALF the RAM matches/exceeds at C1
- Maya #499: 1×3090 + 128G-class RAM SSD-streaming → 4-5 t/s (the trap we avoided)
- r/LocalLLM 144GB VRAM rig: ~40 t/s (3× our VRAM)

## The quants (1017)

| Quant | Size | Notes |
|---|---|---|
| LibertAIDAI 2.9 (ours) | 196G + engram | PPL +1.8% code / +0.4% prose; jinja template; THE layout others copy |
| Mia-AiLab EAQ-v1 2.9 | ~196G | tensor-identical to ours, highest downloads; MTP 4-bit vs our 6-bit |
| 0xSero 2.7 | ~185G | KLD 0.0304, top-1 94.95% — the quality-measured one |
| jayleaton flash17 2.8 | 204G | per-expert mixed widths, coding-weighted; TensorFold-only (won't load elsewhere) |
| bertholomus 4.0 | ~275G | quality headroom, needs 256G+ |
| sfxnz 2.0 MCG+Viterbi | — | NLL 0.139 vs 0.243 unrefit — refit matters at low bpw |
| turboderp's own | — | only V4-Flash-0731, NO 1017 quant from him |

No 3.05/4.05 1017 exists (those are GLM revisions).

## Engines

- **exllamav3 stock: cannot load DeepseekV41 at all.** PR #449 (open, unmerged) adds it + TP + MTP; fixes 3 silent converter bugs (expert-starved calibration, KV pool reset, hc_pre carry). 46.4 t/s on PRO6000+3×170HX
- **TensorFold (Zig)**: the Spark king (57-94 t/s TP2) — and the source of our Recall drafter idea
- **vllm-exl3 plugin**: exists, needs pinned backport images; diffbot 3bpw on 2×PRO6000 Blackwell = 224-308 t/s (hardware, not engine)
- **KTransformers**: no support, issue #2213 open
- **llama.cpp**: GGUF conversion merged, EXL3 none. LM Studio: GGUF only

## Actionable for us

1. **0xSero 2.7bpw** (KLD 0.0304, -11G RAM): candidate A/B — quality actually measured, would ease our 128G pressure
2. **MCG+Viterbi refit technique** (sfxnz): if we ever go lower than 2.7
3. **PR #449's 3 converter-bug fixes**: our 2.9bpw was converted before them — worth a KL check of our own pack (hc_pre carry bug could be in it; our gates pass but a KL panel would quantify)
4. **Engram trap confirmed**: every EXL3 repo excludes tables; ours correctly sources from base shards
5. **10×3090 Maya datapoint** (97 t/s): scaling is near-linear in expert bandwidth — validates our PCIe-wall diagnosis
6. Prefill lever from 0xSero D130: **cached-prefix TTFT 200s → 0.76s** — our slots do this for live convos; nothing to add

## Verdict

Our engine work is at/above the published frontier for this hardware class. The only external artifact worth pulling is 0xSero's 2.7 quant for a RAM-pressure A/B, and PR #449's bug list as a quality audit checklist.
