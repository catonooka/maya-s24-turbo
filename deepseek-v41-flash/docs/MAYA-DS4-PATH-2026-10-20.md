# The maya-ds4 path: why it resolves our both-cards blocker (2026-10-20 research)

## Our blocker, precisely
vLLM PP2 dies at `initialize_kv_cache`: DSV41 shares compressed KV across
layers (kv_source_layer_ids [2,8,14,20], index_source [+24,28,32,36], engram
[1,14]) — sharing groups span ANY pipeline cut, vllm_exl3 has no boundary
hop. EP2 = all-to-all per layer (latency-bound, worse than single). TP2 =
svh width-shard vs pack layout (hard error). Dual instances = 2x197G > RAM.

## How maya-ds4 (upstream, v1.1.18+, 2026-10-20) resolves EACH of these
Engine: src/ds4/ds4_model.{hpp,cu} — "an EXL3 model over the machine's GPUs,
layer-split, with the routed experts that do not fit in VRAM on the CPU lane".

1. **Cross-layer KV solved by architecture**: the engine keeps each token's
   state as a 128-token window ring PER LAYER + append-only compressed KV of
   the source layers (ds4_model.hpp header comment). No vLLM KV allocator to
   fight — the hand-off at a split boundary is just the window state +
   compressed KV passed through host-mapped pinned memory. The exact thing
   vllm_exl3 lacks, they have as the native data structure.
2. **Layer split, not EP/TP-all-to-all**: contiguous layer ranges per card;
   a token crosses between cards once per window (~hundreds of KB through
   pinned host memory), NOT per layer. No NVLink, no P2P, no NCCL needed —
   "cards on x4 or x1 slots work" (MULTI_GPU.md). Our dead NVLink pair is
   FINE for this.
3. **Equal-GPU pairs ALSO get tp=2 (DEFAULT)**: each of our two 3090s takes
   half the heads + half of every expert FFN, summed through host-mapped
   memory (ds4_model.hpp Options.tp = 2, threads). So BOTH mechanisms stack:
   layer-split across cards + tensor-parallel within layers.
4. **Expert capacity doubles**: each card caches experts for its own layers
   (or its FFN share under tp) — 2x the hot-expert VRAM cache vs our single
   GPU, plus the CPU lane + PCIe streaming lane for the rest.
5. **DSpark drafter built in**: loads mtp.* blocks from the same EXL3 files
   (drafter_pack.py), confidence-gated spec decode, keeps itself only where
   it measures faster.
6. **Engram handled natively**: pack.py writes engram.bin (hash tables) from
   the model's tokenizer; the engine serves engram rows with a row cache
   (eg_hits/eg_reads stats) — the 190G tables we already have on disk.

## Local assets map (zero downloads)
- EXL3 2.9bpw 197G = the LibertAIDAI pack maya's README lists as a supported
  model format (README "DeepSeek-V4.1-Flash-1017" table, row 4).
- Engram shards 47/48 (190G) = "DeepSeek's Engram tables" the same row needs.
- pack.py: `python tools/ds4/pack.py --model ~/models/<dir>` writes
  <model>/maya-ds4/ (geometry + tensor map + engram.bin + tokenizer) BESIDE
  the safetensors — "no weights copied". Needs `tokenizers` python pkg only.
- Speed reference: our own rig (catonooka, 3955WX 247G 2x3090 NVLink row)
  = 29.2 t/s decode on GLM EXL3-3.05 with the split pipeline (v1.1.9).
  ds4 is a newer engine on the same skeleton. No published dual-3090 number
  for ds4 1017 yet — WE will be the first row.

## Build/run plan (in flight)
1. git worktree at origin/main (456a5ca) -> ~/maya-ds4-main (done; the
   tuned-exl3-2x3090 branch + local mods untouched)
2. cmake build maya-ds4 for sm_86 (running)
3. tools/ds4/pack.py --model ~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
   (writes ~small pack dir beside the shards; verify engram vocab check passes
   with our tokenizer.json)
4. maya-ds4 --pack <model>/maya-ds4 --gpus 0,1 --serve  (+ tp auto / threads)
5. Bench vs L1b vLLM (10.9 single / 17.2 agg): expect the layer-split + tp
   shape to beat it; DSpark on top if it calibrates in our favor.

## Fallbacks if the EXL3 path stumbles
- The GGUF path (pack_gguf.py) exists but would need a GGUF we deleted —
  NOT re-downloading; EXL3 path is the local one.
- If ds4 EXL3 decode on ONE card already ~= our vLLM L1b, the split is pure
  upside; if it's slower single-card, the split must make up >1x to win.
