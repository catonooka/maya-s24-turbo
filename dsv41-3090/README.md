# DSV41 Flash EXL3-2.9bpw — 2x3090 serving tracker

Repo: same git as project-maya (branch `tuned-exl3-2x3090`), folder `dsv41-3090/`.
Model: ~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw (386G: 43 EXL3 shards + sidecars
+ engram shards 47/48 sha256-verified, moved to ~/models/DeepSeek-V4.1-Flash-engram/).
Hardware: 2x RTX 3090 24G, **247G RAM measured** (profile said 128G — wrong), 1TB NVMe.
Stack: sybil-solutions/dsv41-flash-offload image D141 (digest c247bad…, pulled via
aria2 16-conn at 23 MB/s + verify + docker load; single docker pull crawled 0.7 MB/s).

## Current state
- Container `dsv41` on GPU0, API 127.0.0.1:8000, served name `deepseek-v4.1-flash`
- Engram disk tier ACTIVE both layers (384M rows each), CPU tier 35 layers (AVX2),
  EC VRAM mirror cache ~6.25 GiB (506 slots), fp8 KV, CUDA graphs FULL_DECODE_ONLY
- Smoke: 17*19 -> 323 (matches quant maker's reference)
- Launch: `bash dsv41-3090/serve-dsv41.sh [extra vllm args]`

## Pack adaptation for the 2.9bpw quant (vs D010's 3.0bpw)
1. `prepare_pack.py --source` pointed at our 2.9bpw dir; engram shards moved to
   sibling `DeepSeek-V4.1-Flash-engram/` (recipe expects that name for symlinks)
2. Dropped 43 `layers.N.ffn.gate.bias_vl` keys from pack index (text-only serve;
   vLLM mapper renames `.gate.bias` but not `.bias_vl` -> KeyError)
3. config layer_bits: {18-22: 2} (K2 routed experts) + {40-42: 4} (DSpark draft)
   — first attempt dropped the 40-42 entries, draft load shape-mismatched
4. verify-pack DIFFs are expected (different quant than the measured D010 hashes)

## Decode throughput log (250-tok essay bench, single stream)
| config | tok/s | note |
|---|---|---|
| first boot | 5.4 | CPU tier silent: only 35/40 layers registered (K2 layers bail) but DSV41_CT_LAYERS=40 default never reached `start()` |
| `--cpus 30` | 5.4 | docker CPU quota was NOT the bottleneck |
| `DSV41_CT_LAYERS=35` | 8.5 | CPU tier + EC live |
| after EC warm | 11.0 | EC 506 slots x 12.7 MiB (margin 1500) |
| EC margin 300 warm | 11.0-11.75 | EC 600 slots; hit rate 38.7% and rising; cpu tier: 3.1 cpu_experts/call, gpu_wait 0.77ms |
| + DSpark k=3 spec | 5.3 | NET LOSS: MoE draft streams its own experts over PCIe; acceptance 1.8/step doesn't cover it. Reverted. |

Prefill: ~375 tok/s (4k-token prompt, 8192 chunk, DMA staged + hybrid CPU tail).

Live tier stats (step 700): calls 24360, hits 60339 (38.7%), misses 95591,
cpu_experts 75178 (3.09/call), host_jobs 23680, gpu_wait_ms/call 0.765, timeouts 0.

## Root causes found
- `ct/ct_vllm.py register()` skips layers with `_exl3_cold_k != 3`; our 5 K2 layers
  (18-22) stay pure zero-copy PCIe. CPU AVX2 kernel ft_core.h hardcodes BITS=3.
- py-spy showed main thread in `cudaStreamSynchronize` the whole decode: step time
  is GPU-side PCIe streaming of cold experts, not Python/CPU.
- DSpark on this stack: draft model = MoE too -> 3 extra expert-streaming passes
  per accepted step. On DGX Spark (LPDDR UMA) that's cheap; on 3090 PCIe it isn't.

## Next levers (untried)
- [ ] Extend CPU tier to K2 layers (patch ft_core.h BITS template or pad trellis
      K2->K3 with zero words — trellis is bit-packed, padding needs care)
- [ ] EXL3_HOST_COLD_PACK: serve experts from a 2bpw cold pack (repo supports it:
      `_cold_pack_bits()` reads bits from the cold pack config) — quality tradeoff,
      LibreertAIDAI also ships a 3.5bpw; a 2bpw cold pack would halve PCIe bytes
- [ ] GPU1 is idle: second container serving another stream, or TP=2 experiment
      (vllm-exl3 host-expert path is TP1-oriented; Mia kit does TP2 on UMA only)
- [ ] Engram: row_store RWF_NOWAIT two-tier (research notes §3) — current disk tier
      already stages rows via host callback; measure its share of step time first
- [ ] Tuning knobs: DSV41_CT_* policy (TZC/THIT/A/B/TOK), DSV41_EC_SLOTS bigger,
      EXL3_HOST_DMA_SLOTS/BATCH for prefill
