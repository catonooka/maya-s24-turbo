# maya-ds4 tuning results — 2026-10-21 (final checkpoint)

## DSpark conf sweep (tp2)
| conf | warm decode |
|---|---|
| 0.4 | 21.4-22.5 |
| **0.5 (default)** | **26.1-28.2** |
| 0.6 | 24.9-27.1 (burns more drafts/kept) |

## tp1 layer-split vs tp2 tensor-parallel — THE FORK
| config | decode | prefill @6K | prefill @25K | VRAM experts |
|---|---|---|---|---|
| tp2 | 26-28.2 | **379-384** | **516** | 3504 (GPU1 idles 7.7G) |
| **tp1** | **32.1-35.1** | 89 | 144 | 3071 (both cards packed ~20.8G) |

tp1 = each card owns whole experts for its layer range (CUDA0: L0-42,
CUDA1: L19-39+head): decode 35 but prefill serialized across the split.
tp2 = both cards share every layer: prefill flies, decode a step slower.

**Chosen default: tp1** (agent/chat workload = short prompts, long decode).
Switch = one flag in maya-ds4.json (`--tp 1|2`) + restart.

## Expert profile
The engine persists what it learns: expert-profile.bin in the pack updated
at runtime (grew 61K->66K after our benches; counts scaled, PC-specific
routing accumulates). Boot placement gets better with use. KEEP the pack's
profile file — do not delete.

## Final numbers (tp1 default)
- decode single: 32-35 t/s (3.4x vLLM L1b 10.4)
- prefill: use tp2 when loading long docs (one restart), tp1 for chat
- context 262K, DSpark on, conf 0.5
