# My home LLM rig: why I picked each part

I run DeepSeek-V4.1-Flash, a 320B MoE model with 262k context, at home on two used RTX 3090s. It decodes at 26 to 32 tokens per second and reads a 100k token prompt in about two minutes. People ask what's inside, so here is a short sharing of the part choices and what each one actually does for LLM inference.

**GPUs: 2x RTX 3090 (used)**

The cheapest 24GB cards you can find. VRAM size is the number one spec for local LLMs. The engine keeps the hottest experts in VRAM and streams the rest from RAM, so more VRAM means more experts stay on the fast path. Two cards also split the model's layers, and prefill chunks run across both. Both are power capped at 300W because during decode the cards wait on memory, not power, so the cap is free money saved on the bill.

**Mainboard: ASUS Pro WS WRX80E-SAGE**

Two things matter here. Eight memory channels feed the CPU the experts that do not fit in VRAM, and full x16 Gen4 slots for both GPUs keep the streaming path wide. A normal desktop board with two memory channels would cut the RAM tier's speed and hurt decode. It also has seven x16 slots, so adding a third 3090 later needs no new parts.

**RAM: 247GB (8x 32GB)**

The model needs about 197GB quantized. With 247GB pinned, every expert lives in RAM and nothing reads from disk during generation. Without this, cold experts hit the SSD and speed collapses. If you build something similar, get the fastest modules your board takes, mine are 2400 MT/s in a 3200 board and that is a small regret.

**CPU: Threadripper PRO 3955WX, 16 cores**

The CPU computes part of the experts directly in RAM while the GPUs fetch the rest. The engine measured one expert at about 0.5ms on CPU versus 0.36ms over PCIe and splits the work between both lanes. Sixteen real cores matter more than thread count, turning on SMT for the lane actually made things slower in my tests. One honest note: this chip has no AVX-512, which is the known ceiling on decode speed for this setup.

**How it works together**

The engine splits each layer's experts across VRAM, pinned RAM and streaming KV cache. Decode is bound by RAM bandwidth and PCIe, prefill by GPU compute, and the huge context fits thanks to streaming KV. Result: 26-32 t/s decode, 737-830 t/s prefill, 262k context, all quality gated.

Config and all benchmarks are public: github.com/catonooka
