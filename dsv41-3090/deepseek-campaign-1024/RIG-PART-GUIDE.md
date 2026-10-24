# Building a 320B-MoE LLM Rig for $600 of Used GPUs — Why Every Part Earns Its Seat

I run DeepSeek-V4.1-Flash (a 320B-parameter Mixture-of-Experts model, 262k context) at home, on two used RTX 3090s. The stack decodes at ~26-32 tokens/s and prefills a 100k-token prompt at ~830 tokens/s. That doesn't happen because of any single part — it happens because each component covers a specific bottleneck of LLM inference on consumer hardware. Here's the part-by-part breakdown of what actually does the work.

## The core problem: a 197 GB model in a 48 GB GPU budget

The quantized model weighs ~197 GB. Two 3090s hold 48 GB total. So the engine (Project Maya, a tiered-MoE serving engine) splits every layer's experts across three tiers: the hottest ~1,450 experts pinned in VRAM, all ~13,000 remaining experts pinned in system RAM, and KV cache streamed so even 262k-token contexts fit without a giant VRAM pool. Every decoding token touches ~40 layers, and the tier layout turns the whole machine into one memory hierarchy. That framing decides what each part must be good at.

## GPUs: 2× RTX 3090 — the only sane $/VRAM card

The 3090 is the cheapest 24 GB of VRAM you can buy, and for LLM inference VRAM capacity beats everything: more resident experts = fewer trips over PCIe = faster decode. Two specific things matter:

- **Memory bandwidth (936 GB/s per card)**: whatever experts live in VRAM are read every token. Bandwidth sets the hard floor for the VRAM-resident part of decode.
- **24 GB at used-market prices**: we fit dense weights + ~1,450 hot experts + streaming KV with ~237 MiB to spare. The pool is genuinely full — the boot log proves it.

One card is a waterblocked Founders Edition (fans off at 300 W, 61°C), the other an air-cooled replacement after the first one went flaky — a reminder that used 3090s need burn-in testing. Both are power-capped at 300 W: during decode the cards are expert-fetch-bound, not power-bound, so the cap costs nothing and saves real money on electricity (capped-idle is a priority on a rig that runs 24/7).

Not chasing flashier cards was deliberate. Newer consumer cards with 16 GB would mean less resident experts and slower decode. Tensor-core FP8 paths (Hopper/Blackwell) are irrelevant here — the quant is EXL3, and everything FP8-specific is hardware-gated off on Ampere anyway.

## Motherboard: WRX80E-SAGE — 8 memory channels and 7 PCIe slots

The boring part that quietly decides everything. Two things it provides:

- **8-channel DDR4 memory.** This is the big one. The RAM tier holds ~176 GB of experts, and decode is bound by how fast the CPU lane can read them. Memory bandwidth = inference speed for the 87% of the model that doesn't fit in VRAM. A typical desktop board gives you 2 channels; this gives 8.
- **PCIe lanes (128 from WRX80)**: both GPUs run at full x16 Gen4. The PCIe links are the second load-bearing path — every expert not in VRAM is fetched over them, and prefill chunks pipeline across both cards. Starving a card to x4 would throttle exactly the path the engine depends on.

Plus room to grow: 7 x16 slots means a third 3090 later drops in without replacing anything — and a third card is the measured next decode jump (+24 GB of experts ≈ +50% residency).

## RAM: 247 GB (8× 32 GB) — the tier that makes 320B possible

You cannot serve this model without enough RAM to pin every expert. 247 GB means:

- Every expert is RAM-resident — zero SSD reads during decode (verified in the engine logs, 0.00 disk reads/token warm). With less RAM, cold experts would hit NVMe and decode would collapse to single digits.
- The CPU lane computes ~28-35% of experts per token directly in RAM, overlapping with PCIe fetches of the rest.

One honest miss: the modules are 2400 MT/s in a 3200 MT/s board (buying used, this is what showed up). Zen 2 lacks AVX-512, so the CPU lane is ALU-bound anyway — faster RAM would help the fetch path, but the compute path caps the gain. If I rebuilt today I'd pay the difference for 3200 MT/s.

## CPU: Threadripper PRO 3955WX — 16 cores picked for a job most builders ignore

The CPU has exactly one inference job: run the expert lane. The engine measures it at boot — one expert costs ~0.5 ms on the CPU vs ~0.36 ms over PCIe — and splits work between the two lanes accordingly. What matters:

- **16 physical cores = 16 lane threads.** More full-rate cores, not SMT threads — measured, SMT lane threads made decode *worse* (21.6 → 13.5 t/s; the siblings belong to the disk/Engram readers).
- **Enough cores left over** for the serving stack, the pinned-memory copy engines, and the DSpark drafter without stealing lane cores.
- The 3955WX specifically because it's the cheapest way into 8-channel memory + 128 PCIe lanes. The CPU compute itself is honestly the weakest link (no AVX-512), and it's the documented reason decode tops out where it does.

## PSU: sized for the spike, efficient at the idle that matters most

Two 3090s (up to 370 W stock limit each) + a 280 W CPU + fans can transient-spike past 900 W during prefill bursts — a quality unit sized above that peak keeps the rig from "randomly" rebooting mid-generation. But the real PSU story for an LLM rig is the *other* 22 hours of the day: this box idles most of the time, and electricity is money. High efficiency at the ~150-200 W idle draw is what actually shows up on the bill.

## The payoff: what the combination buys

Measured, cold-cache, quality-gated (7/7 functional battery incl. 30k-token needle retrieval):

- **Decode:** 26-29 t/s fresh, 30.8-32.5 t/s warm (DSpark speculative drafting active)
- **Prefill:** 737 t/s @33k tokens, ~830 t/s @103k (the multi-GPU prompt-skip we verified active adds +8% at length, answers bit-exact)
- **262k context** served without KV pressure, vision tower included

Every one of those numbers is bottlenecked by a different part: decode by the CPU lane + RAM bandwidth + PCIe, prefill by GPU compute, context by the streaming KV design. That's the whole point of picking parts this way — on a workload this weird, the boring components (motherboard channels, RAM capacity, PSU efficiency) carry as much of the result as the GPUs.

*Config, benchmarks, and every rejected experiment: github.com/catonooka (maya-s24-turbo).*
