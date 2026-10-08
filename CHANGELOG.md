# Changelog

Every release is on GitHub (Releases) with these notes; every published change moves the last number. Update: `git pull`, then `./setup.sh` (Windows:
`START-MAYA.bat`) - it recompiles only what changed and starts.

## v1.0.9 - 2026-10-08

Switching between conversations no longer re-reads them: Maya keeps the last few on the SSD.

- **Conversation slots:** when a request doesn't continue the conversation the engine holds - another chat, or an
  agent tool's side request (a title, a sub-task) - that conversation is first set aside in a file on the SSD (its
  attention caches and recurrent state, about 0.15 GB plus 21 KB a token: 0.2 GB at 3K tokens, 0.8 GB at 30K), and a
  later request that continues it takes it back instead of reading its whole prompt again. On 1x Tesla V100 with
  Maya-S, going back to a 3,500-token conversation after another one: prefill (reading the prompt) 10.2 s -> 2.2 s;
  setting it aside 0.09 s, taking it back 0.06 s. A conversation taken back from its file continues token for token as
  it would from memory, and each one still recalls its own details after others ran in between (one GPU, and the
  two-GPU split with the MTP block). Requests still run one at a time. Up to 4 conversations of 1,024 tokens or more, 16 GB on disk, never
  below 8 GB free; `STRATA_GLM_SLOTS=0` turns it off (README > Tuning).
- **Prefill:** the disk read-ahead buffer grows to 3% of free RAM, up to 96 experts (was 2%, up to 64): on two GPUs
  the second one no longer waits on the SSD between layers (2x Tesla V100 with 30 GB RAM: +1% prefill; with 64 GB RAM,
  which reads the SSD less, unchanged).
- README: a table of speeds measured by users (`--bench`), starting with 2x TITAN RTX.
- `--bench` / `--report`: a prompt-only request's engine line no longer shows a decode speed in every case it has
  none.

## v1.0.8 - 2026-10-08

`--bench` warms up before it measures prefill, so speed reports compare fairly.

- **`--bench`:** one unmeasured prompt is read before the 2k and 8k prefill measurements. The 2k figure used to be
  the first prompt after start-up, with the caches still cold, and could read lower than the 8k one (152 vs 238
  tokens/s on a Titan RTX pair). Reported by @dummerjindabin (#8).
- **`--bench` and `--report`:** a request that only read a prompt shows just its prefill time in the engine lines,
  not a meaningless decode speed (it read "96,000 tokens/s").

## v1.0.7 - 2026-10-08

RTX 20-series cards (Turing) no longer crash on long prompts.

- **Fix (RTX 20-series, Titan RTX, Quadro RTX):** any prompt longer than about a thousand tokens stopped the engine, and
  the server restarted it with the conversation lost. The prefill attention asked for 66 KB of shared memory, more
  than the 64 KB Turing GPUs have, and the failure went unnoticed until the engine crashed. Both attention kernels
  now keep the cached rows in shared memory in their 16-bit form - the same values and arithmetic in half the space
  (34 KB, which every NVIDIA card gives without asking). On Turing, decode at long context also gets the faster
  attention it was silently skipping. Other cards: the same answers token for token, the same speed. Found and
  diagnosed by @dummerjindabin (#8).
- **AMD groundwork (#2, by @boxwrench):** the signals the GPU sends the CPU (the expert requests the CPU lane answers,
  the doorbells) are published past AMD's GPU cache, as in Strata #697, and a new HIP test replays the real routing
  handoff. The Linux AMD setup itself is in review (#7). NVIDIA: unchanged.

## v1.0.6 - 2026-10-08

A standard speed test: `./maya.sh --bench` (Windows: `START-MAYA.bat --bench`).

- **`--bench`** measures the installed model on your machine in a few minutes, with Maya stopped: decode on the same
  three questions everywhere (after a warm-up answer) and prefill at 2k and 8k tokens, through the engine the way the
  dashboard starts it. It writes `maya-bench.txt` with the results and the engine's per-token breakdown; `--report`
  includes it - so speed reports from different machines can be compared.
- README: single-GPU speeds (1x Tesla V100 32 GB, 64 GB RAM: decode up to 19 tokens/s, prefill up to 370 tokens/s).

## v1.0.5 - 2026-10-08

Maya-M is out: the larger quant (116 GB), closer to the full model token by token.

- **Maya-M** on Hugging Face (`Maya-M/`): IQ2_S gate/up experts with error-feedback rounding, IQ3_XXS down
  projections, IQ3_S in the most sensitive layers, from Z.ai's FP8 release with the FP8 model's own statistics,
  calibrated toward tool calls and front-end code. Against the FP8 model: 23% lower KL divergence than Maya-S
  (0.329 vs 0.428), the same next token 86.2% of the time (Maya-S 83.3%), and 97.9% of its zero-shot accuracy - the
  same as Maya-S (multiple-choice tasks do not separate the two). Set it up with
  `./setup.sh --setup --model Maya-M` (Windows: `START-MAYA.bat --setup --model Maya-M`); downloads are checked
  against their published sha256. Maya-S stays the default. Results: `bench/results/MAYA-M.md`.
- **Decode:** the dense matrix-vector products run in kernels compiled for their weight type (Q6_K, Q8_0, Q4_K,
  Q5_K) - 10% faster per product, bit-identical results; it shows on cards that hold most experts in VRAM.
- `--model` now takes that download even when an earlier setup used `--gguf-dir`.

## v1.0.4 - 2026-10-08

Faster prefill again (how fast Maya reads your prompt), and more room for experts at long context.

- **Prefill attention on the tensor cores:** the prompt's attention (the absorbed MLA over each token's selected
  positions) now runs on the GPU's tensor cores, with the next positions loaded while the current ones compute - FP16
  operands with F32 accumulation, as flash attention (within ~1e-3 of before). RTX 20-series cards keep the previous
  kernel automatically. Prefill on 2x Tesla V100 32 GB, prompts of 2k / 8k / 16k / 30k tokens: 273 / 428 / 487 / 506
  -> 286 / 469 / 538 / 561 tokens/s (since v1.0.1: +19% / +42% / +56%). On one of those GPUs: 8k-token prompt 226 ->
  243 tokens/s.
- **The attention cache in FP16:** half the memory, so more of the GPU holds experts - at the default 32K context
  0.66 -> 0.47 GB a GPU, at 128K about 0.8 GB more for experts on each GPU (faster decode at long context).
- Quality: the same long texts scored before and after (6,000 tokens read, the next 400 scored) differ by +0.8% in
  likelihood, less than two runs of the same engine differ from each other (+1.4%) - no measurable change.

## v1.0.3 - 2026-10-07

A one-command report for problems and speeds: `./maya.sh --report` (Windows: `START-MAYA.bat --report`).

- **`--report`** writes `maya-report.txt` in the Maya folder: your GPUs (VRAM, driver, PCIe link), CPU, RAM, disks,
  your Maya setup (model, context, GPUs) and the engine log's speed lines - how the model is split across VRAM, RAM and
  the SSD, and where each token's time goes. Attach it when you report a problem or a speed, so the engine can be
  tuned for your machine. Nothing is sent anywhere; your home folder shows as `~` and no API key is included.

## v1.0.2 - 2026-10-07

Faster prefill (how fast Maya reads your prompt): up to 46% on two GPUs and up to 2x on one.

- **Prefill speed:** the experts a prompt needs that are neither in VRAM nor in RAM are read from the SSD ahead of
  time into a deeper buffer (the reader was keeping the NVMe at about a quarter of its speed), the next layer's
  experts are read while the current one computes, and the prompt is cut into bigger pieces on bigger cards (fewer
  times every expert is fetched). Prefill measured on 2x Tesla V100 32 GB with 30 GB RAM, prompts of 2k / 8k / 16k /
  30k tokens: 240 / 331 / 345 / - -> 273 / 428 / 487 / 506 tokens/s. On one of those GPUs: prefill of an 8k-token
  prompt 115 -> 226 tokens/s. Answers are unchanged in quality (the same text, up to rounding).
- **Support Project Maya:** [buymeacoffee.com/peasantsmith](https://buymeacoffee.com/peasantsmith) (README, and the
  Sponsor button on GitHub).
- README: how Maya grew out of Strata; exported chats are named `maya-chat-*.md`.

## v1.0.1 - 2026-10-07

- **Windows (experimental):** `START-MAYA.bat` sets Maya up the way `./maya.sh` does on Linux - Python, the
  engine and the image encoder compiled with Visual Studio 2022 Build Tools and CUDA 12.8, the model download, the
  dashboard. The engine reads the model's experts with unbuffered parallel reads and sizes its pinned RAM to what
  Windows allows (RAM + page file). It compiles on Windows; it has not been run on a Windows PC with an NVIDIA GPU
  yet - tell us how it runs. See README > Windows.
- **Fix:** a second GPU big enough to hold all of its experts (e.g. 64 GB cards) stopped the start with "the pinned
  RAM tier did not allocate".
- **Fix:** temperature 0 is greedy and deterministic again.

## v1.0.0 - 2026-10-07

The first release.

- **Maya-S** (96.5 GB): Project Maya's compact quant of GLM-5.3-Flash, made from Z.ai's FP8 release. It keeps
  97.9% of the FP8 model's zero-shot accuracy (ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, PIQA).
- The engine: the model's experts tiered across VRAM, pinned RAM and the SSD, one GPU or two that share the layers,
  MTP speculative decoding, a thinking budget, images on demand.
- The installer (`./setup.sh`), the dashboard and the OpenAI / Anthropic compatible API.
