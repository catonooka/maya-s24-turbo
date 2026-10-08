# Changelog

Every release is on GitHub (Releases) with these notes; every published change moves the last number. Update: `git pull`, then `./setup.sh` (Windows:
`START-MAYA.bat`) - it recompiles only what changed and starts.

## v1.0.13 - 2026-10-08

Tool calls work: GLM-5.3-Flash's function calls reach your apps and coding agents instead of ending the request.

- **Tool calls (issue #5, reported by @xldistance):** GLM-5.x writes its calls in its own form
  (`<tool_call>NAME<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>`), and the server read only another
  model family's form, so every tool call ended the request with "malformed tool call" - in the chat's tools, MCP,
  and every OpenAI or Anthropic client (Claude Code, agents). Both forms are read now, whole and streamed (the call's
  name as soon as it is written, then its arguments piece by piece), and checked end to end: the call, its result
  in the next turn, both APIs.
- **A call only quoted in the answer stays text:** a `<tool_call>` inside a code block or inline code (an example the
  model shows) is never run - an example `rm -rf` must not reach your shell (from Strata #1058).
- **A stuck engine is restarted:** an engine that says nothing for 90 s and in that time uses no CPU, disk or GPU is
  ended, the request gets an error, and the next request starts it again; a silent engine that is working (a long
  prompt on a slow PC) is never ended (`STRATA_ENGINE_STALL_S`, `0` = off; from Strata #1317).
- **The server, from Strata 0.1.41:** up to 256 connections wait to be accepted (an agent opening many at once got
  "connection reset"; `STRATA_HTTP_BACKLOG`); a request body is checked before it is read (a bad Content-Length is a
  400, one over 256 MiB a 413 - `STRATA_MAX_BODY_MIB`), relays' chunked bodies are read, and a malformed request is
  a 400 with its traceback in the server log instead of a dropped connection; `--api-key` takes several keys
  (`KEY1,KEY2`, or a list in the config).
- `--env` now also reaches these server settings (the rest still go to the engine).
- Checked: 92 server tests (12 new for tool calls, 10 for the rest); the engine is unchanged from v1.0.12.

## v1.0.12 - 2026-10-08

Maya runs on up to 16 GPUs and reads long prompts much faster on one GPU, contributed by @needmorevram.

- **Up to 16 GPUs (#12 by @needmorevram):** `--gpus 0,1,2,...` takes up to 16 GPUs (two before). Two GPUs split the
  layers in the middle as before; with more, each GPU takes a share sized to its free VRAM (`STRATA_GLM_SPLIT` or the
  config's `"layer_split"` pin it). Measured by @needmorevram on nine GPUs (8x RTX 5060 Ti 16 GB + an RTX 3090):
  decode (writing the answer) about 28 tokens/s, prefill (reading the prompt) 1,150-1,210 tokens/s on 8K-token
  prompts.
- **Prefill on one GPU:** bigger prompt chunks (up to 32,768 tokens, sized from the free VRAM); each expert's output
  added in as soon as it's computed, so a chunk holds about twice the tokens; the least-used RAM-tier experts computed
  on the CPU while the GPU loads the rest; experts copied to the GPU while the attention runs; faster sparse-attention
  scoring; a tensor-core attention kernel for Ampere and newer. 1x Tesla V100 with Maya-S: 2K / 8K / 16K-token prompts
  267 / 373 / 362 -> 263 / 578 / 620 tokens/s. 2x V100: 293 / 485 / 553 -> 313 / 541 / 674 tokens/s.
- **Decode with most experts in RAM:** hot RAM-tier experts move into VRAM in the background while it answers
  (`STRATA_GLM_PROMOTE`), and on two-socket machines the RAM tier is spread over both sockets' memory. On
  @needmorevram's RTX 3090 with GSQ-RCO 3.5-bit: 15.3 -> 18.7-19.9 tokens/s. 1x and 2x V100 with Maya-S: unchanged
  (17.1 and about 27.6 tokens/s).
- **GSQ-RCO 3.5-bit download:** a community quant (137.1 GB, by pfeifferj, MIT) in the setup's model question, or
  `--model GSQ-RCO-3.5bit`; checked against its sha256. It is not a Project Maya quant and is not measured by us.
- **`./maya.sh --calibrate`:** measures decode with a few CPU-lane settings (how many RAM-tier experts go over PCIe
  instead of to the CPU, and how many CPU threads) and keeps one only if it is more than 3% faster (README > Tuning).
- **`--models-dir`** names the folder downloaded models go to (`--data-dir` before; older configs still work).
- **Our follow-ups to #12:**
  - Builds for Volta / Turing (V100, RTX 20) and Windows again: the new attention kernel compiles only for Ampere and
    newer, the page-cache drop is Linux-only, and the AVX-512 kernel converts FP16 in a way MSVC accepts.
  - One GPU beside a 6-core CPU kept its decode speed: the CPU's share of each token is cut into about 48 pieces in
    all (`STRATA_GLM_CPU_SPLIT`). One piece per thread, as submitted, took 1x V100 from 17.0 to 14.3 tokens/s; now
    17.1. Machines with 40 or more threads still stream one piece each.
  - The tokenizer remembers what it has read: a 220K-token agent prompt is tokenized in 0.23 s instead of 0.89 s, and
    in 0.013 s when it is sent again with a new turn (the same token ids).
- **Checked:** the engine's parity tests pass on 1x and 2x V100. On held-out text the model's loss stays within
  v1.0.11's run-to-run noise; on one GPU it now varies a little more from run to run.

## v1.0.11 - 2026-10-08

Maya runs on AMD GPUs (experimental): RX 7900 XT / XTX and Radeon AI PRO R9700 / RX 9070 on Linux, contributed by
@boxwrench.

- **AMD (experimental, #7 by @boxwrench):** `./maya.sh --backend hip` builds Maya's engine with ROCm 7 for RX 7900
  XT / XTX (gfx1100) and Radeon AI PRO R9700 / RX 9070 (gfx1201) - Linux, one GPU, text only. Measured by
  @boxwrench with Maya-S: on the R9700, decode (writing the answer) about 20 tokens/s and prefill (reading the
  prompt) up to 490-560 tokens/s; on the RX 7900 XT, prefill up to about 410 tokens/s. Setup and measurements:
  docs/AMD_MAYA.md.
- **Prefill on AMD:** the prompt projections go through hipBLASLt with tuning tables for each card, and the prompt
  runs in bigger sub-batches on cards with room (`STRATA_GLM_PREFILL_SUB`; NVIDIA keeps 256).
- The groundwork, the GPU-to-CPU signal fix, came in v1.0.7 (#2).
- NVIDIA: unchanged - the same tokens as v1.0.10 on 2x Tesla V100, every target builds.
- We have no AMD hardware ourselves; these results are the contributor's. Tell us how it runs on yours (`--bench`
  and `--report` learn AMD in a follow-up).

## v1.0.10 - 2026-10-08

RTX 20-series cards now read prompts on their tensor cores too.

- **Prefill on Turing (RTX 20-series, Titan RTX, Quadro RTX):** the tensor-core prompt attention now keeps its context
  in registers on these cards, so it fits their 64 KB of shared memory. They had been using the slower F32 kernel.
  2x TITAN RTX, an 18,476-token prompt: 62 s -> 58 s of prefill (reading the prompt). By @dummerjindabin (#10).
  Other cards keep the kernel they had; that variant's arithmetic is only correct on Turing and newer (a V100 got the
  attention wrong with it), so it is used only where the other one doesn't fit. A V100 gives the same tokens as
  before.
- **Prefill for 4-bit (Q4_K) weights:** they are converted for the tensor cores the fast way, as 5- and 6-bit ones
  already were - the same values. A test quant with 4-bit attention, 1x V100: 224 / 291 / 285 ->
  268 / 368 / 358 tokens/s at 2k / 8k / 16k, the same tokens. Maya-S and Maya-M are unchanged.

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
