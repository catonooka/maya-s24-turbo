<h1 align="center">Project Maya</h1>

<p align="center"><b>Run GLM-5.3-Flash - a 321-billion-parameter AI model - on your own GPU(s)</b><br>
One or two NVIDIA GPUs, AMD (experimental) · Linux, Windows (experimental) · chat in the browser, pictures, OpenAI- and
Anthropic-compatible API</p>

<p align="center"><a href="https://buymeacoffee.com/peasantsmith">☕ Support Project Maya - buy me a coffee</a></p>

Maya runs **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** (zai-org, MIT license): a mixture-of-
experts model with 321 B parameters, of which about 18 B work on each token, and a context of up to 1 M tokens.
Models this size normally need a server with hundreds of GB of GPU memory. Maya's engine keeps the most-used experts
on your GPU(s), the next ones in RAM and the rest on your NVMe SSD, and moves them as the conversation needs them.
Nothing leaves your machine.

Maya grew out of [Strata](https://github.com/Niko1221/Strata) (MIT): its engine started from Strata's and was rewritten
for GLM-5.3-Flash (the expert tiers across VRAM, RAM and SSD, the two-GPU split, MTP decoding), and its server and
dashboard started from Strata's and were reworked for Maya (a new dashboard, images on demand, the thinking budget).

**AMD (experimental):** Linux on RX 7900 XT / XTX and R9700 / RX 9070, one GPU, text only - see
[docs/AMD_MAYA.md](docs/AMD_MAYA.md).

## The models: Maya-S and Maya-M

Maya installs **Maya-S**, Project Maya's own compact quant of GLM-5.3-Flash (96.5 GB,
[on Hugging Face](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)), made for PCs with a smaller memory
pool across RAM and VRAM. It is made from Z.ai's FP8 release -
the precision the model is served at - with statistics from the FP8 model itself and error-feedback rounding of the
experts, and it keeps the model's MTP block, which drafts tokens ahead (speculative decoding on two GPUs).

**It keeps 97.9% of the full FP8 model's accuracy** on zero-shot tasks (ARC-Easy, ARC-Challenge, HellaSwag,
WinoGrande, PIQA; 400 questions each, the same for both models). On held-out text it picks the same next token as the
FP8 model 83% of the time, and it writes long answers (6,000-14,000 tokens) without looping.
Details: [bench/results/MAYA-S.md](bench/results/MAYA-S.md).

**Maya-M** (116 GB) is the larger quant, made for PCs with a bigger memory pool across RAM and VRAM: more bits where
they count - IQ2_S gate/up experts, IQ3_XXS down projections and IQ3_S in the most sensitive layers - with the same
FP8 statistics and error-feedback rounding, calibrated toward tool calls and front-end code. It is closer to the FP8
model than Maya-S token by token: 23% lower KL divergence, and it picks the same next token as the FP8 model 86%
of the time; on the zero-shot tasks both keep 97.9% of the FP8 model's accuracy. Set it
up with `./setup.sh --setup --model Maya-M` (Windows: `START-MAYA.bat --setup --model Maya-M`).
Details: [bench/results/MAYA-M.md](bench/results/MAYA-M.md).

## How fast is it?

Measured with Maya-S. A token is about ¾ of a word. `./maya.sh --bench` measures your machine the same way.

| Machine | Decode (writing the answer) | Prefill (reading your prompt) |
| --- | ---: | ---: |
| **2x Tesla V100 32 GB** (PCIe 3), Xeon E5-2690 v4, 30 GB RAM, one NVMe | **up to 40 tokens/s** | **up to 560 tokens/s** |
| **1x Tesla V100 32 GB** (PCIe 3), Core i5-12600T, 64 GB RAM, one NVMe | **up to 19 tokens/s** | **up to 370 tokens/s** |

- The speed holds with context: the attention's selection step is linear in the context length, so a 60K-token
  conversation keeps answering fast.
- The first answers after a start are the slowest: the expert caches fill with the experts your conversations use.
- Every machine is different: the engine adapts to the GPUs, RAM and SSD it finds, so your speed depends on your
  hardware. A second GPU in a narrow slot (PCIe x4) still helps: the engine measures each card's link and lets the
  CPU compute more of that card's RAM-tier experts instead of copying them over.

**Measured by users** with `./maya.sh --bench` (Maya-S, 32K context). Send yours: `--bench`, then `--report`, in a
[GitHub issue](https://github.com/mw00/project-maya/issues).

| Machine | Decode (writing the answer) | Prefill (reading your prompt) | By |
| --- | ---: | ---: | --- |
| **2x NVIDIA TITAN RTX 24 GB** (Turing; the second card in a PCIe 3 x4 slot), Core i5-12490F, 48 GB RAM | 13.5 tokens/s (mean of 3 answers) | 238 tokens/s (8K-token prompt) | @dummerjindabin (v1.0.6 with the v1.0.7 fix) |

## What you need

| | |
| --- | --- |
| **GPU** | NVIDIA, compute capability 7.0 or newer (V100 and newer); one GPU, or two that share the model (each holds half of the layers). The engine fills whatever VRAM you have with the most-used experts: more VRAM is faster. Measured: 1 and 2x V100 32 GB; by users: 2x TITAN RTX (above). **AMD (experimental):** RX 7900 XT / XTX and Radeon AI PRO R9700 / RX 9070, one GPU, text only ([docs/AMD_MAYA.md](docs/AMD_MAYA.md)). |
| **RAM** | It runs with **32 GB** (the machine in the table above has 30 GB). More RAM keeps more experts close and is faster; what does not fit is read from the SSD while it answers. |
| **Disk** | **~100 GB free on a fast NVMe SSD** (the model is 96.5 GB, its pictures encoder 1.1 GB, and the engine reads from the model while it answers). Not a hard disk. |
| **System** | Linux (x86-64, CPU with AVX2), NVIDIA driver, CUDA toolkit 12.x (CUDA 13 can be used for Turing and newer, but it no longer compiles for Volta/V100), g++, Python 3.10+. Windows 10/11: experimental, with Visual Studio 2022 Build Tools instead of g++ ([Windows](#windows)). Not WSL2. AMD: Linux with ROCm 7 instead of the NVIDIA driver and CUDA. |

The installer checks all of this and prints the exact command for anything missing. It installs nothing
system-wide by itself.

## Install

**You need:** an NVIDIA GPU (V100 / RTX 20 or newer) on Linux (Windows: [experimental](#windows)), ~100 GB free on
an NVMe SSD, a current NVIDIA driver
and the CUDA toolkit (12.x for a V100; the engine is compiled for your GPU). Everything else - Python, the engine, the
model - is set up for you, the way Strata does it. On an AMD RX 7900 XT / XTX or R9700 / RX 9070 (experimental, Linux,
ROCm 7): `./maya.sh --backend hip --gpu 0 --check` first, then [docs/AMD_MAYA.md](docs/AMD_MAYA.md).

1. Get Project Maya:
   ```sh
   git clone https://github.com/mw00/project-maya.git && cd project-maya
   ```
   (or [download it](https://github.com/mw00/project-maya/archive/refs/heads/main.zip) and unzip it).
2. Run **`./setup.sh`** (the same as `./maya.sh`).
3. Answer a few questions - or just press Enter each time for the recommended choice: which GPUs, how much context,
   pictures. Then it downloads and builds everything (it shows each download first; you can stop and it picks up
   where it left off) and **starts the model**. Open the dashboard at `http://127.0.0.1:8080`.

**Next time**, just run `./setup.sh` again: it starts right away, nothing is downloaded twice. Ctrl+C stops it.

**Updating:** `git pull`, then `./setup.sh`: it recompiles only what changed and starts.

### What the first run does

It takes 20-40 minutes plus the download:

1. checks the PC (GPUs, driver, CUDA toolkit, compiler, RAM, CPU);
2. asks which GPUs to use (both, when there are two) and how much context (32K recommended);
3. installs its Python packages into `.venv` and gets llama.cpp's source at a pinned commit (it lists both and asks);
4. compiles the engine for your GPU(s) (10-30 minutes, once);
5. **the model**: it shows the source, the size (96.5 GB) and the exact `curl` commands, and downloads only when you
   answer `y`; every file is checked against its published sha256. You can run the commands yourself instead, or use
   files you already have: `./maya.sh --gguf-dir DIR`;
6. builds the *pack* - the engine's index of the model files, about 1 GB, written into the model folder;
7. **pictures**: compiles the image encoder (10-20 minutes, once) and fetches its files (1.1 GB, shown and asked
   first); `--no-vision` skips it;
8. writes `maya-<model>.json` and `run-maya-<model>.sh`, and starts the dashboard.

> **The start takes a few minutes**: the engine pins most of the free RAM (all but about 6 GB) for its expert tier
> and warms its caches. Other programs get little RAM while Maya runs.

### Options

| Option | |
| --- | --- |
| `--setup` | set up again (other GPUs, context, model folder) |
| `--check` | only check the PC |
| `--gguf-dir DIR` | use GLM-5.3-Flash GGUF files you already have (the folder must be writable: the pack goes inside it) |
| `--data-dir DIR` | where a downloaded model goes (default `../Maya-data`); put it on the NVMe |
| `--download-model` | download the model without asking (the commands and size are still printed) |
| `--no-vision` | text only: no image encoder |
| `--gpu N` / `--gpus 0,1` | one GPU, or two that split the model |
| `--context N` | context length in tokens: 8192, 32768 (default), 65536, 131072 |
| `--port N`, `--host 0.0.0.0 --api-key KEY` | another port; reachable from your network (always set a key) |
| `--env KEY=VALUE` | an engine setting kept in the config (see [Tuning](#tuning)) |
| `--host-compiler g++-12` | when your g++ is newer than your CUDA accepts ("unsupported GNU version") |
| `--rebuild`, `--repack` | compile the engine / build the pack again |
| `--yes` | the recommended answers (the model download still needs `--download-model`) |

## Using it

- **In the browser:** `http://127.0.0.1:8080` - **Chat**, and a live **Monitor** of the model, the expert caches
  and your GPU/CPU/RAM.
- **Your apps and coding agents:** an "OpenAI-compatible" provider with the base URL `http://127.0.0.1:8080/v1`
  (any model name; any API key unless you set one). Anthropic's API: `http://127.0.0.1:8080/v1/messages`
  (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
- **Thinking:** off, low, medium (the default) or high, in the chat menu or the request's "reasoning effort". A
  reasoning block is capped at 32K tokens, then the answer follows (`"thinking_budget"` in the config; 0 = no cap).
- **Pictures:** attach one in the chat, or send `image_url` parts (OpenAI) / `image` blocks (Anthropic). The image
  encoder runs only while a new picture is read - a second or two to start, in GPU memory the model lends it - so
  the model keeps its whole GPU cache the rest of the time. On the first start Maya measures how much memory the
  encoder needs on your GPU and picks the largest picture size that fits it well (`vision-memory.json`).
- **From another device:** `./maya.sh --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** others wait their turn.

## Tuning

The engine sizes itself: it splits the layers across two GPUs when it has two, fills each GPU's free VRAM with
experts, sizes its RAM tier from the free RAM, and measures the CPU against the PCIe link at start to decide how many
RAM experts the CPU computes itself. These settings change that (put them in the config with `--env`, or into its
`"env"` block):

| Setting | Default | |
| --- | --- | --- |
| `STRATA_GLM_RAM_HEADROOM_GB` | 6 | RAM left free for the system when the RAM tier is sized |
| `STRATA_GLM_RAM_GB` | from free RAM | a fixed RAM-tier size in GB |
| `STRATA_GLM_SPLIT` | middle (+2) with 2 GPUs | the first layer of the second GPU; `0` = one GPU |
| `STRATA_GLM_CPU_LANE` | one thread per physical core | CPU threads for RAM-tier experts; `0` = off |
| `STRATA_GLM_USAGE` | `<pack>/expert_usage.txt` | where your expert usage is kept between starts (the warm-up loads your experts first); `0` = off |
| `STRATA_GLM_SLOTS` | 4 | conversations kept aside on the SSD, so switching back to one doesn't re-read its prompt; `0` = off |
| `STRATA_GLM_SLOT_MIN`, `STRATA_GLM_SLOT_GB`, `STRATA_GLM_SLOT_DIR` | 1024, 16, `<pack>/slots` | the shortest conversation kept aside (tokens), their total size on disk (GB), and the folder |
| `STRATA_GLM_TIMING`, `STRATA_GLM_POOL_STATS` | off | `1` = timing and cache statistics in the engine log |

## Something went wrong?

**Reporting a problem or your speed:** run **`./maya.sh --report`** (Windows: `START-MAYA.bat --report`) right after
a slow answer or the error, and attach the `maya-report.txt` it writes in the Maya folder. It holds your GPUs, CPU,
RAM and disks, your Maya setup and the engine's speed lines - where the time goes, token by token - so the engine
can be tuned for your machine. Nothing is sent anywhere; your home folder shows as `~` and no API key is included.
For a speed report, also run **`./maya.sh --bench`** with Maya stopped (Windows: `START-MAYA.bat --bench`): a
standard test of a few minutes - decode on three questions, prefill at 2k and 8k tokens - that writes
`maya-bench.txt`, which the report then includes.

- **"nvcc ... cannot build for these GPUs"** - Volta needs CUDA 12.x; Blackwell needs 12.8 or newer. Several toolkits
  can be installed side by side; the installer takes the newest that fits.
- **"unsupported GNU version"** while compiling - run `./maya.sh --setup --host-compiler g++-12` (install `g++-12`
  first).
- **Slow, and the SSD is busy all the time** - not enough free RAM for the experts: close programs, or add RAM. A
  hard disk instead of an NVMe SSD is very slow.
- **The download stopped** - run `./maya.sh` again (or the printed `curl` command): it continues where it stopped.
- **Port 8080 is in use** - Maya (or another server) is already running; `--port 8081` starts another one.
- The engine's log is `maya-<model>.log` in the Maya folder.

## Windows

Experimental: the same installer sets Maya up natively on Windows 10/11 (64-bit), and the engine and the image
encoder compile there with Visual Studio 2022 and CUDA 12.8. Maya is developed and measured on Linux and has not
been run on a Windows PC with an NVIDIA GPU yet, so tell us how it runs on yours.

1. Install once: the NVIDIA driver,
   [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive),
   [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop
   development with C++", and 64-bit Python 3.10+ (`winget install -e --id Python.Python.3.12 --scope user`).
2. `git clone https://github.com/mw00/project-maya.git` (or download the zip), then double-click
   **`START-MAYA.bat`**. It takes the same options as `./maya.sh` (`START-MAYA.bat --check`, `--setup`, ...).

- Set Windows' page file to "System managed" (System > About > Advanced system settings > Performance > Virtual
  memory). Windows charges every allocation on the graphics card to RAM + page file too, and Maya pins tens of GB
  of RAM for its experts.
- Put the model on an NVMe SSD: `START-MAYA.bat --setup --data-dir D:\Maya-data`.
- Not WSL2: Strata measured that WSL2's GPU driver pins only about 1 GB of RAM, and Maya pins tens of GB. Run
  `START-MAYA.bat` in Windows itself.

## Support

Project Maya is free and open source. If it is useful to you, you can support its development:
**[buymeacoffee.com/peasantsmith](https://buymeacoffee.com/peasantsmith)**. Thank you!

## Credits and license

- **Built on [Strata](https://github.com/Niko1221/Strata)** (MIT License, Copyright (c) 2026 Niko1221 and the Strata
  contributors) - the code Maya's engine, server and dashboard grew from - **and on
  [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT License, Copyright (c) 2023-2026 The ggml
  authors) - the quantization formats, the CPU dot products and the prompt path's MMQ kernels, built from a pinned
  commit (`third_party/ggml/LICENSE`).
- The model: [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by zai-org (MIT); Maya-S and the image
  encoder file are made from zai-org's released weights and keep its license.
- The dashboard's font: Outfit (SIL Open Font License 1.1, `serve/web/fonts/OFL.txt`).
- Maya is open source under the [MIT License](LICENSE); the notices of Strata and ggml stay with every copy.
