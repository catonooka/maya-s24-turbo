#!/usr/bin/env python3
"""Project Maya - set up and start GLM-5.3-Flash on your own GPU(s).
CUDA: Linux; Windows (experimental). HIP: experimental Linux gfx1100/gfx1201, text only.

    ./maya.sh                 the first run sets everything up and starts the dashboard; later runs just start it
    ./maya.sh --setup         set up again (other GPUs, another context length, another model folder)
    ./maya.sh --check         only check this PC
    ./maya.sh --backend hip --gpu 0 --check    check an RX 7900 XT / XTX with system ROCm 7

On Windows START-MAYA.bat takes the same options.  Both make the private Python environment (.venv, the way Strata's
setup.sh does) and run this file.  It reuses Strata's installer (setup.py) for the PC checks,
pip, llama.cpp's source and resumable downloads.

What the first run does (each step is skipped when it is already done):

  1. checks the PC: NVIDIA GPU(s) of compute capability 7.0+, driver, CUDA toolkit (nvcc), the C++ compiler (g++;
     on Windows Visual Studio 2022's Build Tools), CMake, RAM, CPU; HIP checks AMD gfx1100/gfx1201 and ROCm 7 instead
  2. asks: which GPUs (one, or two that split the layers), how much context
  3. Python packages into .venv, llama.cpp's source at the pinned commit (it lists them and asks first)
  4. compiles the engine (`build/strata`, or `build-hip/strata`) for your GPU(s): 10-30 minutes, once
  5. the model: GGUF files you already have (--gguf-dir), or a download it shows you first - the exact commands
     and the size - and starts only after you answer y (or pass --download-model)
  6. builds the pack (the engine's index of the GGUF files) inside the model folder
  7. images: compiles the vision encoder (`build-vision/bin/strata-vision`) and fetches the model's vision files
     (1.1 GB, shown and asked first like the model; --no-vision and HIP skip it)
  8. writes maya-<model>.json and run-maya-<model>.sh (.bat on Windows), and starts the dashboard on
     http://127.0.0.1:8080

Nothing is installed system-wide: a missing tool is reported with the command that installs it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
import setup as S  # noqa: E402  Strata's installer: PC checks, pip, llama.cpp, downloads (nothing runs on import)
from setup import ask, fail, ok, run, say, step, warn  # noqa: E402

WIN = S.WIN
ME = "START-MAYA.bat" if WIN else "./maya.sh"      # how this is started, for the messages
ROOT = S.ROOT
BUILD = ROOT / "build"
EXE = BUILD / ("strata.exe" if WIN else "strata")
STAMP = BUILD / "MAYA-BUILD.json"                  # what the engine in build/ was compiled from and for
VBUILD = ROOT / "build-vision"
VEXE = VBUILD / "bin" / ("strata-vision.exe" if WIN else "strata-vision")  # the image encoder (llama.cpp's mtmd)
VSTAMP = VBUILD / "MAYA-BUILD.json"
MIN_CC = 70                                        # Volta (V100) and newer (the GLM path; see arch_setting)
PY_PACKAGES = list(S.PY_PACKAGES)                  # (pillow: pictures in formats other than JPEG/PNG/BMP/GIF)
CONTEXTS = [8192, 32768, 65536, 131072]
DEFAULT_CONTEXT = 32768
MODEL_NAME = "glm-5.3-flash"
SAMPLING = {"temperature": 1.0, "top_p": 0.95}     # the dashboard's and the API's defaults for requests that set none
EFFORT = "medium"                                  # thinking level for requests that name none
HF = "https://huggingface.co/{repo}/resolve/{revision}/{folder}/{file}"
# The models the installer can download: Project Maya's own quants, made from Z.ai's FP8 release (their model card
# has the measurements against it).  Another glm5-next GGUF can be used with --gguf-dir (experimental).  "sha256" per
# file name: every download is verified.  "vision": the image encoder's files (the mmproj, made from the official
# vision tower, and the tokenizer it reads its markers with).
MODELS = {
    "Maya-S-v2-IQ2_XXS": {
        "about": "Maya-S, Project Maya's compact quant: error-feedback-rounded IQ2_XXS gate/up experts, "
                 "IQ2_S/IQ3_XXS down projections, Q6_K attention, the MTP draft block, made from Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-S-v2-IQ2_XXS",
        "file": "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-{i:05d}-of-{n:05d}.gguf", "shards": 3, "download_gb": 96.5,
        "sha256": {
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf":
                "a507f2b7b25e04624ee55c631d3b281caea7cfffc747e5247970cd5a7ea91b3f",
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00002-of-00003.gguf":
                "2d65d88a69f8dc124c8d24bd33b33161ddab218918b4253ad4f500aeede84df5",
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00003-of-00003.gguf":
                "a6a981c4fee7a53d78bdbd97d8f465d48ddf439cf938ff90617f395e0351b5a6"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-M": {
        "about": "Maya-M, Project Maya's larger quant: error-feedback-rounded IQ2_S gate/up experts, IQ3_XXS down "
                 "projections (IQ3_S in the most sensitive layers), Q6_K attention, the MTP draft block, made from "
                 "Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-M",
        "file": "GLM-5.3-Flash-Maya-M-{i:05d}-of-{n:05d}.gguf", "shards": 3, "download_gb": 116.0,
        "sha256": {
            "GLM-5.3-Flash-Maya-M-00001-of-00003.gguf":
                "3ac0f066ec45af3432d59b33de49bdfb29156432b627240b35769c7d02cc6c02",
            "GLM-5.3-Flash-Maya-M-00002-of-00003.gguf":
                "285951d2afa0cd98285b03d0dc4aa68d83daf1a6f4a2594fe40b0cdd27ade485",
            "GLM-5.3-Flash-Maya-M-00003-of-00003.gguf":
                "ebf1ce713f71207747e10eeebed87597969d8b5e1d9dd817420d2c2f7ca51e0e"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
}
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
PACK_FILES = ("index.txt", "native_experts.txt", "dense.bin", "tokenizer/vocab.json", "tokenizer/merges.txt",
              "tokenizer/token_type.json", "tokenizer/chat_template.jinja")


# ------------------------------------------------------------------------------------------------ small helpers
def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def configs() -> list:
    """The installed models' configs, the most recently used first."""
    return sorted(ROOT.glob("maya-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def mem_gb() -> tuple:
    """(total, available) RAM in GiB - the engine sizes its RAM tier from MemAvailable (on Windows ullAvailPhys)."""
    if WIN:
        m = S._memory_status()
        return m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024 / 2**30
    except (OSError, ValueError):
        pass
    return info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0)


def existing(path: Path) -> Path:
    """`path`, or its nearest folder that exists (where it would be created)."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def rotational(path: Path) -> bool:
    """True when `path` is on a spinning disk (best effort: findmnt + lsblk; on Windows the drive's MediaType)."""
    if WIN:
        drive = existing(path).drive[:1]
        return drive.isalpha() and S.out(
            ["powershell", "-NoProfile", "-Command", "Get-PhysicalDisk | Where-Object DeviceId -eq "
             f"(Get-Partition -DriveLetter {drive}).DiskNumber | Select-Object -ExpandProperty MediaType"]).strip() == "HDD"
    src = S.out(["findmnt", "-no", "SOURCE", "--target", str(existing(path))]).strip().split("[")[0]
    rota = S.out(["lsblk", "-ndo", "ROTA", src]).split() if src.startswith("/dev/") else []
    return bool(rota) and rota[0] == "1"


def tool_version(exe: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)", S.out([exe, "--version"]))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def venv_tool(name: str):
    p = Path(sys.executable).parent / (name + (".exe" if WIN else ""))
    return str(p) if p.exists() else None


def pick_cmake():
    """CMake 3.24 or newer: the one pip put into .venv first, then the system's."""
    for c in (venv_tool("cmake"), shutil.which("cmake")):
        if c and tool_version(c) >= (3, 24):
            return c
    return None


def gpu_label(g) -> str:
    if g.get("vendor") == "amd":
        return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB, {g['arch']})"
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB, compute capability {S.cc(g)})"


def select_build_backend(backend: str) -> None:
    """Separate build folders and stamps prevent reusing a binary from the other backend."""
    global BUILD, EXE, STAMP
    BUILD = ROOT / ("build-hip" if backend == "hip" else "build")
    EXE = BUILD / ("strata.exe" if WIN else "strata")
    STAMP = BUILD / "MAYA-BUILD.json"


def check_hip_pc(a) -> dict:
    if WIN or not sys.platform.startswith("linux") or S.is_wsl():
        fail("Maya's experimental HIP backend requires native Linux")
    found = S.amd_gpus()
    usable = [g for g in found if g["arch"] in S.AMD_ARCHS]
    for g in found:
        say(f"    {gpu_label(g)} - " + ("can be used" if g in usable else "not supported by Maya's HIP build"))
    if not usable:
        fail("no supported AMD GPU found", "this port targets RX 7900 XT / XTX (gfx1100) and RX 9070 / AI PRO R9700 (gfx1201)")
    if a.gpus:
        fail("Maya's HIP port currently uses one GPU", "select it with --gpu N")
    chosen = next((g for g in usable if g["index"] == a.gpu), None) if a.gpu is not None else max(
        usable, key=lambda g: g["vram_gb"])
    if chosen is None:
        fail(f"GPU {a.gpu} is not a supported AMD card")
    root = Path(os.environ.get("ROCM_PATH") or "/opt/rocm").resolve()
    if not (root / "llvm/bin/clang++").exists() or not list((root / "lib").glob("libhipblas.so*")):
        fail("ROCm's HIP compiler and hipBLAS are required", "install ROCm 7, or set ROCM_PATH to its root")
    if tool_version(str(root / "bin/hipcc")) < (7, 0):
        fail("Maya's HIP port requires ROCm 7 or newer")
    if not shutil.which("c++"):
        fail("a C++ compiler is needed", "Ubuntu/Debian: sudo apt install build-essential")
    cpu, avx2, avx512 = S.cpu_info()
    if not avx2:
        fail(f"the CPU ({cpu}) needs AVX2 for the expert lane")
    total, avail = mem_gb()
    ok(f"using {gpu_label(chosen)}; experimental HIP, one GPU, text only")
    ok(f"ROCm: {root}; CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2'})")
    ok(f"RAM: {total:.0f} GB, {avail:.0f} GB available now")
    select_build_backend("hip")
    return {"backend": "hip", "gpus": [chosen], "archs": ["gfx1100", "gfx1201", "gfx1151"], "rocm": str(root)}


def nvcc_range(archs) -> tuple:
    """The CUDA toolkit versions that can build for these GPUs: (lowest, first too new or None)."""
    lo = (12, 8) if max(archs) >= 100 else (12, 0)  # Blackwell needs 12.8
    hi = (13, 0) if min(archs) < 75 else None       # CUDA 13 no longer compiles for Volta (sm_70)
    return lo, hi


def find_nvcc(archs, given=None) -> tuple:
    """(nvcc, version) for these GPUs: --nvcc, else the newest toolkit that can build for them (Strata's find_nvcc
    takes the newest of all, but CUDA 13 cannot build for Volta), else the newest found (reported as unfit), else
    (None, None)."""
    def version(c):
        v = re.search(r"release (\d+)\.(\d+)", S.out([c, "--version"]))
        return (int(v.group(1)), int(v.group(2))) if v else None
    if given:
        return given, version(given)
    return pick_nvcc([(c, version(c)) for c in dict.fromkeys(c for c in nvcc_candidates() if c and Path(c).exists())],
                     archs)


def nvcc_candidates() -> list:
    """Every place a CUDA toolkit's nvcc may be (some do not exist): PATH, CUDA_PATH, the default install folders."""
    exe = "nvcc.exe" if WIN else "nvcc"
    cands = [shutil.which("nvcc"), os.environ.get("CUDA_PATH") and str(Path(os.environ["CUDA_PATH"]) / "bin" / exe)]
    if WIN:
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        return cands + [str(p / "bin" / exe) for p in sorted(base.glob("v*"))]
    cands += [str(p / "bin" / exe) for p in sorted(Path("/usr/local").glob("cuda*"))]
    return cands + [str(p / "bin" / exe) for p in sorted(Path("/opt").glob("cuda*"))] + ["/usr/bin/nvcc"]


def pick_nvcc(found, archs) -> tuple:
    """Of [(nvcc, version)], the newest that can build for `archs`, else the newest, else (None, None)."""
    found = sorted([f for f in found if f[1]], key=lambda f: f[1], reverse=True)
    lo, hi = nvcc_range(archs)
    fit = [f for f in found if f[1] >= lo and (hi is None or f[1] < hi)]
    return (fit or found or [(None, None)])[0]


def toolkit_hint(archs) -> str:
    if WIN:
        return ("NVIDIA's CUDA Toolkit 12.8 for Windows (the driver can stay as it is): "
                "https://developer.nvidia.com/cuda-12-8-0-download-archive" +
                (" - Volta (V100) needs 12.x, CUDA 13 dropped it" if min(archs) < 75 else ""))
    if min(archs) < 75:
        return ("Volta (V100) needs CUDA 12.x - Ubuntu 24.04: sudo apt-get install -y nvidia-cuda-toolkit (CUDA 12.0), "
                "or NVIDIA's cuda-toolkit-12-8 package: https://developer.nvidia.com/cuda-12-8-0-download-archive")
    if max(archs) >= 100:
        return "NVIDIA's cuda-toolkit-12-8 (or newer) package: https://developer.nvidia.com/cuda-downloads"
    return ("Ubuntu 24.04: sudo apt-get install -y nvidia-cuda-toolkit (CUDA 12.0), or NVIDIA's packages: "
            "https://developer.nvidia.com/cuda-downloads")


def cuda_lib_dirs(nvcc: str) -> list:
    """The toolkit's library folders for the engine's LD_LIBRARY_PATH (none for a distribution's /usr/bin/nvcc); on
    Windows its DLL folders (bin, and bin\\x64 in CUDA 13) for the engine's PATH."""
    if WIN:
        b = Path(nvcc).parent
        return [str(p) for p in (b, b / "x64") if p.is_dir()]
    base = Path(nvcc).resolve().parent.parent
    if base == Path("/usr"):
        return []
    return [str(p) for p in (base / "lib64", base / "targets" / "x86_64-linux" / "lib") if p.is_dir()]


# ------------------------------------------------------------------------------------------------ 1. the PC
def choose_gpus(a, found) -> list:
    """The GPUs the model runs on: --gpu / --gpus, else asked when two can share it (both recommended)."""
    byid = {g["index"]: g for g in found}
    usable = [g for g in found if int(g["arch"]) >= MIN_CC]
    if a.gpus or a.gpu is not None:
        try:
            want = [int(x) for x in str(a.gpus).split(",") if x.strip()] if a.gpus else [a.gpu]
        except ValueError:
            fail(f"--gpus takes GPU numbers as nvidia-smi numbers them, e.g. --gpus 0,1, not {a.gpus!r}")
        if not 1 <= len(want) <= 2 or len(set(want)) != len(want):
            fail("Maya runs on one GPU or splits the model across two: --gpu 0, or --gpus 0,1")
        for i in want:
            if i not in byid or int(byid[i]["arch"]) < MIN_CC:
                fail(f"GPU {i} cannot be used" + ("" if i in byid else " (not found)"),
                     "use one of: " + ", ".join(str(g["index"]) for g in usable))
        return [byid[i] for i in want]
    if not usable:
        fail("none of the GPUs can run Maya", "it needs an NVIDIA GPU of compute capability 7.0 or newer (V100 and newer)")
    best = sorted(usable, key=lambda g: (-round(g["vram_gb"]), g["index"]))
    if len(best) == 1:
        return best
    pair = best[:2]
    say()
    say("  Maya can run on one GPU, or split the model's layers across two: each then caches the experts of its own")
    say("  layers, so together they hold about twice as many (V100s: ~50 tok/s benchmark on two, ~24 on one).")
    say(f"  1) {gpu_label(pair[0])} + {gpu_label(pair[1])} together   (recommended)")
    say(f"  2) {gpu_label(best[0])} only")
    if len(best) > 2:
        say("     (the engine splits across two GPUs at most: --gpus picks another pair)")
    return pair if ask("Which GPUs?", ["1", "2"], "1", a.yes or a.check) == "1" else best[:1]


def check_pc(a) -> dict:
    step(1, "checking this PC")
    if a.backend == "hip":
        return check_hip_pc(a)
    if not (WIN or sys.platform.startswith("linux")):
        fail("Project Maya runs on Linux and Windows", "the engine needs an NVIDIA GPU and CUDA")
    if WIN:
        warn("Windows support is new (experimental): Maya is developed and measured on Linux - tell us how it runs")
    if S.is_wsl():
        warn("this is WSL2, which Maya does not support: Strata measured that WSL2's driver pins only about 1 GB of "
             "RAM for the GPU, and Maya's RAM tier pins tens of GB. Use a native Linux install.")
    found = S.gpus()
    if not found:
        fail("no NVIDIA GPU found (nvidia-smi did not answer)",
             ("install the NVIDIA driver (https://www.nvidia.com/drivers)" if WIN else
              "install the NVIDIA driver (Ubuntu: sudo ubuntu-drivers install)") + f", restart, and run {ME} again")
    say("  NVIDIA GPUs:")
    for g in found:
        say(f"    {gpu_label(g)} - " + ("can be used" if int(g["arch"]) >= MIN_CC else
                                       "too old (Maya needs compute capability 7.0 or newer)"))
    chosen = choose_gpus(a, found)
    archs = sorted({int(g["arch"]) for g in chosen})
    vram = sum(g["vram_gb"] for g in chosen)
    ok("using " + " + ".join(gpu_label(g) for g in chosen) +
       (": the model's layers are split across both" if len(chosen) > 1 else ""))
    if vram < 23:
        say(f"  {vram:.0f} GB of VRAM in total: the engine fills it with the most-used experts and serves the rest from "
            "RAM and the SSD - give it a try (more VRAM is faster; tell us your speed)")
    problems = []                                      # (what is wrong, how to fix it): all of them at once

    nvcc, nv = find_nvcc(archs, a.nvcc)
    lo, hi = nvcc_range(archs)
    want = f"CUDA {lo[0]}.{lo[1]} or newer" + (f" but older than {hi[0]}.0 (CUDA 13 dropped Volta)" if hi else "")
    if nvcc is None:
        problems.append((f"the CUDA toolkit (nvcc) is not installed; these GPUs need {want}", toolkit_hint(archs)))
    elif nv is None:
        problems.append((f"{nvcc} does not say its version (nvcc --version)", "pass the toolkit's nvcc with --nvcc"))
    elif nv < lo or (hi is not None and nv >= hi):
        problems.append((f"nvcc {nv[0]}.{nv[1]} ({nvcc}) cannot build for these GPUs; they need {want}",
                         toolkit_hint(archs) + " (toolkits can be installed side by side: the newest that fits is "
                                                "used, or pass one with --nvcc)"))
    else:
        ok(f"CUDA toolkit {nv[0]}.{nv[1]}: {nvcc}")

    drv = min(S.driver_major(g) for g in chosen)
    need_drv = 580 if nv and nv >= (13, 0) else 525
    if drv < need_drv:
        problems.append((f"the NVIDIA driver {chosen[0]['driver']} is too old: {need_drv} or newer is needed",
                         ("" if WIN else "Ubuntu: sudo ubuntu-drivers install, then restart; or ") +
                         "https://www.nvidia.com/drivers"))
    else:
        ok(f"NVIDIA driver {chosen[0]['driver']}")

    if WIN:
        vcvars = S.find_vcvars()
        if vcvars is None:
            problems.append(("the C++ compiler (Visual Studio 2022 Build Tools, 'Desktop development with C++') is "
                             "not installed", "https://visualstudio.microsoft.com/visual-cpp-build-tools/ - in the "
                             "installer tick 'Desktop development with C++' (CUDA 12 needs 2019 or 2022, not 2026)"))
        else:
            ok(f"C++ compiler: Visual Studio Build Tools ({vcvars})")
    elif a.host_compiler:
        if shutil.which(a.host_compiler) is None:
            problems.append((f"--host-compiler {a.host_compiler} is not installed",
                             f"Ubuntu/Debian: sudo apt-get install -y {Path(a.host_compiler).name}"))
        else:
            ok(f"C++ compiler for nvcc: {a.host_compiler}")
    elif shutil.which("g++") is None:
        problems.append(("the C++ compiler (g++) is not installed", "Ubuntu/Debian: sudo apt-get install -y build-essential"))
    else:
        ok("C++ compiler: " + (S.out(["g++", "--version"]).splitlines() or ["g++"])[0])

    sc = pick_cmake() or shutil.which("cmake")
    scv = tool_version(sc) if sc else (0, 0)
    if scv >= (3, 24):
        ok(f"CMake {scv[0]}.{scv[1]}")
    else:
        ok((f"CMake {scv[0]}.{scv[1]} is older than 3.24" if sc else "no CMake on PATH") +
           ": a current one goes into .venv in step 3 (pip)")
    ok(f"Python {sys.version.split()[0]} ({sys.prefix})")

    total, avail = mem_gb()
    msg = f"RAM: {total:.0f} GB, {avail:.0f} GB available now"
    ok(msg + ("" if total >= 60 else " - it runs with 32 GB; more RAM keeps more experts close and is faster"))
    pf = S.page_file_gb()
    if pf is not None and pf < vram:
        warn(f"Windows' page file is {pf:.0f} GB: under Windows every allocation on the graphics card is also charged "
             f"to RAM + page file, so the engine's RAM tier shrinks by up to the {vram:.0f} GB of VRAM it fills. Set "
             "it to \"System managed\" (or larger than your VRAM): System > About > Advanced system settings > "
             "Performance > Advanced > Virtual memory")
    cpu, avx2, avx512 = S.cpu_info()
    if not avx2:
        problems.append((f"the CPU ({cpu}) has no AVX2", "the engine's CPU expert lane and ggml need AVX2"))
    else:
        ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2'}, {os.cpu_count()} threads)")

    if problems:
        say()
        for what, how in problems:
            say(f"  [X]  {what}")
            say(f"       {how}")
        fail("something Maya needs is missing (above)",
             f"install it and run {ME} again - this script installs nothing system-wide")
    select_build_backend("cuda")
    return {"backend": "cuda", "gpus": chosen, "archs": archs, "nvcc": nvcc,
            "vcvars": str(S.find_vcvars()) if WIN else None}


# ------------------------------------------------------------------------------------------------ 2. choices
def choose_context(a, prev_ctx) -> int:
    if a.context:
        return a.context
    default = prev_ctx if prev_ctx in CONTEXTS else DEFAULT_CONTEXT
    say()
    say("  Context length = how much text the model sees at once (the chat, files, tool output). A longer one takes")
    say("  VRAM from the expert cache, so answers get a little slower:")
    for i, c in enumerate(CONTEXTS, 1):
        say(f"  {i}) {c // 1024}K tokens" + ("   (recommended)" if c == DEFAULT_CONTEXT else ""))
    pick = ask("Context?", [str(i) for i in range(1, len(CONTEXTS) + 1)], str(CONTEXTS.index(default) + 1), a.yes)
    return CONTEXTS[int(pick) - 1]


# ------------------------------------------------------------------------------------------------ 3. tools
def tools_step(a) -> Path:
    """The Python packages (into .venv) and llama.cpp's source at the pinned commit - listed, and asked, first."""
    step(3, "Python packages and llama.cpp's source")
    stamp = Path(sys.prefix) / ".strata-pip.json"          # S.pip_install's record of what it installed
    have = read_json(stamp) if stamp.exists() else []
    need = [p for p in PY_PACKAGES if p not in (have if isinstance(have, list) else [])]
    llama = Path(a.llama_dir).expanduser().resolve() if a.llama_dir else ROOT / "third_party" / "llama.cpp"
    llama_ok = (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir()
    if a.llama_dir and not llama_ok:
        fail(f"{llama} is not a llama.cpp checkout (its ggml/ and gguf-py/ folders are missing)")
    downloads = []
    if need:
        downloads.append("Python packages from PyPI into .venv: " + ", ".join(need) + " (roughly 50-100 MB)")
    if not llama_ok:
        downloads.append(f"llama.cpp's source at the pinned commit {S.LLAMA_CPP_COMMIT[:10]} from GitHub, "
                         f"{S.LLAMA_CPP_ZIP} (tens of MB): ggml for the engine build, gguf-py for the pack builder")
    if downloads:
        say("  This step downloads:")
        for d in downloads:
            say("    - " + d)
        if ask("  Download them now?", ["y", "n"], "y", a.yes) != "y":
            fail("nothing was downloaded", f"run {ME} again when you are ready (or pass --llama-dir with a "
                                           f"llama.cpp checkout at commit {S.LLAMA_CPP_COMMIT[:10]})")
    S.pip_install(PY_PACKAGES, ", ".join(PY_PACKAGES))
    if not llama_ok:
        llama = S.get_llama_cpp()
    ok(f"llama.cpp {S.LLAMA_CPP_COMMIT[:10]}: {llama}")
    return llama


# ------------------------------------------------------------------------------------------------ 4. the engine
def cached_generator():
    """The CMake generator build/ was configured with before (another one cannot be used there), or None."""
    m = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", (BUILD / "CMakeCache.txt").read_text(errors="replace"), re.M) \
        if (BUILD / "CMakeCache.txt").exists() else None
    return m.group(1).strip() if m else None


def arch_setting(archs) -> str:
    """CMAKE_CUDA_ARCHITECTURES for these GPUs.  CMakeLists.txt refuses an explicit arch below 75 (Strata's own
    qwen4exp floor), but the GLM path is developed and measured on Volta V100s (sm_70) - so for those the build
    asks CMake for this machine's own GPUs ("native", which the guard leaves to CMake), with CUDA_VISIBLE_DEVICES
    limited to the chosen cards.  RELEASE-CHECKLIST.md has the one-line CMake fix that makes this unnecessary."""
    return ";".join(str(x) for x in archs) if min(archs) >= 75 else "native"


def compile_engine(archs, gpu_ids, nvcc, host_compiler, llama: Path, src: str, soft=False) -> dict | None:
    """cmake configure + build of the `strata` target in build/; the exact commands are printed as they run.
    soft: a failure warns and returns None (the engine already there keeps working) instead of stopping."""
    def stop(what):
        msg = f"the engine build stopped while {what} (the reason is above)"
        hint = ("common causes: 'unsupported Microsoft Visual Studio version' - CUDA 12 needs Visual Studio 2019 or "
                "2022 (its Build Tools are enough);" if WIN else
                "common causes: 'unsupported GNU version' - install an older g++ your CUDA accepts (e.g. g++-13) and run "
                "again (it is found by itself), or pass --host-compiler g++-13;") + (
                "\n       'Unsupported gpu architecture compute_70' - CUDA 13 cannot build for Volta, "
                "install CUDA 12.x;\n       the compiler killed (out of memory) - close programs and run it again "
                "(it continues)")
        if soft:
            warn(msg + "; starting the engine compiled before")
            return None
        fail(msg, hint)

    cmake = pick_cmake()
    if cmake is None:
        return stop(f"looking for CMake 3.24+ (pip installs one into .venv: run {ME} --setup)")
    gen = cached_generator()
    ninja = venv_tool("ninja") or shutil.which("ninja")
    conf = [cmake]
    if ninja and gen in (None, "Ninja"):
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    cuda_archs = arch_setting(archs)
    conf += ["-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
             "-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
             "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF",
             f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={llama}"]
    if host_compiler:
        conf.append(f"-DCMAKE_CUDA_HOST_COMPILER={shutil.which(host_compiler) or host_compiler}")
    # "native" is what CMake sees: only the chosen cards, numbered as nvidia-smi numbers them
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(str(i) for i in gpu_ids))
    # nvcc takes 2-4 GB per job on the big kernels: half the threads, and no more jobs than RAM / 4 GB
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(mem_gb()[0] // 4) or 2))
    build = [cmake, "--build", str(BUILD), "--target", "strata", "-j", str(jobs)]
    say("  Compiling the engine for " + ", ".join(f"sm_{x}" for x in archs) +
        " (10-30 minutes the first time, a few minutes after an update) ...")
    if cuda_archs == "native":
        say("  (CMakeLists.txt refuses an explicit sm_70 - Strata's own floor is sm_75 - so CMake is asked for this")
        say(f"  machine's GPUs instead: CMAKE_CUDA_ARCHITECTURES=native with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
    stopped = cmake_steps(conf, build, env, "build-maya.bat")
    if stopped:
        return stop(stopped)
    meta = {"src": src, "archs": archs, "gpu_ids": list(gpu_ids), "nvcc": nvcc, "host_compiler": host_compiler,
            "llama": str(llama), "lib_dirs": cuda_lib_dirs(nvcc), "date": time.strftime("%Y-%m-%d %H:%M")}
    STAMP.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok(f"engine compiled: {EXE}")
    return meta


def cmdline(cmd) -> str:
    """One command as a line of a Windows .bat."""
    return " ".join(f'"{x}"' if re.search(r'[\s;&|<>^()]', str(x)) else str(x) for x in cmd)


def shell_join(cmd) -> str:
    """One command as it is typed here: a POSIX shell, or Windows' cmd."""
    return cmdline(cmd) if WIN else shlex.join(cmd)


def cmake_steps(conf, build, env, bat_name: str) -> str | None:
    """CMake's configure, then its build (once more when that stops: it continues where it stopped).  None when both
    worked, else what stopped.  On Windows both run in a .bat that first calls Visual Studio's vcvars64.bat - the
    compiler's environment - the way Strata's setup.py builds there (cmake_build)."""
    if not WIN:
        if run(conf, env=env, check=False).returncode != 0:
            return "configuring"
        if run(build, env=env, check=False).returncode != 0:
            say("  (the build stopped - trying it once more: it continues where it stopped)")
            if run(build, env=env, check=False).returncode != 0:
                return "compiling"
        return None
    vcvars = S.find_vcvars()
    if vcvars is None:
        return "looking for Visual Studio's C++ compiler (the Build Tools with 'Desktop development with C++')"
    bat = ROOT / bat_name
    bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{cmdline(conf)} || exit /b 3\r\n{cmdline(build)} && exit /b 0'
                   '\r\necho   (the build stopped - trying it once more: it continues where it stopped)\r\n'
                   f'{cmdline(build)} || exit /b 4\r\n', encoding="utf-8")
    say(f"  > {bat}   (Visual Studio's environment, then:)")
    say("  > " + cmdline(conf))
    say("  > " + cmdline(build))
    rc = subprocess.call(["cmd", "/c", str(bat)], env=env)
    return None if rc == 0 else "compiling" if rc == 4 else "configuring"


PROBE_CU ="""#include <cmath>
#include <string>
#include <vector>
__global__ void k(float* x) { x[0] = rsqrtf(x[0]) + sinf(x[1]) + expf(x[2]) + __expf(x[3]) + sqrtf(x[4]); }
int main() { std::vector<std::string> v(1); return (int) std::cos(0.0) - 1 + (int) v.size() - 1; }
"""


def toolchain_for(archs, nvcc_given: str | None, hc_given: str | None) -> tuple:
    """(nvcc, host compiler or None for the default) that compile the engine's kind of CUDA here.  Every installed
    toolkit that can build for these GPUs (newest first; --nvcc alone if given) with the default g++, then each
    installed g++-N (newest first; --host-compiler alone if given), on a small file the way the engine compiles:
    C++20 and the math functions the kernels use.  A toolkit refuses a g++ newer than it supports, and a newer
    glibc's own rsqrtf/sinpi declarations clash with older CUDA headers - nothing but trying tells which pair works
    on a given system.  On Windows the compiler is Visual Studio's (in its vcvars64.bat environment): only the
    toolkits are tried."""
    import tempfile

    def version(c):
        v = re.search(r"release (\d+)\.(\d+)", S.out([c, "--version"]))
        return (int(v.group(1)), int(v.group(2))) if v else None
    if nvcc_given:
        nvccs = [nvcc_given]
    else:
        seen, found = set(), []
        for c in nvcc_candidates():
            if c and Path(c).exists() and os.path.realpath(c) not in seen:
                seen.add(os.path.realpath(c))
                found.append((c, version(c)))
        lo, hi = nvcc_range(archs)
        nvccs = [c for c, v in sorted((f for f in found if f[1]), key=lambda f: f[1], reverse=True)
                 if v >= lo and (hi is None or v < hi)]
    hcs = [None] if WIN else [hc_given] if hc_given else [None] + sorted(
        {q.name for d in ("/usr/bin", "/usr/local/bin") for q in Path(d).glob("g++-[0-9]*")
         if re.fullmatch(r"g\+\+-\d+", q.name)}, key=lambda n: -int(n[4:]))
    vcvars = S.find_vcvars() if WIN else None
    say("  Finding a CUDA toolkit and C++ compiler that build the engine here ...")
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "probe.cu"
        src.write_text(PROBE_CU, encoding="utf-8")
        for nv in nvccs:
            for c in hcs:
                cmd = [nv, "-std=c++20", "-c", str(src), "-o", str(Path(t) / "probe.o")] + \
                      (["-ccbin", shutil.which(c) or c] if c else [])
                if vcvars:                             # cl.exe is on PATH only inside Visual Studio's environment
                    bat = Path(t) / "probe.bat"
                    bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{cmdline(cmd)}\r\n', encoding="utf-8")
                    cmd = ["cmd", "/c", str(bat)]
                if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
                    ok(f"{nv} with " + ("Visual Studio's C++ compiler" if WIN else c or "the default g++"))
                    return nv, c
    warn("no installed CUDA toolkit and C++ compiler pair compiled a test file: the build will show why (" +
         toolkit_hint(archs) + ")")
    return (nvccs[0] if nvccs else nvcc_given), hc_given


def compile_engine_hip(pc: dict, llama: Path, src: str, soft=False) -> dict | None:
    root = Path(pc["rocm"])
    cmake = pick_cmake()
    if not cmake:
        fail("CMake 3.24 or newer is needed")
    env = dict(os.environ, HIP_PLATFORM="amd", HIP_COMPILER="clang", HIP_RUNTIME="rocclr",
               ROCM_PATH=str(root), HIP_PATH=str(root))
    env["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "llvm/bin"), env.get("PATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(root / "lib"), env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    # Keep compiler-cache writes inside this project, including when run in a workspace sandbox.
    env.setdefault("CCACHE_DIR", str(BUILD / ".ccache"))
    conf = [cmake, "-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
            "-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_PREFILL_MMQ=ON",
            "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF",
            f"-DCMAKE_HIP_ARCHITECTURES={';'.join(pc['archs'])}",
            f"-DCMAKE_HIP_COMPILER={root / 'llvm/bin/clang++'}", f"-DCMAKE_PREFIX_PATH={root}",
            f"-DSTRATA_GGML_DIR={llama}"]
    jobs = max(1, min(4, (os.cpu_count() or 4) // 2))
    say("  Compiling Maya for AMD " + ", ".join(pc["archs"]) + " ...")
    if cmake_steps(conf, [cmake, "--build", str(BUILD), "--target", "strata", "-j", str(jobs)], env, ""):
        if soft:
            warn("HIP rebuild failed; starting the previous engine")
            return None
        fail("the HIP engine build failed", "see the compiler output above")
    meta = {"backend": "hip", "src": src, "archs": pc["archs"], "rocm": str(root),
            "gpu_ids": [g["index"] for g in pc["gpus"]], "llama": str(llama),
            "lib_dirs": [str(root / "lib")], "date": time.strftime("%Y-%m-%d %H:%M")}
    STAMP.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok(f"engine compiled: {EXE}")
    return meta


def build_step(a, pc, llama: Path) -> dict:
    step(4, "the engine")
    src = S.source_hash(S.ENGINE_SOURCES)              # src/, include/, third_party/ggml, CMakeLists.txt
    meta = read_json(STAMP)
    if pc.get("backend") == "hip":
        if (EXE.exists() and not a.rebuild and meta.get("backend") == "hip" and meta.get("src") == src
                and meta.get("archs") == pc["archs"] and meta.get("rocm") == pc["rocm"]):
            ok(f"HIP engine already compiled: {EXE}")
            return meta
        return compile_engine_hip(pc, llama, src)
    if (EXE.exists() and not a.rebuild and meta.get("src") == src and set(pc["archs"]) <= set(meta.get("archs", []))
            and (not a.nvcc or a.nvcc == meta.get("nvcc"))
            and (not a.host_compiler or a.host_compiler == meta.get("host_compiler"))):
        ok(f"engine already compiled for " + ", ".join(f"sm_{x}" for x in meta["archs"]) + f": {EXE}")
        return meta
    nvcc, hc = toolchain_for(pc["archs"], a.nvcc, a.host_compiler)
    pc["nvcc"] = nvcc or pc["nvcc"]
    return compile_engine(pc["archs"], [g["index"] for g in pc["gpus"]], pc["nvcc"], hc, llama, src)


def refresh_engine(cfg: dict) -> None:
    """At a start: the engine's source changed since it was compiled (a git pull) - compile what changed."""
    meta = read_json(STAMP)
    if not meta or Path(cfg["exe"]).resolve() != EXE.resolve():
        return
    src = S.source_hash(S.ENGINE_SOURCES)
    if meta.get("src") == src:
        return
    if cfg.get("backend") == "hip":
        root = Path(meta.get("rocm") or "/opt/rocm")
        llama = Path(meta.get("llama") or ROOT / "third_party/llama.cpp")
        if not (root / "llvm/bin/clang++").exists() or not (llama / "ggml/CMakeLists.txt").exists():
            warn("HIP compiler or llama.cpp source is missing; starting the previous engine")
            return
        pc = {"rocm": str(root), "archs": meta["archs"],
              "gpus": [{"index": i} for i in cfg.get("gpu", [0])]}
        compile_engine_hip(pc, llama, src, soft=True)
        return
    say("  The engine's source changed since it was compiled (an update): compiling what changed ...")
    nvcc, llama = meta.get("nvcc"), Path(meta.get("llama") or ROOT / "third_party" / "llama.cpp")
    if not nvcc or not Path(nvcc).exists() or not (llama / "ggml" / "CMakeLists.txt").exists():
        warn("the compiler or llama.cpp's source it was built with is gone: starting the engine compiled before "
             f"({ME} --setup --rebuild compiles it again)")
        return
    gpu_ids = meta.get("gpu_ids") or (cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [0])
    compile_engine(meta["archs"], gpu_ids, nvcc, meta.get("host_compiler"), llama, src, soft=True)


# ------------------------------------------------------------------------------------------------ 5. the model
def shard_set(first: Path) -> list:
    """All files of a split GGUF from its first one (<name>-00001-of-0000N.gguf), or just the file."""
    m = SHARD_RE.search(first.name)
    if not m:
        return [first]
    n, stem = int(m.group(2)), first.name[:m.start()]
    return [first.with_name(f"{stem}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def find_first_shard(d: Path):
    c = sorted(d.glob("*-00001-of-*.gguf")) or sorted(p for p in d.glob("*.gguf") if not SHARD_RE.search(p.name))
    return c[0] if c else None


def incomplete(path: Path):
    """Why this GGUF file is not whole, or None when it is: as long as its own tensor directory says (the same test
    as Strata's check_shards, without stopping).  Reads the header only."""
    if not path.exists():
        return "missing"
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(path)
    except Exception as e:                             # noqa: BLE001 - a partial header raises anything
        return f"not a whole GGUF file ({e})"
    need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
    have = path.stat().st_size
    return None if have >= need else f"short: {have:,} of {need:,} bytes"


def quant_of(first: Path) -> str:
    m = re.match(r"GLM-5\.3-Flash-(.+?)(-\d{5}-of-\d{5})?\.gguf$", first.name, re.I)
    return m.group(1) if m else first.parent.name


def sha256_ok(path: Path, want: str) -> bool:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest() == want.lower()


def offer_download(a, quant: str, d: Path, shards: list) -> bool:
    """Shows what would be downloaded - the source, the size, the exact commands - and downloads only after a yes
    (or --download-model).  False: nothing was downloaded."""
    m = MODELS[quant]
    missing = [s for s in shards if incomplete(s)]
    urls = {s: HF.format(repo=m["repo"], revision=m["revision"], folder=m["folder"], file=s.name) for s in shards}
    on_disk = sum(s.stat().st_size for s in shards if s.exists()) / 1e9
    remaining = max(0.0, m["download_gb"] - on_disk)
    free = shutil.disk_usage(existing(d)).free / 1e9
    curl = shutil.which("curl")
    cmds = [["mkdir"] + ([] if WIN else ["-p"]) + [str(d)]] + \
        [["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(s), urls[s]] for s in missing]
    say(f"  {quant}: {m['about']}.")
    say(f"  Source: https://huggingface.co/{m['repo']} (folder {m['folder']}/); the files' own license applies.")
    say(f"  The model is {m['download_gb']:.1f} GB in {len(shards)} files; {len(missing)} still to download, about "
        f"{remaining:.0f} GB, into")
    say(f"    {d}   ({free:.0f} GB free there)")
    if free < remaining + 3:
        fail(f"not enough free space in {d}: about {remaining + 3:.0f} GB are needed (the model + its pack)",
             f"free some space, or put the model on another drive: {ME} --setup --data-dir " +
             (r"D:\Maya-data" if WIN else "/path/on/nvme"))
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: the engine reads experts from these files while it answers - use an "
             "NVMe SSD (--data-dir)")
    say("  The exact commands (resumable: running them again continues an interrupted download):")
    for c in cmds:
        say("    " + shell_join(c))
    say("  You can also run them yourself (or download the files any other way into that folder, or pass")
    say(f"  --gguf-dir <folder with the files>), then run {ME} again.")
    if not curl:
        warn("curl is not installed" + ("" if WIN else " (sudo apt-get install -y curl)") + ": if you say yes, the "
             "same URLs are downloaded with Python instead (also resumable)")
    if not a.download_model and ask(f"  Download about {remaining:.0f} GB now with these commands? (y/n)",
                                    ["y", "n"], "n", False) != "y":
        say()
        say(f"Nothing was downloaded. When the files are in place, run {ME} again "
            f"({ME} --download-model downloads them without asking).")
        return False
    d.mkdir(parents=True, exist_ok=True)
    for s in missing:
        if curl:
            cmd = ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(s), urls[s]]
            if run(cmd, check=False).returncode != 0:
                fail(f"the download of {s.name} stopped (the reason is above)",
                     f"run {ME} --download-model again: it continues where it stopped")
        else:
            S.download(urls[s], s)
        why = incomplete(s)
        if why:
            fail(f"{s.name} after the download: {why}", f"delete it and run {ME} --download-model again")
        want = m["sha256"].get(s.name)
        if want:
            say(f"  checking {s.name}'s sha256 ...")
            if not sha256_ok(s, want):
                fail(f"{s.name}: the sha256 does not match the published one", "delete it and download it again")
        ok(f"{s.name} downloaded")
    return True


def model_step(a, data: Path):
    """(model folder, shards, quant name), or None when the files are not there and were not downloaded."""
    step(5, "the model (GLM-5.3-Flash, GGUF)")
    if a.gguf_dir:
        d = Path(a.gguf_dir).expanduser().resolve()
        first = find_first_shard(d) if d.is_dir() else None
        if first is None:
            fail(f"no GGUF file in {d}", "--gguf-dir takes the folder that holds the GLM-5.3-Flash .gguf files")
        shards, quant = shard_set(first), quant_of(first)
        for s in shards:
            why = incomplete(s)
            if why:
                fail(f"{s.name}: {why}", f"finish copying or downloading it, then run {ME} again")
        if quant not in MODELS:
            warn(f"{quant}: experimental - only {', '.join(MODELS)} has been measured with Maya (README.md)")
    else:
        quant = a.model or next(iter(MODELS))
        m = MODELS[quant]
        d = data / "models" / f"glm-5.3-flash-{quant}".lower()
        shards = [d / m["file"].format(i=i, n=m["shards"]) for i in range(1, m["shards"] + 1)]
        if any(incomplete(s) for s in shards) and not offer_download(a, quant, d, shards):
            return None
    from gguf_reader import GGUFFile
    arch = str(GGUFFile(shards[0]).metadata.get("general.architecture", ""))
    if arch not in ("glm5-next", "glm5next"):
        fail(f"{shards[0].name} is a {arch!r} model, not GLM-5.3-Flash (glm5-next)")
    gb = sum(s.stat().st_size for s in shards) / 1e9
    ok(f"{quant}: {len(shards)} file(s), {gb:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: expect slow answers (the engine reads experts from it) - an NVMe SSD "
             "is strongly recommended")
    return d, shards, quant


# ------------------------------------------------------------------------------------------------ 6. the pack
def pack_step(a, d: Path, shards: list, llama: Path) -> Path:
    step(6, "the pack (the engine's index of the model files)")
    # the engine finds the GGUF files at <pack>/.. (src/core/glm_model.cu, load_pack): the pack lives in their folder
    pack = d / "pack"
    if not a.repack and all((pack / f).exists() for f in PACK_FILES):
        ok(f"pack already built: {pack}")
        return pack
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack is written next to the GGUF files (the engine finds them at <pack>/..)",
             "link the .gguf files into a writable folder (" + ("mklink /H DIR\\<file> <file>, for each file" if WIN
                                                                else "mkdir -p DIR && ln -s /path/to/*.gguf DIR/") +
             ") and pass "
             "--gguf-dir DIR")
    say("  Writing the index of every tensor, the small float weights (about 1 GB) and the tokenizer; the experts")
    say("  stay in the GGUF files (a few minutes) ...")
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    # --compat-bf16: the GLM pack stores the projections its kernels read as BF16 (router, indexer, absorbed MLA,
    # KDA gates) in BF16 even where the GGUF quantized them or keeps them F32 (tools/iq_pack.py GLM5NEXT_BF16)
    run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
         "--compat-bf16"], env=env)
    missing = [f for f in PACK_FILES if not (pack / f).exists()]
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    ok(f"pack: {pack}")
    return pack


# ------------------------------------------------------------------------------------------------ 7. images
def fetch(url: str, dst: Path, want: str | None) -> None:
    """One file, resumable (curl -C -, else Python), then its sha256 when one is published."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        if run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(dst), url], check=False).returncode != 0:
            fail(f"the download of {dst.name} stopped (the reason is above)", f"run {ME} --setup again: it "
                                                                              "continues where it stopped")
    else:
        S.download(url, dst)
    if want and not sha256_ok(dst, want):
        dst.unlink(missing_ok=True)
        fail(f"{dst.name}: the sha256 does not match the published one (the file was removed)",
             f"run {ME} --setup again to download it again")


def compile_vision(pc, meta, llama: Path, src: str) -> bool:
    """strata-vision: llama.cpp's mtmd at the pinned commit with GLM-5.3-Flash's vision tower added
    (tools/vision/glm5next_patch.py, idempotent), for the same GPUs and toolkit as the engine."""
    cmake = pick_cmake()
    if cmake is None:
        return False
    run([sys.executable, str(ROOT / "tools" / "vision" / "glm5next_patch.py"), str(llama)])
    gen = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", (VBUILD / "CMakeCache.txt").read_text(errors="replace"), re.M) \
        if (VBUILD / "CMakeCache.txt").exists() else None
    ninja = venv_tool("ninja") or shutil.which("ninja")
    conf = [cmake]
    if ninja and (gen is None or gen.group(1).strip() == "Ninja"):
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    archs = meta.get("archs") or pc["archs"]
    conf += ["-S", str(ROOT / "tools" / "vision"), "-B", str(VBUILD), "-DCMAKE_BUILD_TYPE=Release",
             f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON",
             "-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(str(x) for x in archs),
             f"-DCMAKE_CUDA_COMPILER={meta.get('nvcc') or pc['nvcc']}"]
    if meta.get("host_compiler"):
        conf.append(f"-DCMAKE_CUDA_HOST_COMPILER={shutil.which(meta['host_compiler']) or meta['host_compiler']}")
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(mem_gb()[0] // 4) or 2))
    say("  Compiling the vision encoder (llama.cpp's image library with ggml's CUDA kernels: 10-20 minutes, once) ...")
    if cmake_steps(conf, [cmake, "--build", str(VBUILD), "--target", "strata-vision", "-j", str(jobs)], None,
                   "build-maya-vision.bat"):
        return False
    VSTAMP.write_text(json.dumps({"src": src, "archs": archs, "llama": str(llama),
                                  "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1), encoding="utf-8")
    return VEXE.exists()


def vision_step(a, pc, meta, llama: Path, d: Path, quant: str) -> dict | None:
    """The config's "vision" entry - the image encoder and its files - or None (images off).  The server starts the
    encoder only while a request's new pictures are encoded, in GPU memory the model lends it, and measures on the
    first start how much that is on this GPU (serve/server.py, vision_footprint)."""
    step(7, "images (the vision encoder)")
    if pc.get("backend") == "hip":
        ok("HIP port: text only; vision is not enabled")
        return None
    m = (MODELS.get(quant) or {}).get("vision")
    if a.no_vision:
        ok("skipped (--no-vision): the model reads text only")
        return None
    if m is None:
        warn(f"no vision files are published for {quant}: the model reads text only")
        return None
    vd = d / m["folder"]
    files = {k: vd / m[k] for k in ("mmproj", "vocab")}
    missing = [p for p in files.values() if not p.exists() or (m["sha256"].get(p.name) and p.stat().st_size == 0)]
    if missing:
        src_url = {p: HF.format(repo=MODELS[quant]["repo"], revision=MODELS[quant]["revision"], folder=m["folder"],
                                file=p.name) for p in missing}
        say(f"  The vision encoder's files ({m['download_gb']:.2f} GB) from https://huggingface.co/"
            f"{MODELS[quant]['repo']} (folder {m['folder']}/), into {vd}:")
        for p in missing:
            say("    " + shell_join(["curl", "-L", "--fail", "-C", "-", "-o", str(p), src_url[p]]))
        if not (a.download_model or a.yes) and ask("  Download them now? (y/n)", ["y", "n"], "y", False) != "y":
            warn(f"not downloaded: the model reads text only ({ME} --setup asks again)")
            return None
        for p in missing:
            fetch(src_url[p], p, m["sha256"].get(p.name))
            ok(f"{p.name} downloaded")
    src = S.source_hash(("tools/vision",))
    vmeta = read_json(VSTAMP)
    if not (VEXE.exists() and not a.rebuild and vmeta.get("src") == src
            and set(meta.get("archs") or pc["archs"]) <= set(vmeta.get("archs", []))):
        if not compile_vision(pc, meta, llama, src):
            warn("the vision encoder did not compile (the reason is above): the model reads text only - "
                 f"{ME} --setup --rebuild tries again")
            return None
    ok(f"vision encoder: {VEXE}")
    return {"exe": str(VEXE), "mmproj": str(files["mmproj"]), "model": str(files["vocab"]), "gpu": True}


# ------------------------------------------------------------------------------------------------ 8. config + start
def parse_env(items) -> dict:
    env = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            fail(f"--env takes KEY=VALUE, e.g. --env STRATA_GLM_RAM_HEADROOM_GB=4, not {it!r}")
        env[k] = v
    return env


def write_run_script(cfg_path: Path, port: int) -> Path:
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port)]
    if WIN:
        script = ROOT / f"run-{cfg_path.stem}.bat"
        script.write_text(f'@echo off\r\nrem starts the Project Maya dashboard and API (written by maya.py; {ME} does '
                          f'the same)\r\ncd /d "{ROOT}" || exit /b 1\r\n{cmdline(cmd)} %*\r\n', encoding="utf-8")
        return script
    script = ROOT / f"run-{cfg_path.stem}.sh"
    script.write_text("#!/bin/sh\n# starts the Project Maya dashboard and API (written by maya.py; ./maya.sh does the "
                      "same)\ncd " + shlex.quote(str(ROOT)) + " || exit 1\nexec " + shlex.join(cmd) + ' "$@"\n',
                      encoding="utf-8")
    script.chmod(0o755)
    return script


def write_config(a, pc, meta, pack: Path, quant: str, ctx: int, data: Path, vision: dict | None) -> Path:
    step(8, "the configuration and the start script")
    port = a.port or 8080
    cfg = {"exe": str(EXE), "args": ["--glm-pack", str(pack), "--max-context", str(ctx)], "cwd": str(ROOT),
           "tokenizer": str(pack / "tokenizer"), "model_name": MODEL_NAME, "gpu": [g["index"] for g in pc["gpus"]],
           "sampling": dict(SAMPLING), "reasoning_effort": EFFORT, "lib_dirs": meta.get("lib_dirs") or [],
           "port": port}
    env = parse_env(a.env)
    if pc.get("backend") == "hip":
        cfg["backend"] = "hip"
        # Leave space for the Linux desktop. The engine sizes its prompt chunk from
        # the available prompt-memory budget; forcing 256 here severely slows HIP.
        hip_env = {"STRATA_GLM_SPLIT": "0", "STRATA_GLM_RESERVE_MB": "3072", "STRATA_GLM_RAM_HEADROOM_GB": "16"}
        g = pc["gpus"][0]
        # Larger prompt sub-batches feed the matrix cores much better (7900 XT: ~250 -> ~410 tok/s); they need
        # a bigger prompt budget, borrowed from the expert pool only while a prompt runs.
        if g.get("vram_gb", 0) >= 20:
            hip_env.update({"STRATA_GLM_PREFILL_SUB": "1024", "STRATA_GLM_PREFILL_MB": "4096"})
        # The prompt projections' hipBLASLt solutions measured on this architecture (tools/hip). The engine
        # refuses a table made for another hipBLASLt version and keeps plain hipBLAS.
        tables = sorted((ROOT / "tools" / "hip").glob(f"{g.get('arch', '')}-glm-hipblaslt-*.txt"))
        if tables:
            hip_env["STRATA_HIPBLASLT_TUNING"] = str(tables[-1])
        env = {**hip_env, **env}
    if env:
        cfg["env"] = env
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if vision:
        cfg["vision"] = vision
    suffix = "-hip" if pc.get("backend") == "hip" else ""
    cfg_path = ROOT / f"maya-{quant.lower()}{suffix}.json"
    cfg["log"] = str(cfg_path.with_suffix(".log"))
    cfg["installer"] = {"data_dir": str(data), "quant": quant,
                        "gguf_dir": str(Path(a.gguf_dir).expanduser().resolve()) if a.gguf_dir else None,
                        "written": time.strftime("%Y-%m-%d %H:%M")}
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    ok(f"config: {cfg_path}")
    ok(f"start script: {write_run_script(cfg_path, port)}")
    return cfg_path


def start(cfg_path: Path, a) -> int:
    cfg = read_json(cfg_path)
    select_build_backend(cfg.get("backend", "cuda"))
    args = cfg.get("args") or []
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    for p, what in ((Path(cfg.get("exe", "")), "the engine"), (pack, "the pack"),
                    (Path(cfg.get("tokenizer", "")) / "vocab.json", "the tokenizer")):
        if p is None or not p.exists():
            fail(f"{cfg_path.name}: {what} is missing ({p})", f"run {ME} --setup to repair it")
    cfg_path.touch()                                   # the most recently used model
    refresh_engine(cfg)
    port = a.port or cfg.get("port") or 8080
    host = a.host or cfg.get("host") or "127.0.0.1"
    key = a.api_key or cfg.get("api_key")
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port)]
    if a.host:
        cmd += ["--host", a.host]
    if a.api_key:
        cmd += ["--api-key", a.api_key]
    if a.gpus or a.gpu is not None:                    # this start only, on these cards
        cmd += ["--gpu", str(a.gpus or a.gpu)]
    if WIN or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        cmd.append("--open")                           # a desktop: the browser opens when the model is ready
    here = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    say()
    say("  " + "-" * 100)
    say(f"  Starting {cfg.get('model_name', MODEL_NAME)} ({cfg_path.name}).")
    say(f"  Dashboard:        http://{here}:{port}/")
    say(f"  API (OpenAI):     http://{here}:{port}/v1        (any model name; the API key: "
        + ("the one you set)" if key else "any)"))
    say(f"  API (Anthropic):  http://{here}:{port}/v1/messages")
    if host not in ("127.0.0.1", "localhost", "::1"):
        say("  Other devices:    the server prints this machine's addresses when it is ready" +
            ("" if key else " - WARNING: no API key, anyone on your network can use it (--api-key KEY)"))
    say("  Loading takes a few minutes: the engine pins up to all but about 6 GB of the free RAM for its expert tier")
    say("  (STRATA_GLM_RAM_HEADROOM_GB changes the 6) and warms its caches - the first answers are the slowest.")
    say("  Ctrl+C (or closing this terminal) stops it. Engine log: " + str(cfg.get("log", "")))
    say("  " + "-" * 100)
    return subprocess.call(cmd)


# ------------------------------------------------------------------------------------------------ the report
# the engine log's lines that tell where the time goes: how the model was split across VRAM / RAM / the SSD, the
# prompt path's chunks, and the per-token breakdown ("glm stat": VRAM hits, RAM fetches, disk reads, CPU lane)
REPORT_LINES = re.compile(r"glm fast:|glm prefill: CUDA|glm split|glm stat|glm slots|glm prefill: \d|ERR|error|failed|"
                          r"out of memory", re.I)
STAT_DECODE = re.compile(r"glm stat: decode ([\d.]+) ms/tok")


def speed_lines(text) -> list:
    """The log lines REPORT_LINES picks.  A request that only read a prompt (one token out) has no decode to report:
    its "glm stat" line keeps only the prompt part, not a meaningless decode speed (96,000 tokens/s, issue #8)."""
    picked = []
    for x in text:
        if not REPORT_LINES.search(x) or "warming the expert tiers" in x:
            continue
        m = STAT_DECODE.search(x)   # no decode: under 1 ms a token, or no expert touched (one prompt token out)
        if m and (float(m.group(1)) < 1.0 or "vram hit 0.00%" in x) and "| prompt " in x:
            x = "glm stat (prompt only): prompt " + x.split("| prompt ", 1)[1]
        picked.append(x.strip())
    return picked


def report(version: str) -> int:
    """--report: maya-report.txt in the Maya folder - this PC, the installed setup and the engine log's speed lines,
    for a bug or speed report.  Nothing is sent anywhere; the home folder is written as ~ and API keys are left out."""
    home = str(Path.home())
    lines = []

    def add(title, text=""):
        lines.append(f"## {title}")
        lines.extend(str(text).rstrip().replace(home, "~").splitlines() or ["(nothing)"])
        lines.append("")

    git = S.out(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"]).strip()
    add("Maya", f"version {version}, commit {git or '?'}, Python {sys.version.split()[0]}, "
                f"{sys.platform}{' (WSL)' if S.is_wsl() else ''}")
    import platform
    add("System", f"{platform.platform()}")
    add("GPUs", S.out(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,"
                                     "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,"
                                     "temperature.gpu", "--format=csv"]) or "nvidia-smi did not answer")
    cpu, avx2, avx512 = S.cpu_info()
    total, avail = mem_gb()
    pf = S.page_file_gb()
    add("CPU and RAM", f"{cpu}, {os.cpu_count()} threads, {'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'}\n"
                       f"RAM {total:.1f} GB, {avail:.1f} GB available now" +
                       (f", page file {pf:.0f} GB" if pf is not None else ""))
    if WIN:
        disks = S.out(["powershell", "-NoProfile", "-Command", "Get-PhysicalDisk | Format-Table -AutoSize "
                                                              "FriendlyName,MediaType,BusType,Size | Out-String -Width 200"])
    else:
        disks = S.out(["lsblk", "-d", "-o", "NAME,MODEL,ROTA,TRAN,SIZE"])
    add("Disks", disks)
    stamp = read_json(STAMP)
    add("Engine build", json.dumps({k: stamp.get(k) for k in ("archs", "nvcc", "host_compiler", "date")})
        if stamp else "not compiled yet")
    have = configs()
    if not have:
        add("Setup", f"no maya-*.json yet: {ME} has not finished a setup here")
    for c in have:
        cfg = read_json(c)
        args = cfg.get("args") or []
        pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
        ctx = args[args.index("--max-context") + 1] if "--max-context" in args[:-1] else "?"
        d = pack.parent if pack else None
        where = ""
        if d is not None and d.exists():
            gb = sum(p.stat().st_size for p in d.glob("*.gguf")) / 1e9
            where = (f"\nmodel folder {d}: {gb:.1f} GB of GGUF, {shutil.disk_usage(d).free / 1e9:.0f} GB free, "
                     f"{'a spinning hard disk' if rotational(d) else 'not a hard disk'}")
        add(f"Setup {c.name}", f"model {(cfg.get('installer') or {}).get('quant', '?')}, context {ctx}, "
                               f"GPUs {cfg.get('gpu')}, images {'on' if cfg.get('vision') else 'off'}, "
                               f"settings {json.dumps(cfg.get('env') or {})}" + where)
        log = Path(cfg.get("log") or c.with_suffix(".log"))
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace").splitlines()
            picked = speed_lines(text)
            add(f"Engine log {log.name}: the speed and memory lines (last 80 of {len(picked)})", "\n".join(picked[-80:]))
            add(f"Engine log {log.name}: the last 25 lines", "\n".join(text[-25:]))
        else:
            add(f"Engine log {log.name}", "not written yet (start Maya once and ask it something)")
    bench_txt = ROOT / "maya-bench.txt"
    if bench_txt.exists():
        add("Benchmark (maya-bench.txt, from --bench)", bench_txt.read_text(encoding="utf-8", errors="replace"))
    out = ROOT / "maya-report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say()
    ok(f"written: {out}")
    say("  Attach this file to your report (GitHub issue, X, or wherever you are asking). It has this PC's hardware,")
    say("  your Maya setup and the engine's speed lines - no API key, and your home folder is shown as ~.")
    say("  Best: run it right after a slow answer, so the log has that answer's numbers.")
    return 0


# ------------------------------------------------------------------------------------------------ the benchmark
# the same three questions on every machine (greedy, thinking off), after one warm-up answer, so speeds compare
BENCH_TOPICS = [
    "Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.",
    "Explain how photosynthesis works, step by step, for a high-school student.",
    "Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.",
]


def bench(cfg_path: Path, version: str) -> int:
    """--bench: a standard speed test of the installed model on this PC - decode (writing an answer) on three
    questions and prefill (reading a prompt) at 2k and 8k tokens of this folder's docs - through the engine alone, the
    way the dashboard starts it.  Writes maya-bench.txt (--report includes it).  A few minutes."""
    import urllib.request
    cfg = read_json(cfg_path)
    port = cfg.get("port") or 8080
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        fail(f"Maya is running (port {port}): the benchmark needs the GPU memory it holds",
             "stop it (Ctrl+C in its window), then run --bench again")
    except OSError:
        pass
    sys.path.insert(0, str(ROOT))
    import serve.server as SV                      # the server's own engine command and environment
    import strata_tokenizer as ST
    args = cfg.get("args") or []
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    tp = Path(cfg.get("tokenizer") or (pack / "tokenizer" if pack else ""))
    if not (tp / "vocab.json").exists():
        fail(f"{cfg_path.name}: the tokenizer is missing ({tp})", f"run {ME} --setup to repair it")
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    names = [None] * len(vocab)
    for t, i in vocab.items():
        names[i] = t
    tok = ST.Tokenizer(names, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tp / "token_type.json").read_text()))
    ctx = int(args[args.index("--max-context") + 1]) if "--max-context" in args[:-1] else 32768
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) +
                       sorted((ROOT / "bench" / "results").glob("*.md")) if p.exists())
    doc = tok.encode(text)
    lens = [n for n in (2048, 8192) if n + 64 <= ctx and n <= len(doc)]
    log_path = ROOT / "maya-bench.log"
    step(1, f"benchmark: {cfg_path.name} (loading the model first - a minute or a few)")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen([cfg["exe"], "--serve"] + SV.engine_args(cfg), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, text=True, env=SV.child_env(cfg),
                             cwd=cfg.get("cwd") or str(ROOT), bufsize=1)

        def gen(ids, n_new):
            p.stdin.write(f"GEN {n_new} temperature=0 " + ",".join(map(str, ids)) + "\n")
            p.stdin.flush()
            while True:
                line = p.stdout.readline()
                if not line:
                    fail("the engine stopped during the benchmark", f"its log: {log_path}")
                if line.startswith("DONE") or line.startswith("ERR"):
                    return line.split()

        while True:
            line = p.stdout.readline()
            if not line:
                fail("the engine stopped while loading", f"its log: {log_path}")
            if line.startswith("READY"):
                break
        ok(f"loaded in {time.time() - t0:.0f} s")
        chat = lambda q: tok.encode("[gMASK]<sop><|user|>\n" + q + "<|assistant|>\n</think>", parse_special=True)
        gen(chat("Say hello in five languages."), 64)          # warm-up: the first answer after a start is slower
        results = []
        for q in BENCH_TOPICS:
            f = gen(chat(q), 256)
            if f[0] == "DONE" and float(f[4]) > 0:
                results.append(("decode", q, int(f[1]) / float(f[4]) * 1000.0))
                say(f"  decode  {results[-1][2]:6.1f} tokens/s   {q[:60]}")
        if lens:                                            # warm-up: the first prompt after a start is slower
            gen(doc[-lens[0]:], 1)                          # (cold caches, issue #8); its opening is not reused below
        for i, n in enumerate(lens):
            f = gen(doc[i * 997:i * 997 + n], 1)            # different openings: nothing reused between them
            if f[0] == "DONE" and float(f[3]) > 0:
                results.append(("prefill", f"{n} tokens", n / float(f[3]) * 1000.0))
                say(f"  prefill {results[-1][2]:6.0f} tokens/s   a {n}-token prompt")
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
    stats = speed_lines(log_path.read_text(encoding="utf-8", errors="replace").splitlines())
    dec = [r[2] for r in results if r[0] == "decode"]
    gpus = ", ".join(f"{g['name']} {g['vram_gb']:.0f} GB" for g in S.gpus()) or "?"
    total, _ = mem_gb()
    lines = [f"Project Maya v{version} benchmark, {time.strftime('%Y-%m-%d %H:%M')}",
             f"GPUs: {gpus}; RAM {total:.0f} GB; CPU {S.cpu_info()[0]}",
             f"setup: {cfg_path.name}, context {ctx}, GPUs {cfg.get('gpu')}",
             f"decode (writing the answer): mean {sum(dec) / max(1, len(dec)):.1f} tokens/s over {len(dec)} answers of "
             f"256 tokens (greedy, thinking off)"]
    lines += [f"  {r[2]:6.1f} tokens/s  {r[1]}" for r in results if r[0] == "decode"]
    lines += [f"prefill (reading the prompt): {r[2]:.0f} tokens/s, {r[1]}" for r in results if r[0] == "prefill"]
    lines += ["", "engine lines:"] + [x.replace(str(Path.home()), "~") for x in stats[-40:]]
    out = ROOT / "maya-bench.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say()
    for x in lines[:4 + len(results)]:
        say("  " + x)
    ok(f"written: {out} - attach it (with {ME} --report) to a speed report")
    return 0


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cuda", "hip"], help="GPU backend (HIP: experimental gfx1100/gfx1201 on Linux)")
    ap.add_argument("--setup", action="store_true", help="set up again instead of starting the installed model")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--no-start", action="store_true", help="set up, but do not start the dashboard")
    ap.add_argument("--yes", action="store_true",
                    help="take the recommended answers (the model download still needs --download-model)")
    ap.add_argument("--download-model", action="store_true",
                    help="download the model without asking (the commands and the size are still printed)")
    ap.add_argument("--model", choices=list(MODELS), help=f"which download (default: {next(iter(MODELS))})")
    ap.add_argument("--no-vision", action="store_true", help="text only: no image encoder (saves its build and "
                                                             "its 1.1 GB of files)")
    ap.add_argument("--gguf-dir", help="use GLM-5.3-Flash GGUF files you already have: the folder with all of them "
                                       "(it must be writable - the pack is written inside it)")
    ap.add_argument("--data-dir", help="where a downloaded model goes (default: Maya-data next to this folder); use a "
                                       "fast NVMe SSD with ~100 GB free")
    ap.add_argument("--gpu", type=int, help="run on this one GPU (CUDA: nvidia-smi; HIP: KFD topology order)")
    ap.add_argument("--gpus", help="split the model across these two GPUs, e.g. 0,1")
    ap.add_argument("--context", type=int, help=f"context length in tokens (default {DEFAULT_CONTEXT})")
    ap.add_argument("--port", type=int, help="the dashboard's and the API's port (default 8080)")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this machine only (default), 0.0.0.0 = also "
                                   "other devices on your network (set --api-key too)")
    ap.add_argument("--api-key", help="require this key from API clients and the dashboard")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="an engine setting kept in the config, e.g. STRATA_GLM_RAM_HEADROOM_GB=4 (README-MAYA.md)")
    ap.add_argument("--nvcc", help="the CUDA toolkit's nvcc to build with (default: the newest that fits your GPUs)")
    ap.add_argument("--host-compiler", help="the C++ compiler nvcc uses, e.g. g++-12 (when the default g++ is newer "
                                            "than your CUDA accepts)")
    ap.add_argument("--llama-dir", help="a llama.cpp checkout at the pinned commit, instead of downloading its source")
    ap.add_argument("--rebuild", action="store_true", help="compile the engine again")
    ap.add_argument("--repack", action="store_true", help="build the pack again")
    ap.add_argument("--report", action="store_true", help="write maya-report.txt - this PC, the setup and the engine's "
                                                          "speed lines - to attach when you report a problem or a "
                                                          "speed (nothing is sent anywhere)")
    ap.add_argument("--bench", action="store_true", help="a standard speed test of the installed model (a few "
                                                         "minutes, with Maya stopped): decode on three questions and "
                                                         "prefill at 2k / 8k tokens; writes maya-bench.txt")
    a = ap.parse_args()
    version = (HERE / "VERSION").read_text(encoding="utf-8").strip() if (HERE / "VERSION").exists() else "?"
    say(f"Project Maya v{version} - GLM-5.3-Flash on your own GPU(s). Built on Strata (MIT) and ggml/llama.cpp "
        "(MIT).")
    if a.report:
        return report(version)
    if a.bench:
        have = configs()
        if not have:
            fail("Maya is not set up here yet", f"run {ME} first")
        return bench(have[0], version)

    have = configs()
    if a.backend:
        have = [p for p in have if read_json(p).get("backend", "cuda") == a.backend]
    else:
        a.backend = "hip" if have and read_json(have[0]).get("backend") == "hip" else "cuda"
    setting_up = a.setup or a.check or a.no_start or a.gguf_dir or a.model or a.rebuild or a.repack or a.download_model
    if have and not setting_up:
        pick = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {c.name}")
            pick = have[int(ask("Which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        return start(pick, a)
    prev = read_json(have[0]) if have else {}
    inst = prev.get("installer") or {}
    if not a.model:                                    # --model asks for that download, not the files set up before
        a.gguf_dir = a.gguf_dir or inst.get("gguf_dir")
    prev_args = prev.get("args") or []
    prev_ctx = int(prev_args[prev_args.index("--max-context") + 1]) if "--max-context" in prev_args[:-1] else None

    pc = check_pc(a)                                   # 1
    if a.check:
        say()
        if a.backend == "hip":
            say(f"HIP prerequisites found. Run {ME} --backend hip --gpu {pc['gpus'][0]['index']} to try the experimental port.")
        else:
            say(f"This PC can run Maya. Run {ME} without --check to set it up.")
        return 0
    step(2, "your choices")                            # 2
    ctx = choose_context(a, prev_ctx)
    ok(f"context: {ctx} tokens")
    data = (Path(a.data_dir).expanduser().resolve() if a.data_dir else
            Path(inst["data_dir"]) if inst.get("data_dir") else ROOT.parent / "Maya-data")
    if not a.gguf_dir:
        ok(f"model folder: {data / 'models'}")
    llama = tools_step(a)                              # 3
    meta = build_step(a, pc, llama)                    # 4
    got = model_step(a, data)                          # 5
    if got is None:
        return 0
    d, shards, quant = got
    pack = pack_step(a, d, shards, llama)              # 6
    vision = vision_step(a, pc, meta, llama, d, quant)  # 7
    cfg_path = write_config(a, pc, meta, pack, quant, ctx, data, vision)   # 8
    if a.no_start:
        say()
        say(f"All set. Start it with {ME} (or " + (f"run-{cfg_path.stem}.bat" if WIN else f"./run-{cfg_path.stem}.sh")
            + ").")
        return 0
    return start(cfg_path, a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
