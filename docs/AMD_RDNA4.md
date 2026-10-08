# R9700 / RX 9070: experimental Maya HIP

The Linux HIP installer accepts RX 7900 XT/XTX (`gfx1100`) and RX 9070 /
Radeon AI PRO R9700 (`gfx1201`). It uses one GPU, text input, and system ROCm 7.
The shared backend changes come from Strata
[`d5ea713`](https://github.com/Niko1221/Strata/tree/d5ea7133741e67743c0e886bb426c0ce8d69cf6c).
The R9700 runs the full model (Maya-S, about 20 tok/s answers and 490-560 tok/s
prompts, every expert in VRAM or a pinned RAM tier).

```sh
./maya.sh --backend hip --gpu 1 --check
```

Use the KFD GPU number printed by setup; the example machine has a 7900 XT
at GPU 0 and an R9700 at GPU 1. Setup chooses the supported card with the most
VRAM when `--gpu` is omitted. See [AMD_MAYA.md](AMD_MAYA.md) for model setup.

## One binary, three code targets

Setup compiles `gfx1100;gfx1201;gfx1151`. CMake also accepts space-separated
lists and architecture feature suffixes. The runtime strips the feature suffix
and checks exact membership in the compiled list, then requires wave32.
Its diagnostics print the actual architecture and the compiled targets.

`gfx1151` **builds, untested**. Code is included for later Strix Halo work, but
setup does not enable that APU here. Unified-memory accounting and hardware
validation are separate requirements.

A manual build, using Maya's existing pinned llama.cpp checkout:

```sh
maya_rocm_root="$(realpath "${ROCM_PATH:-/opt/rocm}")"
cmake -S . -B build-multi -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  '-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201;gfx1151' \
  -DCMAKE_HIP_COMPILER="$maya_rocm_root/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$maya_rocm_root" \
  -DCMAKE_PREFIX_PATH="$maya_rocm_root" \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build-multi --target strata strata-device hip_expert_cache_staging \
  hip_intrinsics hip_handoff hip_glm_handoff hip_glm_prefill_attention iq1_s_parity \
  glm_hc_parity glm_kda_parity glm_ffn_parity glm_dsa_parity glm_layer_parity \
  glm_model_test hip_prefill_mmq_parity -j 3
```

The handoff target and the tested publication fences were merged in v1.0.7 (#2).

## Validation

Native Ubuntu, Ryzen 7 9800X3D, 192 GB DDR5, ROCm 7.2.1 / Clang 22;
RX 7900 XT 20 GiB and AI PRO R9700 32 GiB. The same engine binary contains
64 bundled code objects for each of gfx1100, gfx1201 and gfx1151.

```sh
python3 tools/test_maya_hip.py
python3 tools/test_hip_archs.py
env -u ROCR_VISIBLE_DEVICES LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=1 \
  ctest --test-dir build-multi --output-on-failure --timeout 60 \
  -R '^(glm_(hc|kda|ffn|dsa|layer)_parity|glm_model_test|hip_intrinsics|hip_(glm_handoff|glm_prefill_attention|handoff|device_selftest|prefill_mmq_parity|expert_cache_staging)|iq1_s_parity)$'
env -u ROCR_VISIBLE_DEVICES LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=1 \
  STRATA_GLM_HC1=1 build-multi/glm_hc_parity --selftest
env -u ROCR_VISIBLE_DEVICES LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=1 \
  STRATA_GLM_HC2=1 build-multi/glm_hc_parity --selftest
```

Both cards pass 14/14 GPU checks; HC1/HC2 also pass on the R9700. The intrinsics
test compares signed dot4, overflow, shuffles, and 65,536 packed-byte pairs and
permutations to CPU references. Six setup tests and five architecture-parser
tests pass. Selecting the uncompiled gfx1036 iGPU is refused before a kernel
launch. CUDA was not compile-checked because the test host has no `nvcc`.

The copied gfx1201 hipBLASLt tables match versions `100202` and `100500`.
They serve the inherited tuning loader. Maya's GLM prompt projections can now
use the separate `gfx1201-glm-hipblaslt-100202.txt` table through that loader;
shapes without a tuned row continue through hipBLAS. Its quantized expert path
still uses GGML MMQ.
