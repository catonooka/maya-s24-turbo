"""Maya's HIP setup regressions; no GPU, network or model downloads required."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya


class HipSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rocm = self.root / "rocm"
        (self.rocm / "llvm/bin").mkdir(parents=True)
        (self.rocm / "llvm/bin/clang++").touch()
        (self.rocm / "lib").mkdir()
        (self.rocm / "lib/libhipblas.so").touch()
        self.gpus = [
            {"index": 0, "arch": "gfx1100", "vendor": "amd", "name": "RX 7900 XT", "vram_gb": 20},
            {"index": 1, "arch": "gfx1201", "vendor": "amd", "name": "R9700", "vram_gb": 32},
            {"index": 2, "arch": "gfx1036", "vendor": "amd", "name": "iGPU", "vram_gb": 1},
        ]
        for ctx in (
            patch.object(maya, "ROOT", self.root),
            patch.object(maya, "BUILD", self.root / "build"),
            patch.object(maya, "EXE", self.root / "build/strata"),
            patch.object(maya, "STAMP", self.root / "build/MAYA-BUILD.json"),
            patch.object(maya, "WIN", False),
            patch.dict(os.environ, {"ROCM_PATH": str(self.rocm)}),
            patch.object(maya.S, "amd_gpus", return_value=self.gpus),
            patch.object(maya.S, "gpus", side_effect=AssertionError("HIP must not query NVIDIA")),
            patch.object(maya.S, "is_wsl", return_value=False),
            patch.object(maya.S, "cpu_info", return_value=("test CPU", True, True)),
            patch.object(maya, "mem_gb", return_value=(192, 128)),
            patch.object(maya, "tool_version", return_value=(7, 2)),
            patch.object(maya.shutil, "which", return_value="/usr/bin/c++"),
        ):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.a = SimpleNamespace(backend="hip", gpu=0, gpus=None, no_vision=False, env=[],
                                 port=8099, host=None, api_key=None, gguf_dir=None)

    def test_selects_larger_supported_discrete_card(self):
        self.a.gpu = None
        pc = maya.check_pc(self.a)
        self.assertEqual(pc["gpus"], [self.gpus[1]])
        self.assertEqual(maya.EXE, self.root / "build-hip/strata")

    def test_rejects_unsupported_card_and_multi_gpu(self):
        self.a.gpu = 2
        with self.assertRaises(SystemExit):
            maya.check_pc(self.a)
        self.a.gpu = 0
        self.a.gpus = "0,1"
        with self.assertRaises(SystemExit):
            maya.check_pc(self.a)

    def test_config_selects_hip_and_preserves_user_tuning(self):
        pc = maya.check_pc(self.a)
        self.a.env = ["STRATA_GLM_RESERVE_MB=4096"]
        p = maya.write_config(self.a, pc, {"lib_dirs": [str(self.rocm / "lib")]},
                              self.root / "pack", "test", 8192, self.root / "data", None)
        cfg = json.loads(p.read_text())
        self.assertEqual(cfg["backend"], "hip")
        self.assertEqual(cfg["gpu"], [0])
        self.assertEqual(cfg["env"]["STRATA_GLM_SPLIT"], "0")
        self.assertEqual(cfg["env"]["STRATA_GLM_RESERVE_MB"], "4096")
        self.assertEqual(cfg["exe"], str(self.root / "build-hip/strata"))
        self.assertNotIn("vision", cfg)

    def test_config_prompt_defaults_and_tuning_table(self):
        tables = self.root / "tools/hip"
        tables.mkdir(parents=True)
        (tables / "gfx1100-glm-hipblaslt-100202.txt").write_text("STRATA_HIPBLASLT_TUNING_V1 gfx1100 100202\n")
        (tables / "gfx1201-glm-hipblaslt-100202.txt").write_text("STRATA_HIPBLASLT_TUNING_V1 gfx1201 100202\n")
        pc = maya.check_pc(self.a)
        self.a.env = ["STRATA_GLM_PREFILL_SUB=512"]
        p = maya.write_config(self.a, pc, {"lib_dirs": [str(self.rocm / "lib")]},
                              self.root / "pack", "test", 8192, self.root / "data", None)
        env = json.loads(p.read_text())["env"]
        self.assertNotIn("STRATA_GLM_PREFILL_CHUNK", env)
        self.assertEqual(env["STRATA_GLM_PREFILL_SUB"], "512")   # the user's setting wins
        self.assertEqual(env["STRATA_GLM_PREFILL_MB"], "4096")
        self.assertEqual(env["STRATA_HIPBLASLT_TUNING"], str(tables / "gfx1100-glm-hipblaslt-100202.txt"))

    def test_hip_build_enables_mmq_and_never_cuda(self):
        pc = maya.check_pc(self.a)
        maya.BUILD.mkdir()
        with patch.object(maya, "pick_cmake", return_value="cmake"), \
                patch.object(maya, "cmake_steps", return_value=None) as build:
            meta = maya.compile_engine_hip(pc, self.root / "llama", "source-sha")
        conf, _, env, _ = build.call_args.args
        self.assertIn("-DSTRATA_ENABLE_HIP=ON", conf)
        self.assertIn("-DSTRATA_ENABLE_CUDA=OFF", conf)
        self.assertIn("-DSTRATA_PREFILL_MMQ=ON", conf)
        self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201;gfx1151", conf)
        self.assertEqual(env["ROCM_PATH"], str(self.rocm))
        self.assertEqual(meta["backend"], "hip")

    def test_hip_skips_vision_without_downloading(self):
        with patch.object(maya.S, "download", side_effect=AssertionError("no vision downloads")):
            self.assertIsNone(maya.vision_step(self.a, {"backend": "hip"}, {}, self.root, self.root, "test"))


if __name__ == "__main__":
    unittest.main()
