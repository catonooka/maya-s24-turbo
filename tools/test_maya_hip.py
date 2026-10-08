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

    def test_rejects_unsupported_card_and_bad_gpu_lists(self):
        self.a.gpu = 2
        with self.assertRaises(SystemExit):
            maya.check_pc(self.a)
        self.a.gpu = None
        for bad in ("0,2", "0,1,2", "1,1", "x"):
            self.a.gpus = bad
            with self.assertRaises(SystemExit):
                maya.check_pc(self.a)

    def test_two_gpu_split_config(self):
        tables = self.root / "tools/hip"
        tables.mkdir(parents=True)
        for arch in ("gfx1100", "gfx1201"):
            (tables / f"{arch}-glm-hipblaslt-100202.txt").write_text(f"STRATA_HIPBLASLT_TUNING_V1 {arch} 100202\n")
        self.a.gpu = None
        self.a.gpus = "0,1"
        pc = maya.check_pc(self.a)
        self.assertEqual([g["index"] for g in pc["gpus"]], [1, 0])   # the larger card first
        p = maya.write_config(self.a, pc, {"lib_dirs": [str(self.rocm / "lib")]},
                              self.root / "pack", "test", 8192, self.root / "data", None)
        cfg = json.loads(p.read_text())
        self.assertEqual(cfg["gpu"], [1, 0])
        self.assertNotIn("STRATA_GLM_SPLIT", cfg["env"])
        self.assertEqual(cfg["env"]["STRATA_HIPBLASLT_TUNING"],
                         f"{tables / 'gfx1201-glm-hipblaslt-100202.txt'}:{tables / 'gfx1100-glm-hipblaslt-100202.txt'}")

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
        self.assertEqual(env["STRATA_GLM_RESERVE_MB"], "3072")
        self.assertEqual(env["STRATA_GLM_RAM_HEADROOM_GB"], "16")
        self.assertEqual(env["STRATA_HIPBLASLT_TUNING"], str(tables / "gfx1100-glm-hipblaslt-100202.txt"))

    def test_gpus_with_an_apu_is_rejected(self):
        self.gpus.append({"index": 3, "arch": "gfx1151", "vendor": "amd", "name": "Radeon 8060S", "vram_gb": 4})
        self.a.gpu = None
        self.a.gpus = "1,3"
        with self.assertRaises(SystemExit):
            maya.check_pc(self.a)
        self.a.gpus = None
        self.a.gpu = 3
        self.assertEqual(maya.check_pc(self.a)["gpus"], [self.gpus[3]])

    def test_strix_halo_defaults_use_system_ram_not_vram(self):
        self.gpus.append({"index": 3, "arch": "gfx1151", "vendor": "amd",
                          "name": "AMD Radeon (gfx1151)", "vram_gb": 0.5})
        self.a.gpu = 3
        tables = self.root / "tools/hip"
        tables.mkdir(parents=True)
        table = tables / "gfx1151-glm-hipblaslt-100202.txt"
        table.write_text("STRATA_HIPBLASLT_TUNING_V1 gfx1151 100202\n")
        pc = maya.check_pc(self.a)
        self.assertEqual(pc["gpus"], [self.gpus[3]])
        self.assertTrue(maya.hip_unified_memory(pc["gpus"][0]))
        for ram, budget in ((64, "4096"), (96, "6144"), (128, "6144")):
            with self.subTest(ram=ram), patch.object(maya, "mem_gb", return_value=(ram, ram - 8)):
                p = maya.write_config(self.a, pc, {}, self.root / "pack", "test", 8192,
                                      self.root / "data", None)
                env = json.loads(p.read_text())["env"]
                self.assertEqual(env["STRATA_GLM_SPLIT"], "0")
                self.assertEqual(env["STRATA_GLM_RESERVE_MB"], "1024")
                self.assertEqual(env["STRATA_GLM_RAM_HEADROOM_GB"], "16")
                self.assertEqual(env["STRATA_GLM_PREFILL_SUB"], "1024")
                self.assertEqual(env["STRATA_GLM_PREFILL_MB"], budget)
                self.assertEqual(env["STRATA_HIPBLASLT_TUNING"], str(table))
                self.assertNotIn("STRATA_GLM_POOL_GB", env)
                self.assertNotIn("STRATA_GLM_RAM_GB", env)
                self.assertNotIn("STRATA_GLM_PREFILL_CHUNK", env)

    def test_strix_halo_explicit_settings_win(self):
        self.gpus.append({"index": 3, "arch": "gfx1151", "vendor": "amd",
                          "name": "Radeon 8060S", "vram_gb": 112})
        self.a.gpu = 3
        explicit = {"STRATA_GLM_RESERVE_MB": "2048", "STRATA_GLM_RAM_HEADROOM_GB": "8",
                    "STRATA_GLM_POOL_GB": "84", "STRATA_GLM_RAM_GB": "4",
                    "STRATA_GLM_PREFILL_SUB": "512", "STRATA_GLM_PREFILL_MB": "3072",
                    "STRATA_HIPBLASLT_TUNING": "/custom/table.txt"}
        self.a.env = [f"{k}={v}" for k, v in explicit.items()]
        p = maya.write_config(self.a, maya.check_pc(self.a), {}, self.root / "pack", "test", 8192,
                              self.root / "data", None)
        env = json.loads(p.read_text())["env"]
        for k, v in explicit.items():
            self.assertEqual(env[k], v)

    def test_discrete_r9700_defaults_unchanged(self):
        self.a.gpu = 1
        p = maya.write_config(self.a, maya.check_pc(self.a), {}, self.root / "pack", "test", 8192,
                              self.root / "data", None)
        self.assertEqual(json.loads(p.read_text())["env"], {
            "STRATA_GLM_SPLIT": "0", "STRATA_GLM_RESERVE_MB": "3072", "STRATA_GLM_RAM_HEADROOM_GB": "16",
            "STRATA_GLM_PREFILL_SUB": "1024", "STRATA_GLM_PREFILL_MB": "4096"})
        self.assertFalse(maya.hip_unified_memory(self.gpus[1]))

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
