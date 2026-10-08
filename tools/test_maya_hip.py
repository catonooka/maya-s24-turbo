"""Maya's HIP setup regressions; no GPU, network or model downloads required."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

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

    def installed_config(self, backend="hip", gpu=None):
        tp = self.root / "tokenizer"
        tp.mkdir(exist_ok=True)
        (tp / "vocab.json").write_text('{"hello": 0}')
        (tp / "merges.txt").write_text("")
        (tp / "token_type.json").write_text("[1]")
        # A stale CUDA executable in a HIP config must not select the CUDA build.
        cfg = {"exe": str(self.root / "build/strata"), "args": ["--max-context", "16384"],
               "tokenizer": str(tp), "gpu": [0] if gpu is None else gpu,
               "env": {"STRATA_GLM_PREFILL_SUB": "512", "STRATA_GLM_RAM_GB": "60"},
               "lib_dirs": [str(self.rocm / "lib")]}
        if backend == "hip":
            cfg["backend"] = backend
        path = self.root / f"maya-test-{backend}.json"
        path.write_text(json.dumps(cfg))
        return path

    def run_bench(self, path):
        proc = MagicMock()
        proc.stdout.readline.side_effect = ["READY\n"] + ["DONE 256 0 1000 2000\n"] * 7
        tok = MagicMock()
        tok.encode.return_value = list(range(9000))
        # Exercise the real server's engine_args/child_env without optional tokenizer/template packages.
        modules = {"strata_tokenizer": SimpleNamespace(Tokenizer=MagicMock(return_value=tok)),
                   "serve.frontend": MagicMock()}
        with patch("urllib.request.urlopen", side_effect=OSError("no server")), \
                patch.dict(sys.modules, modules), \
                patch.object(maya.S, "out", side_effect=AssertionError("bench must not run GPU queries")), \
                patch.object(maya.subprocess, "Popen", return_value=proc) as popen, \
                patch.dict(os.environ, {}, clear=True):
            self.assertEqual(maya.bench(path, "test"), 0)
        proc.wait.assert_called_once_with(timeout=120)
        self.assertIn("QUIT\n", [call.args[0] for call in proc.stdin.write.call_args_list])
        text = (self.root / "maya-bench.txt").read_text()
        self.assertIn("128.0 tokens/s over 3 answers", text)
        self.assertIn("2048 tokens/s, 2048 tokens", text)
        self.assertIn("8192 tokens/s, 8192 tokens", text)
        return popen.call_args, text

    def test_bench_uses_hip_build_and_config_env(self):
        call, text = self.run_bench(self.installed_config(gpu=[1, 0]))
        self.assertEqual(maya.BUILD, self.root / "build-hip")
        self.assertEqual(maya.STAMP, self.root / "build-hip/MAYA-BUILD.json")
        self.assertEqual(call.args[0][:2], [str(self.root / "build-hip/strata"), "--serve"])
        self.assertEqual(call.args[0][-2:], ["--layer-split", "auto"])
        env = call.kwargs["env"]
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "1,0")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
        self.assertNotIn("CUDA_DEVICE_ORDER", env)
        self.assertEqual(env["STRATA_GLM_PREFILL_SUB"], "512")
        self.assertEqual(env["STRATA_GLM_RAM_GB"], "60")
        self.assertEqual(env["LD_LIBRARY_PATH"], str(self.rocm / "lib"))
        self.assertIn("GPUs: R9700 32 GB, RX 7900 XT 20 GB", text)
        self.assertIn("GPUs [1, 0]", text)

    def test_bench_cuda_command_and_env_unchanged(self):
        path = self.installed_config(backend="cuda", gpu=[1, 0])
        cfg = json.loads(path.read_text())
        cfg["exe"] = str(self.root / "custom-cuda/strata")
        path.write_text(json.dumps(cfg))
        with patch.object(maya.S, "gpus", return_value=[{"name": "V100", "vram_gb": 16}]), \
                patch.object(maya.S, "amd_gpus", side_effect=AssertionError("CUDA must not query AMD")):
            call, text = self.run_bench(path)
        self.assertEqual(call.args[0][:2], [cfg["exe"], "--serve"])
        self.assertEqual(maya.BUILD, self.root / "build")
        self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1,0")
        self.assertEqual(call.kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertNotIn("HIP_VISIBLE_DEVICES", call.kwargs["env"])
        self.assertIn("GPUs: V100 16 GB", text)

    def test_bench_single_hip_gpu(self):
        call, text = self.run_bench(self.installed_config(gpu=0))
        self.assertEqual(call.kwargs["env"]["HIP_VISIBLE_DEVICES"], "0")
        self.assertNotIn("--layer-split", call.args[0])
        self.assertIn("GPUs: RX 7900 XT 20 GB;", text)

    def run_report(self, outputs=None, tools=(), backend=None, cfg_path=None):
        outputs = outputs or {}

        def out(cmd):
            self.assertNotEqual(Path(cmd[0]).name, "nvidia-smi", "HIP report must not query NVIDIA")
            return outputs.get(Path(cmd[0]).name, "")

        with patch.object(maya.S, "out", side_effect=out) as query, \
                patch.object(maya.S, "page_file_gb", return_value=None), \
                patch.object(maya.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}" if name in tools else None):
            self.assertEqual(maya.report("test", backend, cfg_path), 0)
        return (self.root / "maya-report.txt").read_text(), query

    def test_report_rocm_smi_json_and_engine_speed_lines(self):
        cfg_path = self.installed_config()
        log = cfg_path.with_suffix(".log")
        speed = ["glm fast: CUDA0 expert pool 12 GB", "glm prefill: CUDA0 chunks of 1024 tokens",
                 "glm prefill: HIP0 chunks of 1024 tokens", "glm prefill: 2048 tokens at 413 tok/s",
                 "glm stat: decode 50.0 ms/tok | prompt 400 tok/s",
                 "glm stat: decode 0.1 ms/tok vram hit 0.00% | prompt 413 tok/s"]
        log.write_text("\n".join(speed) + "\n")
        (self.root / "build-hip").mkdir()
        (self.root / "build-hip/MAYA-BUILD.json").write_text(json.dumps({
            "backend": "hip", "archs": ["gfx1100", "gfx1201"], "rocm": str(self.rocm), "date": "test-date"}))
        (self.rocm / ".info").mkdir()
        (self.rocm / ".info/version").write_text("7.2.1\n")
        # SMI names differ from KFD: keep both sources without assuming their device order matches.
        smi = {"card0": {"Card series": "AMD Radeon RX 7900 XT", "VRAM Total Memory (B)": str(20 * 2**30)},
               "card1": {"Card series": "AMD Radeon AI PRO R9700", "VRAM Total Memory (B)": str(32 * 2**30)},
               "system": {"Driver version": "6.14.0-37"}}
        text, query = self.run_report({"rocm-smi": json.dumps(smi)}, tools=("rocm-smi",))
        self.assertIn("GPU 0 (RX 7900 XT, 20 GB, gfx1100)", text)
        self.assertIn("GPU 1 (R9700, 32 GB, gfx1201)", text)
        self.assertIn("AMD Radeon RX 7900 XT", text)
        self.assertIn("AMD Radeon AI PRO R9700", text)
        self.assertIn(str(32 * 2**30), text)
        self.assertIn("6.14.0-37", text)
        self.assertIn(f"path {self.rocm}\nversion 7.2.1", text)
        self.assertIn('"backend": "hip"', text)
        self.assertIn("test-date", text)
        self.assertIn("the speed and memory lines (last 80 of 6)", text)
        for line in maya.speed_lines(speed):
            self.assertIn(line, text)
        query.assert_any_call(["/usr/bin/rocm-smi", "--showproductname", "--showmeminfo", "vram",
                               "--showdriverversion", "--json"])

    def test_report_kfd_fallback_without_smi_or_rocm_version(self):
        self.installed_config()
        text, query = self.run_report()
        self.assertIn("GPU 0 (RX 7900 XT, 20 GB, gfx1100)", text)
        self.assertIn("GPU 1 (R9700, 32 GB, gfx1201)", text)
        self.assertIn("amdgpu driver:", text)
        self.assertIn("kernel", text)
        self.assertIn("using KFD topology", text)
        self.assertIn(f"path {self.rocm}", text)
        self.assertIn("version unknown", text)
        self.assertFalse(any(Path(c.args[0][0]).name in ("rocm-smi", "amd-smi", "hipcc")
                             for c in query.call_args_list))

    def test_report_amd_smi_after_bad_rocm_smi_json(self):
        self.installed_config()
        smi = [{"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7900 XT", "target_graphics_version": "gfx1100"},
                "vram": {"size": "20480 MB"}, "driver": {"driver_version": "6.14.0", "rocm_version": "7.2.1"}}]
        text, query = self.run_report({"rocm-smi": "not JSON", "amd-smi": json.dumps(smi)},
                                      tools=("rocm-smi", "amd-smi"))
        self.assertIn("amd-smi (tool GPU indices)", text)
        self.assertIn("AMD Radeon RX 7900 XT", text)
        self.assertIn("20480 MB", text)
        self.assertIn("6.14.0", text)
        self.assertIn("7.2.1", text)
        query.assert_any_call(["/usr/bin/amd-smi", "static", "--json"])

    def test_report_kfd_fallback_for_empty_or_invalid_smi_json(self):
        self.installed_config()
        for response in ("", "warning: no access", "{}", "[]", "null", '"error"'):
            with self.subTest(response=response):
                text, _ = self.run_report({"rocm-smi": response, "amd-smi": response},
                                          tools=("rocm-smi", "amd-smi"))
                self.assertIn("GPU 1 (R9700, 32 GB, gfx1201)", text)
                self.assertIn("using KFD topology", text)

    def test_report_finds_smi_in_rocm_bin_and_hipcc_version(self):
        self.installed_config()
        (self.rocm / "bin").mkdir()
        (self.rocm / "bin/rocm-smi").touch()
        (self.rocm / "bin/hipcc").touch()
        text, query = self.run_report({"rocm-smi": '{"system": {"Driver version": "6.14.0"}}',
                                       "hipcc": "HIP version: 7.2.1"})
        self.assertIn("rocm-smi (tool GPU indices)", text)
        self.assertIn("version HIP version: 7.2.1", text)
        query.assert_any_call([str(self.rocm / "bin/hipcc"), "--version"])

    def test_report_hip_without_config_or_gpus(self):
        with patch.object(maya.S, "amd_gpus", return_value=[]):
            text, _ = self.run_report(backend="hip")
        self.assertIn("no AMD GPUs found in KFD topology", text)
        self.assertIn("## ROCm", text)
        self.assertIn("not compiled yet", text)

    def test_report_explicit_config_selects_hip_env_and_log(self):
        self.installed_config(backend="cuda")
        path = self.installed_config().rename(self.root / "chosen.json")
        cfg = json.loads(path.read_text())
        root = self.root / "chosen-rocm"
        (root / ".info").mkdir(parents=True)
        (root / ".info/version").write_text("7.2.2\n")
        cfg["env"]["ROCM_PATH"] = str(root)
        path.write_text(json.dumps(cfg))
        path.with_suffix(".log").write_text("glm stat: decode 50.0 ms/tok | prompt 413 tok/s\n")
        text, _ = self.run_report(cfg_path=path)
        self.assertIn(f"path {root}\nversion 7.2.2", text)
        self.assertIn("Setup chosen.json", text)
        self.assertIn("Engine log chosen.log: the speed and memory lines", text)
        self.assertNotIn("Setup maya-test-cuda.json", text)

    def test_report_cuda_query_and_stamp_unchanged(self):
        self.installed_config(backend="cuda")
        (self.root / "build").mkdir()
        stamp = {"archs": [70], "nvcc": "/usr/local/cuda/bin/nvcc", "host_compiler": "g++-12", "date": "test-date"}
        (self.root / "build/MAYA-BUILD.json").write_text(json.dumps(stamp))
        with patch.object(maya.S, "out", return_value="NVIDIA V100") as query, \
                patch.object(maya.S, "page_file_gb", return_value=None), \
                patch.object(maya.S, "amd_gpus", side_effect=AssertionError("CUDA must not query AMD")):
            self.assertEqual(maya.report("test"), 0)
        query.assert_any_call(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,"
                                             "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,"
                                             "temperature.gpu", "--format=csv"])
        text = (self.root / "maya-report.txt").read_text()
        self.assertIn(json.dumps(stamp), text)
        self.assertNotIn("## ROCm", text)

    def test_cli_backend_selects_hip_config_for_bench(self):
        hip = self.installed_config()
        cuda = self.installed_config(backend="cuda")
        with patch.object(maya, "configs", return_value=[cuda, hip]), \
                patch.object(sys, "argv", ["maya.py", "--backend", "hip", "--bench"]), \
                patch.object(maya, "bench", return_value=0) as bench:
            self.assertEqual(maya.main(), 0)
        bench.assert_called_once_with(hip, (maya.HERE / "VERSION").read_text().strip())

    def test_cli_explicit_config_for_bench_and_report(self):
        path = self.installed_config()
        for action in ("bench", "report"):
            with self.subTest(action=action), \
                    patch.object(sys, "argv", ["maya.py", "--backend", "hip", f"--{action}", "--config", str(path)]), \
                    patch.object(maya, action, return_value=0) as run:
                self.assertEqual(maya.main(), 0)
                self.assertIn(path, run.call_args.args)

    def test_cli_backend_never_benchmarks_cuda_config_as_hip(self):
        self.installed_config(backend="cuda")
        with patch.object(sys, "argv", ["maya.py", "--backend", "hip", "--bench"]), \
                patch.object(maya, "bench", side_effect=AssertionError("must not start the CUDA engine")):
            with self.assertRaises(SystemExit):
                maya.main()


if __name__ == "__main__":
    unittest.main()
