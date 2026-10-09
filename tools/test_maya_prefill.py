"""Maya's prompt-chunk setting in the configs it writes (Strata's --prefill engine arg: auto, kept when edited) and
Strata's --prefill tips; no GPU, network or model downloads required.

    python -m unittest tools.test_maya_prefill
"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import maya
from gguf_writer import GGUFWriter


def gguf(path: Path, names: list[str]) -> Path:
    w = GGUFWriter()
    w.add("general.architecture", "glm5next")
    for n in names:
        w.add_f32(n, np.zeros(4, dtype=np.float32))
    return w.write(path)


class PrefillConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "models" / "mine"
        (self.model / "pack").mkdir(parents=True)
        # a "model": its embedding and many layers' tensors
        self.first = gguf(self.model / "GLM-5.3-Flash-test.gguf",
                          ["token_embd.weight"] + [f"blk.{i}.ffn_norm.weight" for i in range(250)])
        for ctx in (
            patch.object(maya, "ROOT", self.root),
            patch.object(maya, "EXE", self.root / "build/strata"),
            patch.object(maya, "CAL_STORE", self.root / "calibration.json"),
            patch.object(maya, "WIN", False),
            patch.object(maya, "say"),
            patch.object(maya, "step"),
            patch.object(maya.S, "gpus", return_value=[]),
            patch.object(maya.S, "cpu_info", return_value=("test CPU", True, True)),
            patch.object(maya.S, "ram_gb", return_value=177),
            patch.object(maya, "mem_gb", return_value=(177.0, 150.0)),
        ):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.a = SimpleNamespace(env=[], port=None, host=None, api_key=None, gguf_dir=None)

    def write(self, gpus):
        pc = {"gpus": [{"index": i, "name": "GPU", "vram_gb": 24} for i in gpus]}
        p = maya.write_config(self.a, pc, {}, self.model / "pack", "test", 32768, self.root / "models", None,
                              ("local", self.first))
        return p, json.loads(p.read_text())["args"]

    def test_prefill_auto_is_written_and_an_edit_kept(self):
        for gpus in ([0], [0, 1]):
            p, args = self.write(gpus)
            self.assertEqual(args[args.index("--prefill") + 1], "auto")
            p.unlink()
        p, args = self.write([0])
        c = json.loads(p.read_text())
        c["args"][c["args"].index("--prefill") + 1] = "32768"  # a bigger chunk, set by hand
        p.write_text(json.dumps(c))
        _, again = self.write([0])
        self.assertEqual(again[again.index("--prefill") + 1], "32768")
        self.assertEqual(again.count("--prefill"), 1)

    def test_prefill_tips_follow_strata(self):
        self.assertEqual(maya.prefill_tips(["--prefill", "auto"], 177, 1), [])      # one GPU: auto takes 32768
        self.assertEqual(maya.prefill_tips(["--prefill", "32768"], 32, 1), [])
        self.assertIn("--prefill 32768", maya.prefill_tips(["--prefill", "auto"], 177)[0])
        self.assertEqual(maya.prefill_tips(["--prefill", "auto"], 64), [])
        self.assertIn("warning", maya.prefill_tips(["--prefill", "32768"], 32)[0])
        self.assertEqual(maya.prefill_tips(["--prefill", "32768"], 177), [])
        self.assertEqual(maya.prefill_tips(["--prefill", "4096"], 32), [])



if __name__ == "__main__":
    unittest.main()
