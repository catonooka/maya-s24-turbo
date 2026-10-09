"""tools/test_maya_models.py - the setup's model downloads: what a setup is offered (Project Maya's four quants, the
24 GB recommendation) and that every download names a hash per file.  No GPU, no network."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya  # noqa: E402


class ModelChoice(unittest.TestCase):
    def choose(self, inst, vram_gb=32.0):
        args = SimpleNamespace(gguf_dir=None, model=None, yes=True)
        lines = []
        with tempfile.TemporaryDirectory() as d, \
                patch.object(maya.S, "gpus", return_value=[{"vram_gb": vram_gb}]), \
                patch.object(maya, "say", side_effect=lambda *a: lines.append(" ".join(map(str, a)))):
            kind, quant = maya.choose_model(args, Path(d), inst)
        return kind, quant, "\n".join(lines)

    def test_new_setup_offers_the_four_maya_quants(self):
        kind, quant, text = self.choose({})
        self.assertEqual((kind, quant), ("download", "Maya-S-v2-IQ2_XXS"))
        for q in ("Maya-S-v2-IQ2_XXS:", "Maya-S24:", "Maya-M:", "Maya-L:"):
            self.assertIn(q, text)
        self.assertNotIn("GSQ-RCO", text)

    def test_24gb_cards_get_maya_s24(self):
        self.assertEqual(self.choose({}, vram_gb=24.0)[1], "Maya-S24")

    def test_a_setup_of_a_model_no_longer_offered_gets_the_recommendation(self):
        # GSQ-RCO 3.5-bit is no longer a download: setting up again recommends a Maya quant (--gguf-dir keeps the files)
        kind, quant, text = self.choose({"quant": "GSQ-RCO-3.5bit"})
        self.assertEqual(quant, "Maya-S-v2-IQ2_XXS")
        self.assertNotIn("GSQ-RCO", text)

    def test_an_earlier_maya_download_stays_the_default(self):
        self.assertEqual(self.choose({"quant": "Maya-L"})[1], "Maya-L")

    def test_every_download_names_a_hash_per_shard(self):
        for q, m in maya.MODELS.items():
            names = {m["file"].format(i=i, n=m["shards"]) for i in range(1, m["shards"] + 1)}
            self.assertEqual(set(m["sha256"]), names, q)
            self.assertTrue(all(len(h) == 64 for h in m["sha256"].values()), q)


class RestartAfterUpdate(unittest.TestCase):
    """The dashboard's Update ends the server with UPDATE_EXIT: maya.py starts the new version - the same model and
    settings, no setup flags, no question - and nothing else (no download, no pack)."""

    def start(self, rc, argv=("maya.py", "--setup", "--model", "Maya-L", "--port", "8090")):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for f in ("strata", "pack/x", "tok/vocab.json"):
                (d / f).parent.mkdir(parents=True, exist_ok=True)
                (d / f).write_text("x")
            cfg = d / "maya-test.json"
            cfg.write_text('{"exe": "%s", "tokenizer": "%s", "args": ["--glm-pack", "%s"]}'
                           % ((d / "strata").as_posix(), (d / "tok").as_posix(), (d / "pack").as_posix()))
            a = SimpleNamespace(port=8090, host=None, api_key=None, gpu=None, gpus="0,1", backend="cuda")
            with patch.object(maya, "refresh_engine") as refresh, patch.object(maya, "say"),                     patch.object(maya.sys, "argv", list(argv)),                     patch.object(maya.subprocess, "call", return_value=rc) as call,                     patch.object(maya.os, "execv") as execv, patch.object(maya, "WIN", False):
                out = maya.start(cfg, a)
            return out, call, execv, refresh

    def test_update_exit_starts_the_new_version(self):
        _, call, execv, refresh = self.start(maya.UPDATE_EXIT)
        self.assertEqual(call.call_args.kwargs["env"]["MAYA_RESTART_ON_UPDATE"], "1")
        refresh.assert_called_once()                    # (the new maya.py compiles what changed when it starts)
        argv = execv.call_args.args[1]
        self.assertEqual(argv[1:3], [str(maya.HERE / "maya.py"), "--yes"])
        self.assertNotIn("--setup", argv)               # no setup: the same model, nothing downloaded or packed
        self.assertNotIn("--model", argv)
        for flag, v in (("--port", "8090"), ("--gpus", "0,1"), ("--backend", "cuda")):
            self.assertEqual(argv[argv.index(flag) + 1], v)

    def test_other_exits_end_as_before(self):
        out, _, execv, _ = self.start(0)
        self.assertEqual(out, 0)
        execv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
