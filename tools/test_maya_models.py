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


if __name__ == "__main__":
    unittest.main()
