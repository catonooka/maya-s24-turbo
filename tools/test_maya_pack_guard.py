"""Maya's guard against a pack whose model files changed after it was built (#155: built while a download manager was
still filling preallocated files): the record in the pack, the check of packs from before it, the sha256 a model file
is checked against once, and the setup building such a pack again; no GPU, network or model downloads required.

    python -m unittest tools.test_maya_pack_guard
"""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya


class PackGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.shards = [self.d / f"GLM-5.3-Flash-test-0000{i}-of-00002.gguf" for i in (1, 2)]
        for s in self.shards:
            s.write_bytes(b"GGUF" + bytes(1024))
        self.pack = self.d / "pack"
        for f in maya.PACK_FILES:
            (self.pack / f).parent.mkdir(parents=True, exist_ok=True)
            (self.pack / f).write_text("")
        (self.pack / "native_experts.txt").write_text(
            f"# native experts (absolute offsets in {self.shards[0].name}, ...)\n")

    def age(self, path: Path, seconds: float):
        t = path.stat().st_mtime + seconds
        os.utime(path, (t, t))

    def test_recorded_pack_sees_a_changed_file(self):
        maya.stamp_pack(self.pack, self.shards)
        self.assertIsNone(maya.pack_stale(self.pack, self.shards))
        with open(self.shards[1], "ab") as f:          # the download manager wrote more of it
            f.write(b"x")
        self.assertIn(self.shards[1].name, maya.pack_stale(self.pack, self.shards))

    def test_recorded_pack_sees_a_rewritten_file_of_the_same_size(self):
        maya.stamp_pack(self.pack, self.shards)
        self.age(self.shards[0], 5)
        self.assertIn(self.shards[0].name, maya.pack_stale(self.pack, self.shards))

    def test_pack_from_before_the_record(self):
        for s in self.shards:                            # the files older than the pack: fine
            self.age(s, -3600)
        self.assertIsNone(maya.pack_stale(self.pack, self.shards))
        self.age(self.shards[1], 3600 + 600)             # a file written after the pack
        self.assertIn(self.shards[1].name, maya.pack_stale(self.pack, self.shards))

    def test_sha256_is_read_once(self):
        import hashlib
        want = hashlib.sha256(self.shards[0].read_bytes()).hexdigest()
        self.assertTrue(maya.sha256_checked(self.shards[0], want))
        with patch.object(maya, "sha256_ok", side_effect=AssertionError("read again")):
            self.assertTrue(maya.sha256_checked(self.shards[0], want))
        with open(self.shards[0], "ab") as f:            # changed: read again, and no longer the published file
            f.write(b"x")
        self.assertFalse(maya.sha256_checked(self.shards[0], want))

    def test_setup_builds_a_stale_pack_again(self):
        built = []

        def fake_run(cmd, **kw):
            built.append(cmd)
            return SimpleNamespace(returncode=0)

        a = SimpleNamespace(repack=False)
        maya.stamp_pack(self.pack, self.shards)
        with patch.object(maya, "run", side_effect=fake_run), patch.object(maya, "say"), \
                patch.object(maya, "ok"), patch.object(maya, "warn"), patch.object(maya, "step"):
            self.assertEqual(maya.pack_step(a, self.d, self.shards, self.d), self.pack)
            self.assertEqual(built, [])                  # unchanged: kept
            with open(self.shards[0], "ab") as f:
                f.write(b"x")
            maya.pack_step(a, self.d, self.shards, self.d)
            self.assertEqual(len(built), 1)              # changed: built again
            self.assertIsNone(maya.pack_stale(self.pack, self.shards))   # and recorded anew


if __name__ == "__main__":
    unittest.main()
