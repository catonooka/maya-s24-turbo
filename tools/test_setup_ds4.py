"""tools/test_setup_ds4.py - the setup's two models: GLM-5.3-Flash or DeepSeek-V4.1-Flash-1017 (maya.py) - which one the
options pick, the DeepSeek download's manifest and URLs, its config (what a setup again keeps) and its start command.
No GPU, no network.

    python -m unittest tools.test_setup_ds4 -v
"""
from pathlib import Path
import json
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402


def args(**kw):
    base = dict(model=None, model_dir=None, gguf_dir=None, draft=None, backend="cuda", yes=True, port=None, host=None,
                api_key=None, env=[], no_vision=False, vram_reserve_mib=None, parallel=None, gpus=None, gpu=None,
                ds4_quant=None)
    base.update(kw)
    return SimpleNamespace(**base)


class Family(unittest.TestCase):
    def test_options_pick_the_model(self):
        self.assertEqual(maya.choose_family(args(model=maya.DS4), {}), "deepseek")
        self.assertEqual(maya.choose_family(args(model="Maya-L"), {}), "glm")
        self.assertEqual(maya.choose_family(args(model_dir="/x"), {}), "deepseek")
        self.assertEqual(maya.choose_family(args(gguf_dir="/x"), {}), "glm")
        self.assertEqual(maya.choose_family(args(draft="dflash2"), {}), "glm")
        self.assertEqual(maya.choose_family(args(backend="hip"), {}), "glm")   # (asked: the default, GLM)

    def test_default_is_the_model_before(self):
        ds4 = {"exe": "/m/build/maya-ds4", "args": ["--pack", "/m/maya-ds4", "--max-context", "65536"]}
        glm = {"exe": "/m/build/maya", "args": ["--glm-pack", "/m/pack"]}
        with patch.object(maya.S, "gpus", return_value=[]), patch.object(maya, "say", lambda *a: None):
            self.assertEqual(maya.choose_family(args(), ds4), "deepseek")
            self.assertEqual(maya.choose_family(args(), glm), "glm")
            self.assertEqual(maya.choose_family(args(), {}), "glm")
        self.assertEqual(maya.config_family(ds4), "deepseek")
        self.assertEqual(maya.config_family(glm), "glm")
        self.assertEqual(maya.config_family({"installer": {"family": "deepseek"}}), "deepseek")


class Manifest(unittest.TestCase):
    def test_every_file_pinned(self):
        m = maya.ds4_model()
        self.assertGreater(len(m["files"]), 45)
        for n in ["config.json", "model.safetensors.index.json", "tokenizer.json"] + \
                 [f"model-{i:05d}-of-00043.safetensors" for i in range(1, 44)] + \
                 ["model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"]:
            self.assertIn(n, m["files"])
        for n, size in m["files"].items():
            self.assertNotIn("/", n)                      # (the downloads are flat: no subfolders)
            self.assertGreater(size, 0)
            self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")
        self.assertAlmostEqual(m["download_gb"], 413.9, delta=0.5)

    def test_urls(self):
        m = maya.ds4_model()
        rev = re.compile(r"/resolve/[0-9a-f]{40}/")
        u = maya.file_url(m, "model-00001-of-00043.safetensors")
        self.assertTrue(u.startswith("https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/"))
        self.assertRegex(u, rev)
        u = maya.file_url(m, "model-00047-of-00048.safetensors")   # the Engram tables: DeepSeek's own repo
        self.assertTrue(u.startswith("https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/"))
        self.assertRegex(u, rev)
        self.assertTrue(u.endswith("/model-00047-of-00048.safetensors"))


class Config(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.patches = [patch.object(maya, "ROOT", self.root), patch.object(maya, "say", lambda *a: None),
                        patch.object(maya, "ok", lambda *a: None), patch.object(maya, "step", lambda *a: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def write(self, a, old=None):
        if old is not None:
            (self.root / "maya-ds4.json").write_text(json.dumps(old))
        pc = {"gpus": [{"index": i} for i in range(3)]}
        p = maya.write_config_ds4(a, pc, {"lib_dirs": ["/cuda/lib64"]}, Path("/m/DS/maya-ds4"), 131072,
                                  Path("/models"), Path("/m/DS"))
        return p, json.loads(p.read_text())

    def test_fresh(self):
        p, c = self.write(args())
        self.assertEqual(p.name, "maya-ds4.json")
        self.assertEqual(c["args"], ["--pack", str(Path("/m/DS/maya-ds4")), "--max-context", "131072", "--tp", "2"])
        self.assertEqual(c["exe"], str(maya.DS4_EXE))
        self.assertEqual(c["tokenizer"], str(Path("/m/DS/maya-ds4/tokenizer")))
        self.assertEqual((c["sampling"], c["reasoning_effort"]), (maya.DS4_SAMPLING, maya.DS4_EFFORT))
        self.assertEqual(c["gpu"], [0, 1, 2])
        self.assertEqual(c["installer"]["family"], "deepseek")
        self.assertEqual(maya.config_family(c), "deepseek")
        self.assertIn("maya.py", (self.root / ("run-maya-ds4" + (".bat" if maya.WIN else ".sh"))).read_text())

    def test_again_keeps_what_it_does_not_ask(self):
        old = {"args": ["--pack", "/old/maya-ds4", "--max-context", "262144", "--tp", "1", "--prefill-chunk", "4096"],
               "reasoning_effort": "max", "sampling": {"temperature": 0.6}, "host": "0.0.0.0", "api_key": "k",
               "env": {"DS4_RING_MB": "800"}}
        _, c = self.write(args(no_vision=True, vram_reserve_mib=900, env=["DS4_CHUNK_MIN=256"]), old)
        a = c["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "131072")    # (asked now)
        self.assertEqual(a[a.index("--tp") + 1], "1")
        self.assertEqual(a[a.index("--prefill-chunk") + 1], "4096")
        self.assertEqual(a[a.index("--reserve-mib") + 1], "900")
        self.assertEqual(a[a.index("--vision") + 1], "off")
        self.assertEqual((c["reasoning_effort"], c["sampling"], c["host"], c["api_key"]),
                         ("max", {"temperature": 0.6}, "0.0.0.0", "k"))
        self.assertEqual(c["env"], {"DS4_RING_MB": "800", "DS4_CHUNK_MIN": "256"})

    def test_gguf_config(self):
        pc = {"gpus": [{"index": 0}]}
        p = maya.write_config_ds4(args(), pc, {}, Path("/m/G/maya-ds4"), 65536, Path("/models"), Path("/m/G"),
                                  "gguf-q8")
        c = json.loads(p.read_text())
        self.assertEqual(c["args"][c["args"].index("--vision") + 1], "off")   # (no vision tower in a GGUF)
        self.assertEqual(c["installer"]["ds4_quant"], "gguf-q8")
        self.assertEqual(maya.config_family(c), "deepseek")


class Gguf(unittest.TestCase):
    def test_manifest_and_urls(self):
        rev = re.compile(r"/resolve/[0-9a-f]{40}/")
        for q, gb in (("gguf-q8", 508.4), ("gguf-q4", 500.7)):
            m = maya.ds4_gguf_model(q)
            self.assertEqual((m["format"], m["folder"]), ("ds4-gguf", maya.DS4_QUANTS[q]))
            self.assertEqual(len(m["files"]), 9)
            for n, size in m["files"].items():
                self.assertNotIn("/", n)
                self.assertTrue(n.endswith(".gguf"))
                self.assertGreater(size, 0)
                self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(m["download_gb"], gb, delta=0.5)
            u = maya.file_url(m, sorted(m["files"])[0])
            self.assertTrue(u.startswith("https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/"))
            self.assertRegex(u, rev)
            self.assertIn("/%s/" % maya.DS4_QUANTS[q], u)

    def test_gguf_config_with_the_vision_tower(self):
        pc = {"gpus": [{"index": 0}]}
        d = Path(tempfile.mkdtemp())
        (d / "maya-ds4").mkdir()
        (d / "maya-ds4" / "ds4.txt").write_text("hidden_size 4096\nvision.num_hidden_layers 32\nimage_token_id 1\n")
        quiet = [patch.object(maya, "ROOT", d), patch.object(maya, "say", lambda *a: None),
                 patch.object(maya, "ok", lambda *a: None), patch.object(maya, "step", lambda *a: None)]
        for q in quiet:
            q.start()
        try:
            for a, vision, off in ((args(), None, False), (args(), False, True), (args(no_vision=True), True, True)):
                p = maya.write_config_ds4(a, pc, {}, d / "maya-ds4", 65536, Path("/models"), d, "gguf-q8", vision)
                self.assertEqual("--vision" in json.loads(p.read_text())["args"], off, (a, vision))
        finally:
            for q in quiet:
                q.stop()

    def test_choice_and_folder(self):
        with patch.object(maya, "say", lambda *a: None), patch.object(maya.S, "UI", None):
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant="gguf-q4"), {}), "gguf-q4")
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant=None), {"installer": {"ds4_quant": "exl3"}}), "exl3")
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant=None), {}), "maya-s")   # (published: the default)
        self.assertEqual(maya.ds4_dir(args(), Path("/models"), {}, "gguf-q8"),
                         Path("/models") / "DeepSeek-V4.1-Flash-1017-GGUF" / "UD-Q8_K_XL")
        self.assertEqual(maya.ds4_dir(args(), Path("/models"), {}, "exl3"), Path("/models") / maya.DS4)


class Vision(unittest.TestCase):
    """A GGUF's vision tower (step 7): DeepSeek's shard 1 and config.json pinned in download_gguf.json, the picture
    markers' 30 KB of shard 2 pinned row by row, and the step's paths - skipped, already there, made, failed."""

    def test_manifest(self):
        m = maya.ds4_vision_model()
        self.assertEqual(m["repo"], "deepseek-ai/DeepSeek-V4.1-Flash-1017")
        self.assertEqual(m["files"], {"model-00001-of-00048.safetensors": 970533624, "config.json": 3311})
        self.assertEqual(m["sha256"]["model-00001-of-00048.safetensors"],
                         "13bd121075f0b68cb7e8e8021a1011d42a81a83f557a2602fd9c07d52fddb0ec")
        for n in m["files"]:
            self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")
            u = maya.file_url(m, n)
            self.assertTrue(u.startswith("https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/"
                                         "1bae0e1088993b9b82de08f59de5be5a34e42170/"), u)
        man = json.loads(maya.DS4_GGUF_MANIFEST.read_text())
        self.assertEqual(man["vision"]["revision"], man["drafter"]["revision"])   # (one checkpoint)
        mk = man["vision"]["marks"]
        rows = mk["rows"]
        self.assertEqual(sorted(rows), ["image_end", "image_newline", "image_start"])
        spans = sorted((o, o + nb) for o, nb, _ in rows.values())
        self.assertTrue(all(nb == 5120 * 2 for _, nb, _ in rows.values()))
        self.assertEqual([a for a, _ in spans[1:]], [b for _, b in spans[:-1]])   # (back to back: one range)
        self.assertEqual(spans[-1][1], mk["bytes"])                               # (the file's last 30 KB)
        self.assertLess(spans[-1][1] - spans[0][0], 1 << 20)

    def setUp(self):
        self.models = Path(tempfile.mkdtemp())
        self.d = self.models / "DeepSeek-V4.1-Flash-1017-GGUF" / "UD-Q8_K_XL"
        self.d.mkdir(parents=True)
        self.v = maya.ds4_vision_dir(self.d)
        self.quiet = [patch.object(maya, "say", lambda *a: None), patch.object(maya, "ok", lambda *a: None),
                      patch.object(maya, "warn", lambda *a: None), patch.object(maya, "step", lambda *a: None),
                      patch.object(maya, "mem_gb", lambda: (64.0, 50.0))]
        for q in self.quiet:
            q.start()

    def tearDown(self):
        for q in self.quiet:
            q.stop()

    def test_folder_is_shared_by_the_quants(self):
        self.assertEqual(self.v, self.models / "DeepSeek-V4.1-Flash-1017-GGUF" / "vision")
        self.assertEqual(maya.ds4_vision_dir(self.d.parent / "UD-Q4_K_XL"), self.v)

    def test_no_vision_downloads_nothing(self):
        with patch.object(maya, "offer_download", side_effect=AssertionError("asked")):
            self.assertFalse(maya.ds4_vision_step(args(no_vision=True), {}, self.d))
        self.assertFalse(self.v.exists())

    def test_already_there(self):
        self.v.mkdir()
        (self.v / "vision.bin").write_bytes(b"x")
        (self.v / "vision.json").write_text("{}")
        with patch.object(maya, "offer_download", side_effect=AssertionError("asked")):
            self.assertTrue(maya.ds4_vision_step(args(), {}, self.d))

    def fake_download(self, a, what, d, files, m):
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            f.write_bytes(b"s" * 8)
        return True

    def fake_model(self):
        import hashlib
        return {"files": {"model-00001-of-00048.safetensors": 8, "config.json": 8},
                "sha256": {n: hashlib.sha256(b"s" * 8).hexdigest() for n in ("model-00001-of-00048.safetensors",
                                                                              "config.json")},
                "repo": "r", "revision": "v", "folder": "", "download_gb": 0.0, "about": "x"}

    def test_made_then_the_shard_is_removed(self):
        def pack(cmd, check=True):
            self.assertIn("vision_pack.py", str(cmd[1]))
            self.assertEqual(cmd[cmd.index("--dir") + 1], str(self.v))
            (self.v / "vision.bin").write_bytes(b"v")
            (self.v / "vision.json").write_text("{}")
            return SimpleNamespace(returncode=0)
        with patch.object(maya, "ds4_vision_model", self.fake_model), \
                patch.object(maya, "offer_download", self.fake_download), patch.object(maya, "run", pack):
            self.assertTrue(maya.ds4_vision_step(args(), {}, self.d))
        self.assertFalse((self.v / "model-00001-of-00048.safetensors").exists())
        self.assertTrue((self.v / "vision.bin").exists())

    def test_failed_markers_keep_the_shard_and_read_text(self):
        with patch.object(maya, "ds4_vision_model", self.fake_model), \
                patch.object(maya, "offer_download", self.fake_download), \
                patch.object(maya, "run", lambda cmd, check=True: SimpleNamespace(returncode=1)):
            self.assertFalse(maya.ds4_vision_step(args(), {}, self.d))
        self.assertTrue((self.v / "model-00001-of-00048.safetensors").exists())   # (not downloaded again next time)

    def test_not_downloaded(self):
        with patch.object(maya, "offer_download", lambda *a: False):
            self.assertFalse(maya.ds4_vision_step(args(), {}, self.d))

    def test_pack_rebuilt_when_the_tower_arrives(self):
        pack = self.d / "maya-ds4"
        (pack / "tokenizer").mkdir(parents=True)
        (self.d / "m-00001-of-00001.gguf").write_bytes(b"g")
        for f in ("ds4.txt", "tensors.txt", "engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja"):
            (pack / f).write_text("hidden_size 4096\n")
        maya.stamp_pack(pack, maya.ds4_shards(self.d))
        rec = json.loads((pack / maya.PACK_SOURCES).read_text())
        rec["tools"] = maya.S.source_hash(maya.DS4_GGUF_PACK_TOOLS)
        (pack / maya.PACK_SOURCES).write_text(json.dumps(rec))
        calls = []
        with patch.object(maya, "run", lambda cmd, **k: calls.append(cmd)):
            maya.ds4_pack_step(args(repack=False), self.d, gguf=True)
        self.assertEqual(calls, [])                    # (built, and no tower to add)
        self.assertFalse(maya.ds4_pack_has_vision(pack))
        self.v.mkdir()
        (self.v / "vision.json").write_text("{}")
        with patch.object(maya, "run", lambda cmd, **k: calls.append(cmd)):
            maya.ds4_pack_step(args(repack=False), self.d, gguf=True, announce=False)
        self.assertEqual(len(calls), 1)                # (built again: the tower arrived after it)
        self.assertEqual(calls[0][calls[0].index("--vision") + 1], str(self.v))


class MayaQuants(unittest.TestCase):
    """Maya's quants of DeepSeek: offered once download_gguf.json lists their files, downloaded into
    <...-GGUF>/Maya-S, their Engram tables from unsloth's Engram-only shard into <...-GGUF>/Engram (hard-linked
    from an unsloth folder that already holds them)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.quiet = [patch.object(maya, n, lambda *a, **k: None) for n in ("say", "ok", "warn", "step")]
        for q in self.quiet:
            q.start()

    def tearDown(self):
        for q in self.quiet:
            q.stop()

    def unpublished(self):
        man = json.loads(maya.DS4_GGUF_MANIFEST.read_text(encoding="utf-8"))
        man["maya"]["revision"] = ""
        for q in man["maya"]["quants"].values():
            q["files"] = {}
        p = self.tmp / "download_gguf_unpublished.json"
        p.write_text(json.dumps(man), encoding="utf-8")
        return patch.object(maya, "DS4_GGUF_MANIFEST", p)

    def test_pinned_maya_s(self):
        m = maya.ds4_gguf_model("maya-s")
        self.assertEqual((m["folder"], len(m["files"])), ("Maya-S", 4))
        self.assertRegex(m["revision"], r"^[0-9a-f]{40}$")
        self.assertAlmostEqual(m["download_gb"], 178.19, delta=0.01)
        for n in m["files"]:
            self.assertRegex(n, r"^DeepSeek-V4\.1-Flash-1017-Maya-S-IQ2_XXS-0000[1-4]-of-00004\.gguf$")
            self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")
        self.assertFalse(maya.ds4_maya_published("maya-m"))
        d = maya.ds4_maya_drafter_model()
        self.assertEqual((d["revision"], sorted(d["files"])), (m["revision"], ["dspark.bin", "dspark.json"]))

    def published(self):
        man = json.loads(maya.DS4_GGUF_MANIFEST.read_text(encoding="utf-8"))
        man["maya"]["revision"] = "0123456789abcdef0123456789abcdef01234567"
        man["maya"]["quants"]["Maya-S"]["files"] = {
            "Maya-S/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00001-of-00002.gguf": [48_000_000_000, "a" * 64],
            "Maya-S/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00002-of-00002.gguf": [30_000_000_000, "b" * 64]}
        p = self.tmp / "download_gguf.json"
        p.write_text(json.dumps(man), encoding="utf-8")
        return patch.object(maya, "DS4_GGUF_MANIFEST", p)

    def test_not_offered_before_they_are_published(self):
        with self.unpublished(), patch.object(maya.S, "UI", None):
            self.assertFalse(maya.ds4_maya_published("maya-s"))
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant=None), {}), maya.DS4_DEFAULT_QUANT)
            with self.assertRaises(SystemExit):
                maya.ds4_gguf_model("maya-s")

    def test_published(self):
        with self.published(), patch.object(maya.S, "UI", None):
            self.assertTrue(maya.ds4_maya_published("maya-s"))
            self.assertFalse(maya.ds4_maya_published("maya-m"))
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant=None), {}), "maya-s")   # (recommended)
            self.assertEqual(maya.choose_ds4_quant(args(ds4_quant=None), {"installer": {"ds4_quant": "gguf-q8"}}),
                             "gguf-q8")
            m = maya.ds4_gguf_model("maya-s")
            self.assertEqual((m["format"], m["folder"], len(m["files"])), ("ds4-gguf", "Maya-S", 2))
            self.assertAlmostEqual(m["download_gb"], 78.0, delta=0.01)
            u = maya.file_url(m, sorted(m["files"])[0])
            self.assertEqual(u, "https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF/resolve/"
                                "0123456789abcdef0123456789abcdef01234567/Maya-S/"
                                "DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00001-of-00002.gguf")
        self.assertEqual(maya.ds4_dir(args(), Path("/models"), {}, "maya-s"),
                         Path("/models") / "DeepSeek-V4.1-Flash-1017-GGUF" / "Maya-S")

    def test_drafter_from_mayas_repo_once_pinned(self):
        with self.unpublished():
            self.assertIsNone(maya.ds4_maya_drafter_model())   # (no revision pinned: DeepSeek's shards instead)
        with self.published():
            m = maya.ds4_maya_drafter_model()
            self.assertEqual(sorted(m["files"]), ["dspark.bin", "dspark.json"])
            self.assertEqual(m["folder"], "dspark")
            self.assertTrue(maya.file_url(m, "dspark.bin").endswith("/dspark/dspark.bin"))
            d = self.tmp / "dspark"
            with patch.object(maya, "offer_download", return_value=True) as dl, \
                    patch.object(maya, "run", side_effect=AssertionError("converted")):
                self.assertTrue(maya.ds4_drafter_step(args(), d))
            self.assertEqual(dl.call_count, 1)

    def test_engram_model(self):
        m = maya.ds4_engram_model()
        self.assertEqual(sorted(m["files"]), ["DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00008-of-00009.gguf"])
        self.assertAlmostEqual(m["download_gb"], 202.8, delta=0.1)
        u = maya.file_url(m, sorted(m["files"])[0])
        self.assertRegex(u, r"^https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/[0-9a-f]{40}/"
                            r"UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00008-of-00009\.gguf$")

    def test_engram_linked_from_an_unsloth_folder(self):
        import hashlib
        base = self.tmp / "DeepSeek-V4.1-Flash-1017-GGUF"
        (base / "UD-Q4_K_XL").mkdir(parents=True)
        names = ["DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00008-of-00010.gguf",
                 "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00009-of-00010.gguf"]
        body = {n: n.encode() * 3 for n in names}
        for n in names:                                # (UD-Q4_K_XL's names, the same bytes)
            (base / "UD-Q4_K_XL" / n.replace("UD-Q8_K_XL", "UD-Q4_K_XL")).write_bytes(body[n])
        m = {"about": "x", "format": "ds4-engram", "repo": "r", "revision": "v", "folder": "UD-Q8_K_XL",
             "files": {n: len(body[n]) for n in names},
             "sha256": {n: hashlib.sha256(body[n]).hexdigest() for n in names}, "download_gb": 0.0}
        with patch.object(maya, "ds4_engram_model", return_value=m), \
                patch.object(maya, "offer_download", side_effect=AssertionError("asked")):
            self.assertTrue(maya.ds4_engram_step(args(), base / "Engram"))
        for n in names:
            self.assertTrue(Path.samefile(base / "Engram" / n, base / "UD-Q4_K_XL" / n.replace("UD-Q8_K_XL", "UD-Q4_K_XL")))

    def test_engram_downloaded_when_nowhere(self):
        m = {"about": "x", "format": "ds4-engram", "repo": "r", "revision": "v", "folder": "UD-Q8_K_XL",
             "files": {"e-00008.gguf": 4}, "sha256": {"e-00008.gguf": "0" * 64}, "download_gb": 0.0}
        with patch.object(maya, "ds4_engram_model", return_value=m), \
                patch.object(maya, "offer_download", return_value=False) as dl:
            self.assertFalse(maya.ds4_engram_step(args(), self.tmp / "Engram"))
        self.assertEqual(dl.call_count, 1)


class Rename(unittest.TestCase):
    """v1.1.16's program names: an earlier version's programs renamed in place, the stamps and configs follow."""

    def test_programs_and_stamp(self):
        root = Path(tempfile.mkdtemp())
        ext = ".exe" if maya.WIN else ""
        (root / "build").mkdir()
        (root / "build-vision" / "bin").mkdir(parents=True)
        (root / "build" / ("strata" + ext)).write_text("glm")
        (root / "build" / ("strata-ds4" + ext)).write_text("ds4")
        (root / "build-vision" / "bin" / ("strata-vision" + ext)).write_text("vision")
        (root / "build" / "MAYA-BUILD.json").write_text(json.dumps({"src": "s", "targets": ["strata", "strata-ds4"]}))
        with patch.object(maya, "ROOT", root), patch.object(maya, "VBUILD", root / "build-vision"):
            maya.rename_programs()
            maya.rename_programs()   # (again: nothing left to do)
        self.assertEqual((root / "build" / ("maya" + ext)).read_text(), "glm")
        self.assertEqual((root / "build" / ("maya-ds4" + ext)).read_text(), "ds4")
        self.assertEqual((root / "build-vision" / "bin" / ("maya-vision" + ext)).read_text(), "vision")
        self.assertFalse((root / "build" / ("strata" + ext)).exists())
        self.assertEqual(json.loads((root / "build" / "MAYA-BUILD.json").read_text())["targets"], ["maya", "maya-ds4"])

    def test_config(self):
        m = Path(tempfile.mkdtemp())
        (m / "build").mkdir()
        (m / "build-vision" / "bin").mkdir(parents=True)
        for p in ("build/maya.exe", "build/maya-ds4", "build-vision/bin/maya-vision"):
            (m / p).write_text("")
        cfg = {"exe": str(m / "build" / "strata.exe"), "vision": {"exe": str(m / "build-vision/bin/strata-vision")}}
        self.assertTrue(maya.fix_config_names(cfg))
        self.assertEqual(Path(cfg["exe"]).name, "maya.exe")
        self.assertEqual(Path(cfg["vision"]["exe"]).name, "maya-vision")
        self.assertFalse(maya.fix_config_names(cfg))
        ds4 = {"exe": str(m / "build" / "strata-ds4")}
        self.assertEqual(maya.config_family(ds4), "deepseek")   # (read before it is fixed too)
        self.assertTrue(maya.fix_config_names(ds4))
        self.assertEqual(Path(ds4["exe"]).name, "maya-ds4")
        other = {"exe": str(m / "elsewhere" / "strata")}   # (no program under the new name there: kept)
        self.assertFalse(maya.fix_config_names(other))
        self.assertEqual(Path(other["exe"]).name, "strata")


@unittest.skipUnless(maya.shutil.which("curl"), "no curl")
class Download(unittest.TestCase):
    """The model download (fetch_files): files fetched side by side, each resumed from its size on disk, and a file
    that keeps stopping without progress ends the setup (it continues on the next run)."""

    def files(self, n=3, size=300_000):
        src, dst = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        data = {f"m-{i:05d}.gguf": bytes((i * 7 + k) % 251 for k in range(size + i)) for i in range(n)}
        for name, b in data.items():
            (src / name).write_bytes(b)
        m = {"files": {name: len(b) for name, b in data.items()}, "download_gb": 0.001}
        return src, dst, data, m

    def test_resumes_each_file_from_its_size(self):
        src, dst, data, m = self.files()
        first = sorted(data)[0]
        (dst / first).write_bytes(data[first][:100_000])   # (half way: an earlier run stopped there)
        shards = [dst / name for name in sorted(data)]
        urls = {s: (src / s.name).as_uri() for s in shards}
        with patch.object(maya, "hf_name", lambda mm, s: s.name):
            maya.fetch_files(m, shards, shards, urls)
        for s in shards:
            self.assertEqual(s.read_bytes(), data[s.name])

    def test_a_file_that_never_arrives_stops_the_setup(self):
        src, dst, data, m = self.files(n=1)
        s = dst / sorted(data)[0]
        with patch.object(maya, "hf_name", lambda mm, x: x.name), patch.object(maya, "DL_STUCK", 1), \
                self.assertRaises(SystemExit):
            maya.fetch_files(m, [s], [s], {s: (src / "nothing-here.gguf").as_uri()})


class Start(unittest.TestCase):
    def test_server_command(self):
        d = Path(tempfile.mkdtemp())
        pack = d / "maya-ds4"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "tokenizer" / "vocab.json").write_text("{}")
        exe = d / "build" / "maya-ds4"
        exe.parent.mkdir()
        exe.write_text("")
        cfg = {"exe": str(exe), "args": ["--pack", str(pack), "--max-context", "65536", "--tp", "2"],
               "tokenizer": str(pack / "tokenizer"), "model_name": maya.DS4_MODEL_NAME, "gpu": [0, 1], "port": 8099}
        p = d / "maya-ds4.json"
        p.write_text(json.dumps(cfg))
        quiet = [patch.object(maya, "say", lambda *a: None), patch.object(maya, "ok", lambda *a: None),
                 patch.object(maya, "refresh_engine", lambda c: None), patch.object(maya.S, "UI", None)]
        for q in quiet:
            q.start()
        try:
            cmd, env, info = maya.server_command(p, args())
        finally:
            for q in quiet:
                q.stop()
        self.assertIn("--config", cmd)
        self.assertEqual(cmd[cmd.index("--port") + 1], "8099")
        self.assertEqual(info["context"], "65536")
        self.assertNotIn("--expert-profile", json.loads(p.read_text())["args"])   # (GLM's profile is not DeepSeek's)


if __name__ == "__main__":
    unittest.main()
