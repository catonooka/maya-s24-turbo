"""tools/test_maya_names.py - the settings' names: MAYA_* on disk and on screen, the older STRATA_* still read.

    python -m unittest tools.test_maya_names -v
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402
import maya_names as N  # noqa: E402


class Names(unittest.TestCase):
    def test_environment(self):
        env = {"MAYA_GLM_CPU_LANE": "8", "STRATA_GLM_SPLIT": "0", "STRATA_GLM_PREFILL_MB": "1", "MAYA_GLM_PREFILL_MB": "2",
               "PATH": "/bin"}
        N.alias_env(env)
        self.assertEqual(env["STRATA_GLM_CPU_LANE"], "8")       # the name the code reads
        self.assertEqual(env["STRATA_GLM_SPLIT"], "0")          # an older name alone still counts
        self.assertEqual(env["STRATA_GLM_PREFILL_MB"], "2")     # both: the MAYA_ one wins
        self.assertEqual(env["PATH"], "/bin")

    def test_config_round_trip(self):
        disk = {"exe": "x", "env": {"MAYA_GLM_CPU_LANE": "8", "STRATA_GLM_SPLIT": "0", "HIP_VISIBLE_DEVICES": "0"}}
        cfg = N.env_in(json.loads(json.dumps(disk)))
        self.assertEqual(cfg["env"], {"STRATA_GLM_CPU_LANE": "8", "STRATA_GLM_SPLIT": "0", "HIP_VISIBLE_DEVICES": "0"})
        out = N.env_out(cfg)
        self.assertEqual(out["env"], {"MAYA_GLM_CPU_LANE": "8", "MAYA_GLM_SPLIT": "0", "HIP_VISIBLE_DEVICES": "0"})
        self.assertEqual(cfg["env"]["STRATA_GLM_SPLIT"], "0")   # (the copy written; the config in memory unchanged)
        self.assertEqual(N.env_out({"exe": "x"}), {"exe": "x"})

    def test_setup_reads_and_writes_configs_so(self):
        p = Path(tempfile.mkdtemp()) / "maya-x.json"
        p.write_text(json.dumps({"env": {"MAYA_GLM_RESERVE_MB": "3072"}}))
        self.assertEqual(maya.read_json(p)["env"], {"STRATA_GLM_RESERVE_MB": "3072"})
        self.assertEqual(N.shown("STRATA_GLM_RESERVE_MB"), "MAYA_GLM_RESERVE_MB")


if __name__ == "__main__":
    unittest.main()
