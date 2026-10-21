"""Exp 3 (2026-10-20): EC expert-demand persistence across boots (maya usage-file idea).

Maya persists expert hit counts and seeds its VRAM tier at start; our EC re-learns
from zero every boot (64-step warmup + slow score accumulation -> the well-known
cold 38% -> warm 47% gap). Port: dump EC["score"] to a JSON sidecar every replan
cycle (cheap: L x E floats, ~560 KB), and seed the score matrix from it at
_ec_alloc time. Decay semantics unchanged; persistence only sets the INITIAL
state, so a workload change still relearns (scores decay at 0.9/step).

Env:
  DSV41_EC_PROFILE   path to the score sidecar (unset = off). Writable volume!
"""
import json
import os

import torch


def install(CT):
    path = os.environ.get("DSV41_EC_PROFILE", "")
    if not path:
        return False
    L = len(CT.S.get("layers", []))
    E = CT.S["layers"][0]._ct["E"] if L else 0
    if not L or not E:
        CT._log("ECprof: no layers registered; OFF")
        return False

    # --- load side: seed scores at alloc time ---
    orig_alloc = CT._ec_alloc

    def alloc_seeded():
        orig_alloc()
        if CT.EC["N"] <= 0 or CT.EC.get("score") is None:
            return
        try:
            with open(path) as f:
                prof = json.load(f)
            sc = torch.tensor(prof["score"], dtype=torch.float64).reshape(L, E)
            if sc.shape == CT.EC["score"].shape:
                CT.EC["score"].copy_(sc)
                CT._log(f"ECprof: seeded score from {path} (sum {float(sc.sum()):.0f})")
            else:
                CT._log(f"ECprof: shape mismatch {tuple(sc.shape)} != {tuple(CT.EC['score'].shape)}; not seeded")
        except FileNotFoundError:
            CT._log(f"ECprof: no sidecar at {path} (first boot; will write)")
        except Exception as exc:
            CT._log(f"ECprof: load failed {exc!r}")

    CT._ec_alloc = alloc_seeded

    # --- dump side: write sidecar after each real replan ---
    orig_step = CT.ec_step
    n = {"i": 0}

    def step_dumping():
        before = CT.EC["admits"] + CT.EC["evicts"]
        orig_step()
        n["i"] += 1
        if CT.EC.get("score") is None or CT.EC["N"] <= 0:
            return
        if CT.EC["admits"] + CT.EC["evicts"] == before and n["i"] % 64 != 0:
            return  # only write when residency moved (or every 64th step as heartbeat)
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"L": L, "E": E, "score": CT.EC["score"].tolist()}, f)
            os.replace(tmp, path)
        except Exception as exc:
            CT._log(f"ECprof: dump failed {exc!r}")

    CT.ec_step = step_dumping
    CT._log(f"ECprof: persistence on {path}")
    return True
