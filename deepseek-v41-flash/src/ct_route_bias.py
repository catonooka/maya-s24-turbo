"""Exp 1 (2026-10-20): route bias toward EC-resident experts (maya TRUNK_KEEP / Xolotl route-bias idea).

Maya bends the router toward experts already in VRAM within a measured KL budget
(Xolotl, same model class: bias 0.5/0.3 = 21.7 t/s vs 9.7 exact, 62.9% picks kept).
Port onto the sybil stack: after each ec_step() (the EC's own replan cadence), add a
residency bonus to every layer's gate.e_score_correction_bias IN PLACE.

Graph-safe: dsv4_topk's Triton kernel takes the bias tensor by address; a captured
graph replays the launch and reads whatever bytes are there now — the same property
the EC's pointer tables rely on. The home copy b0 is kept pristine so bias=0
restores byte-identical routing.

S["layers"] holds the FusedMoE experts modules; the bias Parameter lives on the
parent FFN (model.layers[N].ffn.gate.e_score_correction_bias). We recover the
experts->parent map with a one-shot gc walk at start() (model fully built then).

Env:
  DSV41_RB   bonus as a multiple of the bias tensors' mean std (0 / unset = off).
             0.25 = gentle near-tie tilt, 0.5 = maya-class aggressive, 1.0 = strong.
"""
import gc
import os

import torch
import torch.nn as nn


def _find_biases(CT):
    """{li: bias.data} for every registered experts module that has a biased parent FFN."""
    reg = {id(l): li for li, l in enumerate(CT.S.get("layers", []))}
    if not reg:
        return {}
    out = {}
    for o in gc.get_objects():
        try:
            if not isinstance(o, nn.Module):
                continue
            b = getattr(getattr(o, "gate", None), "e_score_correction_bias", None)
            if b is None or getattr(b, "numel", lambda: 0)() == 0:
                continue
            # find which registered experts module (if any) lives under this FFN
            for child in o.modules():
                if id(child) in reg:
                    out[reg[id(child)]] = b.data
                    break
        except Exception:
            continue
    return out


def install(CT):
    """Wrap CT.ec_step so the residency bonus follows every EC replan. Returns True if active."""
    mult = float(os.environ.get("DSV41_RB", "0"))
    if mult <= 0:
        return False
    biases = _find_biases(CT)
    if not biases:
        CT._log("RB: no e_score_correction_bias reachable from registered layers; OFF")
        return False
    base = {li: b.detach().clone() for li, b in biases.items()}
    stds = [float(b.detach().float().std()) for b in biases.values()]
    bonus = mult * (sum(stds) / len(stds))
    E = CT.S["layers"][0]._ct["E"]
    dev = next(iter(biases.values())).device

    orig_step = CT.ec_step
    armed = {"n": 0, "swaps": 0}

    def step_with_bias():
        orig_step()
        if not CT.EC.get("where"):
            return
        sw = CT.EC["admits"] + CT.EC["evicts"]
        if sw == armed["swaps"] and armed["n"]:
            return  # residency unchanged since the last apply; skip the rewrite
        armed["swaps"] = sw
        armed["n"] += 1
        mask = torch.zeros(len(CT.S["layers"]), E, dtype=torch.float32, device=dev)
        for flat in CT.EC["where"]:
            li, e = divmod(flat, E)
            mask[li, e] = 1.0
        bm = bonus * mask
        for li, b0 in base.items():
            b = biases.get(li)
            if b is not None:
                b.copy_(b0 + bm[li].to(b.dtype))

    CT.ec_step = step_with_bias
    CT._log(f"RB: installed mult={mult} bonus={bonus:.5f} ({len(biases)} layers, E={E})")
    return True
