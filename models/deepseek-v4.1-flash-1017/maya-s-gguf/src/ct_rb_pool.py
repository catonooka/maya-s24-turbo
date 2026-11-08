"""Exp 6 (2026-10-20): FIXED-POOL route bias (Xolotl STRATA_RB_POOL=M shape).

Exp 1 failed because the bonus tracked the CHURNING EC residency (feedback loop,
hit 47->30%). Xolotl's fix shape: bias only within a STABLE pool — every pick is
kept inside the exact top-M of that layer (Max-Rank, arXiv:2412.00099), so KL is
bounded by construction and the target set never moves under the router.

Port: at start(), learn each layer's top-M experts from the EC profile sidecar
(DSV41_EC_PROFILE, written by ct_ec_profile); build a STATIC bias tensor
b0 + bonus*topM_mask ONCE (no per-step rewrites, zero steady-state cost, no
churn). M ~ EC slots/layer keeps the pool aligned with what the cache can hold.

Env:
  DSV41_RBP_MULT  bonus multiple of bias-std (0/unset = off); 0.5 = aggressive.
  DSV41_RBP_M     pool size per layer (default: EC slots // num layers).
"""
import gc
import json
import os

import torch
import torch.nn as nn


def _find_biases(CT):
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
            for child in o.modules():
                if id(child) in reg:
                    out[reg[id(child)]] = b.data
                    break
        except Exception:
            continue
    return out


def install(CT):
    mult = float(os.environ.get("DSV41_RBP_MULT", "0"))
    if mult <= 0:
        return False
    prof_path = os.environ.get("DSV41_EC_PROFILE", "")
    if not prof_path:
        CT._log("RBP: no DSV41_EC_PROFILE; fixed-pool bias OFF")
        return False
    try:
        with open(prof_path) as f:
            prof = json.load(f)
        score = torch.tensor(prof["score"], dtype=torch.float64)
    except Exception as exc:
        CT._log(f"RBP: profile load failed {exc!r}; OFF")
        return False
    biases = _find_biases(CT)
    if not biases:
        CT._log("RBP: no bias tensors; OFF")
        return False
    L, E = score.shape
    m = int(os.environ.get("DSV41_RBP_M", "0")) or max(1, int(os.environ.get("DSV41_EC_SLOTS", "420")) // max(1, L))
    m = min(m, E)
    stds = [float(b.detach().float().std()) for b in biases.values()]
    bonus = mult * (sum(stds) / len(stds))

    applied = 0
    for li, b in biases.items():
        if li >= L:
            continue
        topm = torch.topk(score[li], m).indices
        mask = torch.zeros_like(b)
        mask[topm] = 1.0
        b.add_((bonus * mask).to(b.dtype))  # one-shot, static, never rewritten
        applied += 1
    CT._log(f"RBP: fixed pool M={m} bonus={bonus:.5f} applied to {applied} layers (from {prof_path})")
    return True
