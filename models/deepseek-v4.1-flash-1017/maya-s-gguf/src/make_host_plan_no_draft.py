#!/usr/bin/env python3
"""Build a draft-pinned host-plan: layers 35-39 (dspark target 37-39 + margin) removed from
the host plan so their experts load as regular VRAM weights (7.06 GiB) instead of UVA streaming.
Writes host-plan-no-draft.json next to the original; the pack config is switched by DSV41 env."""
import json, os

src = '~/models/DSV41-EXL3-3090-local/host-plan.json'
dst = '~/models/DSV41-EXL3-3090-local/host-plan-no-draft.json'
hp = json.load(open(src))
PIN_FROM = 35  # keep 0-34 in host RAM (35 CT layers == DSV41_CT_LAYERS)
out = {k: v for k, v in hp.items() if int(k.split('.')[2]) < PIN_FROM}
json.dump(out, open(dst, 'w'))
print('kept %d layers (0-%d), pinned %d to VRAM -> %s' % (len(out), PIN_FROM - 1, len(hp) - len(out), dst))
