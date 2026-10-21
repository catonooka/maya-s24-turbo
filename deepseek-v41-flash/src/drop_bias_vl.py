#!/usr/bin/env python3
"""Drop VL-only leftover keys the LLM-only loader has no params for.

Our 2.9bpw source conversion kept `layers.N.ffn.gate.bias_vl` (43 keys, VL gate
bias). The d141 vLLM deepseek_v4_1 loader with --language-model-only has no
such params -> KeyError at load. The campaign's D010 pack (Mia source) did not
carry these keys. Remove them from the pack's weight_map to match that layout.
Idempotent; rewrites model.safetensors.index.json in place.
"""
import json, sys
from pathlib import Path

pack = Path(sys.argv[1] if len(sys.argv) > 1 else "~/models/DSV41-EXL3-3090-local")
idxp = pack / "model.safetensors.index.json"
idx = json.loads(idxp.read_text())
wm = idx["weight_map"]
drop = sorted(k for k in wm if k.endswith(".ffn.gate.bias_vl"))
for k in drop:
    del wm[k]
idxp.write_text(json.dumps(idx, indent=2))
print(f"dropped {len(drop)} bias_vl keys; weight_map now {len(wm)} entries")
