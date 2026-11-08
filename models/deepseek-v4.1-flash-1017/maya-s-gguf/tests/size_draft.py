#!/usr/bin/env python3
"""Size the mtp.* (DSpark draft) tensors in the pack."""
import json, os, struct, math

root = '~/models/DSV41-EXL3-3090-local'
idx = json.load(open(os.path.join(root, 'model.safetensors.index.json')))
wm = idx['weight_map']
files = sorted(set(wm[k] for k in wm if k.startswith('mtp.')))
total = 0
for f in files:
    p = os.path.join(root, f)
    with open(p, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        if k.startswith('mtp.'):
            total += math.prod(v['shape']) * 2  # int16/fp16 dominant dtype width
print('draft (mtp.*) total: %.2f GiB across %d shards' % (total / 2**30, len(files)))
