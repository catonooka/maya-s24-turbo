"""splice.py - a copy of a split GGUF with some tensors replaced by those of another GGUF (same names and shapes,
another quant type).  Every shard keeps its metadata (split.* included) and its tensor order; replaced tensors take
the other file's type and bytes, the rest are copied byte for byte.

    python splice.py --src MODEL-00001-of-00003.gguf --with ATTN.gguf [ATTN2.gguf ...] --out OUTDIR --name NEWBASE
"""
import argparse
import glob
import os
import re
import sys

import gguf

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--with", dest="repl", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True, help="the new file base name, e.g. GLM-5.3-Flash-Maya-S24-test")
ap.add_argument("--title", help="a new general.name")
a = ap.parse_args()

rep = {}
for p in a.repl:
    for t in gguf.GGUFReader(p).tensors:
        rep[t.name] = t
print(f"{len(rep)} replacement tensors", flush=True)

shards = sorted(glob.glob(re.sub(r"-00001-of-", "-*-of-", a.src)))
os.makedirs(a.out, exist_ok=True)
used = set()
arch = gguf.GGUFReader(shards[0]).fields[gguf.Keys.General.ARCHITECTURE].contents()   # only shard 1 names it
for sp in shards:
    r = gguf.GGUFReader(sp)
    suffix = re.search(r"-\d{5}-of-\d{5}\.gguf$", sp).group(0)
    op = os.path.join(a.out, a.name + suffix)
    w = gguf.GGUFWriter(op, arch=arch, endianess=r.endianess)
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name == gguf.Keys.General.NAME and a.title:
            val = a.title
        w.add_key_value(field.name, val, vt, sub_type=st)
    plan = []
    for t in r.tensors:
        s = rep.get(t.name)
        if s is not None:
            if list(s.shape) != list(t.shape):
                sys.exit(f"{t.name}: shape {list(s.shape)} vs {list(t.shape)}")
            used.add(t.name)
            plan.append(s)
        else:
            plan.append(t)
    for t in plan:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in plan:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(f"{op}: {len(plan)} tensors, {sum(1 for t in plan if t.name in rep)} replaced, "
          f"{os.path.getsize(op) / 1e9:.2f} GB", flush=True)
missing = set(rep) - used
if missing:
    sys.exit(f"{len(missing)} replacement tensors not in the model: {sorted(missing)[:5]}")
print("done", flush=True)
