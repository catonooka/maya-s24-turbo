#!/usr/bin/env python3
"""Tokenize a prose prompt with the pack's tokenizer -> ids file for maya-ds4."""
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file("~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/maya-ds4/tokenizer/tokenizer.json")
prompt = "Write a vivid paragraph about a lighthouse keeper who discovers the sea has begun receding overnight, leaving boats stranded and fish glittering in the moonlight. Describe what he does next."
ids = tok.encode(prompt).ids
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prose_ids.i32"
with open(out, "wb") as f:
    f.write(bytes.join(b"", [i.to_bytes(4, "little") for i in ids]))
print(f"{len(ids)} ids -> {out}")
