# GLM-5.3-Flash Maya-S24 (IQ2_XXS_S, 89G) — SPEED tier (3x RTX 3090 + 247G DRAM)

Pack: `~/Maya-data/models/Maya-S24/` (peasantsmith/GLM-5.3-Flash-Maya-GGUF,
Maya-S24/* 89GiB/3 shards + vision/* vocab+mmproj needed by --setup).

## SERVING config: `s24-dflash-p32k.json`
`bash models/glm-5.3-flash/maya-s24/run-s24.sh` (or from ~/maya-glm3).
env: RAM_RESIDENT, KV_INT8, PROMOTE_MIN=6, SPEC_SAMPLE, SPEC_CANCEL, PCIE_SHARE=0.25,
MISS_SKIP_W=0.14, MISS_SKIP_FROM=23, FETCH_WIDE, P2P, PF_CE; args:
`--dflash .../DFlash2-Q4_K_M.gguf --prefill 32768`; ctx 262144.

## Measured (official --bench)
- decode 87.8 mean (86.9 code / 80.9 prose / 95.5 html); client spec=8:
  code 94 / prose 77
- prefill 699-720 @2k / 1002-1009 @8k / ~926 t/s sustained @100k

## Ladder
| config | decode |
|---|---|
| MTP (MTP_KEEP=2) | 80.6 |
| + DFlash2 | 85.6 |
| **+ prefill 32768** | **87.8 KEEPER** |
| PROMOTE_MIN=8 | 83.8 rejected |

- DFlash2 > MTP on 3 cards (opposite of 4-card #349 — pipelined window economics).
- PROMOTE_MIN 6 optimal.

Reports: ../reports/ (glm3-s24-100push, glm3-3x3090-v1156).
