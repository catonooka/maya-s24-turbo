# GLM-5.3-Flash Maya-XS (IQ1_S, 73.4G) — REJECTED

Pack kept on disk: `~/Maya-data/models/Maya-XS-IQ1_S/` (69G) — pending keep/delete.

## Verdict (2026-11-08)
Benched 50.0 mean (61.3 code / 50.5 expository / 38.1 html), prefill 819 @2k.
REJECTED — the pack curve is U-shaped on sm86:

EXL3-3.05 125G = 53.3 → **Maya-S24 89G = 87.8 (the knee)** → Maya-XS 73G = 50.0

Below ~2.3bpw, IQ1_S dequant compute per byte eats the byte savings.
Smaller is NOT faster. S24 is the optimal quant on this rig.

(Downloaded to test the llamAmpere "fewer expert bytes" hypothesis —
see ../reports/llamAmpere-research.md; hypothesis disproven by measurement.)
