# L2 results (2026-10-19) — python-level per-step admission is NOT the lever

## What was tried
- L2v1 (DSV41_EC_UNION=1): want-set = this step's live routing union, admitted count-ranked
  every step, hysteresis kept. RESULT: 6.3 t/s, hit 20%, 127,194 admits / 126,774 evicts.
  Classic thrash: union ~6*35=210 pairs/step single-stream (~840 at N=4) > 420 slots; every
  admit copies 12.7 MiB H2D; 96 admits/step ~= 1.2 GB EXTRA PCIe per step — worse than the
  misses it removed.
- L2v2 (UNION=0, EVERY=1, ADMIT=48): stock decayed-score ranking but replanning every step.
  RESULT: 8.2-9.6 t/s, hit 11-13%, admits froze at 420 (hysteresis deadlock: decayed scores
  stay positive so victims never leave `want`; and no `pending` activations logged). Also
  counts.cpu() stream-sync every step (~0.3ms) + topk over 13,440 pairs on the python thread.

## Conclusion
Stock EC (EVERY=16, decayed topk) is already the right python-level policy; J&M's finding
that policy changes buy <0.1% holds here too. Admission LATENCY was never the real cost —
the real costs are (a) SM-load UVA streaming at ~22-25 GB/s vs copy engine ~56 GB/s, and
(b) re-streaming of non-resident actives every step. Both are C++-level.

## Best-known config (restored): serve-ec-ab.sh 4 + MARGIN 800 + SLOTS 420 + ADMIT 96
11.05 t/s single, 13.5 aggregate @N=4, hit ~47% (EVERY=4).

## Next: L3 (the real kernel work, port of maya win_experts)
Design stays as DEDUP-KERNEL-DESIGN.md: grouped per-step launch, dequant-once per expert,
copy-engine staged fetch keyed off the (already on-device) counts union. This is where the
1.75x/2.4x/3.05x union multipliers and the 2x copy-engine bandwidth live.
