# Blocker resolution log (live-updated)

Legend: ✅ resolved (measured), 🟡 partial, ❌ closed not-viable, ⬜ queued

## B13 Prefill (was 375 t/s)
🟡 PARTIAL: PR152 DMA env (EXL3_HOST_DMA=1 + EXL3_DMA_GEMM=1, both confirmed ACTIVE in logs)
→ 4k prefill 375 → **432 t/s (+15%)**. Their 575-765 needs their full shape (65536-len bench,
1GiB KV, FP16 wo_a overlay pack, EPYC DDR4-3200). Remaining gap = D107 chunking + pack overlay.

## Decode tracker (single / agg@N4)
- stock broken tier: 5.4
- CT_LAYERS fix + EC warm: **11.0 / —**
- L1b 420 slots EVERY=4: **11.05 / 13.5** ← still best
- PR152 env full (DMA+GEMM+FAT+elastic900): 10.5 / 12.4, hit 42% — decode REGRESSION ~0.5
  vs L1b; prefill +15%. Root cause: elastic EC releases/rewarms the pool around bursts and
  margin 900 leaves fewer live slots in steady decode.
- margin300+DMA (non-elastic): N=4 OOM (reconstruct_hgemm 300MB) — B12 still open at low margin.
→ OPERATOR DECISION: DMA env is a prefill win and a small decode loss. For a chat-oriented
  single-stream deployment keep L1b (11.05/13.5). For long-prompt workloads keep DMA env.
  Kept serving config = L1b for now.

## B12 engram wkv OOM @N=4 with low margin
🟡 margin 900 + elastic = stable at N=4 (12.4 agg, gate 323 ✓). margin 300 OOMs. Not fully
resolved — capacity tradeoff documented; real fix = L3 kernel freeing the EC-vs-graphs fight.

## B6 draft-expert streaming (assumed cause of 5.3 t/s DSpark)
❌ CLOSED — MISDIAGNOSIS: mtp.* experts were NEVER host-pinned (host-plan covers model.layers.*
only) → they load as normal VRAM weights (2.11 GiB). The 5.3 t/s was entirely B7: verify-window
union re-streaming + draft serialization. No draft-pinning work needed; spec decode is blocked
SOLELY on L3 grouped fetch. (Verified: host-plan keys + pack tensor scan + code path
_host_plan_for -> None -> regular Parameter.)

## L3 pre-work bound experiments (2026-10-19 evening)
❌ ngram spec T=3 WITH cuda graphs (captured 4/4, rejection sampler warmed): prose 5.6 /
   doc-copy 6.5 t/s — identical to the old eager numbers. Eager-overhead hypothesis DEAD.
   Spec slowdown = fundamental (per-step routed-expert count scales with window, EC churn).
❌ decode-DMA forced (EXL3_HOST_DMA_MIN_TOKENS=1, mode-1 stages ALL 384 cold experts in
   8-slot rotation): 1.43-1.48 t/s single, 5.5 agg @N4 — catastrophic. Mode-1 decode staging
   is strictly wrong design for tiny batches.
→ L3 is now PRECISELY SCOPED: mode-2 (active-experts-only) decode staging, made graph-safe
   (pre-pinned index ring + device-side table update = maya pattern; C++ hooks _dma2_issue/
   tabs/index_copy already exist in image). This is the build. Ceiling math: ~53% of expert
   bytes are cold misses; CE-staged at ~2x UVA efficiency overlapped behind attention →
   est +30-60% decode. Cold-pack 2.0bpw sidecar = separate ~+1-1.5 t/s for a ~135G download
   (only if disk/quality budget allows; research: cold-only 2.0 is the accepted hot/cold trade).

## B7 spec decode
❌ CLOSED at T=3 with graphs (see above). Revisit only after L3 staged fetch, if ever.


## B1/B2/B3/B4/B5/B8/B9/B10/B11: ⬜ queued (L3 grouped kernel is the single next lever)


