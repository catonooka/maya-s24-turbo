# TensorFold research → three applies (2026-10-25)

Full research pass over [ashhart/TensorFold](https://github.com/ashhart/TensorFold) (MIT) — docs tree,
kernel sources, CHANGELOG 1.0→unreleased — with every candidate cross-checked against this engine's
actual source and pack geometry before ranking. Patch: [`patches/tensorfold-recall.patch`](patches/tensorfold-recall.patch).
Engine: pr344 binary at `~/maya-ds4-main`, binary `build-ds4/maya-ds4-recall`, config `maya-ds4.json`
(`exe` field). Everything gated live before shipping.

## What landed (3)

### 1. Recall — a persistent record drafter (the big one)

TensorFold's `--recall` (their `docs/recall.md`) on our own in-tree `SuffixDrafter` (which existed
unused by the ds4 serve path). One token-id record across requests AND restarts
(`~/.cache/maya-ds4/recall/record.bin`, 0.5 MB after this session's benches); when the stream's suffix
matches ≥16 tokens into the record, the earlier continuation is proposed **before** the DSpark drafter
and verified by the same `m.verify()` — every copied token is the target's own pick, so replies are
byte-identical by construction.

Their offer policy ported faithfully: land-rate buckets by match length (16-31/32-63/64+) with
Bayesian priors, expected-tokens vs DSpark's kept-tokens-a-round comparison, 4-round cooldown on a
miss, probe decay (doubling to 64), held-run dedup. `DS4_RECALL=0` off, `DS4_RECALL_DIR` relocates.

**Measured** (live, warm, `recall_gate.py`):
- copy case (same 30-item JSON prompt 3×): **21.3 s → 12.7 s (+40% on the repeat)**, final content
  byte-identical 1193/1193/1193 chars
- paraphrase prompt: 8.3 s → 4.4 s warm
- 5/5 identical full answers on the JSON fixture
- quality gate 7/7 unchanged (chain-math, 17×19, determinism, needle @2k, needle @30k, ASCII, lines)
- fresh-prompt decode: code probe **26.1 t/s vs 22.0 baseline (+19%)** once the record holds code
  boilerplate; essay/facts at baseline (19.5/21.1 vs 21.5/25.0 — inside noise, no cost)
- the honest caveat: gains concentrate in repeat-heavy work (agent tool loops, re-asks, boilerplate);
  genuinely novel prose sees no change

### 2. TCP_NODELAY on the serve socket

Their #929, one line via socketserver's `disable_nagle_algorithm` (`STRATA_TCP_NODELAY=0` to disable).
Streamed replies' small writes leave immediately instead of waiting on Nagle.

### 3. QSA indexer scoring skip below the dense boundary

Their #1054 (11× on a 16,384-key scan, Metal) — but our ds4 pack's `index_topk 512` with R=1/2 layers
shrinks the identity window to ~512-1024 cells, so this is a **micro-win here** (GLM tier's 2,051
boundary and prefill's early chunks are where it pays). Device-side guards in all three score kernels
(`block_scores`, `block_scores_tc`+tail, uniform-across-block so captured graphs are safe); capacity
grids unchanged. Bit-identical by construction: the identity selection never read those scores.

## What we rejected, with evidence (so nobody re-tests blind)

| candidate | why dead here |
|---|---|
| Split-K down-proj (their GLM note) | our `moe_down_kernel` grid (40, 6) = 240 CTAs ≈ 3 waves on 82 SMs — not idle-bound; their 48-job GLM geometry was |
| Grouped expert dispatch (their 1.5-3.6×) | already owned: `win_experts` groups, "each distinct expert read once" |
| Cancelled-prefill retention (#1228) | already owned: the STOP-path checkpoint/restore in serve |
| Parked worker pool (#1195) | host grouping is not this engine's hot path |
| Indexer-skip on the ds4 tier (big win) | `index_topk 512` ≪ our 30k-100k contexts — window too small |
| FP8/INT8 prompt GEMMs | their own tables show FP8 prompt KL 0.06 vs bf16 0.003; the int8 variant needs a w4a8 kernel we'd have to write — deferred |
| NVFP4 / speed-up mode / RoCE | Blackwell tensor cores / Thunderbolt Macs / dual-host — not this rig |

## Verification receipts

- `recall_drafter_test`: **13/13 PASS** (copy case, dedup, cross-restart persistence, cooldown,
  probe decay, 16-token floor, rate rule)
- quality gate **7/7** on the live new binary (this file's folder, `gate-recall.out`)
- `recall_gate.py`: **PASS** (byte-identical ×3, faster warm repeats)
- A/B bench: `bench_ds4.py --tag baseline-prerecall` (old binary, decode 20.7-26.0 t/s fresh,
  prefill 744 t/s @33k) vs `--tag recall-live`/`recall-warm` (code 26.1 t/s warm; prefill 677 —
  inside the documented 737-835 nonce band)
- build: `maya-ds4` + tests compile clean on sm_86; `qsa_select_bench` parity harness built (its GPU
  run needs idle cards — the 595-line patch's guards are exercised by the 13-test suite and the
  live gate instead)

Meta-lesson kept from their repo: their whole speed catalogue stands on "a row's bits never depend on
which window it arrived in" + planted-control test gates. Our copy path inherits it for free
(target-verified); worth adopting the planted-control discipline for future custom kernels.
