# S3 — Class-Tagged Chunk Staging for Mixed-K EXL3 Packs

Branch `mixed-lane` @ `ab301658` (working tree of `src/ds4/ds4_model.cu` was dirty/mtime-touched
during analysis; **line numbers below are from this inspection and drift ±15 — anchors quoted**).

Goal: let mixed-K EXL3 packs (DS4.1-Flash 0xSero 2.7bpw: K2 41% / K3 48.5% / K4 10% / K5 35
experts per layer) run **chunked prefill** instead of 16-token windows (125 tok/s on 40k prompts;
single-class 2.9bpw did 737–870 tok/s in chunks of 8192). Design chosen: **per-expert class tag
in the staging pointers' low bits** (the "S5 tag" the code already anticipates at
`exl3.cu:857`). This is a design doc; no source was modified.

---

## 1. The chunk prefill path today (single-class quant, CPU-lane experts)

Entry → per-device → per-layer:

| Step | Function (file:line) | What happens |
|---|---|---|
| Run entry | `Impl::forward_chunks` (`ds4_model.cu:~12599`) | splits the prompt into `chunk`-token chunks; calls `plan_streams()` then `run_dev_chunks` per device |
| Stream plan | `Impl::plan_streams` (`ds4_model.cu:11564`) | `l.host_list` = CPU-lane experts in **expert-index order** (test at `11569`: `slot[e]<0 \|\| slot[e]>=pv_n`); `x.vs_pin[e]` = VRAM slot if `slot[e] ∈ [0,pv_n)` else `pv_n + j++` (`11574`). **Virtual slot (vs) = dense position**: VRAM experts occupy vs `[0, pv_n)`, streamed experts `[pv_n, n_vs)` |
| Batch plan | `Impl::run_dev_chunks` (`ds4_model.cu:11756`, batch loop `~11819`) | `S.b` = Batches `{k, il, j0, n}` over `host_list`, `B = ring_n/kParts` experts each |
| Staging | `Impl::copy_ahead` (`ds4_model.cu:3996`) | per batch: (in-place quants) readers fill pinned `c.stage` slots; then `stream_expert` per expert DMAs its blob (or member part) into ring slot `c.ring + (half*(ring_n/P) + i) * ring_stride` on copy stream `c.cs`; `cudaEventRecord(c.filled[half])` |
| Expert copy | `Impl::stream_expert` (`ds4_model.cu:3822`) | NH==1/nm==1: pieces from the host region (`hh`/`hl` offsets, single-class) or **mixed branch already present** (`3860-3876`: `lay_e = clay[cls[e]]`, pieces at `esrc` offsets, dst at the class layout's offsets). NH>1 branches (`3893-3932`) are **class-0 only** |
| Layer compute | `Impl::layer_chunk` (`ds4_model.cu:11181`) | routes (`route`, `moe_sort` at `11464` sorts picks by `vs_dev` → `c.bounds` per virtual slot, `row_tok`/`pos` gather); then interleaves per batch: wait `filled[half]` → **ring** `run_set` (`11544`, `wbase = c.ring + half*(ring_n/P)*ring_stride`, `stride = ring_stride`) → `copy_ahead` next → **VRAM** share `run_vram` (`11551`, `11557`) |
| VRAM experts | `Impl::run_vram` (`ds4_model.cu:4437`) | splits `[i0,i1)` at `L0.n_main`: run_set over `x.vram_base` (`stride = unit(x)`) + tail-borrow range over `x.xbase` |
| Kernel launch | `Impl::run_set` (`ds4_model.cu:4445`) | builds `x3::SetArgs` (`include/strata/kernels/exl3.hpp`): `sa.wbase/stride` (blob = `wbase + e*stride`), `sa.lay = l.nm>1 ? l.hlay : l.lay` (`4458`); **mixed special case `4459-4462`**: `if (l.n_cls>1 && wbase == l.vram_base) { sa.mix = l.mix; sa.eptr = l.d_maddr + i0; }` → `x3::set_gate_up/set_down` take the per-expert-pointer path |
| GPU kernels | `exl3.cu: set_gate_up 2318 / set_down 2402` | `sa.mix.n>0` → portable `set_mix_kernel` (`exl3.cu:1436`, blob = `a.eptr[e]`, class = `mix_cls(mx,ptr)`); else sm80+ tensor-core tiles (`tc::GArgs`, single layout), pre-sm80 portable `hip_set_kernel` |

Not class-sensitive (verified): `Src::x_stage`/`x_stage1` (`844-853` — KV/IK row hand-off staging
between groups), Engram staging (`emb_host/eg_host`), yoco prompt-skip two-pass, group hand-offs
`c.hand` (h\|pre\|topk\|cand), `chunk_allreduce`, `hc_mix_chunk`, shared-expert GEMMs,
`moe_combine` (order-independent `acc += y*rw` per pick).

### Exactly where "one class per layer" breaks for mixed-K

1. **`plan_streams` (`11569`, `11574`)** — assumes `slot[e]` is a **dense** slot index in
   `[0, n_slots)`. Mixed layers set `slot[e] = e` for VRAM experts (sparse expert ids,
   `place_experts` at `5250`: "a mixed layer: slot e = expert e") and `pv_n = n_vram`
   (`5222`). Consequences: a VRAM expert with id ≥ `n_vram` fails the `slot < pv_n` test →
   lands in `host_list` with `l.host[e] == nullptr` → `stream_expert` copies from NULL; and the
   vs↔bounds↔`d_maddr` index correspondence collapses.
2. **`run_set` ring call (`11544` + `4458`)** — `sa.lay = l.lay` = **class 0's layout**
   (`2401`: "class 0 as the layer's layout") and no `sa.mix` for the ring: every streamed expert
   would be decoded with class-0 K regardless of its true class. `sa.stride = ring_stride` is
   fine (ring slots are `align256(l.blob)` and `l.blob` = **largest** class's blob, `2402`).
3. **`run_vram` (`4437-4442`)** — splits at `n_main` and addresses VRAM blobs as
   `vram_base + i*stride`; mixed arenas are per-class cudaMallocs with scattered `maddr[e]`
   (`5236-5256`), so only the `eptr` path is valid — and the whole `[0, n_exp)` range must go
   through it, not `[0, pv_n)`.
4. **`layer_chunk` (`11462`, `11543`, `11550/11557`)** — `n_vs = pv_n + host_list.size()`,
   `v0 = pv_n + bt.j0`, `v_to` shares over `[0, pv_n)` — all assume the dense single-class vs.
5. **`stream_expert` NH>1 (`3893-3932`)** — class-0-only geometry (`l.lay.k2[r]`, `hl` offsets);
   mixed CPU-lane host parts are **raw regions** (`hh[e][X] = file_part(l,e)` or packed
   `split_host`, `5330-5332`), only the NH==1 branch (`3860-3876`) knows the `esrc`/`clay`
   mapping. `NH` is picked at `3273` before any mixed-awareness.
6. **The boot gates** — `2629-2633` (`chunk = prompt_par && !any_mixed ? … : 0`; note: at this
   point `L` is still empty — this gate is inert on first load, the split planner at `~2757`
   already runs with `chunk>0`) and **`2831-2843`, the real gate** ("the layouts are known only
   now") which forces `chunk = 0` and prints the windows message.

## 2. How decode (windows of 16) already handles mixed classes — the S1 machinery

- `route_publish` (`~13579`) writes a **per-pick pointer** `d.plan_ptr[t][i]` from
  `l.vram_tab` (mixed = `maddr[e]` or 0). Grouped window kernels
  `win_gu_mix_kernel` / `win_down_mix_kernel` (`exl3.cu:946/976`) take each group's pointer and
  derive its class **from the address**: `mix_cls(mx, ptr)` (`exl3.cu:850-858`) — inside class
  arena `[lo[c], hi[c])` → class c; **outside every arena → fallback "first real class", with
  the comment "a staged pointer, S5 tags it" (`857`)** — the exact hook S3 uses.
- ptr == 0 picks (CPU lane) are computed on the host: `serve_lane` → `cpu_experts`
  (`~9473`), whose `part0` reads each expert's parts at `esrc` offsets with
  `clay[cls[e]].k2` — per-expert class dispatch on the host; results fold into the combine
  kernel via `cpu_part`/`cpu_flag` (`exl3.cu:820-823`).
- S5 (`DS4_MIX_SWAP`): `cache_queue` (`~8231`) picks the cold victim **of the same class**
  (`cls[e] != cls[best] → skip`), `cache_loop` (`~7606`) repacks the region to the class layout
  and DMAs it **into the victim's arena slot**; the switch (`~8131`: "addresses trade places")
  rewrites `maddr/slot/host/hh` + `vram_tab`/`d_maddr` on the device. Staged experts therefore
  always stay **inside the class arenas** — that is why range classification has been enough so
  far, and why the chunk ring (outside every arena) is the first thing that needs the tag.
- EXL3 mixed never uses: the PCIe lane staging (`lane_class` paths are MXFP4-gated,
  `~13596`), the helper GPU (`helper_select` skips non-MXFP4, `~8821`), recent slots
  (`rc_base` only set in the single-class branch `~5367`), the flow (GGUF-only, `fl` requires
  `kFmtMXFP4` at `~3338`), fat/light set paths (MXFP4-only, `4476`).
- Tensor-parallel **within** a layer is refused for mixed at load (`4884-4887`: "runs
  single-member (layer split) for now") → S3 only needs to work for layer split (`--tp 1`,
  N GPUs, contiguous layer ranges), which is the campaign's rig shape.

**Pattern to reuse:** every expert carries its own device pointer; the kernel derives the class
from the pointer. The chunk path just has to carry pointers too (it currently carries
`wbase + e*stride`), and tag the ones that live outside the arenas.

## 3. The design: class tag in the staging pointers' low bits

Ring slots are `align256(l.blob)`-strided and 256-aligned → **low 8 bits of every ring pointer
are free**. `kMaxClasses = 32` fits in 5 bits; use a full byte for margin.

- **Tag rule:** staged pointer = `ring_slot_addr | (cls + 1)` (tag ∈ [1, 32]); **0 = untagged**
  (arena pointers from `cudaMalloc` are 256-aligned → tag 0 → range classification unchanged).
- **`exl3.cu` changes (small, contained):**
  - `mix_cls` (`850`): check the tag first:
    `if (p & 0xffull) return (int)(p & 0xffull) - 1;` before the existing range loop.
  - Mask the blob base wherever a mix kernel forms it:
    `const uint8_t* blob = (const uint8_t*)(ptr & ~0xffull);`
    in `moe_gate_up_mix_kernel` (`893`), `moe_down_mix_kernel` (`919`), `win_gu_mix_kernel`
    (`953`), `win_down_mix_kernel` (`982`), `set_mix_kernel` (`1447`),
    `wmma_set_mix_kernel` (`1652`).
  - Nothing else: `kmix`, `mixed_ok`, `mix_launch`, all launch paths identical; the tag-0 path
    is bit-identical to today (single-class packs never see `sa.mix.n > 0`).
- **Host-side changes (`ds4_model.cu`):**
  1. **`plan_streams`** — mixed-aware vs scheme: for `n_cls > 1` layers,
     `host_list` = experts with `slot[e] < 0` **only**; `vs_pin[e] =` `e` for VRAM experts
     (dense over `[0, n_exp)` — matches `d_maddr` indexing exactly, since `slot[e]=e`),
     `n_exp + j` for CPU-lane experts. `n_vs = n_exp + host_list.size()` (384 + lane ≤ 2048
     `c.bounds` — assert).
  2. **`layer_chunk`** — `n_vs` (`11462`), `v0 = n_exp + bt.j0` (`11543`), `v_to` shares over
     `[0, n_exp)` (`11550`, `11557`) when `n_cls > 1`.
  3. **`run_vram`** — when `l.n_cls > 1`: one `run_set` over `[i0, i1)` with the eptr path (no
     `n_main`/`xbase` split — mixed has no borrowed tail).
  4. **`run_set`** — replace the implicit `wbase == l.vram_base` test (`4459`) with an explicit
     `const unsigned long long* eptr` parameter (+ its vs origin): VRAM callers pass
     `l.d_maddr` (index = vs = expert id); the ring caller passes the per-batch tagged-pointer
     array. `sa.mix = l.mix` whenever `eptr != nullptr`.
  5. **`copy_ahead`** — after a batch's `stream_expert` copies, fill
     `c.ring_ptr_pin[abs_slot] = dst | (cls[e] + 1)` and issue **one** `cudaMemcpyAsync` of the
     batch's slice `pinned → c.ring_ptr_dev` on `c.cs`, **before** `cudaEventRecord(filled[half])`.
     Stream-ordered: the compute stream's wait on `filled[half]` also covers the pointer copy —
     no new sync, no host stall.
  6. **`alloc_chunks`** — `dalloc(c.ring_ptr_dev, ring_n * 8)` + `halloc(c.ring_ptr_pin, ring_n
     * 8)` when any mixed layer exists (alloc unconditionally is fine too, ~1 KB).
  7. **`stream_expert` / NH** — force `NH = 1` at (`3273`) when any layer is mixed
     (`n_cls` is known: `expert_layout` ran in the layer loop above). The NH==1 mixed branch is
     already correct; NH>1 mixed (raw regions split by class geometry) is left as follow-up.
  8. **Remove the gates** — drop `!any_mixed` at `2631`; delete the forcing block at
     `2837-2843` (keep a verbose one-liner that mixed now chunks).
  9. **S5 interlock** — guard the mixed switch (`~8131`, and the queue decision at `~8231`)
     with `&& !prompt_active` where `prompt_active` is an atomic set/cleared by
     `forward_chunks` around the run. Today switches only land on decode ticks so this is
     belt-and-braces: a mid-run `maddr/slot/host` change would desync `vs_pin`, `d_maddr` and
     the ring pointer table.
- **What must NOT change:** flow, fat/light, helper, lane staging (all MXFP4-gated and thus
  dead for EXL3 mixed); `x3::SetArgs` layout (already has `mix`/`eptr`); single-class
  addressing (`mix.n == 0` everywhere); the S5 swap itself (arenas stay class-pure and
  in-range → tag 0).
- **Losslessness:** identical weight bytes (the ring copy uses the same `esrc`→class-layout
  repack as the arena upload in `cache_loop`), identical per-expert arithmetic
  (`set_mix_kernel` = the same `block_gemv`/`mix_gemv` math as the window kernels, incl.
  `case 10` K5 via `mix_gemv`'s default, `exl3.cu:879`). Only `moe_combine`'s accumulation
  order can differ (batch interleaving) — the same reassociation the single-class chunk path
  already exhibits vs windows. Tag bits are masked before any weight access.

### Alternative considered (recorded for the perf follow-up, not S3)

**Per-class sub-batches:** order `host_list` and the vs space class-major; each class arena is
a uniform `n_c × align256(cblob[c])` region, so per-class `run_set` calls could use the
**non-mix tensor-core tile path** (`sa.lay = clay[c]`, `sa.stride = align256(cblob[c])`) with
zero kernel changes. Rejected for S3: it spreads class logic across `plan_streams`/`run_vram`/
`run_set`/batch building, fragments every batch into ≤4 launches, and its slot-order
addressing is more exposed to S5 swaps (positions trade places inside an arena; `eptr` is
exactly the decoupling S5 introduced). It is the natural S4 if the portable `set_mix_kernel`
throughput disappoints (hybrid: tiles for arena experts + tagged ring for streamed ones).

## 4. Every site that must change

| # | Site | Change |
|---|---|---|
| 1 | `src/ds4/ds4_model.cu:2629-2633` | gate (a): drop `! any_mixed` (currently inert — `L` empty here — make it honest) |
| 2 | `src/ds4/ds4_model.cu:2831-2843` | gate (b): delete the `chunk = 0` forcing + windows message; keep verbose note |
| 3 | `src/ds4/ds4_model.cu:11564` `plan_streams` | mixed vs scheme: `host_list` = `slot[e]<0` only; `vs = e` (VRAM) / `n_exp + j` (lane) |
| 4 | `src/ds4/ds4_model.cu:11462` `layer_chunk` | `n_vs` mixed-aware (`n_exp + host_list.size()`), assert ≤ 2047 |
| 5 | `src/ds4/ds4_model.cu:11543-11557` `layer_chunk` | `v0 = n_exp + bt.j0`; `v_to` over `[0, n_exp)`; ring `run_set` call gains the eptr array |
| 6 | `src/ds4/ds4_model.cu:4437` `run_vram` | mixed → single eptr `run_set` over `[i0,i1)`, no `n_main` split |
| 7 | `src/ds4/ds4_model.cu:4445` `run_set` | explicit `eptr` param replaces `wbase == l.vram_base` test (`4459-4462`) |
| 8 | `src/ds4/ds4_model.cu:3996` `copy_ahead` | write tagged `ring_ptr_pin[slot] = dst \| (cls+1)` + slice H2D on `c.cs` before `filled[half]` |
| 9 | `src/ds4/ds4_model.cu:3614` `alloc_chunks` | alloc `c.ring_ptr_dev` / `c.ring_ptr_pin` (`ring_n × 8` each); add the two fields to `Chunk` (`~697`) |
| 10 | `src/ds4/ds4_model.cu:3273` | `NH = 1` when any mixed layer (guard the NH>1 class-0 path of `stream_expert`) |
| 11 | `src/ds4/ds4_model.cu:~8131 / ~8231` | S5 mixed switch/queue: `&& !prompt_active` (new atomic around `forward_chunks`) |
| 12 | `src/kernels/cuda/exl3.cu:850` `mix_cls` | tag-first: `if (p & 0xff) return (p & 0xff) - 1;` |
| 13 | `src/kernels/cuda/exl3.cu:893, 919, 953, 982, 1447, 1652` | `blob = (const uint8_t*)(ptr & ~0xffull)` in the six mix kernels |

Unchanged-but-load-bearing (verify in review, do not edit): `stream_expert` NH==1 mixed branch
(`3860-3876`), `place_experts` arena packing (`5218-5266`), `run_set`'s zero-row early-return
(`set_mix_kernel` `1442-1445` — makes interleaved lane experts in the VRAM range free),
`moe_sort` (vs-agnostic), `l.blob` = max class blob (ring sizing), TP-within-layer refusal
(`4884-4887`).

## 5. VRAM cost

- **New device memory:** `c.ring_ptr_dev` = `ring_n × 8` B per GPU. With `DS4_RING_MB=400` and
  `ring_stride ≈ align256(max-class blob)` (~2.5–3.5 MB at 2.7bpw, FFN 2048/member … rig:
  ~7 MB full expert) → `ring_n` ≈ 60–160 → **≈ 0.5–1.3 KB per GPU**. Pinned mirror the same on
  the host. No new arenas, no new staging buffers, no per-class rings.
- **Existing budget reused:** the ring is already sized by the **largest** class's blob, so a
  staged K2 expert wastes `max_cblob − cblob[K2]` of its slot — staging efficiency
  ≈ `mean(cblob)/max(cblob)` (~70–85% for this pack's K2/K3/K4/K5 mix). Bounded by the existing
  `DS4_RING_MB` budget; chunk buffers (`lay_chunk`) unchanged.
- Compute cost (not VRAM): mixed layers' chunk experts run the **portable** `set_mix_kernel`
  (the mixed path never uses the sm80 tensor-core tiles — `exl3.cu:2320` "the portable kernels
  everywhere"). Still expected ≫ windows (all picks on-GPU, staging overlapped); if not, the
  per-class-sub-batch S4 above restores tiles for arena experts.

## 6. Test plan

1. **Lossless vs windows (the S1 path):** mixed 2.7bpw pack, short prompt (~200 tok):
   `--prefill-chunk 0` (windows) vs `DS4_CHUNK=64` and `DS4_CHUNK=256` — compare final logits
   and ~50 greedy continuations (tolerance for `moe_combine` reassociation only, e.g.
   rel. `|Δ| < 1e-2`; greedy tokens must match). Also `DS4_CHUNK=16` (minimum) to stress many
   chunks, empty batches and the batch/bounds edges.
2. **K5 coverage:** confirm chunked prefill routes the 35-expert K5 classes — use the existing
   `dump_y` hook in `run_set` (`~4515`) on a K5-bearing layer; outputs must be non-zero and
   match the window path's per-expert `y` (both go through `mix_gemv` `case 10`).
3. **Single-class regression:** 2.9bpw single-class pack, `DS4_CHUNK=8192`: logits vs the
   pre-patch build (tag=0 → kernels bit-identical; `sa.mix.n == 0` → tile path untouched) +
   prefill tok/s unchanged.
4. **Layer split:** `--tp 1` over the 2×3090 rig (layer ranges): chunked prompt with group
   hand-offs; logits equal to the single-GPU run at the same `DS4_CHUNK`; watch `Model::where`
   (the serve watchdog) for a wedged hand-off. `--tp 2` (intra-layer) must keep refusing mixed
   loads with the existing error.
5. **Prompt-skip path:** a 40k prompt exercising `forward_chunks`' yoco two-pass
   (`kYocoMin 1024`): tokens identical to a forced one-pass run.
6. **S5 interlock:** `DS4_MIX_SWAP=1`: decode windows between chunked prompts (swaps land on
   ticks), then another chunked prompt — `plan_streams` refreshes; no crash, logits stable;
   verify the `prompt_active` guard blocks a mid-run switch.
7. **Prefill bench (the point of S3):** 40k prompt, mixed pack: windows (baseline ~125 tok/s)
   vs chunks at default and `DS4_CHUNK=4096/8192`; record tok/s + `stream_bytes`; acceptance
   per campaign: ≥ 4× windows, target 500+ tok/s. Decode-after-prompt tok/s unchanged.
8. **Boot checks:** the "prompts in windows" message gone for mixed; `NH=1` logged for mixed
   packs; `ds4: L.. n of 384 experts in VRAM in 4 class arenas` lines unchanged.

## 7. Key risks

- **Portable-kernel throughput** for mixed chunk experts (no tensor-core tiles in the mix set
  path) — measured by test 6; mitigation = per-class sub-batch S4 (hybrid tiles+tag).
- **S5 swap vs running chunk** desync of `vs_pin`/`d_maddr`/`hh` — mitigated by the
  `prompt_active` guard (site 11); without it the window is small but real (in-flight swap
  completing during a prompt).
- **NH>1 mixed staging unimplemented** — forced `NH=1` costs host-side read parallelism for
  in-place (file) expert reads on multi-NUMA rigs; RAM-resident CPU lane (this rig's shape:
  regions pinned whole) is unaffected.
- **Ring slot waste** for sub-max classes — bounded, existing budget; tune `DS4_RING_MB` if
  staged-expert throughput dips.
- **`bounds` capacity:** `n_exp + host_list.size()` must stay ≤ 2047 (`c.bounds` / `bounds_pin`
  are 2048 ints) — true for 384-expert DS4 layers (≤ 767); assert anyway.
- **Line drift:** the tree is being actively modified (`ds4_model.cu` dirty during inspection);
  re-anchor by the quoted code strings, not line numbers, when patching.
- **gfx11 `wmma_set_mix_kernel` lacks `case 10`** (K5 falls into `block_gemm_w<8>`) — HIP-only,
  not this rig, but the tag patch touches this kernel; file upstream separately.
