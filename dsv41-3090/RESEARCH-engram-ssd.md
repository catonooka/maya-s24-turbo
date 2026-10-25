# DSV41 Engram trên SSD — research notes (2026-10-18)

Rig: 2x RTX 3090 (48G VRAM), 128G RAM, 1TB NVMe. Bộ model đã tải đủ:
`~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` = 43 EXL3 shards (196.3G) + sidecars
(exl3_k_map.json, quantization_config.json 52M codebook, index) + engram shards 47/48
(189G, sha256 verified). Tổng ~385G. RAM 128G < bảng engram 189G → phải để NVMe + page cache.

## Bảng Engram là gì (thông số chính xác)

- 2 layer (L1, L14), mỗi layer ~384M rows × 256 dim, FP8 E4M3 + E8M0 block-scale (32/group)
- 1 row = 256B weights + 8B scales = 264B
- Mỗi token: 2 layer × 3 n-gram size (2/3/4-gram) × 8 hash heads = **48 random row gathers/token**
- Row id = hash bucket (prime modulus, XOR-multiplied compressed token ids) → **đã là random-access O(1) sẵn, không cần sort theo key**
- N-gram freq theo Zipf → phần lớn row hot, page cache hấp thụ được nhiều
- Tham chiếu: PCIe5 NVMe (Samsung 9100 PRO) random 4K QD1 ~80.6us; 48 gather QD8 ~1.14ms/token
  = ~3.7% của budget decode 30ms (FreeToken issue #444, đo qua WSL2 VHDX còn pessimistic)

## "Pre-compile" theo nghĩa đen — cái gì đáng compile offline

Không repo nào sort bảng theo hash key (vô ích: row id = bucket, fixed-size). Cái người ta
compile trước:

1. **Row-file repack** (Mia `start.sh pack`): safetensors → raw `.bin` per-rank
   (`engram-l{1,14}-r<rank>of{3,4}.bin`), weights + scales ở offset cố định → runtime là
   `pread(row_id * 264)` thuần, không parse header. 1 lần ~10 phút.
2. **Token map offline** (zLLM): compressed token map (129,280 u32) export ra
   `engram_token_map.bin` + multipliers/primes/offsets expand thành static table → runtime
   không chạy tokenizer normalization nữa. Hash front-end = pure integer ops.
3. **Pack layout EXL3** (sybil `pack/prepare_pack.py`): symlink shards + dựng text-only
   config với per-module EXL3 map + expand wo_a/compressor/indexer sang FP16 +
   `host-plan.json`. **Repo này giải đúng cái "pack layout" đang block ~/vllm-030 của mình.**

## Kỹ thuật làm SSD lookup nhanh (xếp theo impact cho rig này)

### 1. Prestage ngoài CUDA graph — nktlabs/dsv41-flash-engram-disk-prestage (HF)
- Vendor path `DSV41_ENGRAM_DISK=1` đọc row *trong* forward pass → buộc eager → 26.1 tok/s
- Fix: move gather vào `prepare_inputs` (chạy eager, ngay trước `run_fullgraph`):
  hash kernel sớm → `preadv` vào pinned buffer → 1 H2D copy vào GPU tensor persistent
  → CUDA graph giữ nguyên vì trong captured region không đụng file
- **26.1 → 80.2 tok/s (1 stream), 449.7 tok/s @ 8 streams, RAM 18G. Pinned RAM = 113.3**
- Có parity mode `ENGRAM_PRESTAGE_CHECK=1` assert byte-identical
- Patch ~450 dòng / 4 file, bind-mount không cần rebuild image

### 2. Graph-safe host callback — sybil-solutions/dsv41-flash-offload (★145, EXL3 trên 1x 3090!)
- `engram_disk.cpp`: mmap bảng (page cachefront) + `cudaLaunchHostFunc` callback,
  OMP parallel memcpy 264B rows vào pinned buffer → H2D → dequant kernel GPU có sẵn
- Không gọi CUDA nào trong callback → launch được từ trong CUDA graph
- Decode 16.4–20.9 tok/s trên 1 con 3090 + 256G DDR4 (262k ctx, fp8 KV 1.5G)
- Prefill staged-DMA 16k chunk: 565–745 tok/s (D061)
- Yêu cầu host 256G (pin 190.5G experts) — mình 128G thì pin hết experts không được,
  nhưng engram tier + graph pattern lấy nguyên được

### 3. RWF_NOWAIT hai tầng — yunwei37.com (DGX Spark TP4)
- Row đã nằm trong page cache → `preadv(RWF_NOWAIT)` đồng bộ, không qua thread pool
- Chỉ row EAGAIN (lạnh) mới trả về blocking pool 32 threads
- Bottleneck thật không phải SSD mà là future/GIL/thread overhead của cached read
- **Hot-read component 8.3x nhanh hơn**; fallback tự động nếu kernel không hỗ trợ

### 4. Bounded direct-mapped RAM cache — 0xSero/sybil 4x-rtx-pro-6000 (`adapter/row_store.cpp`)
- Cache direct-mapped `id % slots`, 264B/slot + key 8B, 256 sharded mutex
- `OFFLOAD_MODE=nvme` = O_DIRECT pread (không double-cache page cache), miss đọc đúng row gốc
- Budget 64G DDR5; decode 200–230 tok/s single stream (4x RTX Pro 6000, DSpark)
- Với 128G RAM mình có thể dành ~50–70G làm cache này (Zipf → hit rate cao)

### 5. Load-time warmup + MADV_RANDOM — zLLM (zhuai.tech, TQ, 8x W7900D CPU-side)
- Lúc load: advise sequential + readahead → touch từng page → set `MADV_RANDOM`
  (runtime random access không thrash readahead). Cold fault dời về lúc boot
- Không dùng mlock (warmup ≠ locked); page vẫn reclaim được dưới áp lực
- Kèm: AVX-512 BF16 packed WKV (300M/layer, 1 lần load phục vụ 30 token),
  NUMA bind fixed worker team (L1/L14 2 socket khác nhau), first-touch theo worker,
  pinned staging gần node GPU khi upload (L14 90ms → 15ms), early projection stage-0
  overlap (decode 19.4 → 20.45 tok/s)
- Bài viết: https://zhuai.tech/blog/deepseek-v41-flash-engram/ (serie, mỗi ngày 1 bài)

### 6. Quantize chính bảng — satindergrewal EngramQ4 (dual RTX Pro 6000)
- Engram table Q4 → 190G → ~95G, decode 89–94 tok/s @2K, GSM8K 96%, quality gates pass
- Lợi ích kép cho SSD: footprint nhỏ hơn = page cache hit rate cao hơn hẳn
- Là derivative hobbyist (abliterated) — số liệu không rigorous, nhưng kỹ thuật Q4 bảng thì valid

### 7. SemiAnalysis (tham khảo, B200 serving)
- Trên B200/B300: SSD offload thua DRAM cả tok/$ lẫn P90 (52M vs 121M tok/$)
- Native UVA (chọn row + dequant trực tiếp trên GPU, tránh CPU round-trip) là hướng đúng
  nhưng chưa optimize được; GDS không bật được
- Kết luận cho datacenter, không áp dụng cho mình (mình không có choice: 128G < 189G)

## Kết luận cho rig 2x3090 / 128G

- Thứ tự ưu tiên nếu tự build: **(1) mmap + page cache + prestage ngoài graph →
  (2) RWF_NOWAIT hai tầng → (3) direct-mapped cache ~50G trong RAM → (4) warmup lúc load**
- Muốn nhanh hơn nữa: EngramQ4-style quantize bảng (halve → cache hit rate tăng mạnh)
- Đường tắt: fork `sybil-solutions/dsv41-flash-offload` — gần nhất với hardware mình
  (3090 sm_86, EXL3, vLLM lazymio backport + vllm_exl3 plugin + Ampere patches).
  Khác biệt duy nhất: nó cần 256G RAM để pin hết 15,360 experts; mình 128G → phải cắt
  expert tier (CPU tier + elastic VRAM cache giữ, phần pin giảm, chấp nhận chậm hơn)
  hoặc chờ build pack riêng với host-plan fit 128G
- Repo đó cũng có sẵn `pack/prepare_pack.py` + `expected-sha256.json` → test pack layout
  cho ~/vllm-030 luôn được

## Links

- https://github.com/sybil-solutions/dsv41-flash-offload (3090 + DDR4 + NVMe, ★145)
- https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000 (row_store.cpp, ★73)
- https://huggingface.co/nktlabs/dsv41-flash-engram-disk-prestage (prestage patch)
- https://www.yunwei37.com/blog/deepseek-v41-engram-nowait-dgx-spark (RWF_NOWAIT)
- https://zhuai.tech/blog/deepseek-v41-flash-engram/ (zLLM, TQ)
- https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks (SGLang TP3, pack per-rank)
- https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks (EXL3 vLLM TP2)
- https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark (TP3 lane, ★109)
- https://github.com/satindergrewal/DeepSeek-v4.1-Flash-EXL3-2.0bpw-Ablit-EngramQ4-SM120-Dual-RTX-Pro-6000 (EngramQ4)
- https://newsletter.semianalysis.com/p/engrams-embedding-entendre-codesign (SSD vs DRAM)
- https://github.com/FlashML-org/FreeToken/issues/444 (access pattern + NVMe benchmark)
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/engram.py (reference impl)
