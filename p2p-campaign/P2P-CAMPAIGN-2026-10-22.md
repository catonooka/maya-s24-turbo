# P2P/vBIOS campaign — full findings (2026-10-22)

## What we built (all working)
1. **aikitoria 615.71.09-p2p fork installed**: BIOS Above-4G + ReBAR +
   `amd_iommu=on iommu=pt` + fork kernel modules (built locally, dropped in
   updates/dkms over DKMS 595). One-shot systemd unit (nv-p2p-oneshot)
   handles install at boot before X. Result: `topo -p2p r` = **OK** (was
   CNS), driver 615.71.09 both sides. Stock backup:
   /root/nvidia-stock-7.0.0-38-generic-20261022-044619
2. **GPU1 vBIOS is the P2P blocker**: FE card (vBIOS 94.02.27.00.0A) BAR1
   caps at 256MB (lspci: supported sizes stop at 256MB). dmesg: "P2P
   mailbox area expected from RM but no valid address is installed gpu=1".
   Transfers fall back to host path 10.9-11 GB/s (measured, p2p_check).
   GPU0 (new card) = 32GB BAR1 fine.

## The ReBAR ROM (researched + verified, ready to flash)
- Official NVIDIA FE ReBAR image: `~/rtx3090-fe-rebar/NVIDIA.RTX3090.24576.210318.rom`
  (999,424 B, MD5 f6d43fccfa0002f13a48642dc43e113b) — matches BOTH NVIDIA's
  own updater manifest (still live: us.download.nvidia.com/Windows/rebar/1.0/
  NVIDIA_ResizableBAR_Firmware_Updater_1.3-x64.exe) and TechPowerUp DB 231642.
- Exact pairing: our 94.02.27.00.0A → same version ReBAR build 2021-03-18
  (version string unchanged; build date is the signal). PG136 SKU10 /
  subsystem 147D confirmed = our card.
- Flash sequence (nvflash_linux, GPU1 = index 1, bus 41:00.0):
  backup `-i1 --save` → flash `-i1 -6` → reboot → verify BAR1 32768 MiB.
  Rollback = reflash backup.
- BLOCKED ON: nvflash Linux binary. TPU (only host) behind anti-bot
  fingerprint wall (/.firewall pipeline, token'd, unfetchable). Windows
  nvflash 5.867 obtained but its .sys driver needs
  IoGetDeviceAttachmentBaseRef — unimplemented in Wine 10, service can't
  start. User asked to grab NVFlash Linux 5.735.0 zip from TPU manually.

## Research conclusions (the honest numbers)
- **Layer-split decode is P2P-immune**: maya's boundary = ONE 64KB/token
  host hop (3.2 MB/s at 50 t/s); issue #333 A/B NVLink vs PCIe halves =
  44.0 vs 46.2 t/s. GLM/DSV41 tp1 decode gains ~nothing from P2P.
- **MAYA_GLM_PREFILL_CALC=1 (PR #270) = the real PCIe win, shipped**:
  waiting GPU computes RAM-tier expert share over its own link. MEASURED
  HERE: GLM prefill 437→569 @1.5K (+30%), 770→880 @6K (+14%), decode
  unchanged, gate 5/5. LIVE on :8081 now.
- **MAYA_GLM_DUAL_PREFILL (PR #121)**: authors measured ~0 on PCIe-only
  (peer copy eats the working GPU's link; +24-28% was NVLink-only). Skip.
- **DSV41 tp2 allreduce = the P2P payoff**: chunk all-reduce host-bounces
  (~1.7ms/copy driver staging, ds4_model.cu xfer_put kernel). With BAR1
  fixed on both cards: engine patch (peer-access xfer) could push 783 →
  900-1100 t/s prefill. Community TP rigs: +19-22% spec-decode, +28%
  prefill with patched driver + NCCL_P2P_LEVEL=SYS.
- llama.cpp GGML_CUDA_P2P: opt-in since Apr 2026 (crash-fix demote);
  helps -sm tensor, not layer split. ik_llama: no P2P. exllamav3: P2P
  probe in to_device (EXLLAMA_NO_P2P_COPY override exists).
- loktar00 quote "P2P within 1% of NVLink on x16 PCIe4": NOT supported by
  his own data (26 GB/s PCIe vs 52 GB/s NVLink = half). Gains are real
  but half-NVLink, not parity.

## Operational gotchas learned
- systemd-oomd kills user-slice GLM loads at RAM pinning (~940s) — system
  slice (`systemd-run --slice=system.slice`) is mandatory (unit glm-serve3)
- dcgm-exporter silently holds /dev/nvidia* — must docker stop before
  rmmod; same for X (isolate multi-user.target)
- DKMS reinstalls stock modules on kernel update → P2P silently dies;
  rebuild fork against new kernel (script pattern in nv-p2p-oneshot)
- NVFlash Windows under Wine: dead end (IoGetDeviceAttachmentBaseRef)
- TPU anti-bot: /.firewall + token + fingerprint pipeline — not worth
  defeating; ask user to download manually

## Current serving state
- GLM EXL3 :8081 (glm-serve3, PREFILL_CALC=1) — 880 t/s @6K prefill, 45-46 decode
- DSV41 :8080 stopped (GPUs needed for GLM; restart = serve-ds4-dual.sh)
- Machine in text mode awaiting flash (X stopped, modules unloaded)
