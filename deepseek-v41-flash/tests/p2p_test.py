import torch, time
a = torch.ones(64*1024*1024, dtype=torch.uint8, device="cuda:0")
b = torch.empty_like(a, device="cuda:1")
torch.cuda.synchronize()
t = time.time()
for _ in range(50):
    b.copy_(a, non_blocking=True)
torch.cuda.synchronize()
dt = time.time() - t
print(f"P2P GPU0->GPU1 64MiB x50: {dt*1000:.0f} ms = {50*64/1024/dt:.1f} GB/s")
can = torch.cuda.can_device_access_peer(0, 1)
print("peer access enabled-able:", can)
