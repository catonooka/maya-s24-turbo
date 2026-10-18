#!/usr/bin/env python3
"""Measure host<->GPU PCIe throughput during decode via CUDA performance counters.
Reads dram+pcie counters from NVML inside the container's GPU process."""
import subprocess, time, statistics

# use nvidia-smi pmon for SM occupancy + dmon for rx/tx PCIe if available
def sample():
    out = subprocess.run(['nvidia-smi', 'dmon', '-s', 'pt', '-d', '1', '-c', '1', '-i', '1'],
                         capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip() and not l.startswith('#')]

print("sampling pcie rx (MB/s) via dmon during decode; start load...")
subprocess.Popen(['python3', '/home/aisever/project-maya/dsv41-3090/bench_batch.py', '8001', '1'],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(8)
vals = []
for _ in range(10):
    for row in sample():
        f = row.split()
        # dmon -s p gives pwr gtemp mtemp only in this driver; try full
        vals.append(row)
    time.sleep(1.5)
print('\n'.join(str(v) for v in vals[:20]))
