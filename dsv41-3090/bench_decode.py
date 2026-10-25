#!/usr/bin/env python3
"""Decode-phase profiler: samples CPU%, GPU% during a generation, then dumps cpu_tier stats."""
import subprocess, threading, time, requests, json

URL = 'http://127.0.0.1:8000/v1/chat/completions'
samples = {'cpu': [], 'gpu': []}
stop = threading.Event()

def sampler():
    while not stop.is_set():
        try:
            out = subprocess.run(['docker','stats','dsv41','--no-stream','--format','{{.CPUPerc}}'],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            samples['cpu'].append(out)
        except Exception as e:
            samples['cpu'].append(f'err {e}')
        try:
            out = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,power.draw',
                                  '--format=csv,noheader','-i','0'],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            samples['gpu'].append(out)
        except Exception as e:
            samples['gpu'].append(f'err {e}')
        time.sleep(2)

t = threading.Thread(target=sampler); t.start()
time.sleep(2)
t0 = time.time()
r = requests.post(URL, json={"model":"deepseek-v4.1-flash",
                             "messages":[{"role":"user","content":"Write a long essay about mountains."}],
                             "max_tokens":250,"chat_template_kwargs":{"enable_thinking":False}})
dt = time.time() - t0
stop.set(); t.join()
ct = r.json()['usage']['completion_tokens']
print(f"decode: {ct} tok in {dt:.1f}s = {ct/dt:.2f} tok/s")
print("CPU% samples:", samples['cpu'][:8])
print("GPU samples:", samples['gpu'][:8])
