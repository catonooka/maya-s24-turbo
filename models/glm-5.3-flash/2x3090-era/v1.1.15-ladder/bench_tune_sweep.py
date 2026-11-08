#!/usr/bin/env python3
"""Sweep strata_tune knobs with the bench.py protocol (2 runs/probe, take run 2).
Usage: bench_tune_sweep.py spec 6,8,10   |   bench_tune_sweep.py missskip 0.12,0.14,0.16"""
import json, time, urllib.request, sys

URL = 'http://127.0.0.1:8080/v1/chat/completions'
PROBES = [
    ('prose', 'Write a 150-word pitch for a space tourism company called Nova.', 400),
    ('facts', 'List 10 facts about Mars, one line each.', 400),
    ('code',  'Write Python: a complete binary search tree class with insert, delete, search. Code only.', 600),
]

def chat(prompt, max_tokens, tune, temp=0.6):
    body = {'model': 'glm-5.3-flash', 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': temp}
    if tune: body['strata_tune'] = tune
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t0
    ct = r.get('usage', {}).get('completion_tokens', 0)
    return ct, dt

def bench(tune, label):
    # warmup per setting (first call = route learning for this tune combo)
    chat('hi', 20, tune)
    tot = []
    for name, p, mt in PROBES:
        for attempt in range(2):
            ct, dt = chat(p, mt, tune)
            if attempt == 1:
                print(f'  {name}: {ct} tok | {ct/dt:.1f} t/s', flush=True)
                tot.append(ct/dt)
    m = sum(tot)/len(tot)
    print(f'{label}: MEAN {m:.1f} t/s', flush=True)
    return m

if __name__ == '__main__':
    kind = sys.argv[1]
    vals = sys.argv[2].split(',')
    out = {}
    for v in vals:
        v = float(v)
        if kind == 'spec':
            tune = {'spec': int(v)}
            label = f'spec={int(v)}'
        elif kind == 'missskip':
            tune = {'miss_skip_w': v, 'miss_skip_from': 23}
            label = f'miss_skip_w={v}'
        else:
            raise SystemExit('kind: spec|missskip')
        out[label] = bench(tune, label)
    print('\nsummary:')
    for k, m in out.items():
        print(f'  {k}: {m:.1f} t/s')
