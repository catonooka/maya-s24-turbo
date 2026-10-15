import json, time, urllib.request
URL = 'http://127.0.0.1:8080/v1/chat/completions'
def chat(prompt, max_tokens=400, tune=None, effort=None):
    body = {'model':'glm-5.3-flash','max_tokens':max_tokens,'temperature':0.6,
            'messages':[{'role':'user','content':prompt}]}
    if tune: body['strata_tune'] = tune
    if effort: body['reasoning_effort'] = effort
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    t0=time.time(); r=json.load(urllib.request.urlopen(req, timeout=900)); dt=time.time()-t0
    u=r.get('usage',{}); ct=u.get('completion_tokens',0)
    return ct, dt, ct/dt
chat('hi', 20)  # warmup
probes = [
    ('prose', 'Write a 150-word pitch for a space tourism company called Nova.', 400),
    ('facts', 'List 10 facts about Mars, one line each.', 400),
    ('code',  'Write Python: a complete binary search tree class with insert, delete, search. Code only.', 600),
]
import sys
label = sys.argv[1] if len(sys.argv)>1 else 'run'
tune = json.loads(sys.argv[2]) if len(sys.argv)>2 else None
tot=[]
for name,p,mt in probes:
    vals=[]
    for attempt in range(2):
        ct,dt,tps = chat(p, mt, tune=tune)
        if attempt==1:
            print(f'[{label}] {name}: {ct} tok | {tps:.1f} t/s | {dt:.1f}s'); tot.append(tps)
print(f'[{label}] MEAN: {sum(tot)/len(tot):.1f} t/s')
