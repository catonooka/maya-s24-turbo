import json, time, urllib.request, threading, sys
URL = 'http://127.0.0.1:8080/v1/chat/completions'
def one(prompt, max_tokens, out, idx):
    body = {'model':'glm-5.3-flash','max_tokens':max_tokens,'temperature':0.6,
            'messages':[{'role':'user','content':prompt}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    t0=time.time(); r=json.load(urllib.request.urlopen(req, timeout=1200)); dt=time.time()-t0
    ct=r.get('usage',{}).get('completion_tokens',0)
    out[idx]=(ct, dt, ct/dt)
def run(n, label):
    prompts = [
        'Write a 150-word pitch for a space tourism company called Nova.',
        'List 10 facts about Mars, one line each.',
        'Write Python: a complete binary search tree class with insert, delete, search. Code only.',
        'Explain how JPEG compression works, step by step.',
    ]
    out=[None]*n
    ths=[threading.Thread(target=one, args=(prompts[i%4], 500, out, i)) for i in range(n)]
    t0=time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall=time.time()-t0
    tot=sum(o[0] for o in out)
    per=' | '.join(f'{o[2]:.1f}' for o in out)
    print(f'[{label}] C{n}: {tot} tok in {wall:.1f}s -> AGG {tot/wall:.1f} t/s (per-stream: {per})')
    return tot/wall
one('hi', 20, [None], 0)  # warmup
run(1, sys.argv[1] if len(sys.argv)>1 else 'par')
run(2, sys.argv[1] if len(sys.argv)>1 else 'par')
run(4, sys.argv[1] if len(sys.argv)>1 else 'par')
