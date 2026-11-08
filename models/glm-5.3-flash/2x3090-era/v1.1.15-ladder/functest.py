"""Functional test suite for maya :8080 — tools, vision, stream, long-ctx, native+zero-arg calls."""
import json, time, urllib.request, sys

BASE = 'http://127.0.0.1:8080'
results = []

def post(path, body, timeout=300, stream=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)

def chat(body, timeout=300):
    r = post('/v1/chat/completions', body, timeout)
    return json.load(r)

def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        results.append((name, 'PASS', f'{time.time()-t0:.1f}s', detail or ''))
        print(f'PASS {name} ({time.time()-t0:.1f}s) {detail or ""}')
    except Exception as e:
        results.append((name, 'FAIL', f'{time.time()-t0:.1f}s', str(e)[:160]))
        print(f'FAIL {name}: {str(e)[:200]}')

# 1. plain chat
def t_plain():
    r = chat({'model':'glm-5.3-flash','max_tokens':100,'temperature':0.6,
              'messages':[{'role':'user','content':'Say exactly: LANE OK'}]})
    txt = r['choices'][0]['message'].get('content','')
    assert len(txt) > 5, txt
    return txt[:40].replace('\n',' ')
check('chat-basic', t_plain)

# 2. native tool call (GLM arg_key form) via forced tool_choice
tools = [{'type':'function','function':{'name':'get_weather','parameters':{'type':'object',
         'properties':{'city':{'type':'string'},'unit':{'type':'string'}},'required':['city']}}}]
def t_tool():
    r = chat({'model':'glm-5.3-flash','max_tokens':300,'temperature':0.6,
              'tools':tools,'tool_choice':{'type':'function','function':{'name':'get_weather'}},
              'messages':[{'role':'user','content':'Weather in Tokyo in celsius?'}]})
    m = r['choices'][0]['message']
    tcs = m.get('tool_calls') or []
    assert tcs, f'no tool_calls: {str(m)[:160]}'
    a = tcs[0]['function']
    assert 'tokyo' in a['arguments'].lower(), a['arguments']
    return f"{a['name']}({a['arguments'][:40]})"
check('tool-forced', t_tool)

# 3. zero-arg tool call (the bug we fixed: bare NAME in tool_call)
tools0 = [{'type':'function','function':{'name':'skills_list','parameters':{'type':'object','properties':{}}}}]
def t_zeroarg():
    r = chat({'model':'glm-5.3-flash','max_tokens':300,'temperature':0.6,
              'tools':tools0,'tool_choice':{'type':'function','function':{'name':'skills_list'}},
              'messages':[{'role':'user','content':'List my skills'}]})
    m = r['choices'][0]['message']
    tcs = m.get('tool_calls') or []
    assert tcs and tcs[0]['function']['name'] == 'skills_list', str(m)[:160]
    return f"zero-arg OK: {tcs[0]['function']['arguments']!r}"
check('tool-zero-arg', t_zeroarg)

# 4. multi-turn with tool result
def t_multiturn():
    r1 = chat({'model':'glm-5.3-flash','max_tokens':300,'temperature':0.6,'tools':tools,
               'messages':[{'role':'user','content':'Weather in Paris?'}]})
    tc = (r1['choices'][0]['message'].get('tool_calls') or [None])[0]
    assert tc, 'no call'
    r2 = chat({'model':'glm-5.3-flash','max_tokens':200,'temperature':0.6,
               'messages':[{'role':'user','content':'Weather in Paris?'},
                           {'role':'assistant','content':'','tool_calls':[{'id':tc['id'],'type':'function',
                            'function':tc['function']}]},
                           {'role':'tool','tool_call_id':tc['id'],'content':'{"temp_c": 18, "cond": "cloudy"}'}]})
    txt = r2['choices'][0]['message'].get('content','')
    assert len(txt) > 3, txt
    return f"second turn OK ({len(txt)} chars)"
check('tool-multiturn', t_multiturn)

# 5. streaming (SSE)
def t_stream():
    body = {'model':'glm-5.3-flash','max_tokens':80,'temperature':0.6,'stream':True,
            'messages':[{'role':'user','content':'Count 1 to 5'}]}
    resp = post('/v1/chat/completions', body, stream=True)
    chunks = 0; got = []
    for line in resp:
        line = line.decode().strip()
        if line.startswith('data: ') and line != 'data: [DONE]':
            d = json.loads(line[6:])
            delta = d['choices'][0].get('delta',{})
            if delta.get('content'): got.append(delta['content'])
            chunks += 1
    assert chunks > 3 and ''.join(got), f'{chunks} chunks'
    return f'{chunks} chunks, {len("".join(got))} chars streamed'
check('streaming-sse', t_stream)

# 6. vision (base64 tiny png)
def t_vision():
    import base64, zlib, struct
    def chunk(t, d):
        c = t + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', 64, 32, 8, 2, 0, 0, 0)
    raw = b''.join(b'\x00' + bytes([255, 0, 0] * 64) for _ in range(32))
    png = base64.b64encode(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr)
                           + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')).decode()
    r = chat({'model':'glm-5.3-flash','max_tokens':100,'temperature':0.6,
              'messages':[{'role':'user','content':[
                  {'type':'text','text':'What is in this image? One sentence.'},
                  {'type':'image_url','image_url':{'url':f'data:image/png;base64,{png}'}}]}]})
    txt = r['choices'][0]['message'].get('content','')
    assert len(txt) > 3, str(r)[:200]
    return f'vision reply: {txt[:40]!r}'
check('vision-image', t_vision)

# 7. long-context probe (~65k tokens: repeat filler, ask a detail from the start)
def t_longctx():
    fact = 'The secret codeword buried in this document is ZEPHYR-77.'
    filler = 'The quick brown fox jumps over the lazy dog near the riverbank at dusk. ' * 1100  # ~66k tok
    msgs = [{'role':'user','content':fact + '\n\n' + filler + '\n\nWhat was the secret codeword mentioned at the very start? Answer with just the codeword.'}]
    t0 = time.time()
    r = chat({'model':'glm-5.3-flash','max_tokens':200,'temperature':0.6,'messages':msgs}, timeout=900)
    txt = (r['choices'][0]['message'].get('content','') or '').upper()
    assert 'ZEPHYR-77' in txt, txt[:100]
    u = r.get('usage',{})
    return f"ctx {u.get('prompt_tokens','?')} tok, found codeword in {time.time()-t0:.0f}s"
check('long-ctx-65k', t_longctx)

# 8. sampling params respected
def t_sampling():
    r = chat({'model':'glm-5.3-flash','max_tokens':10,'temperature':0.0,
              'messages':[{'role':'user','content':'Say OK'}]})
    assert r['choices'][0]['message'].get('content'), 'empty'
    return 'greedy OK'
check('sampling-greedy', t_sampling)

fails = [r for r in results if r[1] == 'FAIL']
print(f'\n=== {len(results)-len(fails)}/{len(results)} PASS ===')
sys.exit(1 if fails else 0)
