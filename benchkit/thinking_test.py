"""Thinking test suite — verifies GLM-5.3-Flash thinking works properly on maya :8080.

Ground truth for "thinking ran" is TOKEN INFLATION (completion_tokens with effort=max
must far exceed effort=off on the same question) — immune to the known parse quirk where
a pre-injected <think> can leave reasoning_content empty on some turns.
"""
import json, time, urllib.request, sys

BASE = 'http://127.0.0.1:8080'
results = []

def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)

def oai(body, timeout=900):
    return json.load(post('/v1/chat/completions', body, timeout))

def oai_level(q, level=None, max_tokens=3000, temperature=0.6, stream=False):
    body = {'model': 'glm-5.3-flash', 'max_tokens': max_tokens, 'temperature': temperature,
            'messages': [{'role': 'user', 'content': q}]}
    if level: body['reasoning_effort'] = level
    r = oai(body)
    m = r['choices'][0]['message']
    u = r.get('usage', {})
    return (m.get('reasoning_content') or ''), (m.get('content') or ''), u.get('completion_tokens', 0)

# Genuinely-hard questions (each needs real multi-step reasoning)
HARD = {
 'marbles': ("I have 3 boxes. Each box contains 5 bags. Each bag holds 4 marbles. "
             "I remove 7 marbles altogether. How many marbles remain? Answer with just the number.",
             lambda t: '53' in t.replace(',', '')),
 'days':    ("Today is Wednesday. What day of the week will it be exactly 100 days from now? "
             "Answer with just the day name.", lambda t: 'friday' in t.lower()),
 'percent': ("What is 17% of 350? Answer with just the number.",
             lambda t: '59.5' in t.replace(',', '')),
 'knaves':  ("On an island knights always tell the truth and knaves always lie. You meet A and B. "
             "A says: 'We are both knaves.' What are A and B exactly? "
             "Answer as 'A=knight|knave B=knight|knave'.",
             lambda t: 'a=knave' in t.lower().replace(' ', '') and 'b=knight' in t.lower().replace(' ', '')),
}

def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        results.append((name, True)); print(f'PASS {name} ({time.time()-t0:.1f}s) {detail or ""}')
    except AssertionError as e:
        results.append((name, False)); print(f'FAIL {name}: {e}')
    except Exception as e:
        results.append((name, False)); print(f'FAIL {name}: {str(e)[:200]}')

# ---------- collect data ----------
print('collecting: off (4q), low (2q), high (2q), max (4q) ...')
data = {}   # data[qname][level] = (rc, txt, tokens)
for qn in HARD:
    rc, txt, tok = oai_level(HARD[qn][0], level='none', max_tokens=600)
    data[qn] = {'none': (rc, txt, tok)}
for qn in ('knaves', 'days'):
    data[qn]['low'] = oai_level(HARD[qn][0], level='low')
for qn in ('knaves', 'days'):
    data[qn]['high'] = oai_level(HARD[qn][0], level='high')
for qn in HARD:
    data[qn]['max'] = oai_level(HARD[qn][0], level='max')

for qn in HARD:
    for lv in ('none', 'low', 'high', 'max'):
        if lv in data[qn]:
            rc, txt, tok = data[qn][lv]
            print(f'  {qn:8s} {lv:5s}: {tok:4d} tok, rc {len(rc):5d} ch, ok={HARD[qn][1](txt)}')

# ---------- T1: thinking runs (token inflation, off vs max) ----------
def t1():
    infl = {qn: data[qn]['max'][2] - data[qn]['none'][2] for qn in HARD}
    print('   inflation (max - none tokens):', infl)
    assert all(v >= 30 for v in infl.values()), f'thinking did not run somewhere: {infl}'
    return f'mean +{sum(infl.values())/4:.0f} tokens'
check('T1-thinking-runs-token-inflation', t1)

# ---------- T2: levels scale ----------
def t2():
    lo = sum(data[q]['low'][2] for q in ('knaves', 'days'))
    hi = sum(data[q]['high'][2] for q in ('knaves', 'days'))
    mx = sum(data[q]['max'][2] for q in ('knaves', 'days'))
    print(f'   tokens low={lo} high={hi} max={mx}')
    assert mx > lo, f'max must exceed low: {mx} vs {lo}'
    assert hi >= lo, f'high below low: {hi} vs {lo}'
    return f'low {lo} <= high {hi} < max {mx}'
check('T2-levels-scale-low-high-max', t2)

# ---------- T3: off is truly off ----------
def t3():
    for qn in ('marbles', 'percent'):
        rc, txt, tok = data[qn]['none']
        assert len(rc.strip()) == 0, f'{qn}: reasoning_content leaked: {rc[:60]!r}'
        assert '<think>' not in txt, f'{qn}: <think> leaked into content'
        assert HARD[qn][1](txt), f'{qn}: wrong answer off: {txt[:60]!r}'
    return 'no reasoning, no <think> leak, answers correct'
check('T3-off-clean-no-leak', t3)

# ---------- T4: reasoning visible (>=2/3 hard qs at max) ----------
def t4():
    vis = [qn for qn in ('knaves', 'days', 'marbles') if len(data[qn]['max'][0].strip()) > 20]
    chars = [len(data[qn]['max'][0]) for qn in HARD]
    print(f'   visible: {vis}, rc chars per q: {chars}')
    assert len(vis) >= 2, f'reasoning_content visible in only {len(vis)}/3'
    return f'{len(vis)}/3 visible, max {max(chars)} chars'
check('T4-reasoning-visible-max', t4)

# ---------- T5: thinking keeps answers correct on traps ----------
def t5():
    bad = [qn for qn in HARD if not HARD[qn][1](data[qn]['max'][1])]
    assert not bad, f'wrong answers at max effort: {bad}'
    return '4/4 hard questions correct'
check('T5-correctness-with-thinking', t5)

# ---------- T6: streaming reasoning deltas ----------
def t6():
    body = {'model': 'glm-5.3-flash', 'max_tokens': 2000, 'temperature': 0.6,
            'reasoning_effort': 'max', 'stream': True, 'stream_options': {'include_usage': True},
            'messages': [{'role': 'user', 'content': HARD['knaves'][0]}]}
    r = post('/v1/chat/completions', body)
    tch = td = cch = cd = 0; usage = None
    for line in r:
        line = line.decode().strip()
        if not line.startswith('data: '): continue
        d = line[6:]
        if d == '[DONE]': break
        j = json.loads(d)
        if j.get('usage'): usage = j['usage']
        for ch in j.get('choices', []):
            delta = ch.get('delta', {})
            if delta.get('reasoning_content'): tch += len(delta['reasoning_content']); td += 1
            if delta.get('content'): cch += len(delta['content']); cd += 1
    print(f'   stream: {td} reasoning deltas ({tch} ch), {cd} content deltas ({cch} ch)')
    assert tch > 30, f'no reasoning streamed ({tch} ch)'
    assert cch > 3 and usage and usage.get('completion_tokens', 0) > 20
    return f'{tch} reasoning chars streamed'
check('T6-streaming-reasoning', t6)

# ---------- T7: Anthropic budget mapping ----------
def t7():
    def anth(budget, max_tokens=3000):
        body = {'model': 'glm-5.3-flash', 'max_tokens': max_tokens, 'temperature': 0.6,
                'thinking': {'type': 'enabled', 'budget_tokens': budget},
                'messages': [{'role': 'user', 'content': HARD['knaves'][0]}]}
        r = json.load(post('/v1/messages', body))
        blocks = r.get('content', [])
        th = sum(len(b.get('thinking', '')) for b in blocks if b.get('type') == 'thinking')
        tx = ''.join(b.get('text', '') for b in blocks if b.get('type') == 'text')
        u = r.get('usage', {})
        return th, tx, u.get('output_tokens', 0)
    th1, tx1, to1 = anth(1000)
    th2, tx2, to2 = anth(12000)
    print(f'   budget 1k: think {th1} ch, {to1} out-tok | budget 12k: think {th2} ch, {to2} out-tok')
    assert HARD['knaves'][1](tx1) and HARD['knaves'][1](tx2), 'wrong answers via anthropic path'
    assert to2 > to1, f'12k budget must think more than 1k: {to2} vs {to1}'
    return f'1k {to1} < 12k {to2} out-tokens, both correct'
check('T7-anthropic-budgets', t7)

# ---------- T8: determinism at temp 0 with thinking ----------
def t8():
    a = oai_level(HARD['days'][0], level='low', temperature=0)
    b = oai_level(HARD['days'][0], level='low', temperature=0)
    assert a[1] == b[1], f'content differs:\n{a[1][:80]!r}\n{b[1][:80]!r}'
    assert HARD['days'][1](a[1]), f'wrong deterministic answer: {a[1][:60]!r}'
    return 'temp-0 identical answers with thinking on'
check('T8-determinism-temp0-thinking', t8)

# ---------- T9: thinking + forced tool call ----------
def t9():
    tools = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {'type': 'object',
             'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
    r = oai({'model': 'glm-5.3-flash', 'max_tokens': 1500, 'temperature': 0.6,
             'reasoning_effort': 'high', 'tools': tools,
             'tool_choice': {'type': 'function', 'function': {'name': 'get_weather'}},
             'messages': [{'role': 'user', 'content': "What's the weather in Tokyo?"}]})
    m = r['choices'][0]['message']
    tcs = m.get('tool_calls') or []
    assert tcs, f'no tool call: {str(m)[:150]}'
    args = json.loads(tcs[0]['function']['arguments'])
    assert 'tokyo' in args.get('city', '').lower(), f'bad args: {args}'
    return f"tool call OK {tcs[0]['function']['name']}({args})"
check('T9-thinking-plus-tools', t9)

# ---------- T10: multi-turn hygiene ----------
def t10():
    msgs = [{'role': 'user', 'content': HARD['marbles'][0]},
            {'role': 'assistant', 'content': '53'},
            {'role': 'user', 'content': 'Say exactly: LANE OK'}]
    r = oai({'model': 'glm-5.3-flash', 'max_tokens': 400, 'temperature': 0,
             'reasoning_effort': 'none', 'messages': msgs})
    m = r['choices'][0]['message']
    assert 'LANE OK' in (m.get('content') or ''), f'bad follow-up: {str(m)[:120]}'
    assert len((m.get('reasoning_content') or '').strip()) == 0, 'reasoning leaked into off-turn'
    return 'follow-up clean after thinking turn'
check('T10-multiturn-hygiene', t10)

# ---------- summary ----------
print()
n_pass = sum(1 for _, ok in results if ok)
print(f'=== {n_pass}/{len(results)} PASS ===')
sys.exit(0 if n_pass == len(results) else 1)
