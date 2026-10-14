"""Thinking-level verification v2: levels must (a) think on a question that needs it,
(b) scale verbosity low -> max (proves effort kwarg reaches the template), (c) stay correct."""
import json, time, urllib.request, sys

BASE = 'http://127.0.0.1:8080'

def oai(body, timeout=900):
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def anth(body, timeout=900):
    req = urllib.request.Request(BASE + '/v1/messages', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

# a question that punishes skipping thinking
TRAP = ('A farmer has 17 sheep. All but 9 run away. How many sheep does he have left? '
        'Then: if he buys 3 more, how many? Answer as "left=X bought=Y".')

def think_and_answer_oai(level=None, max_tokens=4000):
    body = {'model':'glm-5.3-flash','max_tokens':max_tokens,'temperature':0.6,
            'messages':[{'role':'user','content':TRAP}]}
    if level: body['reasoning_effort'] = level
    r = oai(body)
    m = r['choices'][0]['message']
    rc = m.get('reasoning_content') or ''
    return rc, (m.get('content') or '')

def think_and_answer_anth(budget=None, max_tokens=4000):
    body = {'model':'glm-5.3-flash','max_tokens':max_tokens,'temperature':0.6,
            'messages':[{'role':'user','content':TRAP}]}
    if budget: body['thinking'] = {'type':'enabled','budget_tokens':budget}
    r = anth(body)
    blocks = r.get('content', [])
    think = ''.join(b.get('thinking','') for b in blocks if b.get('type')=='thinking')
    text = ''.join(b.get('text','') for b in blocks if b.get('type')=='text')
    return think, text

def ok_answer(txt):
    t = txt.replace(',', '')
    assert ('left=9' in t.replace(' ','') or 'left = 9' in t or ('9' in t and '12' in t)), f'bad: {txt[:150]!r}'

print('--- OpenAI levels (trap question) ---')
res = {}
for level in ['low', 'high', 'max']:
    rc, txt = think_and_answer_oai(level)
    ok_answer(txt)
    res[level] = len(rc.strip())
    print(f'  effort={level}: think {res[level]:5d} chars, answer OK')

rc, txt = think_and_answer_oai()  # default
ok_answer(txt)
print(f'  default:   think {len(rc.strip()):5d} chars, answer OK')

print('--- Anthropic budgets (trap question) ---')
ares = {}
for name, budget in [('low<2k', 1000), ('high<8k', 5000), ('max>8k', 12000)]:
    think, text = think_and_answer_anth(budget)
    ok_answer(text)
    ares[name] = len(think.strip())
    print(f'  budget {name:8s}: think {ares[name]:5d} chars, answer OK')
think, text = think_and_answer_anth()
ok_answer(text)
print(f'  default:          think {len(think.strip()):5d} chars, answer OK')

# --- differentiation gates ---
fails = []
def gate(cond, msg):
    print(f'  [{"PASS" if cond else "FAIL"}] {msg}')
    if not cond: fails.append(msg)

print('--- gates ---')
gate(res['max'] > 50, f"max thinks substantively ({res['max']} chars > 50)")
gate(res['max'] >= res['low'], f"verbosity scales low->max (low {res['low']} <= max {res['max']})")
gate(ares['max>8k'] > 50, f"anthropic max budget thinks substantively ({ares['max>8k']} chars)")
gate(ares['max>8k'] >= ares['low<2k'], f"anthropic verbosity scales ({ares['low<2k']} <= {ares['max>8k']})")

print()
if fails:
    print(f'=== {len(fails)} GATE FAILURE(S) ===')
    sys.exit(1)
print('=== ALL THINKING LEVELS VERIFIED ===')
