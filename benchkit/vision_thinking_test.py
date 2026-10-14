"""Full matrix: thinking levels (off/low/high/max) x both APIs + vision paths.
Run against live maya :8080. Every check asserts hard; soft metrics printed."""
import json, time, urllib.request, base64, zlib, struct, sys

BASE = 'http://127.0.0.1:8080'
results = []

def make_png(w, h, top_rgb, bot_rgb):
    def chunk(t, d):
        c = t + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    raw = b''
    for y in range(h):
        row = top_rgb if y < h // 2 else bot_rgb
        raw += b'\x00' + bytes(row * w)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr)
            + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))

def post(path, body, timeout=600, stream=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)

def oai(body, timeout=600):
    r = post('/v1/chat/completions', body, timeout)
    return json.load(r)

def anth(body, timeout=600):
    r = post('/v1/messages', body, timeout)
    return json.load(r)

def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        results.append((name, 'PASS', f'{time.time()-t0:.1f}s', detail or ''))
        print(f'PASS {name} ({time.time()-t0:.1f}s) {detail or ""}')
    except Exception as e:
        results.append((name, 'FAIL', f'{time.time()-t0:.1f}s', str(e)[:200]))
        print(f'FAIL {name}: {str(e)[:250]}')

Q = 'If a shirt costs $27 and is discounted 20%, what is the sale price? Answer with just the number.'
IMQ = 'What are the two colored halves of this image, top and bottom? Answer in the form "top=X bottom=Y".'

png_small = base64.b64encode(make_png(96, 48, (255, 0, 0), (0, 0, 255))).decode()
png_big = base64.b64encode(make_png(640, 480, (255, 215, 0), (0, 128, 0))).decode()

def reasoning_of(r):
    m = r['choices'][0]['message']
    rc = m.get('reasoning_content') or ''
    return rc, (m.get('content') or '')

def think_tokens_reasonable(rc):  # thinking text present
    assert len(rc.strip()) > 10, f'reasoning_content too short: {rc[:80]!r}'
    return f'{len(rc)} think-chars'

def answer_has_216(rc, txt):
    assert '21.6' in txt or '216' in txt.replace('.', '').replace(',', ''), f'bad answer: {txt[:120]!r}'

# ---------- OpenAI thinking levels ----------
def t_oai_default():
    r = oai({'model':'glm-5.3-flash','max_tokens':900,'temperature':0.6,'messages':[{'role':'user','content':Q}]})
    rc, txt = reasoning_of(r); think_tokens_reasonable(rc); answer_has_216(rc, txt)
    return f'default(high): {len(rc)} chars'
check('oai-thinking-default-high', t_oai_default)

def t_oai(level, name):
    r = oai({'model':'glm-5.3-flash','max_tokens':900,'temperature':0.6,'reasoning_effort':level,
             'messages':[{'role':'user','content':Q}]})
    rc, txt = reasoning_of(r); think_tokens_reasonable(rc); answer_has_216(rc, txt)
    return f'{len(rc)} chars'
check('oai-thinking-low',  lambda: t_oai('low','low'))
check('oai-thinking-high', lambda: t_oai('high','high'))
check('oai-thinking-max',  lambda: t_oai('max','max'))

def t_oai_off_none():
    r = oai({'model':'glm-5.3-flash','max_tokens':400,'temperature':0.6,'reasoning_effort':'none',
             'messages':[{'role':'user','content':Q}]})
    rc, txt = reasoning_of(r)
    assert len(rc.strip()) == 0, f'expected NO reasoning, got {rc[:100]!r}'
    answer_has_216(rc, txt)
    return 'reasoning empty, direct answer'
check('oai-thinking-off-none', t_oai_off_none)

def t_oai_off_kwargs():
    r = oai({'model':'glm-5.3-flash','max_tokens':400,'temperature':0.6,
             'chat_template_kwargs': {'enable_thinking': False},
             'messages':[{'role':'user','content':Q}]})
    rc, txt = reasoning_of(r)
    assert len(rc.strip()) == 0, f'expected NO reasoning, got {rc[:100]!r}'
    answer_has_216(rc, txt)
    return 'reasoning empty'
check('oai-thinking-off-enable_thinking-false', t_oai_off_kwargs)

def t_oai_responses_style():
    r = oai({'model':'glm-5.3-flash','max_tokens':900,'temperature':0.6,
             'reasoning': {'effort': 'low'},'messages':[{'role':'user','content':Q}]})
    rc, txt = reasoning_of(r); think_tokens_reasonable(rc); answer_has_216(rc, txt)
    return f'{len(rc)} chars'
check('oai-thinking-responses-effort-low', t_oai_responses_style)

# ---------- Anthropic thinking levels ----------
def anth_bits(r):
    blocks = r.get('content', [])
    think = ''.join(b.get('thinking','') for b in blocks if b.get('type') == 'thinking')
    text = ''.join(b.get('text','') for b in blocks if b.get('type') == 'text')
    return think, text

def t_anth(level_cfg, name):
    body = {'model':'glm-5.3-flash','max_tokens':900,'temperature':0.6,
            'messages':[{'role':'user','content':Q}]}
    if level_cfg: body['thinking'] = level_cfg
    r = anth(body)
    think, text = anth_bits(r)
    assert len(think.strip()) > 10, f'{name}: no thinking block ({r.get("content")!r:.200})'
    assert '216' in text.replace('.','').replace(',',''), f'bad answer: {text[:100]!r}'
    return f'{len(think)} chars'
check('anth-thinking-default', lambda: t_anth(None, 'default'))
check('anth-thinking-budget-1k-low', lambda: t_anth({'type':'enabled','budget_tokens':1000}, 'low'))
check('anth-thinking-budget-5k-high', lambda: t_anth({'type':'enabled','budget_tokens':5000}, 'high'))
check('anth-thinking-budget-12k-max', lambda: t_anth({'type':'enabled','budget_tokens':12000}, 'max'))

def t_anth_off():
    r = anth({'model':'glm-5.3-flash','max_tokens':400,'temperature':0.6,
              'thinking': {'type': 'disabled'},'messages':[{'role':'user','content':Q}]})
    think, text = anth_bits(r)
    assert len(think.strip()) == 0, f'expected no thinking, got {think[:100]!r}'
    assert '216' in text.replace('.','').replace(',',''), f'bad answer: {text[:100]!r}'
    return 'no thinking block, direct answer'
check('anth-thinking-disabled', t_anth_off)

# ---------- Vision ----------
def t_vision_oai():
    r = oai({'model':'glm-5.3-flash','max_tokens':400,'temperature':0.6,
             'messages':[{'role':'user','content':[
                 {'type':'text','text':IMQ},
                 {'type':'image_url','image_url':{'url':f'data:image/png;base64,{png_small}'}}]}]})
    txt = r['choices'][0]['message']['content'] or ''
    low = txt.lower()
    assert 'top' in low and 'red' in low and 'bottom' in low and 'blue' in low, f'bad: {txt[:150]!r}'
    return txt[:60].replace('\n',' ')
check('vision-openai-image-url', t_vision_oai)

def t_vision_anth():
    r = anth({'model':'glm-5.3-flash','max_tokens':400,'temperature':0.6,
              'messages':[{'role':'user','content':[
                  {'type':'text','text':IMQ},
                  {'type':'image','source':{'type':'base64','media_type':'image/png','data':png_small}}]}]})
    text = ''.join(b.get('text','') for b in r.get('content',[]) if b.get('type')=='text')
    low = text.lower()
    assert 'top' in low and 'red' in low and 'bottom' in low and 'blue' in low, f'bad: {text[:150]!r}'
    return text[:60].replace('\n',' ')
check('vision-anthropic-image-block', t_vision_anth)

def t_vision_big():
    r = oai({'model':'glm-5.3-flash','max_tokens':400,'temperature':0.6,
             'messages':[{'role':'user','content':[
                 {'type':'text','text':'What are the two colored halves of this image, top and bottom? Form: "top=X bottom=Y".'},
                 {'type':'image_url','image_url':{'url':f'data:image/png;base64,{png_big}'}}]}]})
    txt = (r['choices'][0]['message']['content'] or '').lower()
    assert 'top' in txt and ('gold' in txt or 'yellow' in txt) and 'bottom' in txt and 'green' in txt, f'bad: {txt[:150]!r}'
    return '640x480 gold/green ok'
check('vision-large-640x480', t_vision_big)

def t_vision_thinking():
    r = oai({'model':'glm-5.3-flash','max_tokens':1200,'temperature':0.6,'reasoning_effort':'high',
             'messages':[{'role':'user','content':[
                 {'type':'text','text':'Look at the image, think carefully, then answer: which half is RED, top or bottom? One word.'},
                 {'type':'image_url','image_url':{'url':f'data:image/png;base64,{png_small}'}}]}]})
    rc, txt = reasoning_of(r)
    assert len(rc.strip()) > 10, f'no reasoning with vision: {rc[:80]!r}'
    low = txt.lower()
    assert 'top' in low, f'bad answer: {txt[:100]!r}'
    return f'think {len(rc)} chars + correct answer'
check('vision-plus-thinking-high', t_vision_thinking)

def t_vision_off():
    r = oai({'model':'glm-5.3-flash','max_tokens':300,'temperature':0.6,'reasoning_effort':'none',
             'messages':[{'role':'user','content':[
                 {'type':'text','text':'Which half is RED, top or bottom? One word.'},
                 {'type':'image_url','image_url':{'url':f'data:image/png;base64,{png_small}'}}]}]})
    rc, txt = reasoning_of(r)
    assert len(rc.strip()) == 0, f'reasoning leaked: {rc[:80]!r}'
    assert 'top' in txt.lower(), f'bad: {txt[:100]!r}'
    return 'vision + thinking-off ok'
check('vision-plus-thinking-off', t_vision_off)

# ---------- summary ----------
print()
n_pass = sum(1 for r in results if r[1] == 'PASS')
print(f'=== {n_pass}/{len(results)} PASS ===')
for name, st, dt, det in results:
    if st == 'FAIL': print(f'  FAILED: {name}: {det}')
sys.exit(0 if n_pass == len(results) else 1)
