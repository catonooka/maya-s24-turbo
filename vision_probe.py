import json, urllib.request, base64, zlib, struct

def make_png(w=64, h=32):
    def chunk(t, d):
        c = t + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    raw = b''.join(b'\x00' + bytes([255, 0, 0] * w) for _ in range(h))  # solid red
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr)
            + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))

png = base64.b64encode(make_png()).decode()
body = {'model':'glm-5.3-flash','max_tokens':100,'temperature':0.6,
        'messages':[{'role':'user','content':[
            {'type':'text','text':'What is the one dominant color of this image? Answer with one word.'},
            {'type':'image_url','image_url':{'url':f'data:image/png;base64,{png}'}}]}]}
req = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',
                             data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
r = json.load(urllib.request.urlopen(req, timeout=300))
txt = r['choices'][0]['message'].get('content','')
print('VISION REPLY:', txt[:80])
assert 'red' in txt.lower(), txt
print('VISION PASS')
