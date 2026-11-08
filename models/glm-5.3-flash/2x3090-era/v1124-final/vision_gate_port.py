#!/usr/bin/env python3
"""Vision gate (model-agnostic port): color ID, counting, spatial — for any :port/model."""
import base64
import json
import struct
import sys
import urllib.request
import zlib

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm-5.3-flash"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def png(w, h, pix):
    raw = b""
    for row in pix:
        raw += b"\x00" + b"".join(bytes(p) for p in row)
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


RED, BLUE, WHITE, BLACK = (220, 30, 30), (30, 60, 220), (255, 255, 255), (20, 20, 20)

def solid(c):
    return png(64, 64, [[c] * 64 for _ in range(64)])

def squares(n):
    size, gap, canvas = 60, 40, 400
    while 40 + n * (size + gap) > canvas - 40 and size > 10:
        size -= 4
        gap = max(12, gap - 4)
    pix = [[WHITE] * canvas for _ in range(canvas)]
    y0 = (canvas - size) // 2
    for i in range(n):
        x0 = 40 + i * (size + gap)
        for y in range(y0, y0 + size):
            for x in range(x0, x0 + size):
                pix[y][x] = BLACK
    return png(canvas, canvas, pix)

def bars():
    return png(400, 400, [[RED if x < 200 else BLUE for x in range(400)] for _ in range(400)])


TESTS = [
    ("red", solid(RED), "What color is this image? One word.", "red"),
    ("blue", solid(BLUE), "What color is this image? One word.", "blue"),
    ("3 squares", squares(3), "How many black squares? Just the number.", "3"),
    ("5 squares", squares(5), "How many black squares? Just the number.", "5"),
    ("left bar", bars(), "The image is two vertical halves. What color is the LEFT half? One word.", "red"),
]

score = 0
for name, img, prompt, expect in TESTS:
    b64 = base64.b64encode(img).decode()
    body = json.dumps({"model": MODEL, "max_tokens": 700, "temperature": 0.0,
                       "messages": [{"role": "user", "content": [
                           {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                           {"type": "text", "text": prompt}]}]}).encode()
    try:
        r = json.loads(urllib.request.urlopen(
            urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=900).read())
        m = r["choices"][0]["message"]
        text = (m.get("content") or m.get("reasoning_content") or "")
        ok = expect.lower() in text.lower()
        score += ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} — {text[:70].replace(chr(10),' ')}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
print(f"VISION {score}/{len(TESTS)}")
