#!/usr/bin/env python3
"""Vision gate for the maya-ds4 server: generates PNGs in pure stdlib, sends
multimodal chat requests, checks answers. Tests color ID, shape count, size compare."""
import base64
import json
import struct
import urllib.request
import zlib

URL = "http://127.0.0.1:8080/v1/chat/completions"


def png(w, h, pix):
    """pix: list of rows, each row list of (r,g,b)."""
    raw = b""
    for row in pix:
        raw += b"\x00" + b"".join(bytes(p) for p in row)
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


RED = (220, 30, 30)
BLUE = (30, 60, 220)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)

def solid(w, h, c):
    return png(w, h, [[c] * w for _ in range(h)])

def squares(bg, fg, n, size=60, gap=40, canvas=400):
    # fit n squares: shrink size if needed so 40 + n*(size+gap) <= canvas-40
    while 40 + n * (size + gap) > canvas - 40 and size > 10:
        size -= 4
        gap = max(12, gap - 4)
    pix = [[bg] * canvas for _ in range(canvas)]
    y0 = (canvas - size) // 2
    for i in range(n):
        x0 = 40 + i * (size + gap)
        for y in range(y0, y0 + size):
            for x in range(x0, x0 + size):
                pix[y][x] = fg
    return png(canvas, canvas, pix)

def bars(canvas=400):
    """Left half red, right half blue."""
    pix = []
    for y in range(canvas):
        row = []
        for x in range(canvas):
            row.append(RED if x < canvas // 2 else BLUE)
        pix.append(row)
    return png(canvas, canvas, pix)

TESTS = [
    ("solid red 64x64", solid(64, 64, RED), "What color is this image? Answer with just the color word.", "red"),
    ("solid blue 64x64", solid(64, 64, BLUE), "What color is this image? Answer with just the color word.", "blue"),
    ("three squares", squares(WHITE, BLACK, 3), "How many black squares are in this image? Answer with just the number.", "3"),
    ("five squares", squares(WHITE, BLACK, 5), "How many black squares are in this image? Answer with just the number.", "5"),
    ("two bars", bars(), "The image is split into two vertical halves. What color is the LEFT half? One word.", "red"),
]


def ask(prompt, img_png):
    b64 = base64.b64encode(img_png).decode()
    body = json.dumps({
        "model": "deepseek-v4.1-flash", "max_tokens": 700, "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    msg = out["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


score = 0
for name, img, prompt, expect in TESTS:
    try:
        text = ask(prompt, img)
        ok = expect.lower() in text.lower()
        score += ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} (want {expect!r}, got {text[:80]!r})")
    except Exception as e:
        print(f"{name}: ERROR {e}")
print(f"VISION SCORE {score}/{len(TESTS)}")
