import requests, json
U="http://127.0.0.1:8080/v1/chat/completions"
def ask(p, mt=200, t=0.0):
    r=requests.post(U,json={"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":t,"top_p":1.0,"stream":False},timeout=600)
    r.raise_for_status(); j=r.json(); m=j["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning_content") or "")

# 1. arithmetic chain (EXL3 weak spot)
a=ask("Compute step by step: (17*19) + (23*8) - 145. Reply with just the final number.")
print("chain-math:", "PASS" if "362" in a else "FAIL", repr(a.strip()[:50]))
# 2. 17x19 simple
b=ask("What is 17 times 19? Reply with just the number.")
print("17x19:", "PASS" if "323" in b else "FAIL", repr(b.strip()[:30]))
# 3. greedy determinism
d1=ask("Name three primary colors, comma separated.",50); d2=ask("Name three primary colors, comma separated.",50)
print("determinism:", "PASS" if d1==d2 else "FAIL")
# 4. needle @2k (un-patterned)
filler=" ".join(f"Dossier {i}: the fleet meets at harbor {i} at dawn." for i in range(280))
n=ask(filler+" Dossier 250: the fleet password is BANANA-7741. What is the fleet password from Dossier 250? Reply with just the password.")
print("needle2k:", "PASS" if "BANANA-7741" in n else "FAIL", repr(n.strip()[:40]))
# 5. needle @30k (long-context, exercises the skip path over the cut)
import random
words=("harbor lantern sediment glacier magnet orbit quiver thicket cobalt ember raster viaduct monsoon kestrel "
       "azimuth trellis fathom bramble zephyr anvil cascade dovetail ").split()
rng=random.Random(99)
big=" ".join(rng.choice(words) for _ in range(24000))
n2=ask(big+" IMPORTANT: the vault code buried above is ZEPHYR-9042. What is the vault code? Reply with just the code.", mt=50)
print("needle30k:", "PASS" if "ZEPHYR-9042" in n2 else "FAIL", repr(n2.strip()[:40]))
# 6. ASCII garbage
g=ask("In one short sentence, what is a hammer for?")
bad=sum(1 for ch in g if ord(ch)>0x2100)
print("ascii:", "PASS" if bad<3 else "FAIL", repr(g.strip()[:50]))
# 7. exact instruction following
c=ask("Write exactly 4 lines, each line exactly the word 'apple'. Nothing else.",60)
lines=[l for l in c.strip().splitlines() if l.strip()]
print("lines4:", "PASS" if len(lines)==4 and all(l.strip().lower().startswith("apple") for l in lines) else "FAIL", repr(c.strip()[:60]))
