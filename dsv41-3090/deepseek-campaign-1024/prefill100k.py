import requests, random, time
U="http://127.0.0.1:8080/v1/chat/completions"
words=("harbor lantern sediment glacier magnet orbit quiver thicket cobalt ember raster palindrome "
       "viaduct monsoon kestrel azimuth trellis fathom bramble zephyr anvil cascade dovetail ").split()
rng=random.Random(int(time.time()))
filler=" ".join(rng.choice(words) for _ in range(62000))  # ~100k tokens, unique every run
t0=time.time()
r=requests.post(U,json={"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":filler+"\n\nReply with only: done"}],"max_tokens":8,"temperature":0,"stream":False},timeout=1800)
r.raise_for_status()
print("client wall", round(time.time()-t0,1), "s | prompt_tokens", r.json()["usage"]["prompt_tokens"])
