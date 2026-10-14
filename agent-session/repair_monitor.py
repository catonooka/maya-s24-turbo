#!/usr/bin/env python3
"""One repair turn: feed monitor.py bugs back to the model, get fixed file."""
import json, re, os, urllib.request, sys

URL = 'http://127.0.0.1:8080/v1/chat/completions'
HOME = os.path.expanduser('~/maya-longtask')
history = [json.loads(l) for l in open(os.path.join(HOME, 'session.jsonl'))]

# get real log lines for the parser
import subprocess
stat_lines = subprocess.run("grep -a 'glm stat' /home/aisever/project-maya/maya-maya-s24.log | tail -3",
                            shell=True, capture_output=True, text=True).stdout

err = f"""Your monitor.py has two bugs found in live testing. Fix and re-emit the COMPLETE file.

BUG 1 — crash at shutdown:
TypeError: 'Event' object is not callable  (threading.Event stored as self._stop, then called self._stop())

BUG 2 — glm stat parsing yields null decode_tok_s / vram_hit. Your parser does not match the real log lines.
Real lines from /home/aisever/project-maya/maya-maya-s24.log (binary-ish file: contains NUL bytes, use errors='replace'):
{stat_lines}
Expected: decode_tok_s=55.57, vram_hit=81.76 from line 1 style. Note the log file contains NUL bytes and lines
accumulate across boots — parse the LAST matching line. Also 'concurrency' from /v1/status is under
['concurrency']['serving'].

Sample JSONL your run produced (wrong): {{"decode_tok_s": null, "vram_hit": null, "concurrency": null}}

Emit the complete fixed file as ```python:benchkit/monitor.py ...```"""

history.append({'role': 'user', 'content': err})
body = {'model': 'glm-5.3-flash', 'messages': history, 'max_tokens': 12000, 'temperature': 0.3}
req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
r = json.load(urllib.request.urlopen(req, timeout=1200))
answer = r['choices'][0]['message']['content']
print(f"repair turn: {r['usage']['completion_tokens']} tok out")
m = re.search(r"```python:benchkit/monitor\.py\n(.*?)```", answer, re.DOTALL)
if not m:
    print("NO FILE IN ANSWER"); sys.exit(1)
code = m.group(1).rstrip('\n') + '\n'
import ast; ast.parse(code)
open(os.path.join(HOME, 'benchkit/monitor.py'), 'w').write(code)
# append to session for future turns
with open(os.path.join(HOME, 'session.jsonl'), 'a') as f:
    f.write(json.dumps({'role': 'user', 'content': err}) + '\n')
    f.write(json.dumps({'role': 'assistant', 'content': answer}) + '\n')
print("monitor.py replaced, syntax OK")
