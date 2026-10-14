#!/usr/bin/env python3
"""Long-task agent driver: GLM-5.3-Flash builds benchkit stage by stage.
Resumable, self-correcting (validator feedback loop), logs everything."""
import json, time, urllib.request, sys, os, re, traceback

URL = 'http://127.0.0.1:8080/v1/chat/completions'
HOME = os.path.expanduser('~/maya-longtask')
PKG = os.path.join(HOME, 'benchkit')
SESSION = os.path.join(HOME, 'session.jsonl')
STAGES_F = os.path.join(HOME, 'stages_done.json')
DRIVER_LOG = os.path.join(HOME, 'driver.log')

sysmsg = ("You are a senior Python engineer. You are building a benchmark+health toolkit called "
          "benchkit for a local LLM server (OpenAI-compatible, http://127.0.0.1:8080, model "
          "glm-5.3-flash, 262k context). Work strictly stage by stage. When asked to write code, "
          "emit COMPLETE files as fenced blocks: ```python:benchkit/<filename> ...``` — every file "
          "complete and runnable, no placeholders, no 'TODO', no truncation. Keep answers focused.")

STAGES = [
 ("S1 SPEC", "Write benchkit/SPEC.md — the full specification for the toolkit: 5 modules "
   "(bench.py, functest.py, monitor.py, report.py, cli.py). For each: purpose, CLI, key functions, "
   "output formats. Facts to embed: server is OpenAI-compatible at http://127.0.0.1:8080, model "
   "name glm-5.3-flash, /v1/status endpoint exists (JSON with context, uptime_s), engine log at "
   "/home/aisever/project-maya/maya-maya-s24.log has 'glm stat:' health lines. Token counts MUST "
   "come from usage.completion_tokens. Emit the file as ```text:benchkit/SPEC.md ...```"),
 ("S2 CORE", "Now implement benchkit/bench.py per SPEC: single-stream probes (prose/facts/code, "
   "2 runs each, report run-2) + concurrency ladder C1/C2/C4 with threads. CLI: --probes, --conc "
   "1,2,4, --json out. Robust HTTP with retries and timeouts. Emit ```python:benchkit/bench.py```"),
 ("S3 BATTERY", "Implement benchkit/functest.py per SPEC: checks = plain chat, forced tool call "
   "(tools API, tool_choice), zero-arg tool call, multiturn with tool result, SSE streaming "
   "(stream=true), greedy determinism (temp 0 same prompt twice, byte-identical), long-ctx needle "
   "(~15k filler + codeword), ASCII sanity. Each check PASS/FAIL with detail; exit code = fails. "
   "Emit ```python:benchkit/functest.py```"),
 ("S4 MONITOR", "Implement benchkit/monitor.py per SPEC: sampler thread polling /v1/status + "
   "tailing glm stat lines from the engine log, appends JSONL (ts, uptime, decode_tok_s, vram_hit, "
   "concurrency), thresholds: decode<20 alert, server-down alert; --duration, --interval flags; "
   "clean shutdown summary. Emit ```python:benchkit/monitor.py```"),
 ("S5 CLI", "Implement benchkit/report.py (results JSON -> self-contained HTML report: tables for "
   "bench probes/concurrency + functest checks + monitor summary, inline CSS, no external deps) "
   "and benchkit/cli.py + __init__.py + __main__.py exposing: quick (1 probe, C1), full (all probes, "
   "C1/C2/C4 + functest), monitor, report. Emit ALL files as separate fenced blocks."),
 ("S6 ANALYZE", "STAGE6_PLACEHOLDER"),
]

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(DRIVER_LOG, 'a') as f: f.write(line + '\n')

def chat(messages, max_tokens=12000, temperature=0.3, retries=5):
    body = {'model': 'glm-5.3-flash', 'messages': messages, 'max_tokens': max_tokens,
            'temperature': temperature}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            t0 = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=1200))
            dt = time.time() - t0
            msg = r['choices'][0]['message']
            content = msg.get('content') or ''
            usage = r.get('usage', {})
            log(f"    model turn: {usage.get('completion_tokens',0)} tok out, "
                f"{usage.get('prompt_tokens',0)} in, {dt:.0f}s wall, "
                f"{usage.get('completion_tokens',0)/max(dt,0.001):.1f} t/s")
            return content, usage
        except Exception as e:
            log(f"    HTTP attempt {attempt+1} failed: {e}")
            time.sleep(min(30, 5 * (attempt + 1)))
    raise RuntimeError("model unreachable after retries")

FENCE_RE = re.compile(r"```(\w+):benchkit/([\w./-]+)\n(.*?)```", re.DOTALL)

def extract_files(text):
    files = {}
    for m in FENCE_RE.finditer(text):
        lang, name, body = m.groups()
        files[f"benchkit/{name}"] = body.rstrip('\n') + '\n'
    return files

def validate(stage_name, files):
    problems = []
    if not files:
        return ["no ```lang:benchkit/<file> fenced blocks found in answer"]
    for path, body in files.items():
        if path.endswith('.py'):
            try:
                import ast; ast.parse(body)
            except SyntaxError as e:
                problems.append(f"{path}: SyntaxError line {e.lineno}: {e.msg}")
        if 'TODO' in body or 'pass  # placeholder' in body:
            problems.append(f"{path}: contains TODO/placeholder")
        if len(body) < 200 and not path.endswith('__init__.py'):
            problems.append(f"{path}: suspiciously short ({len(body)} chars)")
    return problems

def run_cmd(cmd, timeout=300):
    import subprocess
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=HOME)
        return p.returncode, p.stdout[-3000:], p.stderr[-3000:]
    except subprocess.TimeoutExpired:
        return 124, '', f'timeout after {timeout}s'

# ---- resumable state ----
done = {}
if os.path.exists(STAGES_F):
    done = json.load(open(STAGES_F))
history = []
if os.path.exists(SESSION):
    for line in open(SESSION):
        try: history.append(json.loads(line))
        except Exception: pass
log(f"driver start: {len(done)} stages done, history {len(history)} msgs")

def save_state():
    json.dump(done, open(STAGES_F, 'w'))
    with open(SESSION, 'w') as f:
        for m in history: f.write(json.dumps(m) + '\n')

sys_prompt = {'role': 'system', 'content': sysmsg}
if not history:
    history.append(sys_prompt)

try:
    for idx, (name, prompt) in enumerate(STAGES):
        if done.get(name, {}).get('ok'):
            continue
        if prompt == "STAGE6_PLACEHOLDER":
            rj = os.path.join(HOME, 'results', 'full_results.json')
            data = open(rj).read() if os.path.exists(rj) else '{}'
            fj = os.path.join(HOME, 'results', 'functest_results.json')
            fdata = open(fj).read() if os.path.exists(fj) else '{}'
            prompt = ("I ran 'python -m benchkit full' — the complete results JSONs are inlined below "
                      "between <results> and <functest> tags. Write benchkit/REPORT.md: tables of ALL "
                      "measured numbers, comparison against baselines (single-stream thinking-on mean "
                      "baseline 59.2 t/s; greedy determinism MUST pass; C2/C4 aggregate expected BELOW "
                      "serial on this 2-GPU layer-split rig), plus a verdict section. "
                      "Emit as ```text:benchkit/REPORT.md ...```\n<results>\n" + data[:40000] +
                      "\n</results>\n<functest>\n" + fdata[:20000] + "\n</functest>")
        log(f"=== {name} ===")
        for attempt in range(1, 4):
            history.append({'role': 'user', 'content': f"[{name}] attempt {attempt}. {prompt}"})
            answer, usage = chat(history)
            history.append({'role': 'assistant', 'content': answer})
            files = extract_files(answer)
            problems = validate(name, files)
            if not problems:
                for path, body in files.items():
                    fp = os.path.join(HOME, path)
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    open(fp, 'w').write(body)
                    log(f"    wrote {path} ({len(body)} bytes)")
                done[name] = {'ok': True, 'attempts': attempt}
                save_state()
                log(f"=== {name} PASSED (attempt {attempt}) ===")
                break
            # stage-specific functional gates
            feedback = "VALIDATION FAILED:\n" + "\n".join(problems)
            if name == 'S2 CORE' and files:
                rc, out, err = run_cmd("timeout 240 python3 benchkit/bench.py --probes prose "
                                       "--conc 1 --json /tmp/s2_smoke.json")
                if rc == 0:
                    done[name] = {'ok': True, 'attempts': attempt, 'smoke': 'pass'}
                    save_state(); log(f"=== {name} PASSED + smoke ==="); break
                feedback += f"\nSMOKE RUN exit {rc}:\nSTDOUT:{out[-800:]}\nSTDERR:{err[-800:]}"
            if name == 'S3 BATTERY' and files:
                rc, out, err = run_cmd("timeout 420 python3 benchkit/functest.py")
                passed = out.count('PASS'); failed = out.count('FAIL')
                if rc == 0 and failed == 0 and passed >= 6:
                    done[name] = {'ok': True, 'attempts': attempt, 'checks': f'{passed} pass'}
                    save_state(); log(f"=== {name} PASSED ({passed} checks) ==="); break
                feedback += f"\nFUNCTEST exit {rc}: {passed} PASS / {failed} FAIL\n{out[-1500:]}"
            if name == 'S4 MONITOR' and files:
                rc, out, err = run_cmd("timeout 120 python3 benchkit/monitor.py --duration 20 "
                                       "--interval 5 --out /tmp/s4_smoke.jsonl")
                if rc == 0:
                    import subprocess
                    n = subprocess.run("wc -l < /tmp/s4_smoke.jsonl", shell=True,
                                       capture_output=True, text=True).stdout.strip()
                    if n.isdigit() and int(n) >= 2:
                        done[name] = {'ok': True, 'attempts': attempt, 'samples': n}
                        save_state(); log(f"=== {name} PASSED ({n} samples) ==="); break
                    feedback += f"\nMONITOR wrote only {n} records"
                else:
                    feedback += f"\nMONITOR exit {rc}:\n{out[-500:]}\n{err[-500:]}"
            if name == 'S5 CLI' and files:
                rc, out, err = run_cmd("cd benchkit && timeout 300 python3 -m benchkit quick "
                                       "--json ../results/quick.json 2>&1 || "
                                       "cd .. && timeout 300 python3 -m benchkit quick "
                                       "--json results/quick.json")
                if rc == 0:
                    done[name] = {'ok': True, 'attempts': attempt}
                    save_state(); log(f"=== {name} PASSED (CLI e2e) ==="); break
                feedback += f"\nCLI e2e exit {rc}:\n{out[-800:]}\n{err[-800:]}"
            if name == 'S6 ANALYZE' and files:
                done[name] = {'ok': True, 'attempts': attempt}
                save_state(); log(f"=== {name} PASSED ==="); break
            log(f"    {name} attempt {attempt} failed validation; feeding back")
            history.append({'role': 'user', 'content': feedback + "\nFix and re-emit ALL files for "
                            "this stage, complete."})
        else:
            log(f"=== {name} FAILED after 3 attempts — stopping for manual fix ===")
            break
    else:
        log("ALL STAGES COMPLETE")
        # S6 prep: run full suite
        os.makedirs(os.path.join(HOME, 'results'), exist_ok=True)
        rc, out, err = run_cmd("timeout 1500 python3 -m benchkit full "
                               "--json results/full_results.json", timeout=1600)
        log(f"FULL RUN exit {rc}\n{out[-2000:]}")
        if rc == 0 and os.path.exists(os.path.join(HOME, 'results', 'full_results.json')):
            json.dump({'s6_ready': True}, open(os.path.join(HOME, 's6_ready.json'), 'w'))
        else:
            json.dump({'s6_ready': False, 'err': err[-2000:]}, open(os.path.join(HOME, 's6_ready.json'), 'w'))
finally:
    save_state()
    log("driver exit")
