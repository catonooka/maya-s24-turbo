#!/usr/bin/env python3
"""benchkit.functest — functional battery for the local LLM server (Stage 3).

Checks (each yields PASS/FAIL with a detail string; exit code = number of fails):
  1. plain_chat        — model answers a simple question coherently.
  2. forced_tool       — tools API with tool_choice forces a named tool call.
  3. zero_arg_tool     — tool with empty parameters schema; args must parse as {}.
  4. multiturn_tool    — tool call -> tool result fed back -> final answer uses it.
  5. sse_stream        — stream=true SSE works end-to-end; tokens from usage.
  6. greedy_determinism— temperature 0, same prompt twice, byte-identical output.
  7. long_ctx_needle   — ~15k-token filler with a codeword; model must recall it.
  8. ascii_sanity      — reply is pure printable ASCII.

HARD RULE (per SPEC): token counts come ONLY from usage.completion_tokens.
HTTP layer: retries with exponential backoff, hard timeouts (same style as bench.py).

Usage:
  python -m benchkit.functest [--checks name,...] [--base-url URL] [--model NAME]
                              [--timeout S] [--retries N] [--json] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import urllib.error as urlerror
import urllib.request as urequest

# ---------------------------------------------------------------- constants

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = "glm-5.3-flash"
DEFAULT_TIMEOUT_S = 120
DEFAULT_RETRIES = 3
RETRY_BACKOFF_S = 1.5

# Tool definitions shared by the tool-call checks.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}

ZERO_ARG_TOOL = {
    "type": "function",
    "function": {
        "name": "ping_server",
        "description": "Ping the game server. Takes no arguments.",
        "parameters": {"type": "object", "properties": {}},
    },
}

# Long-context needle: ~15k tokens of filler (~11k words) + a codeword.
NEEDLE_CODEWORD = "MAGNOLIA-7F"
_FILLER_LINE = "The meadow was quiet, the fence posts leaned slightly, and a " \
               "single crow watched the road with patience. "


def _build_needle_prompt() -> str:
    words_per_line = len(_FILLER_LINE.split())  # ~24 words ≈ ~31 tokens
    target_tokens = 15000
    lines = max(1, (target_tokens * 130) // (words_per_line * 100))
    filler = _FILLER_LINE * lines
    return (
        "Read the passage below. Somewhere in it there is a codeword written "
        "as CODEWORD: <value>. Reply with ONLY the codeword value, nothing else.\n\n"
        "=== PASSAGE START ===\n"
        + filler
        + "CODEWORD: %s\n" % NEEDLE_CODEWORD
        + filler
        + "=== PASSAGE END ===\n"
        "What was the codeword value?"
    )


# ---------------------------------------------------------------- HTTP layer


class HttpError(Exception):
    pass


def _post_raw(base_url: str, payload: Dict[str, Any],
              timeout_s: float) -> Dict[str, Any]:
    """POST /v1/chat/completions (non-streaming). Returns parsed JSON."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urequest.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", "replace"))


def post_with_retry(base_url: str, payload: Dict[str, Any], timeout_s: float,
                    retries: int) -> Dict[str, Any]:
    attempt = 0
    last: Optional[Exception] = None
    while attempt <= retries:
        if attempt > 0:
            time.sleep(RETRY_BACKOFF_S * (2 ** (attempt - 1)))
        try:
            return _post_raw(base_url, payload, timeout_s)
        except urlerror.HTTPError as e:
            code = getattr(e, "code", 0)
            last = HttpError("HTTP %s: %s" % (code, e.reason or e))
            if code < 500:
                raise last
        except (urlerror.URLError, OSError, ValueError) as e:
            last = HttpError("network error: %s" % (e,))
        attempt += 1
    raise last or HttpError("request failed")


def post_stream_raw(base_url: str, payload: Dict[str, Any],
                    timeout_s: float) -> Tuple[str, Optional[int]]:
    """POST with stream=true; returns (assembled_text, completion_tokens|None).

    Adds stream_options.include_usage so the final SSE chunk carries usage;
    token count still comes ONLY from usage.completion_tokens (never chunk counts).
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    p = dict(payload)
    p["stream"] = True
    p["stream_options"] = {"include_usage": True}
    req = urequest.Request(
        url,
        data=json.dumps(p).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    text_parts: List[str] = []
    completion_tokens: Optional[int] = None
    with urequest.urlopen(req, timeout=timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str):
                    text_parts.append(piece)
            usage = chunk.get("usage") or {}
            val = usage.get("completion_tokens")
            if isinstance(val, int) and val >= 0:
                completion_tokens = val
    return "".join(text_parts), completion_tokens


def post_stream_with_retry(base_url: str, payload: Dict[str, Any],
                           timeout_s: float, retries: int) -> Tuple[str, Optional[int]]:
    attempt = 0
    last: Optional[Exception] = None
    while attempt <= retries:
        if attempt > 0:
            time.sleep(RETRY_BACKOFF_S * (2 ** (attempt - 1)))
        try:
            return post_stream_raw(base_url, payload, timeout_s)
        except urlerror.HTTPError as e:
            code = getattr(e, "code", 0)
            last = HttpError("HTTP %s: %s" % (code, e.reason or e))
            if code < 500:
                raise last
        except (urlerror.URLError, OSError, ValueError) as e:
            last = HttpError("network error: %s" % (e,))
        attempt += 1
    raise last or HttpError("stream request failed")


def completion_tokens_of(resp: Dict[str, Any]) -> Optional[int]:
    usage = (resp or {}).get("usage") or {}
    val = usage.get("completion_tokens")
    return val if isinstance(val, int) and val >= 0 else None


# ---------------------------------------------------------------- check harness


class CheckResult(dict):
    """{name, passed, detail, latency_ms, completion_tokens, usage_missing}"""


def _record(name: str, passed: bool, detail: str, latency_ms: float,
            completion_tokens: Optional[int]) -> CheckResult:
    return CheckResult(
        name=name,
        passed=passed,
        detail=detail,
        latency_ms=round(latency_ms * 1000.0, 1),
        completion_tokens=completion_tokens,
        usage_missing=completion_tokens is None,
    )


def _first_tool_call(resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for choice in (resp.get("choices") or []):
        msg = choice.get("message") or {}
        tcs = msg.get("tool_calls")
        if tcs:
            return tcs[0]
    return None


# ---------------------------------------------------------------- checks


def check_plain_chat(base_url: str, model: str, timeout_s: float,
                     retries: int) -> CheckResult:
    """1. plain chat: coherent single-turn answer."""
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": "What is the capital of France? Answer in one word."}],
        "max_tokens": 32,
    }
    t0 = time.perf_counter()
    try:
        resp = post_with_retry(base_url, payload, timeout_s, retries)
    except Exception as e:  # noqa: BLE001
        return _record("plain_chat", False, "request failed: %s" % e, 0.0, None)
    lat = time.perf_counter() - t0
    try:
        text = resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        return _record("plain_chat", False, "malformed response shape", lat, None)
    ntok = completion_tokens_of(resp)
    ok = "paris" in text.strip().lower()
    return _record("plain_chat", ok,
                   "reply=%r" % text.strip()[:60], lat, ntok)


def check_forced_tool(base_url: str, model: str, timeout_s: float,
                      retries: int) -> CheckResult:
    """2. tools API + tool_choice forces get_weather call with city arg."""
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": "What's the weather in Kyoto right now? "
                                 "Use the weather tool."}],
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "function",
                        "function": {"name": "get_weather"}},
        "max_tokens": 128,
    }
    t0 = time.perf_counter()
    try:
        resp = post_with_retry(base_url, payload, timeout_s, retries)
    except Exception as e:  # noqa: BLE001
        return _record("forced_tool", False, "request failed: %s" % e, 0.0, None)
    lat = time.perf_counter() - t0
    tc = _first_tool_call(resp)
    if tc is None:
        return _record("forced_tool", False, "no tool_calls in response", lat,
                       completion_tokens_of(resp))
    fn = (tc.get("function") or {})
    name = fn.get("name")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except ValueError:
        return _record("forced_tool", False,
                       "arguments not valid JSON: %r" % (fn.get("arguments"),),
                       lat, completion_tokens_of(resp))
    ok = (name == "get_weather"
          and isinstance(args, dict)
          and "kyoto" in str(args.get("city", "")).lower())
    return _record("forced_tool", ok,
                   "tool=%s args=%s" % (name, args), lat,
                   completion_tokens_of(resp))


def check_zero_arg_tool(base_url: str, model: str, timeout_s: float,
                        retries: int) -> CheckResult:
    """3. zero-arg tool: empty parameters schema; args must parse to {}."""
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": "Ping the game server using the ping tool."}],
        "tools": [ZERO_ARG_TOOL],
        "tool_choice": {"type": "function",
                        "function": {"name": "ping_server"}},
        "max_tokens": 64,
    }
    t0 = time.perf_counter()
    try:
        resp = post_with_retry(base_url, payload, timeout_s, retries)
    except Exception as e:  # noqa: BLE001
        return _record("zero_arg_tool", False, "request failed: %s" % e, 0.0, None)
    lat = time.perf_counter() - t0
    tc = _first_tool_call(resp)
    if tc is None:
        return _record("zero_arg_tool", False, "no tool_calls in response", lat,
                       completion_tokens_of(resp))
    fn = tc.get("function") or {}
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except ValueError:
        return _record("zero_arg_tool", False,
                       "arguments not valid JSON: %r" % (fn.get("arguments"),),
                       lat, completion_tokens_of(resp))
    ok = (fn.get("name") == "ping_server"
          and isinstance(args, dict)
          and len(args) == 0)
    return _record("zero_arg_tool", ok,
                   "tool=%s args=%s" % (fn.get("name"), args), lat,
                   completion_tokens_of(resp))


def check_multiturn_tool(base_url: str, model: str, timeout_s: float,
                         retries: int) -> CheckResult:
    """4. multiturn: tool call -> tool result -> final answer uses the value."""
    msgs: List[Dict[str, Any]] = [
        {"role": "user",
         "content": "What's the weather in Oslo? Use the weather tool, "
                    "then tell me the temperature."},
    ]
    payload1 = {
        "model": model, "messages": msgs, "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "function",
                        "function": {"name": "get_weather"}},
        "max_tokens": 128,
    }
    t0 = time.perf_counter()
    try:
        r1 = post_with_retry(base_url, payload1, timeout_s, retries)
    except Exception as e:  # noqa: BLE001
        return _record("multiturn_tool", False, "turn-1 failed: %s" % e, 0.0, None)
    tc = _first_tool_call(r1)
    if tc is None:
        return _record("multiturn_tool", False,
                       "turn-1 produced no tool_calls",
                       time.perf_counter() - t0, completion_tokens_of(r1))
    msgs.append(r1["choices"][0]["message"])
    tool_reply = json.dumps({"temp_c": 4, "cond": "light snow"})
    msgs.append({"role": "tool",
                 "tool_call_id": tc.get("id", "call_0"),
                 "content": tool_reply})
    payload2 = {
        "model": model, "messages": msgs, "tools": [WEATHER_TOOL],
        "max_tokens": 128,
    }
    try:
        r2 = post_with_retry(base_url, payload2, timeout_s, retries)
    except Exception as e:  # noqa: BLE001
        return _record("multiturn_tool", False, "turn-2 failed: %s" % e,
                       time.perf_counter() - t0, completion_tokens_of(r1))
    lat = time.perf_counter() - t0
    try:
        text = r2["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        return _record("multiturn_tool", False, "turn-2 malformed shape", lat,
                       completion_tokens_of(r2))
    ntok = (completion_tokens_of(r1) or 0) + (completion_tokens_of(r2) or 0)
    ok = ("4" in text and ("snow" in text.lower() or "c" in text.lower()))
    return _record("multiturn_tool", ok,
                   "final=%r" % text.strip()[:60], lat, ntok or None)


def check_sse_stream(base_url: str, model: str, timeout_s: float,
                     retries: int) -> CheckResult:
    """5. stream=true SSE: chunks arrive, assembled text coherent, usage present."""
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": "Count from 1 to 10, space-separated, digits only."}],
        "max_tokens": 64,
    }
    t0 = time.perf_counter()
    try:
        text, ntok = post_stream_with_retry(base_url, payload, timeout_s, retries)
    except Exception as e:  # noqa: BLE001
        return _record("sse_stream", False, "request failed: %s" % e, 0.0, None)
    lat = time.perf_counter() - t0
    digits = [ch for ch in text if ch.isdigit()]
    seq_ok = "".join(digits) == "12345678910"
    ok = seq_ok and bool(text.strip())
    detail = "len=%d digits_ok=%s usage=%s" % (len(text), seq_ok,
                                               "yes" if ntok is not None else "MISSING")
    return _record("sse_stream", ok, detail, lat, ntok)


def check_greedy_determinism(base_url: str, model: str, timeout_s: float,
                             retries: int) -> CheckResult:
    """6. temperature 0, same prompt twice -> byte-identical content."""
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": "Name three primary colors, comma-separated, "
                                 "lowercase, ending with a period."}],
        "max_tokens": 48,
        "temperature": 0,
        "top_p": 1,
    }
    t0 = time.perf_counter()
    try:
        r1 = post_with_retry(base_url, payload, timeout_s, retries)
        r2 = post_with_retry(base_url, payload, timeout_s, retries)
    except Exception as e:  # noqa: BLE001
        return _record("greedy_determinism", False, "request failed: %s" % e,
                       0.0, None)
    lat = time.perf_counter() - t0
    try:
        t1 = r1["choices"][0]["message"]["content"]
        t2 = r2["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return _record("greedy_determinism", False, "malformed response shape",
                       lat, None)
    ntok = completion_tokens_of(r2)
    ok = (t1 == t2) and bool(t1 and t1.strip())
    detail = ("identical" if ok else "DIVERGED: %r vs %r" % (t1, t2))
    return _record("greedy_determinism", ok, detail[:80], lat, ntok)


def check_long_ctx_needle(base_url: str, model: str, timeout_s: float,
                          retries: int) -> CheckResult:
    """7. ~15k-token filler + codeword; model must recall the codeword."""
    prompt = _build_needle_prompt()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    try:
        resp = post_with_retry(base_url, payload, timeout_s, retries)
    except Exception as e:  # noqa: BLE001
        return _record("long_ctx_needle", False, "request failed: %s" % e, 0.0, None)
    lat = time.perf_counter() - t0
    try:
        text = resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        return _record("long_ctx_needle", False, "malformed response shape", lat, None)
    ntok = completion_tokens_of(resp)
    ok = NEEDLE_CODEWORD in text.strip()
    return _record("long_ctx_needle", ok,
                   "reply=%r" % text.strip()[:60], lat, ntok)


def check_ascii_sanity(base_url: str, model: str, timeout_s: float,
                       retries: int) -> CheckResult:
    """8. reply is pure printable ASCII (no mojibake / control chars)."""
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": "Write one short sentence about a lighthouse."}],
        "max_tokens": 96,
    }
    t0 = time.perf_counter()
    try:
        resp = post_with_retry(base_url, payload, timeout_s, retries)
    except Exception as e:  # noqa: BLE001
        return _record("ascii_sanity", False, "request failed: %s" % e, 0.0, None)
    lat = time.perf_counter() - t0
    try:
        text = resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        return _record("ascii_sanity", False, "malformed response shape", lat, None)
    ntok = completion_tokens_of(resp)
    bad = [ch for ch in text if not (32 <= ord(ch) <= 126)]
    ok = (len(bad) == 0) and bool(text.strip())
    detail = ("clean ascii" if ok
              else "non-ascii/control chars: %s" % [hex(ord(c)) for c in bad[:5]])
    return _record("ascii_sanity", ok, detail, lat, ntok)


CHECKS: Dict[str, Callable[[str, str, float, int], CheckResult]] = {
    "plain_chat": check_plain_chat,
    "forced_tool": check_forced_tool,
    "zero_arg_tool": check_zero_arg_tool,
    "multiturn_tool": check_multiturn_tool,
    "sse_stream": check_sse_stream,
    "greedy_determinism": check_greedy_determinism,
    "long_ctx_needle": check_long_ctx_needle,
    "ascii_sanity": check_ascii_sanity,
}

CHECK_ORDER = list(CHECKS)


# ---------------------------------------------------------------- runner + CLI


def run_suite(names: List[str], base_url: str, model: str, timeout_s: float,
              retries: int) -> List[CheckResult]:
    results: List[CheckResult] = []
    for name in names:
        fn = CHECKS[name]
        print("  [%s] running..." % name)
        res = fn(base_url, model, timeout_s, retries)
        results.append(res)
        print("  [%s] %s  (%s ms, %s tokens)  %s"
              % (name, "PASS" if res["passed"] else "FAIL",
                 res["latency_ms"],
                 "n/a" if res["usage_missing"] else res["completion_tokens"],
                 res["detail"]))
    return results


def render_text(results: List[CheckResult]) -> str:
    lines = ["functest battery:"]
    lines.append("  %-20s %-5s %-12s %-18s %s"
                 % ("check", "res", "latency_ms", "completion_tokens", "detail"))
    for r in results:
        lines.append("  %-20s %-5s %-12s %-18s %s"
                     % (r["name"],
                        "PASS" if r["passed"] else "FAIL",
                        r["latency_ms"],
                        "n/a" if r["usage_missing"] else r["completion_tokens"],
                        r["detail"]))
    fails = sum(1 for r in results if not r["passed"])
    lines.append("%d/%d checks passed" % (len(results) - fails, len(results)))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="benchkit.functest",
        description="Functional battery for the local LLM server.",
    )
    p.add_argument("--checks", default=",".join(CHECK_ORDER),
                   help="comma list of checks (default: all)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    p.add_argument("--json", action="store_true", dest="as_json")
    p.add_argument("--out", default=None, help="write JSON payload to FILE")
    args = p.parse_args(argv)

    names = [x.strip() for x in args.checks.split(",") if x.strip()]
    unknown = [x for x in names if x not in CHECKS]
    if unknown:
        print("error: unknown check(s): %s (choose from %s)"
              % (", ".join(unknown), ", ".join(CHECK_ORDER)), file=sys.stderr)
        return 2
    if not names:
        names = list(CHECK_ORDER)

    print("functest: %d checks against %s (model=%s)"
          % (len(names), args.base_url, args.model))
    results = run_suite(names, args.base_url, args.model, args.timeout,
                        args.retries)
    fails = sum(1 for r in results if not r["passed"])
    payload = {
        "meta": {"base_url": args.base_url, "model": args.model,
                 "checks": names},
        "results": [dict(r) for r in results],
        "passed": len(results) - fails,
        "total": len(results),
        "all_passed": fails == 0,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("wrote %s" % args.out)
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(results))
    # Exit code = number of fails (0 = all green).
    return fails


if __name__ == "__main__":
    sys.exit(main())
