#!/usr/bin/env python3
"""benchkit.bench — single-stream probes + concurrency ladder for the local LLM server.

Stage 2 implementation per SPEC:
  * Single-stream phase: probes (prose / facts / code), 2 runs each, run-2 reported.
  * Concurrency ladder: C in {1, 2, 4} (configurable), threads issuing parallel requests.
  * Token counts come ONLY from usage.completion_tokens (never estimated).
  * Robust HTTP: retries with exponential backoff, hard timeouts.

Usage:
  python -m benchkit.bench [--probes prose,facts,code] [--conc 1,2,4]
                           [--base-url URL] [--model NAME] [--max-tokens T]
                           [--timeout S] [--retries N] [--json] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

try:
    import urllib.error as urlerror
    import urllib.request as urequest
except ImportError:  # pragma: no cover
    urlerror = None
    urequest = None

# ---------------------------------------------------------------- constants

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = "glm-5.3-flash"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_S = 120
DEFAULT_RETRIES = 3
RETRY_BACKOFF_S = 1.5  # base; doubles each attempt

# Probe presets: (prompt, max_tokens). Deterministic, fixed synthetic prompts.
PROBES: Dict[str, Dict[str, Any]] = {
    "prose": {
        "prompt": (
            "Write a short descriptive paragraph about a harbor at dawn. "
            "Describe the water, the gulls, and the first light on the masts. "
            "Keep it under 120 words."
        ),
        "max_tokens": 256,
    },
    "facts": {
        "prompt": (
            "List the eight traditional planets of the Solar System in order from "
            "the Sun, one per line, each followed by its approximate diameter in km."
        ),
        "max_tokens": 256,
    },
    "code": {
        "prompt": (
            "Write a Python function `word_freq(text: str) -> dict` that returns "
            "a case-insensitive frequency count of words, sorted by count then "
            "alphabetically. Include a short docstring and one doctest."
        ),
        "max_tokens": 384,
    },
}

DEFAULT_PROBES = ["prose", "facts", "code"]
DEFAULT_CONC = [1, 2, 4]


# ---------------------------------------------------------------- HTTP layer


class HttpError(Exception):
    """Raised when a request ultimately fails after all retries."""


def _post_chat(base_url: str, model: str, payload: Dict[str, Any],
               timeout_s: float) -> Dict[str, Any]:
    """One POST to /v1/chat/completions. Returns parsed JSON. Raises HttpError."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = urequest.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urequest.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", "replace"))


def post_chat_with_retry(base_url: str, model: str, payload: Dict[str, Any],
                         timeout_s: float, retries: int) -> Dict[str, Any]:
    """POST with exponential-backoff retries on network errors and 5xx."""
    attempt = 0
    last_err: Optional[Exception] = None
    while attempt <= retries:
        if attempt > 0:
            time.sleep(RETRY_BACKOFF_S * (2 ** (attempt - 1)))
        try:
            return _post_chat(base_url, model, payload, timeout_s)
        except urlerror.HTTPError as e:
            code = getattr(e, "code", 0)
            last_err = HttpError("HTTP %s: %s" % (code, e.reason or e))
            if code < 500:
                raise last_err  # 4xx: do not retry
        except (urlerror.URLError, OSError, ValueError) as e:
            last_err = HttpError("network error: %s" % (e,))
        attempt += 1
    raise last_err or HttpError("request failed")


# ---------------------------------------------------------------- measurement


def extract_completion_tokens(resp: Dict[str, Any]) -> Optional[int]:
    """HARD RULE: tokens come only from usage.completion_tokens; never estimated."""
    usage = (resp or {}).get("usage") or {}
    val = usage.get("completion_tokens")
    if isinstance(val, int) and val >= 0:
        return val
    return None


def run_one(base_url: str, model: str, probe_name: str, prompt: str,
            max_tokens: int, timeout_s: float, retries: int) -> Dict[str, Any]:
    """One timed single-stream request. Returns a RunRecord dict."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    rec: Dict[str, Any] = {
        "phase": "single",
        "probe": probe_name,
        "run": None,
        "latency_ms": None,
        "completion_tokens": None,
        "tok_s": None,
        "usage_missing": False,
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        resp = post_chat_with_retry(base_url, model, payload, timeout_s, retries)
    except Exception as e:  # noqa: BLE001 — record and continue
        rec["error"] = str(e)
        return rec
    latency_s = time.perf_counter() - t0
    rec["latency_ms"] = round(latency_s * 1000.0, 1)
    ntok = extract_completion_tokens(resp)
    if ntok is None:
        rec["usage_missing"] = True
    else:
        rec["completion_tokens"] = ntok
        rec["tok_s"] = round(ntok / latency_s, 2) if latency_s > 0 else None
    return rec


def single_stream_phase(base_url: str, model: str, probes: List[str],
                        timeout_s: float, retries: int) -> List[Dict[str, Any]]:
    """2 sequential runs per probe; returns all records (run-1 warmup, run-2 reported)."""
    out: List[Dict[str, Any]] = []
    for name in probes:
        spec = PROBES[name]
        for run_idx in (1, 2):
            rec = run_one(base_url, model, name, spec["prompt"],
                          spec["max_tokens"], timeout_s, retries)
            rec["run"] = run_idx
            out.append(rec)
            marker = "" if rec["error"] is None else " ERROR: %s" % rec["error"]
            print("  [single] %-6s run-%d  latency=%s ms  tokens=%s%s"
                  % (name, run_idx, rec["latency_ms"], rec["completion_tokens"], marker))
    return out


def ladder_phase(base_url: str, model: str, probes: List[str],
                 conc_levels: List[int], timeout_s: float,
                 retries: int) -> List[Dict[str, Any]]:
    """Concurrency ladder: for each C, fire C requests in parallel (probes cycled)."""
    out: List[Dict[str, Any]] = []
    for c in conc_levels:
        jobs = []
        for i in range(c):
            name = probes[i % len(probes)]
            spec = PROBES[name]
            jobs.append((name, spec["prompt"], spec["max_tokens"]))
        results: List[Dict[str, Any]] = [None] * c  # type: ignore[list-item]
        lock = threading.Lock()
        start_gate = threading.Event()

        def worker(idx: int, name: str, prompt: str, mt: int) -> None:
            start_gate.wait()
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": mt,
                "stream": False,
            }
            t0 = time.perf_counter()
            rec: Dict[str, Any] = {
                "phase": "ladder", "conc": c, "probe": name,
                "latency_ms": None, "completion_tokens": None,
                "tok_s": None, "usage_missing": False, "error": None,
            }
            try:
                resp = post_chat_with_retry(base_url, model, payload,
                                            timeout_s, retries)
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)
                with lock:
                    results[idx] = rec
                return
            lat = time.perf_counter() - t0
            rec["latency_ms"] = round(lat * 1000.0, 1)
            ntok = extract_completion_tokens(resp)
            if ntok is None:
                rec["usage_missing"] = True
            else:
                rec["completion_tokens"] = ntok
                rec["tok_s"] = round(ntok / lat, 2) if lat > 0 else None
            with lock:
                results[idx] = rec

        with ThreadPoolExecutor(max_workers=c) as pool:
            futs = [pool.submit(worker, i, *job) for i, job in enumerate(jobs)]
            wall_t0 = time.perf_counter()
            start_gate.set()
            for f in futs:
                f.result()
            wall_s = time.perf_counter() - wall_t0

        total_tokens = 0
        usage_missing = 0
        errors = 0
        for rec in results:
            if rec["error"] is not None:
                errors += 1
            if rec["usage_missing"]:
                usage_missing += 1
            elif rec["completion_tokens"] is not None:
                total_tokens += rec["completion_tokens"]
        ladder_tok_s = round(total_tokens / wall_s, 2) if wall_s > 0 else None
        latencies = sorted(r["latency_ms"] for r in results
                           if r["latency_ms"] is not None)
        p50 = latencies[len(latencies) // 2] if latencies else None

        out.extend(results)
        print("  [ladder] C=%-2d  wall=%.2fs  p50_lat=%s ms  tokens=%s  "
              "ladder_tok_s=%s  errors=%d  usage_missing=%d"
              % (c, wall_s, p50, total_tokens, ladder_tok_s, errors, usage_missing))
    return out


# ---------------------------------------------------------------- reporting


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate summary. Single-stream verdict = run-2 of each probe."""
    single_reported = []
    for name in sorted({r["probe"] for r in records if r.get("phase") == "single"}):
        runs = [r for r in records
                if r.get("phase") == "single" and r["probe"] == name]
        run2 = next((r for r in runs if r["run"] == 2), None)
        if run2 is not None:
            single_reported.append(run2)

    def _ok(r: Dict[str, Any]) -> bool:
        return r["error"] is None and not r["usage_missing"]

    lat = sorted(r["latency_ms"] for r in single_reported if _ok(r))
    tps = [r["tok_s"] for r in single_reported if _ok(r) and r["tok_s"] is not None]
    return {
        "single_reported": [
            {"probe": r["probe"], "latency_ms": r["latency_ms"],
             "completion_tokens": r["completion_tokens"], "tok_s": r["tok_s"],
             "usage_missing": r["usage_missing"], "error": r["error"]}
            for r in single_reported
        ],
        "single_latency_ms_p50": (lat[len(lat) // 2] if lat else None),
        "single_mean_tok_s": (round(sum(tps) / len(tps), 2) if tps else None),
        "usage_missing_count": sum(1 for r in records if r["usage_missing"]),
        "error_count": sum(1 for r in records if r["error"] is not None),
    }


def render_text(meta: Dict[str, Any], records: List[Dict[str, Any]],
                summary: Dict[str, Any]) -> str:
    lines = []
    lines.append("bench — model=%s base=%s" % (meta["model"], meta["base_url"]))
    lines.append("")
    lines.append("Single-stream (2 runs each; run-2 is the reported number):")
    lines.append("  %-8s %-12s %-18s %-10s %s"
                 % ("probe", "latency_ms", "completion_tokens", "tok/s", "status"))
    for r in summary["single_reported"]:
        status = "ok" if r["error"] is None and not r["usage_missing"] else (
            "usage_missing" if r["usage_missing"] else "error")
        lines.append("  %-8s %-12s %-18s %-10s %s"
                     % (r["probe"], r["latency_ms"], r["completion_tokens"],
                        r["tok_s"], status))
    lines.append("")
    lines.append("Concurrency ladder:")
    lines.append("  %-6s %-12s %-10s %s" % ("C", "p50_lat_ms", "tok/s", "errors"))
    for c in meta["conc"]:
        recs = [r for r in records if r.get("phase") == "ladder" and r["conc"] == c]
        lats = sorted(r["latency_ms"] for r in recs if r["latency_ms"] is not None)
        p50 = lats[len(lats) // 2] if lats else None
        errs = sum(1 for r in recs if r["error"] is not None)
        lines.append("  %-6d %-12s %-10s %d" % (c, p50, "-", errs))
    lines.append("")
    lines.append("Summary: single_p50=%s ms  mean_tok_s=%s  "
                 "usage_missing=%d  errors=%d"
                 % (summary["single_latency_ms_p50"],
                    summary["single_mean_tok_s"],
                    summary["usage_missing_count"], summary["error_count"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def parse_int_list(text: str, default: List[int]) -> List[int]:
    if not text:
        return list(default)
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    return vals or list(default)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="benchkit.bench",
        description="Benchmark probes + concurrency ladder for the local LLM server.",
    )
    p.add_argument("--probes", default=",".join(DEFAULT_PROBES),
                   help="comma list: prose,facts,code")
    p.add_argument("--conc", default=",".join(str(x) for x in DEFAULT_CONC),
                   help="comma list of concurrency levels, e.g. 1,2,4")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="print JSON instead of the human table")
    p.add_argument("--out", default=None, help="write JSON payload to FILE")
    args = p.parse_args(argv)

    probes = [x.strip() for x in args.probes.split(",") if x.strip()]
    unknown = [x for x in probes if x not in PROBES]
    if unknown:
        print("error: unknown probe(s): %s (choose from %s)"
              % (", ".join(unknown), ", ".join(sorted(PROBES))), file=sys.stderr)
        return 2
    if not probes:
        probes = list(DEFAULT_PROBES)
    conc = parse_int_list(args.conc, DEFAULT_CONC)
    conc = [c for c in conc if c >= 1] or DEFAULT_CONC

    meta = {
        "base_url": args.base_url,
        "model": args.model,
        "probes": probes,
        "conc": conc,
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout,
        "retries": args.retries,
    }

    print("bench: single-stream phase (2 runs per probe)")
    records = single_stream_phase(args.base_url, args.model, probes,
                                  args.timeout, args.retries)
    print("bench: concurrency ladder %s" % conc)
    records.extend(ladder_phase(args.base_url, args.model, probes, conc,
                                args.timeout, args.retries))

    summary = summarize(records)
    payload = {"meta": meta, "runs": records, "summary": summary}

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("wrote %s" % args.out)

    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(meta, records, summary))

    return 0 if summary["error_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
