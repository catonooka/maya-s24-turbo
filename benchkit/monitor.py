#!/usr/bin/env python3
"""benchkit.monitor — health watcher for the local LLM server (Stage 4, fixed).

A sampler thread polls GET /v1/status every --interval seconds and tails the
engine log for 'glm stat:' health lines. Each sample is appended as JSONL:
    {"ts": ..., "uptime_s": ..., "decode_tok_s": ..., "vram_hit": ...,
     "concurrency": ..., "rtt_ms": ..., "alerts": [...]}

Fixed in this revision (live-testing findings):
  * BUG 1: shutdown crash — the stop flag is a threading.Event and is now
    ONLY ever .set()/.is_set()/.wait()'d, never called. Clean shutdown:
    stop -> join -> final drain -> summary.
  * BUG 2: glm stat parsing now matches the REAL engine log lines, e.g.:
        glm stat: decode 17.55 ms/tok (56.98 tok/s) | vram hit 89.76% | ...
    producing decode_tok_s=56.98 and vram_hit=89.76 (percent value).
    The log is binary-ish (contains NUL bytes -> read in binary, decode with
    errors='replace', strip NULs) and lines accumulate across boots, so the
    LAST matching line in each poll window wins.
  * concurrency is read from /v1/status as status['concurrency']['serving'].

Alerts (thresholds):
  * decode_tok_s < 20        -> "low_decode"
  * /v1/status unreachable   -> "server_down"

Usage:
  python -m benchkit.monitor [--interval S] [--duration S] [--log PATH]
                             [--base-url URL] [--out FILE] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import urllib.error as urlerror
import urllib.request as urequest

# ---------------------------------------------------------------- constants

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_LOG_PATH = "/home/aisever/project-maya/maya-maya-s24.log"
DEFAULT_INTERVAL_S = 5.0
DEFAULT_DURATION_S = 0  # 0 = run until Ctrl-C
DECODE_ALERT_THRESHOLD = 20.0  # tok/s; below -> low_decode alert
POLL_TIMEOUT_S = 10.0

_GLM_STAT_MARKER = "glm stat:"

# Real engine line format (one regex per metric, all anchored on the marker
# tail). Examples matched:
#   decode 17.55 ms/tok (56.98 tok/s)
#   vram hit 89.76%
#   ram fetches 10.44/tok
#   promotions 4.78/tok
#   prompt 0.7 ms/tok
#   vram 5599/5726 ram 6784/8374
_RE_DECODE = re.compile(
    r"decode\s+([0-9.]+)\s*ms/tok\s*\(\s*([0-9.]+)\s*tok/s\s*\)")
_RE_VRAM_HIT = re.compile(r"vram hit\s+([0-9.]+)\s*%")
_RE_RAM_FETCH = re.compile(r"ram fetches\s+([0-9.]+)/tok")
_RE_PROMOTIONS = re.compile(r"promotions\s+([0-9.]+)/tok")
_RE_PROMPT = re.compile(r"prompt\s+([0-9.]+)\s*ms/tok")
_RE_VRAM_CAP = re.compile(r"vram\s+([0-9]+)\s*/\s*([0-9]+)\s+ram\s+([0-9]+)\s*/\s*([0-9]+)")


# ---------------------------------------------------------------- HTTP


def poll_status(base_url: str, timeout_s: float) -> Tuple[Optional[Dict[str, Any]], float]:
    """GET /v1/status. Returns (json_or_None, round_trip_ms)."""
    url = base_url.rstrip("/") + "/v1/status"
    req = urequest.Request(url, method="GET")
    t0 = time.perf_counter()
    try:
        with urequest.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        return json.loads(raw.decode("utf-8", "replace")), rtt_ms
    except (urlerror.HTTPError, urlerror.URLError, OSError, ValueError):
        return None, (time.perf_counter() - t0) * 1000.0


def extract_concurrency(status: Optional[Dict[str, Any]]) -> Optional[int]:
    """concurrency lives at status['concurrency']['serving'] on this server."""
    if not isinstance(status, dict):
        return None
    conc = status.get("concurrency")
    if isinstance(conc, dict):
        val = conc.get("serving")
        if isinstance(val, int):
            return val
        if isinstance(val, float) and val.is_integer():
            return int(val)
        return None
    if isinstance(conc, int):
        return conc  # tolerate flat int form
    return None


# ---------------------------------------------------------------- log parsing


def parse_glm_stat(line: str) -> Optional[Dict[str, Any]]:
    """Parse a real 'glm stat:' line into a metrics dict; None if not one.

    Returns e.g. {"decode_ms_tok": 17.55, "decode_tok_s": 56.98,
                  "vram_hit": 89.76, "ram_fetches_per_tok": 10.44,
                  "promotions_per_tok": 4.78, "prompt_ms_tok": 0.7,
                  "vram_used": 5599, "vram_total": 5726,
                  "ram_used": 6784, "ram_total": 8374}
    """
    if not line:
        return None
    # NUL bytes and other binary junk: strip control chars first.
    cleaned = line.replace("\x00", "").replace("\ufffd", "")
    idx = cleaned.find(_GLM_STAT_MARKER)
    if idx < 0:
        return None
    tail = cleaned[idx + len(_GLM_STAT_MARKER):]

    out: Dict[str, Any] = {}

    m = _RE_DECODE.search(tail)
    if m:
        out["decode_ms_tok"] = float(m.group(1))
        out["decode_tok_s"] = float(m.group(2))

    m = _RE_VRAM_HIT.search(tail)
    if m:
        out["vram_hit"] = float(m.group(1))  # percent value, e.g. 89.76

    m = _RE_RAM_FETCH.search(tail)
    if m:
        out["ram_fetches_per_tok"] = float(m.group(1))

    m = _RE_PROMOTIONS.search(tail)
    if m:
        out["promotions_per_tok"] = float(m.group(1))

    m = _RE_PROMPT.search(tail)
    if m:
        out["prompt_ms_tok"] = float(m.group(1))

    m = _RE_VRAM_CAP.search(tail)
    if m:
        out["vram_used"] = int(m.group(1))
        out["vram_total"] = int(m.group(2))
        out["ram_used"] = int(m.group(3))
        out["ram_total"] = int(m.group(4))

    return out or None


class LogTailer:
    """Stateful incremental tailer over a binary-ish, append-across-boots log.

    Reads only new bytes since the last poll (binary mode, byte offsets), so
    NUL bytes never corrupt the offset bookkeeping. Handles rotation/truncation
    (file smaller than remembered offset -> restart from 0).
    """

    def __init__(self, path: str):
        self.path = path
        self._offset = 0
        self._exists = os.path.exists(path)
        if self._exists:
            try:
                self._offset = os.path.getsize(path)
            except OSError:
                self._offset = 0

    def read_new_lines(self) -> List[str]:
        """Return new complete lines since the last call (decoded, NUL-stripped)."""
        if not os.path.exists(self.path):
            if self._exists:  # log vanished (rotation) — reset
                self._exists = False
                self._offset = 0
            return []
        if not self._exists:
            self._exists = True
            try:
                self._offset = os.path.getsize(self.path)
            except OSError:
                self._offset = 0
            return []
        try:
            size = os.path.getsize(self.path)
            if size < self._offset:  # truncated/rotated: start over
                self._offset = 0
            if size == self._offset:
                return []
            with open(self.path, "rb") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
                end_pos = fh.tell()
        except OSError:
            return []

        # Only complete lines; an incomplete trailing line is re-read next poll.
        nl = chunk.rfind(b"\n")
        if nl < 0:
            self._offset = end_pos  # nothing complete yet; skip these bytes
            return []
        complete = chunk[:nl + 1]
        self._offset = end_pos - (len(chunk) - nl - 1)

        text = complete.decode("utf-8", errors="replace")
        return [ln.replace("\x00", "").replace("\ufffd", "")
                for ln in text.splitlines()]


# ---------------------------------------------------------------- sampler thread


class Sampler(threading.Thread):
    """Background thread: one sample per interval; appends to a shared deque.

    Shutdown contract: call stop() (sets the Event), then join(). The thread
    NEVER calls the Event — only .set() / .is_set() / .wait() are used.
    """

    def __init__(self, base_url: str, interval_s: float, log_path: str,
                 timeout_s: float):
        super().__init__(daemon=True, name="benchkit-monitor-sampler")
        self.base_url = base_url
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self.tailer = LogTailer(log_path)
        # Backfill: the engine emits 'glm stat:' only on long requests; carry
        # the last-ever stat line as the baseline so samples between
        # emissions report the last known state instead of null.
        self._last_stat: Dict[str, Any] = {}
        try:
            with open(log_path, "rb") as fh:
                data = fh.read()
            for ln in data.replace(b"\x00", b"").decode("utf-8", "replace").splitlines():
                parsed = parse_glm_stat(ln)
                if parsed:
                    self._last_stat = parsed
        except OSError:
            pass
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._pending: Deque[Dict[str, Any]] = deque()
        self._events: List[Tuple[str, str]] = []
        self._was_down = False

    def stop(self) -> None:
        """Request shutdown. Safe to call from any thread, any number of times."""
        self._stop_evt.set()

    def drain(self) -> List[Dict[str, Any]]:
        """Atomically pop all pending samples."""
        with self._lock:
            out = list(self._pending)
            self._pending.clear()
        return out

    def events(self) -> List[Tuple[str, str]]:
        with self._lock:
            return list(self._events)

    def _sample_once(self) -> Dict[str, Any]:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status, rtt_ms = poll_status(self.base_url, self.timeout_s)
        alerts: List[str] = []
        uptime_s = None
        if status is None:
            alerts.append("server_down")
            if not self._was_down:
                self._events.append(
                    (ts, "server_down: /v1/status unreachable"))
            self._was_down = True
        else:
            self._was_down = False
            uptime_s = status.get("uptime_s")
        concurrency = extract_concurrency(status)

        # Tail the log; lines accumulate across boots, so the LAST matching
        # 'glm stat:' line in the window is the current state.
        latest: Dict[str, Any] = dict(self._last_stat)  # start from backfill
        for line in self.tailer.read_new_lines():
            parsed = parse_glm_stat(line)
            if parsed:
                latest = parsed  # overwrite: last match wins
        self._last_stat = latest  # carry forward between emissions
        decode = latest.get("decode_tok_s")
        if isinstance(decode, (int, float)) and decode < DECODE_ALERT_THRESHOLD:
            alerts.append("low_decode: %.2f tok/s < %s"
                          % (decode, DECODE_ALERT_THRESHOLD))
            self._events.append((ts, "low_decode: %.2f tok/s" % decode))

        return {
            "ts": ts,
            "uptime_s": uptime_s,
            "decode_tok_s": decode,
            "decode_ms_tok": latest.get("decode_ms_tok"),
            "vram_hit": latest.get("vram_hit"),
            "vram_used": latest.get("vram_used"),
            "vram_total": latest.get("vram_total"),
            "ram_fetches_per_tok": latest.get("ram_fetches_per_tok"),
            "promotions_per_tok": latest.get("promotions_per_tok"),
            "prompt_ms_tok": latest.get("prompt_ms_tok"),
            "concurrency": concurrency,
            "rtt_ms": round(rtt_ms, 1),
            "alerts": alerts,
        }

    def run(self) -> None:
        next_t = time.time()
        while not self._stop_evt.is_set():
            try:
                sample = self._sample_once()
                with self._lock:
                    self._pending.append(sample)
            except Exception as e:  # never let the sampler die silently
                with self._lock:
                    self._pending.append({
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "uptime_s": None, "decode_tok_s": None, "vram_hit": None,
                        "concurrency": None, "rtt_ms": None,
                        "alerts": ["sampler_error: %s" % e],
                    })
            next_t += self.interval_s
            sleep_s = max(0.05, next_t - time.time())
            # Event.wait returns True if the stop flag was set -> exit loop.
            if self._stop_evt.wait(sleep_s):
                break


# ---------------------------------------------------------------- output


def render_sample(s: Dict[str, Any]) -> str:
    parts = ["[%s]" % s["ts"]]
    parts.append("up=%s" % ("?" if s.get("uptime_s") is None else s["uptime_s"]))
    decode = s.get("decode_tok_s")
    parts.append("decode=%s" % ("?" if decode is None
                                else "%.2f tok/s" % decode))
    vram = s.get("vram_hit")
    parts.append("vram_hit=%s" % ("?" if vram is None else "%.2f%%" % vram))
    parts.append("conc=%s" % ("?" if s.get("concurrency") is None
                              else s["concurrency"]))
    parts.append("rtt=%s ms" % s.get("rtt_ms"))
    if s.get("alerts"):
        parts.append("ALERTS: %s" % "; ".join(s["alerts"]))
    return " ".join(parts)


def print_summary(samples: List[Dict[str, Any]],
                  events: List[Tuple[str, str]],
                  out_path: Optional[str], elapsed_s: float) -> None:
    decodes = [s["decode_tok_s"] for s in samples
               if isinstance(s.get("decode_tok_s"), (int, float))]
    alert_count = sum(len(s.get("alerts") or []) for s in samples)
    print("\n--- monitor summary ---")
    print("duration:      %.1fs" % elapsed_s)
    print("samples:       %d" % len(samples))
    if decodes:
        print("decode tok/s:  min=%.2f  mean=%.2f  max=%.2f"
              % (min(decodes), sum(decodes) / len(decodes), max(decodes)))
    else:
        print("decode tok/s:  no 'glm stat:' lines observed")
    print("alerts:        %d" % alert_count)
    if events:
        print("events:")
        for ts, msg in events:
            print("  [%s] %s" % (ts, msg))
    if out_path:
        print("jsonl written to: %s" % out_path)


# ---------------------------------------------------------------- CLI


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="benchkit.monitor",
        description="Health monitor: /v1/status + 'glm stat:' log tailing.",
    )
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                   help="seconds between samples (default 5)")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                   help="seconds to run; 0 or omitted = until Ctrl-C")
    p.add_argument("--log", default=DEFAULT_LOG_PATH,
                   help="engine log path to tail for 'glm stat:' lines")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--out", default=None,
                   help="append JSONL samples to FILE")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="also print JSONL lines to stdout")
    args = p.parse_args(argv)

    sampler = Sampler(args.base_url, max(0.5, args.interval), args.log,
                      POLL_TIMEOUT_S)
    sampler.start()
    print("monitor: sampling %s every %.1fs (log=%s); Ctrl-C to stop"
          % (args.base_url, args.interval, args.log))

    out_fh = None
    if args.out:
        out_fh = open(args.out, "a", encoding="utf-8")

    all_samples: List[Dict[str, Any]] = []
    t0 = time.time()

    def emit(samples: List[Dict[str, Any]]) -> None:
        for s in samples:
            all_samples.append(s)
            line = json.dumps(s, ensure_ascii=False)
            if out_fh:
                out_fh.write(line + "\n")
                out_fh.flush()
            if args.as_json:
                print(line)
            else:
                print(render_sample(s))

    try:
        while True:
            elapsed = time.time() - t0
            if args.duration and elapsed >= args.duration:
                break
            emit(sampler.drain())
            time.sleep(min(0.2, max(0.05, args.interval / 4.0)))
    except KeyboardInterrupt:
        pass
    finally:
        # --- clean shutdown: set the Event (never call it), join, final drain.
        sampler.stop()
        sampler.join(timeout=5.0)
        emit(sampler.drain())
        if out_fh:
            out_fh.close()
        print_summary(all_samples, sampler.events(), args.out,
                      time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
