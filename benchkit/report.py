#!/usr/bin/env python3
"""benchkit.report — render saved results into a self-contained HTML report (Stage 5).

Reads the JSON artifacts written by the other benchkit modules:
  * bench payload    (meta / runs / summary)         — from benchkit.bench
  * functest payload (meta / results / passed / ...) — from benchkit.functest
  * monitor JSONL    (one sample object per line)    — from benchkit.monitor

Produces ONE self-contained HTML file: inline CSS, no external assets, no JS
dependencies. Sections: Bench probes, Concurrency ladder, Functest checks,
Monitor summary. Any subset of inputs may be supplied.

Usage:
  python -m benchkit.report [--bench FILE] [--functest FILE] [--monitor FILE]
                            [--out FILE] [--title TEXT]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- loaders


class ReportError(Exception):
    pass


def load_bench(path: str) -> Dict[str, Any]:
    """Validate + load a bench payload (meta/runs/summary)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "meta" not in data or "runs" not in data:
        raise ReportError("%s: not a bench payload (need meta/runs/summary)" % path)
    data.setdefault("summary", {})
    return data


def load_functest(path: str) -> Dict[str, Any]:
    """Validate + load a functest payload (meta/results/passed/total)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "results" not in data:
        raise ReportError("%s: not a functest payload (need results[])" % path)
    data.setdefault("passed", sum(1 for r in data["results"] if r.get("passed")))
    data.setdefault("total", len(data["results"]))
    data.setdefault("meta", {})
    return data


def load_monitor(path: str) -> Dict[str, Any]:
    """Load monitor JSONL (one sample per line) into a summary dict."""
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise ReportError("%s:%d: bad JSONL (%s)" % (path, ln, e))
            if isinstance(obj, dict):
                samples.append(obj)
    decodes = [s["decode_tok_s"] for s in samples
               if isinstance(s.get("decode_tok_s"), (int, float))]
    alerts = [a for s in samples for a in (s.get("alerts") or [])]
    return {
        "path": path,
        "samples": samples,
        "sample_count": len(samples),
        "decode_min": min(decodes) if decodes else None,
        "decode_mean": (round(sum(decodes) / len(decodes), 2)
                        if decodes else None),
        "decode_max": max(decodes) if decodes else None,
        "alert_count": len(alerts),
        "alerts": alerts,
    }


# ---------------------------------------------------------------- HTML rendering

_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial,
       sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.4rem; border-bottom: 2px solid #2b6cb0; padding-bottom: .3rem; }
h2 { font-size: 1.1rem; margin-top: 1.8rem; color: #2b6cb0; }
h3 { font-size: .95rem; margin-top: 1.2rem; }
table { border-collapse: collapse; margin-top: .6rem; background: #fff;
        box-shadow: 0 1px 2px rgba(0,0,0,.08); }
th, td { border: 1px solid #d4d4d4; padding: .35rem .7rem; text-align: left;
         font-size: .85rem; }
th { background: #edf2f7; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: #1a7f37; font-weight: 600; }
.fail { color: #b42318; font-weight: 600; }
.alert { color: #b42318; }
.muted { color: #666; font-size: .8rem; }
.meta { font-size: .85rem; color: #444; margin-top: .2rem; }
.badge { display: inline-block; padding: .15rem .5rem; border-radius: 4px;
         font-size: .75rem; font-weight: 600; }
.badge.ok { background: #d7f0dd; color: #1a7f37; }
.badge.bad { background: #fbe4e2; color: #b42318; }
"""


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _td_num(v: Any, fmt: str = "%s") -> str:
    """Numeric <td>; n/a (muted) when the value is missing."""
    if v is None:
        return '<td class="num muted">n/a</td>'
    if fmt != "%s":
        try:
            return '<td class="num">%s</td>' % (fmt % v)
        except (TypeError, ValueError):
            return '<td class="num muted">n/a</td>'
    return '<td class="num">%s</td>' % _esc(v)


def _run_status(r: Dict[str, Any]) -> str:
    if r.get("error") is not None:
        return "error"
    if r.get("usage_missing"):
        return "usage_missing"
    return "ok"


def _render_bench(bench: Dict[str, Any]) -> str:
    meta = bench.get("meta", {})
    runs = bench.get("runs", [])
    summary = bench.get("summary", {})
    out = ["<h2>Bench</h2>"]
    out.append("<div class='meta'>model=%s  base=%s  probes=%s  conc=%s</div>"
               % (_esc(meta.get("model")), _esc(meta.get("base_url")),
                  _esc(",".join(meta.get("probes", []))),
                  _esc(",".join(str(c) for c in meta.get("conc", [])))))
    # single-stream runs table
    out.append("<h3>Single-stream probes <span class='muted'>"
               "(run-2 is the reported number)</span></h3>")
    out.append("<table><tr><th>probe</th><th>run</th><th>latency_ms</th>"
               "<th>completion_tokens</th><th>tok/s</th><th>status</th></tr>")
    for r in runs:
        if r.get("phase") != "single":
            continue
        status = _run_status(r)
        cls = "pass" if status == "ok" else "fail"
        out.append("<tr><td>%s</td>%s%s%s%s<td class='%s'>%s</td></tr>"
                   % (_esc(r.get("probe")),
                      _td_num(r.get("run")),
                      _td_num(r.get("latency_ms"), "%.1f"),
                      _td_num(r.get("completion_tokens")),
                      _td_num(r.get("tok_s"), "%.2f"),
                      cls, status))
    out.append("</table>")
    # ladder table
    out.append("<h3>Concurrency ladder</h3>")
    out.append("<table><tr><th>C</th><th>latency_ms</th><th>tok/s</th>"
               "<th>status</th></tr>")
    for r in runs:
        if r.get("phase") != "ladder":
            continue
        status = _run_status(r)
        cls = "pass" if status == "ok" else "fail"
        out.append("<tr><td class='num'>%s</td>%s%s<td class='%s'>%s</td></tr>"
                   % (_esc(r.get("conc")),
                      _td_num(r.get("latency_ms"), "%.1f"),
                      _td_num(r.get("tok_s"), "%.2f"),
                      cls, status))
    out.append("</table>")
    # summary block
    out.append("<p class='meta'>Summary: single_p50=%s ms  mean_tok_s=%s  "
               "usage_missing=%d  errors=%d</p>"
               % (_esc(summary.get("single_latency_ms_p50")),
                  _esc(summary.get("single_mean_tok_s")),
                  summary.get("usage_missing_count", 0),
                  summary.get("error_count", 0)))
    return "\n".join(out)


def _render_functest(functest: Dict[str, Any]) -> str:
    out = ["<h2>Functest</h2>"]
    passed = functest.get("passed", 0)
    total = functest.get("total", 0)
    all_ok = passed == total and total > 0
    badge = ('<span class="badge %s">%s</span>'
             % ("ok" if all_ok else "bad", "%d/%d passed" % (passed, total)))
    out.append("<p>%s <span class='meta'>model=%s</span></p>"
               % (badge, _esc(functest.get("meta", {}).get("model"))))
    out.append("<table><tr><th>check</th><th>result</th><th>latency_ms</th>"
               "<th>completion_tokens</th><th>detail</th></tr>")
    for r in functest.get("results", []):
        ok = bool(r.get("passed"))
        out.append("<tr><td>%s</td><td class='%s'>%s</td>%s%s<td class='muted'>%s"
                   "</td></tr>"
                   % (_esc(r.get("name")), "pass" if ok else "fail",
                      "PASS" if ok else "FAIL",
                      _td_num(r.get("latency_ms"), "%.1f"),
                      _td_num(r.get("completion_tokens")),
                      _esc(r.get("detail"))))
    out.append("</table>")
    return "\n".join(out)


def _render_monitor(mon: Dict[str, Any]) -> str:
    out = ["<h2>Monitor</h2>"]
    out.append("<div class='meta'>source=%s</div>" % _esc(mon.get("path")))
    out.append("<table><tr><th>metric</th><th>value</th></tr>")
    out.append("<tr><td>samples</td><td class='num'>%d</td></tr>"
               % mon.get("sample_count", 0))
    out.append("<tr><td>decode tok/s (min/mean/max)</td><td class='num'>%s</td></tr>"
               % _esc("%s / %s / %s" % (mon.get("decode_min"),
                                        mon.get("decode_mean"),
                                        mon.get("decode_max"))))
    n_alerts = mon.get("alert_count", 0)
    out.append("<tr><td>alerts</td><td class='num %s'>%d</td></tr>"
               % ("" if n_alerts == 0 else "alert", n_alerts))
    out.append("</table>")
    samples = mon.get("samples", [])
    if samples:
        out.append("<h3>Samples</h3>")
        out.append("<table><tr><th>ts</th><th>uptime_s</th><th>decode_tok_s</th>"
                   "<th>vram_hit</th><th>concurrency</th><th>rtt_ms</th>"
                   "<th>alerts</th></tr>")
        for s in samples:
            alerts = s.get("alerts") or []
            alert_cell = ("<td class='alert'>%s</td>" % _esc("; ".join(alerts))
                          if alerts else "<td class='muted'>-</td>")
            out.append("<tr><td>%s</td>%s%s%s%s%s%s</tr>"
                       % (_esc(s.get("ts")),
                          _td_num(s.get("uptime_s")),
                          _td_num(s.get("decode_tok_s"), "%.2f"),
                          _td_num(s.get("vram_hit"), "%.3f"),
                          _td_num(s.get("concurrency")),
                          _td_num(s.get("rtt_ms"), "%.1f"),
                          alert_cell))
        out.append("</table>")
    return "\n".join(out)


def render_html(bench: Optional[Dict[str, Any]],
                functest: Optional[Dict[str, Any]],
                monitor: Optional[Dict[str, Any]],
                title: str = "benchkit report") -> str:
    """Assemble the full self-contained HTML document."""
    parts: List[str] = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append("<title>%s</title>" % html.escape(title))
    parts.append("<style>%s</style></head><body>" % _CSS)
    parts.append("<h1>%s</h1>" % html.escape(title))
    verdict = None
    if functest is not None:
        all_ok = functest.get("passed", 0) == functest.get("total", 0)
        verdict = "HEALTHY" if all_ok else "FAILED"
    if verdict:
        cls = "ok" if verdict == "HEALTHY" else "bad"
        parts.append("<p>Verdict: <span class='badge %s'>%s</span></p>"
                     % (cls, verdict))
    if bench is not None:
        parts.append(_render_bench(bench))
    if functest is not None:
        parts.append(_render_functest(functest))
    if monitor is not None:
        parts.append(_render_monitor(monitor))
    if bench is None and functest is None and monitor is None:
        parts.append("<p class='muted'>No input files given; nothing to render.</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------- CLI


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="benchkit.report",
        description="Render saved benchkit results as a self-contained HTML report.",
    )
    p.add_argument("--bench", default=None, help="bench JSON payload file")
    p.add_argument("--functest", default=None, help="functest JSON payload file")
    p.add_argument("--monitor", default=None, help="monitor JSONL file")
    p.add_argument("--out", default=None, help="write HTML to FILE")
    p.add_argument("--title", default="benchkit report")
    args = p.parse_args(argv)

    try:
        bench = load_bench(args.bench) if args.bench else None
        functest = load_functest(args.functest) if args.functest else None
        monitor = load_monitor(args.monitor) if args.monitor else None
    except (ReportError, OSError, ValueError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2

    doc = render_html(bench, functest, monitor, title=args.title)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(doc)
        print("wrote %s (%d bytes)" % (args.out, len(doc)))
    else:
        print(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
