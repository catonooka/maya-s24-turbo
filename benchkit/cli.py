#!/usr/bin/env python3
"""benchkit.cli — unified entry point (Stage 5).

Commands:
  quick    — 1 probe (prose), concurrency C1 only; fast sanity/perf peek.
  full     — all probes, C1/C2/C4 ladder + full functest battery.
  monitor  — health watcher (delegates to benchkit.monitor).
  report   — render saved results as self-contained HTML (delegates to
             benchkit.report).

Exit codes: quick/full return the underlying stage's code (functest fails
count as the exit code); monitor/report delegate transparently.

Usage:
  python -m benchkit quick   [--base-url URL] [--model NAME] [--out FILE] [--json]
  python -m benchkit full    [--base-url URL] [--model NAME]
                             [--bench-out FILE] [--functest-out FILE] [--json]
  python -m benchkit monitor [monitor flags...]
  python -m benchkit report  [report flags...]
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

# --- path bootstrap: make `benchkit` importable no matter where we're run from
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from benchkit import bench        # noqa: E402
from benchkit import functest     # noqa: E402
from benchkit import monitor      # noqa: E402
from benchkit import report       # noqa: E402

USAGE = """benchkit — benchmark + health toolkit for the local LLM server

usage: benchkit <command> [flags...]

commands:
  quick    1 probe (prose), concurrency C1 only
  full     all probes, C1/C2/C4 ladder + full functest battery
  monitor  poll /v1/status + tail 'glm stat:' log lines
  report   render saved results as a self-contained HTML report

run 'benchkit <command> --help' for command-specific flags.
"""

# Flags understood by both bench and functest — passed through on delegation.
COMMON_FLAGS = {"--base-url", "--model", "--timeout", "--retries", "--json"}


def _extract_value(argv: List[str], flag: str) -> Optional[str]:
    """Extract `flag VALUE` from argv if present (does not mutate argv)."""
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
        return argv[i + 1]
    return None


def _split_common(argv: List[str]) -> List[str]:
    """Keep only flags shared with the delegated module (pass-through safe)."""
    keep: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in COMMON_FLAGS:
            keep.append(tok)
            if (i + 1 < len(argv)
                    and not argv[i + 1].startswith("-")
                    and tok != "--json"):
                keep.append(argv[i + 1])
                i += 2
                continue
            i += 1
            continue
        i += 1
    return keep


def cmd_quick(argv: List[str]) -> int:
    """1 probe (prose), C1: bench --probes prose --conc 1."""
    rest = _split_common(argv)
    out = _extract_value(argv, "--out")
    bench_argv = rest + ["--probes", "prose", "--conc", "1"]
    if out:
        bench_argv += ["--out", out]
    print("quick: bench probe=prose C=1")
    return bench.main(bench_argv)


def cmd_full(argv: List[str]) -> int:
    """All probes, C1/C2/C4 ladder + full functest battery."""
    rest = _split_common(argv)
    bench_out = _extract_value(argv, "--bench-out")
    functest_out = _extract_value(argv, "--functest-out")
    print("full: stage 1/2 — bench (all probes, C=1,2,4)")
    rc_bench = bench.main(rest + ["--conc", "1,2,4"]
                          + (["--out", bench_out] if bench_out else []))
    print()  # spacer between stages
    print("full: stage 2/2 — functest battery")
    rc_functest = functest.main(rest
                                + (["--out", functest_out]
                                   if functest_out else []))
    # bench rc is 1 only on request errors; functest rc = number of failing checks
    return max(rc_bench, rc_functest)


def cmd_monitor(argv: List[str]) -> int:
    """Delegate verbatim to benchkit.monitor."""
    return monitor.main(argv)


def cmd_report(argv: List[str]) -> int:
    """Delegate verbatim to benchkit.report."""
    return report.main(argv)


COMMANDS: "dict[str, Tuple[object, str]]" = {
    "quick": (cmd_quick, "1 probe, C1"),
    "full": (cmd_full, "all probes + C1/C2/C4 + functest"),
    "monitor": (cmd_monitor, "health watcher"),
    "report": (cmd_report, "HTML report from saved results"),
}


def dispatch(argv: List[str]) -> int:
    """Parse the first token, delegate, return the module's exit code."""
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print("error: unknown command %r" % cmd, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    fn, _desc = COMMANDS[cmd]
    return fn(argv[1:])


def main(argv: Optional[List[str]] = None) -> int:
    """Top-level entry: dispatch on sys.argv (or the given list)."""
    args = list(sys.argv[1:]) if argv is None else list(argv)
    return dispatch(args)


if __name__ == "__main__":
    sys.exit(main())
