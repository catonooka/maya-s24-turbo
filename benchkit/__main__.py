#!/usr/bin/env python3
"""benchkit.__main__ — allow `python -m benchkit <command> [flags...]`.

Delegates to benchkit.cli.main(), which dispatches:
  quick    — 1 probe (prose), concurrency C1 only
  full     — all probes, C1/C2/C4 ladder + full functest battery
  monitor  — health watcher (poll /v1/status + tail 'glm stat:' lines)
  report   — self-contained HTML report from saved results

Examples:
  python -m benchkit quick --json
  python -m benchkit full --bench-out bench.json --functest-out functest.json
  python -m benchkit monitor --interval 5 --duration 60 --out samples.jsonl
  python -m benchkit report --bench bench.json --functest functest.json \\
      --monitor samples.jsonl --out report.html
"""

import os
import sys

# --- path bootstrap: guarantee `benchkit` is importable as a package even
# when invoked from an arbitrary working directory or an embedded interpreter.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PKG_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Import through the package so __init__ bootstrap runs first.
from benchkit.cli import main  # noqa: E402


def run(argv=None) -> int:
    """Entry wrapper: forwards to benchkit.cli.main and returns its exit code."""
    return main(argv)


if __name__ == "__main__":
    sys.exit(run())
