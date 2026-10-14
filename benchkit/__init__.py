#!/usr/bin/env python3
"""benchkit — benchmark + health toolkit for a local OpenAI-compatible LLM server.

Modules:
  bench      — single-stream probes + concurrency ladder (benchkit.bench)
  functest   — functional battery: chat, tools, streaming, determinism, needle
  monitor    — /v1/status + 'glm stat:' log tailing with alerts
  report     — self-contained HTML report from saved results
  cli        — unified entry point: quick / full / monitor / report

Server facts: http://127.0.0.1:8080, model glm-5.3-flash, 262k context.
Token counts always come from usage.completion_tokens.

This package is importable both as `import benchkit` and via
`python -m benchkit` (see benchkit/__main__.py).
"""

import os
import sys as _sys

# Ensure the package's parent directory is on sys.path so that
# `python /path/to/benchkit/__main__.py`-style invocations and any
# sibling-script imports (e.g. running from another cwd) still resolve
# `benchkit.*` submodules.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in _sys.path:
    _sys.path.insert(0, _PARENT)

__version__ = "1.0.0"
__all__ = ["bench", "functest", "monitor", "report", "cli", "__version__"]


def _lazy(name: str):
    """Import a benchkit submodule on first attribute access."""
    import importlib
    module = importlib.import_module("benchkit.%s" % name)
    globals()[name] = module
    return module


def __getattr__(name):  # PEP 562: lazy submodule access (bench.bench, etc.)
    if name in __all__:
        return _lazy(name)
    raise AttributeError("module 'benchkit' has no attribute %r" % name)


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
