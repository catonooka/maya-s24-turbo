"""ct_vllm.py drop-in: stock sybil cpu_tier + dsv41-dedup experiments chained on top.

Serves as the bind-mount target for /opt/dsv41/ct/ct_vllm.py. Loads the stock
module body from ct_vllm_stock.py (the image's original file, bind-mounted beside
this one), executes it in this module's namespace, then layers the repo's
experiments on the patched objects (route bias wraps ec_step after start()).

Experiment toggles are env-driven; nothing here changes stock behavior when
the env vars are unset.
"""
import os
import sys
import types

_STOCK = os.path.join(os.path.dirname(__file__), "ct_vllm_stock.py")
_src = open(_STOCK).read()
exec(compile(_src, _STOCK, "exec"), globals())  # stock symbols land in THIS module

# ---- experiments (chained after stock; each no-ops when its env is unset) ----


def _chain_experiments():
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import ct_route_bias as RB

        RB.install(sys.modules[__name__])  # passes this module as CT
    except Exception as exc:
        _log(f"route bias install failed: {exc!r}")
    try:
        import ct_ec_profile as ECP

        ECP.install(sys.modules[__name__])
    except Exception as exc:
        _log(f"ec profile install failed: {exc!r}")
    try:
        import ct_rb_pool as RBP

        RBP.install(sys.modules[__name__])
    except Exception as exc:
        _log(f"rb pool install failed: {exc!r}")


_orig_start = start


def start():
    _orig_start()
    _chain_experiments()


_log("dsv41-dedup ct_vllm chained (stock + experiments)")
