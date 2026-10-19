"""tools/maya_names.py - Maya's settings by the names users see, MAYA_*, and the older STRATA_* names the code still
reads: the environment and a config's "env" take either; on disk and on screen they are MAYA_*.

    alias_env(os.environ)    every MAYA_X also sets STRATA_X (MAYA_X wins), so what a program reads and what it starts
                             (the engines) find the setting under the name they read
    env_in(cfg)              a config as read from disk: its "env" under the names the code reads
    env_out(cfg)             a config about to be written: its "env" under the names users see
"""
from __future__ import annotations

OLD, NEW = "STRATA_", "MAYA_"


def shown(name: str) -> str:
    """A setting's name as users see it."""
    return NEW + name[len(OLD):] if name.startswith(OLD) else name


def read_as(name: str) -> str:
    """A setting's name as the code reads it."""
    return OLD + name[len(NEW):] if name.startswith(NEW) else name


def alias_env(env) -> None:
    """In place (os.environ too): every MAYA_X also sets STRATA_X, over an older value."""
    for k, v in list(env.items()):
        if k.startswith(NEW):
            env[read_as(k)] = v


def env_in(cfg):
    """A config as read from disk -> its "env" under the names the code reads (in place; returned).  Where a config
    has a setting under both names, the MAYA_ one counts."""
    e = cfg.get("env") if isinstance(cfg, dict) else None
    if isinstance(e, dict):
        out = {k: v for k, v in e.items() if not k.startswith(NEW)}
        out.update({read_as(k): v for k, v in e.items() if k.startswith(NEW)})
        cfg["env"] = out
    return cfg


def env_out(cfg):
    """A config about to be written -> a copy with its "env" under the names users see."""
    e = cfg.get("env") if isinstance(cfg, dict) else None
    if not isinstance(e, dict):
        return cfg
    return {**cfg, "env": {shown(k): v for k, v in e.items()}}
