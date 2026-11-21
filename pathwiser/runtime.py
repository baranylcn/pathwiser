"""Environment description and command-line parameter parsing shared by the commands."""

from __future__ import annotations

import contextlib
import json
import platform

import numpy as np

from pathwiser import __version__


def environment() -> dict:
    import sklearn

    env = {
        "pathwiser": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for mod in ("sentence_transformers", "torch", "transformers", "lightgbm"):
        with contextlib.suppress(ImportError):
            env[mod.replace("_", "-")] = __import__(mod).__version__
    return env


def parse_params(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param expects KEY=VALUE, got {item!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out
