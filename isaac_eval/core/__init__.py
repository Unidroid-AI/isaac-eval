# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core evaluation engine for Isaac Lab tasks.

The evaluator imports Isaac Lab. Names that need it are resolved on first use
so scorer and benchmark imports stay usable without a simulator.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "IsaacLabEvaluator",
    "EpisodeRunner",
    "EvaluationReporter",
    "get_scorer",
]

_LAZY = {
    "IsaacLabEvaluator": (".evaluator", "IsaacLabEvaluator"),
    "EpisodeRunner": (".episode_runner", "EpisodeRunner"),
    "EvaluationReporter": (".reporter", "EvaluationReporter"),
    "get_scorer": (".scorers", "get_scorer"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
