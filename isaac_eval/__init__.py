# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab evaluation framework for Unidroid.

Runs fixed evaluation campaigns with:
- Deterministic execution
- Versioned scorers for navigation and manipulation
- Comprehensive metric collection
- Structured failure handling

Heavy imports (the evaluator, which pulls in Isaac Lab) are resolved on first
use so benchmark and scorer modules can be imported without a simulator.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "EvalConfig",
    "IsaacLabEvaluator",
    "EvaluationReporter",
    "core",
    "domain",
    "managers",
    "utils",
]

__version__ = "0.1.0"

_LAZY = {
    "EvalConfig": (".domain.config", "EvalConfig"),
    "IsaacLabEvaluator": (".core.evaluator", "IsaacLabEvaluator"),
    "EvaluationReporter": (".core.reporter", "EvaluationReporter"),
    "core": (".core", None),
    "domain": (".domain", None),
    "managers": (".managers", None),
    "utils": (".utils", None),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = importlib.import_module(module_name, __name__)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value
