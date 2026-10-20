# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain models for Isaac Lab evaluation.

This module contains the core domain models:
- Configuration (EvalConfig)
- Metrics (EpisodeMetrics, AggregateMetrics)
- Errors (EvaluationStatus, EvaluationError, etc.)
"""

from .config import EvalConfig
from .errors import (
    EnvironmentError,
    EvaluationError,
    EvaluationRuntimeError,
    EvaluationStatus,
    EvaluationTimeoutError,
)
from .metrics import AggregateMetrics, EpisodeMetrics

__all__ = [
    # Configuration
    "EvalConfig",
    # Metrics
    "EpisodeMetrics",
    "AggregateMetrics",
    # Errors
    "EvaluationStatus",
    "EvaluationError",
    "EnvironmentError",
    "EvaluationRuntimeError",
    "EvaluationTimeoutError",
]

