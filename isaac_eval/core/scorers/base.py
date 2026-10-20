# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for all task-type scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...domain.metrics import AggregateMetrics, EpisodeMetrics


class BaseScorer(ABC):
    """Abstract base for all Unidroid task scorers.

    Every concrete scorer must implement ``compute_score`` and ``to_dict``.
    """

    @abstractmethod
    def compute_score(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        """Compute a final score in [0, 1].

        Args:
            metrics: Aggregate metrics from all evaluated episodes.
            max_episode_steps: Episode step budget (used for time normalization).
            episodes: Individual episode records; required by per-episode scorers.
            max_episode_time_s: Physical time budget in seconds.  When provided,
                scorers that measure time efficiency normalize against seconds
                instead of steps, making the score decimation-invariant.

        Returns:
            Final score in the range [0, 1].
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize scorer configuration for logging / reporting."""
