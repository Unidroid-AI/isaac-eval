# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task status checking for Isaac Lab evaluation.

This module provides task-aware status checking (completion and failure) that
uses the eval_compat wrapper to access environment state in a consistent way.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch


def _read_flag(source: Any, env_idx: int | None) -> bool:
    """One environment's flag from a property or a method returning a bool or a per-env array."""
    value = source() if callable(source) else source
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    flags = np.asarray(value, dtype=bool).reshape(-1)
    if flags.size == 0:
        raise ValueError("task status is empty")
    if env_idx is None:
        if flags.size != 1:
            raise ValueError(f"task status has {flags.size} entries; pass env_idx for a vectorized env")
        return bool(flags[0])
    return bool(flags[env_idx])


def check_task_status(
    env: gym.Env,
    info: dict[str, Any] | None = None,
    env_idx: int | None = None,
    current_success: bool = False,
    current_failure: bool = False,
) -> tuple[bool, bool]:
    """Check task completion and failure status using eval_compat wrapper.

    ``task_completed`` and ``task_failed`` may be properties or methods.

    Args:
        env: The gymnasium environment (may be wrapped with EvalCompatEnv).
        info: Info dictionary from the environment step (unused).
        env_idx: Environment index for vectorized environments (None for single env).
        current_success: Current success status (will be OR'd with new check).
        current_failure: Current failure status (will be OR'd with new check).

    Returns:
        Tuple of (success, failure) booleans indicating task status.

    Raises:
        ValueError: The environment exposes neither flag, or a flag has the wrong shape.
    """
    task_completed = getattr(env, "task_completed", None)
    task_failed = getattr(env, "task_failed", None)
    if task_completed is None and task_failed is None:
        raise ValueError("Environment does not have task completion or failure properties")

    success = current_success or (task_completed is not None and _read_flag(task_completed, env_idx))
    failure = current_failure or (task_failed is not None and _read_flag(task_failed, env_idx))
    return success, failure


def check_success(
    env: gym.Env,
    info: dict[str, Any] | None = None,
    task_name: str | None = None,
    env_idx: int | None = None,
    current_success: bool = False,
) -> bool:
    """Whether an episode has succeeded. ``task_name`` is accepted for older callers and ignored."""
    success, _ = check_task_status(env=env, info=info, env_idx=env_idx, current_success=current_success)
    return success
