# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve an extracted EnvHub bundle to ``nepher.load_env`` without network or a Nepher API key.

A Unidroid validator downloads the bundle through BenchHub and mounts it read-only, for example at
``/benchmark/<benchmark_id>``. Nepher looks environments up as ``<NEPHER_CACHE_DIR>/<env_id>/manifest.yaml``,
so this points ``NEPHER_CACHE_DIR`` at a directory where ``<env_id>`` is the mounted bundle. Worker
processes inherit the variable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

BUNDLE_DIR_ENV = "ISAAC_EVAL_BUNDLE_DIR"
"""Environment variable with the bundle directory, used when ``--bundle-dir`` is not given."""


def bundle_dir_arg(explicit: str | None) -> Path | None:
    """``--bundle-dir``, else ``$ISAAC_EVAL_BUNDLE_DIR``, else None (download with nepher as before)."""
    value = explicit or os.environ.get(BUNDLE_DIR_ENV)
    return Path(value) if value else None


def mount_bundle(bundle: Path, expected_env_id: str | None = None) -> str:
    """Make ``nepher.load_env(<env id>)`` read ``bundle``. Returns the env id from its manifest.yaml.

    Call it before nepher reads its configuration in this process.

    Raises:
        FileNotFoundError: ``bundle`` has no manifest.yaml.
        ValueError: The manifest has no id, or it is not ``expected_env_id``.
    """
    bundle = Path(bundle).resolve()
    manifest = bundle / "manifest.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"bundle {bundle} has no manifest.yaml")
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    env_id = str(data.get("id") or "")
    if not env_id:
        raise ValueError(f"{manifest} has no id")
    if expected_env_id and env_id != expected_env_id:
        raise ValueError(f"bundle {bundle} is EnvHub environment {env_id}, but the config expects {expected_env_id}")
    if bundle.name == env_id:
        cache = bundle.parent
    else:
        cache = Path(tempfile.mkdtemp(prefix="isaac-eval-bundle-"))
        (cache / env_id).symlink_to(bundle, target_is_directory=True)
    os.environ["NEPHER_CACHE_DIR"] = str(cache)
    print(f"[INFO] EnvHub environment {env_id} served from {bundle}", flush=True)
    return env_id
