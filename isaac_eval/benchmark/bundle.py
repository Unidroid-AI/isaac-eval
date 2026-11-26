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

UNIDROID_CACHE_ENV = "UNIDROID_CACHE_DIR"
"""Environment variable overriding ``~/.unidroid/bench-cache`` for ``udcli bench download``."""

UNIDROID_HOME_ENV = "UNIDROID_HOME"
"""Environment variable overriding ``~/.unidroid``."""

EXIT_CONFIG_ERROR = 5
"""Exit code emitted when config or bundle resolution fails before any episode worker starts."""


def unidroid_bench_cache_dir() -> Path:
    """Return the local ``unidroid-cli`` benchmark cache root."""
    override = os.environ.get(UNIDROID_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    home = Path(os.environ.get(UNIDROID_HOME_ENV, Path.home() / ".unidroid")).expanduser()
    return home / "bench-cache"


def bundle_root_of(path: Path) -> Path:
    """Return the directory containing ``manifest.yaml`` (root or single top-level folder)."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"bundle directory does not exist: {resolved}")
    if (resolved / "manifest.yaml").is_file():
        return resolved
    children = [child for child in resolved.iterdir() if child.is_dir() and child.name != "__MACOSX"]
    if len(children) == 1 and (children[0] / "manifest.yaml").is_file():
        return children[0]
    raise FileNotFoundError(f"bundle {resolved} has no manifest.yaml")


def bundle_dir_arg(explicit: str | None) -> Path | None:
    """``--bundle-dir``, else ``$ISAAC_EVAL_BUNDLE_DIR``, else None (download with nepher as before)."""
    value = explicit or os.environ.get(BUNDLE_DIR_ENV)
    return Path(value) if value else None


def resolve_bundle_dir(explicit: str | None, env_id: str | None = None) -> Path | None:
    """Resolve an extracted bundle from CLI arg / env var, then ``~/.unidroid/bench-cache/<env_id>/bundle``."""
    direct = bundle_dir_arg(explicit)
    if direct is not None:
        return bundle_root_of(direct)
    if env_id:
        cached = unidroid_bench_cache_dir() / env_id / "bundle"
        if cached.is_dir():
            try:
                return bundle_root_of(cached)
            except FileNotFoundError:
                pass
    return None


def mount_bundle(bundle: Path, expected_env_id: str | None = None, *, strict_env_id: bool = True) -> str:
    """Make ``nepher.load_env(<env id>)`` read ``bundle``. Returns the env id from its manifest.yaml.

    Call it before nepher reads its configuration in this process. When ``strict_env_id`` is False
    (used by BenchHub mounts where a Unidroid benchmark id may alias a versioned EnvHub id), both
    ``env_id`` and ``expected_env_id`` are linked into the temporary ``NEPHER_CACHE_DIR`` when needed.

    Raises:
        FileNotFoundError: ``bundle`` has no manifest.yaml.
        ValueError: The manifest has no id, or ``strict_env_id`` is True and it is not ``expected_env_id``.
    """
    bundle = bundle_root_of(bundle)
    manifest = bundle / "manifest.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    env_id = str(data.get("id") or "")
    if not env_id:
        raise ValueError(f"{manifest} has no id")
    if expected_env_id and env_id != expected_env_id:
        if strict_env_id:
            raise ValueError(
                f"bundle {bundle} is EnvHub environment {env_id}, but the config expects {expected_env_id}"
            )
        print(
            f"[INFO] BenchHub benchmark {expected_env_id} aliases EnvHub environment {env_id}",
            flush=True,
        )
    aliases = {env_id}
    if expected_env_id:
        aliases.add(expected_env_id)
    if aliases == {bundle.name}:
        cache = bundle.parent
    else:
        cache = Path(tempfile.mkdtemp(prefix="isaac-eval-bundle-"))
        for alias in sorted(aliases):
            try:
                (cache / alias).symlink_to(bundle, target_is_directory=True)
            except OSError:
                if strict_env_id:
                    raise
    os.environ["NEPHER_CACHE_DIR"] = str(cache)
    print(f"[INFO] EnvHub environment {env_id} served from {bundle}", flush=True)
    return env_id
