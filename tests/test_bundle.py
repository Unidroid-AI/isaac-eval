# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A mounted BenchHub bundle is served to nepher.load_env without a download."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from isaac_eval.benchmark.bundle import BUNDLE_DIR_ENV, bundle_dir_arg, mount_bundle

ENV_ID = "franka-tabletop-v13"


@pytest.fixture(autouse=True)
def _restore_nepher_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NEPHER_CACHE_DIR", raising=False)
    monkeypatch.delenv(BUNDLE_DIR_ENV, raising=False)


def make_bundle(path: Path, env_id: str = ENV_ID) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.yaml").write_text(f"id: {env_id}\nname: Franka\ncategory: manipulation\nversion: '13'\n")
    (path / "benchmark.yaml").write_text("version: 1\ntasks: []\n")
    return path


def can_symlink(tmp_path: Path) -> bool:
    try:
        (tmp_path / "probe-link").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        return False
    return True


def test_bundle_dir_arg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    assert bundle_dir_arg(None) is None
    monkeypatch.setenv(BUNDLE_DIR_ENV, str(tmp_path))
    assert bundle_dir_arg(None) == tmp_path
    assert bundle_dir_arg("/benchmark/x") == Path("/benchmark/x")


def test_bundle_named_after_env_is_its_own_cache(tmp_path: Path):
    bundle = make_bundle(tmp_path / ENV_ID)
    assert mount_bundle(bundle, ENV_ID) == ENV_ID
    assert os.environ["NEPHER_CACHE_DIR"] == str(tmp_path.resolve())


def test_benchhub_mount_is_linked_under_the_env_id(tmp_path: Path):
    if not can_symlink(tmp_path):
        pytest.skip("symlinks are not permitted here")
    bundle = make_bundle(tmp_path / "benchmark" / "franka-tabletop")
    mount_bundle(bundle)
    linked = Path(os.environ["NEPHER_CACHE_DIR"]) / ENV_ID
    assert (linked / "benchmark.yaml").read_text() == "version: 1\ntasks: []\n"


def test_wrong_or_missing_bundle(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no manifest.yaml"):
        mount_bundle(tmp_path)
    bundle = make_bundle(tmp_path / ENV_ID)
    with pytest.raises(ValueError, match="config expects franka-tabletop-v12"):
        mount_bundle(bundle, "franka-tabletop-v12")
    assert "NEPHER_CACHE_DIR" not in os.environ


def test_nepher_loads_the_mounted_bundle_offline(tmp_path: Path):
    pytest.importorskip("nepher")
    bundle = make_bundle(tmp_path / ENV_ID)
    mount_bundle(bundle)
    code = ("import json; from nepher import load_env; e = load_env('franka-tabletop-v13', category='manipulation'); "
            "print(json.dumps({'id': e.id, 'path': str(e.cache_path)}))")
    env = {**os.environ, "NEPHER_API_KEY": "", "ENVHUB_API_URL": "http://127.0.0.1:9"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True,
                         cwd=tmp_path)
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    assert loaded["id"] == ENV_ID
    assert Path(loaded["path"]).resolve() == bundle.resolve()
