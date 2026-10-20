# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac process for one GPU. Started by ``evaluate.py --worker``."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

from isaac_eval.benchmark.expand import EpisodeJob, Shard


def main() -> None:
    """Start Isaac Sim once and run every group assigned to this GPU."""
    from isaaclab.app import AppLauncher

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--group", required=True)
    pre_args, _ = pre.parse_known_args()
    payload = json.loads(Path(pre_args.group).read_text(encoding="utf-8"))
    if payload.get("device") and "--device" not in sys.argv:
        sys.argv.extend(["--device", payload["device"]])

    parser = argparse.ArgumentParser(description="Run one evaluation GPU")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    # Lab 3 dropped the --enable_cameras flag. The launcher still reads the attribute.
    if payload.get("enable_cameras"):
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    from isaac_eval.benchmark.lockstep import run_shard
    from isaac_eval.utils.determinism import apply_determinism

    module_name, class_name = payload["cfg_entry"].split(":")
    cfg_cls = getattr(importlib.import_module(module_name), class_name)
    if payload.get("task_module"):
        importlib.import_module(payload["task_module"])
    groups = payload.get("groups") or [{"shards": payload["shards"]}]
    client = None
    runtime = None
    observation: dict | None = None
    output = Path(args_cli.output)
    output.write_text("", encoding="utf-8")
    try:
        if payload.get("runtime") == "brain":
            from isaac_eval.runtime.brain import BrainRuntime
            from brain_comm.client import BrainClient

            # Connect before the first scene so a brain that is still loading is waited for here.
            client = BrainClient(
                payload["socket"],
                timeout_s=float(payload["step_timeout_s"]),
                connect_wait_s=float(payload.get("connect_wait_s", 0.0)),
            )
            observation = observation_request(client.connect().get("observation"))
            runtime = BrainRuntime(client)
        for group in groups:
            shards = group["shards"]
            first = shards[0]
            num_envs = max(len(shard["jobs"]) for shard in shards)
            task_id = first["task_id"]
            scene_id = first["scene_id"]
            print(f"[INFO] scene {task_id} / {scene_id} ({num_envs} envs)", flush=True)
            extra = {"observation": observation} if observation is not None and _accepts(cfg_cls, "observation") else {}
            if first.get("mixed"):
                if not _accepts(cfg_cls, "scenes"):
                    raise RuntimeError(f"{payload['cfg_entry']} cannot build a stage with different objects per env")
                # The stage is built for these jobs, env by env.
                extra["scenes"] = [(job["task_id"], job["scene_id"]) for job in first["jobs"]]
            cfg = cfg_cls(
                task_id=first["task_id"],
                scene=first["scene_id"],
                variant=first["variant"],
                composer=first["composer"],
                pose_jitter_m=first["pose_jitter_m"],
                env_id=payload.get("benchmark_env_id"),
                num_envs=num_envs,
                **extra,
            )
            env = gym.make(payload["task_name"], cfg=cfg)
            unwrapped = env.unwrapped
            if payload.get("runtime") != "brain":
                from isaac_eval.runtime.in_process import InProcessRuntime
                from isaac_eval.utils.policy_loader import load_policy_from_checkpoint

                policy = load_policy_from_checkpoint(payload["policy_path"], payload["task_name"], env)
                runtime = InProcessRuntime(policy)
            group_lines: list[str] = []
            try:
                for shard_payload in shards:
                    apply_determinism(int(shard_payload["jobs"][0]["seed"]))
                    shard = _shard(shard_payload)
                    _prepare_jobs(unwrapped, shard)
                    records = run_shard(
                        unwrapped,
                        runtime,
                        shard,
                        open_loop_horizon=int(payload["open_loop_horizon"]),
                        max_steps=int(payload["max_steps"]),
                    )
                    group_lines.extend(json.dumps(record) for record in records)
            finally:
                env.close()
            # The next scene rebuild is what Kit crashes on. Keep this scene first.
            with output.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(group_lines) + "\n")
    finally:
        if client is not None:
            client.close()
    simulation_app.close()


MAX_CAMERAS = 16
IMAGE_SIDE_RANGE = (16, 4096)
"""Pixel bounds for each side of a requested ``image_hw``."""


def observation_request(raw) -> dict:
    """The brain's camera request, reduced to ``cameras`` and ``image_hw``.

    The brain process runs submission code, so the request is untrusted. Any other
    key, or a value of the wrong type or size, is rejected.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("brain observation request must be a mapping")
    unknown = set(raw) - {"cameras", "image_hw"}
    if unknown:
        raise ValueError(f"brain observation request has unknown keys {sorted(unknown)}")
    request: dict = {}
    if "cameras" in raw:
        cameras = raw["cameras"]
        if (
            not isinstance(cameras, list)
            or len(cameras) > MAX_CAMERAS
            or not all(isinstance(name, str) and name.isidentifier() for name in cameras)
        ):
            raise ValueError(f"brain observation cameras must be at most {MAX_CAMERAS} identifier names")
        request["cameras"] = list(cameras)
    if "image_hw" in raw:
        size = raw["image_hw"]
        low, high = IMAGE_SIDE_RANGE
        if (
            not isinstance(size, list)
            or len(size) != 2
            or not all(type(side) is int and low <= side <= high for side in size)
        ):
            raise ValueError(f"brain observation image_hw must be two integers in [{low}, {high}]")
        request["image_hw"] = list(size)
    return request


def _prepare_jobs(env, shard: Shard) -> None:
    if hasattr(env, "set_episode_jobs"):
        env.set_episode_jobs(shard.jobs)


def _shard(payload: dict) -> Shard:
    jobs = tuple(EpisodeJob(**job) for job in payload["jobs"])
    return Shard(
        task_id=payload["task_id"],
        scene_id=payload["scene_id"],
        composer=payload["composer"],
        variant=payload["variant"],
        pose_jitter_m=payload["pose_jitter_m"],
        jobs=jobs,
        pack=payload.get("pack", ""),
        mixed=bool(payload.get("mixed", False)),
    )


def _accepts(builder, name: str) -> bool:
    """Whether a cfg builder declares a keyword, so tasks without it are called as before."""
    try:
        return name in inspect.signature(builder).parameters
    except (TypeError, ValueError):
        return False
