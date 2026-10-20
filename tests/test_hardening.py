# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from isaac_eval.benchmark.expand import expand, make_shards
from isaac_eval.benchmark.lockstep import run_shard
from isaac_eval.benchmark.manifest import manifest_from_dict
from isaac_eval.benchmark.scheduler import UnitFailed, WorkerSlot, group_shards, run_queue
from isaac_eval.benchmark.worker import observation_request
from isaac_eval.core.scorers import REGISTRY, VALID_VERSIONS_PER_TASK_TYPE, get_scorer
from isaac_eval.domain.config import EvalConfig
from isaac_eval.domain.metrics import AggregateMetrics, EpisodeMetrics

_SUCCESS = {"op": "in_container", "objects": ["banana"], "container": "bowl"}


def _manifest(episodes_per_shard: int = 2):
    task = {"task_id": "t", "scene": "s", "instruction": "go", "success": _SUCCESS}
    return manifest_from_dict({"version": "v", "shard_size": episodes_per_shard, "seed_salt": "x", "tasks": [task]})


def _episode(**extra) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode_id=0, scene="s", seed=1, success=True, steps=100, timeout=False, completion_time=10.0, extra=extra
    )


# --- success checker ---------------------------------------------------------------------------


class _MethodEnv:
    """Exposes the lockstep-style ``task_completed()`` method."""

    def __init__(self, flags):
        self._flags = np.asarray(flags, dtype=bool)

    def task_completed(self):
        return self._flags


def test_task_completed_method_is_called_not_truth_tested():
    from isaac_eval.utils.task_checker import check_success

    env = _MethodEnv([False, True])
    assert check_success(env, env_idx=0) is False
    assert check_success(env, env_idx=1) is True
    assert check_success(_MethodEnv([False]), env_idx=None) is False


def test_vectorized_status_without_env_idx_is_rejected():
    from isaac_eval.utils.task_checker import check_success

    with pytest.raises(ValueError):
        check_success(_MethodEnv([True, False]), env_idx=None)


# --- brain input -------------------------------------------------------------------------------


def test_observation_request_keeps_only_cameras_and_size():
    raw = {"cameras": ["cam_exterior_left", "cam_wrist"], "image_hw": [180, 320]}
    assert observation_request(raw) == raw
    assert observation_request(None) == {}


@pytest.mark.parametrize(
    "raw",
    [
        ["cam"],
        {"cameras": ["cam"], "num_envs": 4096},
        {"cameras": "cam_wrist"},
        {"cameras": ["../stage"]},
        {"image_hw": [180]},
        {"image_hw": [100000, 320]},
        {"image_hw": [180.0, 320]},
    ],
)
def test_observation_request_rejects_anything_else(raw):
    with pytest.raises(ValueError):
        observation_request(raw)


class _ConstantRuntime:
    def __init__(self, chunk):
        self.chunk = chunk

    def begin_episode(self, episode_ids, seeds, instructions):
        return None

    def act(self, obs, step, seed):
        return self.chunk

    def end_episode(self, episode_ids):
        return None


class _IdleEnv:
    def __init__(self, count):
        self.count = count

    def reset(self, *, seed):
        return None

    def get_obs(self):
        return {"joint_pos": np.zeros((self.count, 7), dtype=np.float32)}

    def step(self, action):
        return None

    def task_completed(self):
        return np.zeros(self.count, dtype=bool)

    def task_failed(self):
        return np.zeros(self.count, dtype=bool)

    def instructions(self):
        return [""] * self.count

    def get_raw_state(self):
        return np.zeros((self.count, 3))


@pytest.mark.parametrize(
    "chunk",
    [
        np.zeros((2, 8), dtype=np.float32),
        np.zeros((2, 1, 8), dtype=np.float32),
        np.zeros((3, 2, 8), dtype=np.float32),
        np.full((2, 2, 8), np.nan, dtype=np.float32),
    ],
)
def test_lockstep_rejects_malformed_action_chunks(chunk):
    shard = make_shards(expand(_manifest(), episodes=2), 2)[0]
    with pytest.raises(ValueError):
        run_shard(_IdleEnv(2), _ConstantRuntime(chunk), shard, open_loop_horizon=2, max_steps=4)


# --- crash retry -------------------------------------------------------------------------------


def test_retried_group_is_not_counted_twice():
    groups = group_shards(make_shards(expand(_manifest(), episodes=2), 2))
    ids = [job.job_id for shard in groups[0] for job in shard.jobs]
    assert len(ids) == 2

    def run_unit(slot, unit, attempt):
        rows = [{"job_id": job_id} for job_id in ids]
        if attempt == 0:
            raise UnitFailed("torn line", rows[:1])
        return rows

    records = run_queue([WorkerSlot("0", 0)], [groups], run_unit, retries=1)
    assert sorted(row["job_id"] for row in records) == sorted(ids)


def test_scoring_rejects_repeated_job_ids():
    from isaac_eval.benchmark.aggregate import score_records

    row = {"job_id": "a", "task_id": "t", "success": True, "steps": 10, "completion_time_s": 1.0}
    with pytest.raises(ValueError):
        score_records([row, dict(row)], "manipulation.multitask", "v1", 100, max_episode_time_s=10.0)


# --- scorers -----------------------------------------------------------------------------------


def test_go2_missing_telemetry_never_beats_measured_telemetry():
    scorer = get_scorer("navigation.go2", "v1")
    measured = _episode(course_length_m=20.0, mean_speed=2.5, mean_roll_pitch_rate=1.0, max_speed=3.0)
    bare = _episode(course_length_m=20.0)
    assert scorer._episode_quality(bare, 900, max_episode_time_s=36.0) < scorer._episode_quality(
        measured, 900, max_episode_time_s=36.0
    )


def test_spot_v4_directness_stays_in_unit_range():
    scorer = get_scorer("navigation.spot", "v4")
    assert scorer._directness_quality({"mean_lateral_speed": -0.4}) == pytest.approx(0.2)
    assert scorer._directness_quality({"mean_lateral_speed": 2.0}) == 0.0
    episodes = [_episode(mean_speed=1.0, max_speed=1.2, speed_std=0.1, mean_lateral_speed=-5.0)]
    score = scorer.compute_score(AggregateMetrics.from_episodes(episodes), 900, episodes, max_episode_time_s=36.0)
    assert 0.0 <= score <= 1.0


# --- config ------------------------------------------------------------------------------------


def _config(**overrides) -> EvalConfig:
    fields = {
        "task_name": "Task-v0",
        "task_type": "manipulation.multitask",
        "scoring_version": "v1",
        "category": "manipulation",
        "num_envs": 1,
        "env_scenes": [{"env_id": "e", "scene": 0}],
    }
    fields.update(overrides)
    return EvalConfig(**fields)


def test_config_rejects_a_non_positive_step_cap():
    _config(max_episode_steps=600).validate()
    with pytest.raises(ValueError):
        _config(max_episode_steps=0).validate()


def test_config_accepts_exactly_the_registered_scorers():
    assert {(t, v) for t, versions in VALID_VERSIONS_PER_TASK_TYPE.items() for v in versions} == set(REGISTRY)
    with pytest.raises(ValueError):
        _config(scoring_version="v9").validate()
    with pytest.raises(ValueError):
        _config(task_type="navigation.unknown").validate()
