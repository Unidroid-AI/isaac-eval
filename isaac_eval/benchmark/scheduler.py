# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assign shard groups to GPUs. The shard list itself does not depend on GPU count."""

from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .expand import Shard


@dataclass(frozen=True)
class WorkerSlot:
    """One Isaac process. ``device_id`` is the value for ``CUDA_VISIBLE_DEVICES``."""

    device_id: str
    brain_index: int


def visible_device_ids() -> list[str]:
    """GPU ids this process may use, in ``CUDA_VISIBLE_DEVICES`` order."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and raw.strip() not in ("", "-1", "none"):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        return parts or ["0"]
    return [str(index) for index in range(_nvidia_gpu_count())]


def plan_slots(placement: str = "paired") -> list[WorkerSlot]:
    """Isaac workers for one machine.

    ``paired`` puts worker ``i`` on GPU ``i`` with ``brain-i.sock``. One GPU is one
    worker and one brain. ``split`` keeps that pairing when only one GPU is visible.
    With two or more, the first half are brains and the rest are Isaac workers that
    round-robin across those brains.
    """
    ids = visible_device_ids()
    if placement == "split" and len(ids) >= 2:
        brain_count = max(1, len(ids) // 2)
        sim_ids = ids[brain_count:]
        return [
            WorkerSlot(device_id=device_id, brain_index=index % brain_count)
            for index, device_id in enumerate(sim_ids)
        ]
    return [WorkerSlot(device_id=device_id, brain_index=index) for index, device_id in enumerate(ids)]


def _nvidia_gpu_count() -> int:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1
    if completed.returncode != 0:
        return 1
    count = sum(1 for line in completed.stdout.splitlines() if line.startswith("GPU"))
    return count or 1


# A packed scene plus the brain replica filled about 35 GB of a 48 GB GPU. The next
# scene in that same Kit process is not released, and PhysX then fails with CUDA
# error 2 (out of memory). One scene per process frees the device. Rebuilding past
# about ten scenes in one process also kills Kit in the camera graph (exit 139).
GROUPS_PER_PROCESS = 1


def group_shards(shards: list[Shard]) -> list[list[Shard]]:
    """One group per (task, scene, variant), preserving shard order."""
    groups: list[list[Shard]] = []
    for shard in shards:
        if not groups or groups[-1][0].group_key != shard.group_key:
            groups.append([shard])
        else:
            groups[-1].append(shard)
    return groups


# Aim for this many units per worker, so a slow unit leaves other workers something to take.
UNITS_PER_WORKER = 3


class UnitFailed(RuntimeError):
    """An Isaac process died. ``records`` holds what it finished before it did."""

    def __init__(self, message: str, records: list[dict] | None = None):
        super().__init__(message)
        self.records = records or []


def group_cost(group: list[Shard]) -> int:
    """Relative cost of one group: its episode count."""
    return sum(len(shard.jobs) for shard in group)


def plan_units(groups: list[list[Shard]], workers: int, limit: int = GROUPS_PER_PROCESS) -> list[list[list[Shard]]]:
    """Cut groups into units of work, largest first.

    A unit is what one Isaac process runs. It is small enough that every worker
    gets about :data:`UNITS_PER_WORKER` of them, and never more than ``limit``
    groups. ``limit`` is 1, so the process exits after one scene and frees the GPU.
    Which worker runs which unit is decided at run time, so the plan does not
    depend on the GPU count except through the unit size.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if not groups:
        return []
    ordered = sorted(groups, key=group_cost, reverse=True)
    target = UNITS_PER_WORKER * max(1, workers)
    size = max(1, min(limit, -(-len(ordered) // target)))
    return [ordered[start : start + size] for start in range(0, len(ordered), size)]


def unfinished(groups: list[list[Shard]], records: list[dict]) -> list[list[Shard]]:
    """Groups with at least one job that has no record yet."""
    done = {record["job_id"] for record in records}
    return [group for group in groups if any(job.job_id not in done for shard in group for job in shard.jobs)]


def run_queue(
    slots: list[WorkerSlot],
    units: list[list[list[Shard]]],
    run_unit,
    retries: int = 1,
) -> list[dict]:
    """Run units on idle workers and return every record.

    Args:
        slots: One entry per Isaac worker. Each runs one unit at a time.
        units: Output of :func:`plan_units`.
        run_unit: ``run_unit(slot, groups, attempt)`` returns the unit's records, or raises
            :class:`UnitFailed` carrying the records it finished.
        retries: How many times the unfinished part of a crashed unit is run again.

    Raises:
        RuntimeError: A unit still had unfinished groups after its last attempt.
    """
    pending = deque((groups, 0) for groups in units)
    lock = threading.Lock()
    records: list[dict] = []
    failures: list[BaseException] = []

    def work(slot: WorkerSlot) -> None:
        while True:
            with lock:
                if failures or not pending:
                    return
                groups, attempt = pending.popleft()
            try:
                found = run_unit(slot, groups, attempt)
                missing: list[list[Shard]] = []
                error: BaseException | None = None
            except UnitFailed as exc:
                missing = unfinished(groups, exc.records)
                # A rerun group reports all of its jobs again, so drop its partial records.
                rerun = {job.job_id for group in missing for shard in group for job in shard.jobs}
                found = [record for record in exc.records if record["job_id"] not in rerun]
                error = exc
            except Exception as exc:
                with lock:
                    failures.append(exc)
                return
            with lock:
                records.extend(found)
                if error is None or not missing:
                    continue
                if attempt < retries:
                    print(f"[WARN] {error}. Running {len(missing)} unfinished group(s) again.", flush=True)
                    pending.appendleft((missing, attempt + 1))
                else:
                    failures.append(error)

    with ThreadPoolExecutor(max_workers=max(1, len(slots))) as pool:
        for future in [pool.submit(work, slot) for slot in slots]:
            future.result()
    if failures:
        raise failures[0]
    return records
