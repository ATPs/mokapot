"""Utilities for resolving adaptive worker counts."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .brew import resolve_parallelism

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerPlan:
    """Resolved worker counts for each stage."""

    effective_max_workers: int
    outer_workers: int
    model_n_jobs: int
    auto_parquet_workers: int
    read_workers: int
    confidence_workers: int


def effective_cpu_count() -> int:
    """Return CPU count honoring affinity/cgroup limits when available."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, NotImplementedError):
        return max(1, os.cpu_count() or 1)


def resolve_worker_budget(requested_max_workers: int, policy: str) -> int:
    """Resolve the effective worker budget from policy and request."""
    if requested_max_workers < 1:
        raise ValueError("'max_workers' must be >= 1.")

    if policy not in {"auto", "manual"}:
        raise ValueError("'worker_policy' must be either 'auto' or 'manual'.")

    if policy == "manual":
        return requested_max_workers

    cpu_cap = effective_cpu_count()
    resolved = min(requested_max_workers, cpu_cap)
    if resolved < requested_max_workers:
        LOGGER.info(
            "Capping max_workers from %d to %d based on available CPU budget "
            "(worker_policy=auto, cpu_count=%d).",
            requested_max_workers,
            resolved,
            cpu_cap,
        )
    return resolved


def _build_probe_candidates(max_workers: int) -> list[int]:
    if max_workers < 1:
        return [1]

    out = [1]
    value = 2
    while value < max_workers:
        out.append(value)
        value *= 2
    if out[-1] != max_workers:
        out.append(max_workers)
    return out


def _read_prefix_bytes(path: Path, bytes_to_read: int) -> int:
    try:
        with open(path, "rb") as stream:
            return len(stream.read(bytes_to_read))
    except OSError:
        return 0


def _probe_once(
    paths: Sequence[Path],
    workers: int,
    bytes_per_file: int,
) -> float:
    if not paths:
        return 0.0

    worker_count = min(max(1, workers), len(paths))
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        total_bytes = sum(
            executor.map(
                _read_prefix_bytes,
                paths,
                [bytes_per_file for _ in paths],
            )
        )
    elapsed = max(1e-9, time.perf_counter() - start)
    return total_bytes / elapsed


def probe_io_workers(
    paths: Iterable[Path],
    candidate_workers: Iterable[int],
    *,
    sample_size: int = 8,
    bytes_per_file: int = 256 * 1024,
) -> int:
    """Probe candidate worker counts and choose the best throughput."""
    candidates = sorted({int(c) for c in candidate_workers if int(c) > 0})
    if not candidates:
        raise ValueError(
            "candidate_workers must include at least one integer >= 1."
        )

    existing_files = sorted(
        {Path(path) for path in paths if Path(path).is_file()},
        key=str,
    )
    if not existing_files:
        return candidates[0]

    sample = existing_files[: max(1, sample_size)]

    best_workers = candidates[0]
    best_throughput = -1.0
    for workers in candidates:
        throughput = _probe_once(sample, workers, bytes_per_file)
        LOGGER.debug(
            "I/O probe workers=%d throughput=%.2f MiB/s over %d file(s).",
            workers,
            throughput / 1024**2,
            len(sample),
        )
        if throughput > best_throughput:
            best_throughput = throughput
            best_workers = workers
        elif throughput == best_throughput and workers < best_workers:
            best_workers = workers

    return best_workers


def _resolve_worker_setting(
    value: int | str | None,
    *,
    default_workers: int,
    cap_workers: int,
    worker_policy: str,
    setting_name: str,
) -> int:
    requested = value
    resolved = default_workers
    if value is not None and value != "auto":
        resolved = int(value)

    if resolved < 1:
        raise ValueError(f"'{setting_name}' must be >= 1 or 'auto'.")

    if worker_policy == "auto":
        capped = min(resolved, cap_workers)
        if capped < resolved:
            requested_txt = "auto" if requested in (None, "auto") else str(requested)
            LOGGER.info(
                "Capping %s from %s to %d based on worker budget "
                "(worker_policy=auto).",
                setting_name,
                requested_txt,
                capped,
            )
        resolved = capped

    return max(1, resolved)


def build_worker_plan(
    *,
    psm_files: Sequence[Path],
    max_workers: int,
    folds: int,
    model_n_jobs: int | str,
    worker_policy: str,
    read_workers: int | str,
    confidence_workers: int | str,
    auto_parquet_workers: int | None,
) -> WorkerPlan:
    """Build an adaptive plan of worker counts for each pipeline stage."""
    effective_max_workers = resolve_worker_budget(max_workers, worker_policy)
    outer_workers, inner_workers = resolve_parallelism(
        max_workers=effective_max_workers,
        folds=folds,
        model_n_jobs=model_n_jobs,
    )

    file_count = max(1, len(psm_files))
    io_cap = max(1, min(effective_max_workers, file_count))

    probe_candidates = _build_probe_candidates(io_cap)
    auto_io_workers = probe_io_workers(psm_files, probe_candidates)

    resolved_read_workers = _resolve_worker_setting(
        read_workers,
        default_workers=auto_io_workers,
        cap_workers=io_cap,
        worker_policy=worker_policy,
        setting_name="read_workers",
    )

    resolved_confidence_workers = _resolve_worker_setting(
        confidence_workers,
        default_workers=io_cap,
        cap_workers=io_cap,
        worker_policy=worker_policy,
        setting_name="confidence_workers",
    )

    auto_parquet_setting: int | str
    if auto_parquet_workers is None:
        auto_parquet_setting = "auto"
    else:
        auto_parquet_setting = auto_parquet_workers

    resolved_auto_parquet_workers = _resolve_worker_setting(
        auto_parquet_setting,
        default_workers=auto_io_workers,
        cap_workers=io_cap,
        worker_policy=worker_policy,
        setting_name="auto_parquet_workers",
    )

    return WorkerPlan(
        effective_max_workers=effective_max_workers,
        outer_workers=outer_workers,
        model_n_jobs=inner_workers,
        auto_parquet_workers=resolved_auto_parquet_workers,
        read_workers=resolved_read_workers,
        confidence_workers=resolved_confidence_workers,
    )
