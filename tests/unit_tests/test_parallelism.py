from pathlib import Path

import pytest

from mokapot.parallelism import (
    build_worker_plan,
    effective_cpu_count,
    probe_io_workers,
    resolve_worker_budget,
)


def test_effective_cpu_count_uses_affinity(monkeypatch):
    monkeypatch.setattr(
        "mokapot.parallelism.os.sched_getaffinity",
        lambda _pid: set(range(12)),
    )
    assert effective_cpu_count() == 12


def test_effective_cpu_count_fallback_to_cpu_count(monkeypatch):
    def _raise(_pid):
        raise NotImplementedError

    monkeypatch.setattr("mokapot.parallelism.os.sched_getaffinity", _raise)
    monkeypatch.setattr("mokapot.parallelism.os.cpu_count", lambda: 7)
    assert effective_cpu_count() == 7


def test_resolve_worker_budget_auto_caps(monkeypatch):
    monkeypatch.setattr("mokapot.parallelism.effective_cpu_count", lambda: 80)
    assert resolve_worker_budget(512, "auto") == 80


def test_resolve_worker_budget_manual_keeps_request():
    assert resolve_worker_budget(512, "manual") == 512


def test_probe_io_workers_tie_chooses_smaller(monkeypatch, tmp_path):
    monkeypatch.setattr("mokapot.parallelism._probe_once", lambda *_args: 1.0)
    path_a = tmp_path / "a.pin"
    path_b = tmp_path / "b.pin"
    path_a.write_text("a", encoding="utf-8")
    path_b.write_text("b", encoding="utf-8")
    paths = [path_a, path_b]
    assert probe_io_workers(paths, [8, 4, 1]) == 1


def test_build_worker_plan_auto(monkeypatch):
    monkeypatch.setattr("mokapot.parallelism.effective_cpu_count", lambda: 80)
    monkeypatch.setattr("mokapot.parallelism.probe_io_workers", lambda *_: 16)
    plan = build_worker_plan(
        psm_files=[Path(f"f{i}.pin") for i in range(100)],
        max_workers=512,
        folds=3,
        model_n_jobs="auto",
        worker_policy="auto",
        read_workers="auto",
        confidence_workers="auto",
        auto_parquet_workers=None,
    )
    assert plan.effective_max_workers == 80
    assert plan.outer_workers == 3
    assert plan.model_n_jobs == 26
    assert plan.read_workers == 16
    assert plan.auto_parquet_workers == 16
    assert plan.confidence_workers == 80


def test_build_worker_plan_manual(monkeypatch):
    monkeypatch.setattr("mokapot.parallelism.probe_io_workers", lambda *_: 32)
    plan = build_worker_plan(
        psm_files=[Path(f"f{i}.pin") for i in range(64)],
        max_workers=512,
        folds=3,
        model_n_jobs="auto",
        worker_policy="manual",
        read_workers="auto",
        confidence_workers="auto",
        auto_parquet_workers=48,
    )
    assert plan.effective_max_workers == 512
    assert plan.outer_workers == 3
    assert plan.model_n_jobs == 170
    assert plan.read_workers == 32
    assert plan.auto_parquet_workers == 48
    assert plan.confidence_workers == 64


def test_resolve_worker_budget_invalid_policy():
    with pytest.raises(ValueError):
        resolve_worker_budget(4, "invalid")
