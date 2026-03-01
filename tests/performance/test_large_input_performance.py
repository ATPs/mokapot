"""Optional performance checks for large-input training throughput."""

import os
import time
from pathlib import Path

import numpy as np
import pytest

import mokapot
from mokapot.brew import make_train_sets
from mokapot.parsers.pin import parse_in_chunks

RUN_PERF = os.getenv("MOKAPOT_RUN_PERF_TESTS") == "1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not RUN_PERF,
        reason="Set MOKAPOT_RUN_PERF_TESTS=1 to run optional perf tests.",
    ),
]

# Baselines measured before throughput optimizations on this workload.
BASELINE_SPLIT_S = 2.06
BASELINE_BREW_S_WORKERS_3 = 14.20
BASELINE_BREW_S_WORKERS_1 = 21.80


def _scope_pins():
    pins = [
        Path("data", "scope2_FP97AA.pin"),
        Path("data", "scope2_FP97AB.pin"),
        Path("data", "scope2_FP97AC.pin"),
    ]
    for pin in pins:
        if not pin.exists():
            pytest.skip(f"Missing benchmark dataset file: {pin}")
    return pins


def test_large_input_throughput_targets():
    pins = _scope_pins()
    datasets = mokapot.read_pin(pins, max_workers=3)

    rng = np.random.default_rng(1)
    split_start = time.perf_counter()
    test_folds_idx = [dataset._split(3, rng) for dataset in datasets]
    split_s = time.perf_counter() - split_start

    train_sets = list(
        make_train_sets(
            test_idx=test_folds_idx,
            subset_max_train=None,
            data_size=[len(dataset.spectra_dataframe) for dataset in datasets],
            rng=np.random.default_rng(1),
        )
    )
    extract_start = time.perf_counter()
    _ = parse_in_chunks(
        datasets=datasets,
        train_idx=train_sets,
        chunk_size=200000,
        max_workers=3,
    )
    train_extract_s = time.perf_counter() - extract_start

    model = mokapot.PercolatorModel(rng=1, n_jobs=1)
    brew_start = time.perf_counter()
    mokapot.brew(
        datasets=datasets,
        model=model,
        max_workers=3,
        folds=3,
        rng=1,
    )
    brew_s_workers_3 = time.perf_counter() - brew_start

    model = mokapot.PercolatorModel(rng=1, n_jobs=1)
    brew_start = time.perf_counter()
    mokapot.brew(
        datasets=datasets,
        model=model,
        max_workers=1,
        folds=3,
        rng=1,
    )
    brew_s_workers_1 = time.perf_counter() - brew_start

    # Target checks from the optimization plan.
    assert split_s <= BASELINE_SPLIT_S / 10.0
    assert brew_s_workers_3 <= BASELINE_BREW_S_WORKERS_3 * 0.80
    assert brew_s_workers_1 <= BASELINE_BREW_S_WORKERS_1 * 1.05

    # Expose additional timing context in assertion output when enabled.
    assert train_extract_s > 0.0
