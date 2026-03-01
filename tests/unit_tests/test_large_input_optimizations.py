import gzip
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import mokapot
from mokapot.config import Config


def test_config_parses_temp_and_auto_parquet_flags(tmp_path):
    cfg = Config(
        main_args=[
            "--temp",
            str(tmp_path),
            "--no-auto-parquet",
            str(Path("data", "10k_psms_test.pin")),
        ]
    )
    assert cfg.temp == tmp_path
    assert cfg.auto_parquet is False


def test_config_parses_force_flag(tmp_path):
    cfg = Config(
        main_args=[
            "--temp",
            str(tmp_path),
            "--force",
            str(Path("data", "10k_psms_test.pin")),
        ]
    )
    assert cfg.force is True


def test_read_pin_auto_parquet_trigger_by_file_count(tmp_path):
    src_pin = Path("data", "10k_psms_test.pin")
    in_a = tmp_path / "a.pin"
    in_b = tmp_path / "b.pin"
    in_a.write_bytes(src_pin.read_bytes())
    in_b.write_bytes(src_pin.read_bytes())

    datasets = mokapot.read_pin(
        [in_a, in_b],
        max_workers=2,
        temp_dir=tmp_path,
        auto_parquet=True,
        auto_parquet_min_bytes=10**12,
        auto_parquet_min_files=1,
        auto_parquet_workers=2,
    )
    assert len(datasets) == 2
    assert all(
        dataset.get_default_extension() == ".parquet" for dataset in datasets
    )
    parquet_files = sorted((tmp_path / "input_parquet").glob("*.parquet"))
    assert len(parquet_files) == 2
    complete_marker = tmp_path / "input_parquet" / "_auto_parquet_complete"
    assert complete_marker.exists()
    status_path = tmp_path / "input_parquet" / "_auto_parquet_status.json"
    assert status_path.exists()
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["status"] == "complete"


def test_read_pin_auto_parquet_reuses_temp_contents(tmp_path):
    src_pin = Path("data", "10k_psms_test.pin")
    in_a = tmp_path / "a.pin"
    in_b = tmp_path / "b.pin"
    in_a.write_bytes(src_pin.read_bytes())
    in_b.write_bytes(src_pin.read_bytes())
    inputs = [in_a, in_b]

    _datasets = mokapot.read_pin(
        inputs,
        max_workers=2,
        temp_dir=tmp_path,
        auto_parquet=True,
        auto_parquet_min_bytes=1,
        auto_parquet_min_files=1,
        auto_parquet_workers=2,
        force=False,
    )
    parquet_files = sorted((tmp_path / "input_parquet").glob("*.parquet"))
    assert len(parquet_files) == 2
    before = {path.name: path.stat().st_mtime_ns for path in parquet_files}

    _datasets = mokapot.read_pin(
        inputs,
        max_workers=2,
        temp_dir=tmp_path,
        auto_parquet=True,
        auto_parquet_min_bytes=1,
        auto_parquet_min_files=1,
        auto_parquet_workers=2,
        force=False,
    )
    parquet_files = sorted((tmp_path / "input_parquet").glob("*.parquet"))
    after = {path.name: path.stat().st_mtime_ns for path in parquet_files}
    assert before == after


def test_read_pin_auto_parquet_force_rebuilds(tmp_path):
    src_pin = Path("data", "10k_psms_test.pin")
    in_a = tmp_path / "a.pin"
    in_b = tmp_path / "b.pin"
    in_a.write_bytes(src_pin.read_bytes())
    in_b.write_bytes(src_pin.read_bytes())
    inputs = [in_a, in_b]

    _datasets = mokapot.read_pin(
        inputs,
        max_workers=2,
        temp_dir=tmp_path,
        auto_parquet=True,
        auto_parquet_min_bytes=1,
        auto_parquet_min_files=1,
        auto_parquet_workers=2,
        force=False,
    )
    parquet_files = sorted((tmp_path / "input_parquet").glob("*.parquet"))
    before = {path.name: path.stat().st_mtime_ns for path in parquet_files}

    time.sleep(1.1)
    _datasets = mokapot.read_pin(
        inputs,
        max_workers=2,
        temp_dir=tmp_path,
        auto_parquet=True,
        auto_parquet_min_bytes=1,
        auto_parquet_min_files=1,
        auto_parquet_workers=2,
        force=True,
    )
    parquet_files = sorted((tmp_path / "input_parquet").glob("*.parquet"))
    after = {path.name: path.stat().st_mtime_ns for path in parquet_files}
    assert any(after[name] > before[name] for name in before)


def test_read_pin_auto_parquet_disabled(tmp_path):
    src_pin = Path("data", "10k_psms_test.pin")
    in_a = tmp_path / "a.pin"
    in_a.write_bytes(src_pin.read_bytes())

    datasets = mokapot.read_pin(
        [in_a],
        max_workers=1,
        temp_dir=tmp_path,
        auto_parquet=False,
        auto_parquet_min_files=1,
        auto_parquet_min_bytes=1,
    )
    assert len(datasets) == 1
    assert datasets[0].get_default_extension() != ".parquet"


def test_read_pin_auto_parquet_with_traditional_pin_gz(tmp_path):
    src_pin = Path("data", "phospho_rep1.traditional.pin")
    gz_pin = tmp_path / "traditional.pin.gz"
    with src_pin.open("rb") as src_fh, gzip.open(gz_pin, "wb") as out_fh:
        shutil.copyfileobj(src_fh, out_fh)

    datasets = mokapot.read_pin(
        [gz_pin],
        max_workers=2,
        temp_dir=tmp_path,
        auto_parquet=True,
        auto_parquet_min_bytes=1,
        auto_parquet_min_files=1,
        auto_parquet_workers=2,
    )
    assert len(datasets) == 1
    assert datasets[0].get_default_extension() == ".parquet"
    assert len(datasets[0].spectra_dataframe) > 0


def test_read_pin_auto_parquet_proteins_stays_string(tmp_path):
    pin = tmp_path / "mixed_proteins.pin"
    pin.write_text(
        "\n".join(
            [
                "SpecId\tLabel\tScanNr\tExpMass\tMass\tfeature_1\tPeptide\tProteins",
                "1\t1\t1\t500.2\t500.2\t2.1\tPEPTIDE\t191889",
                "2\t-1\t2\t501.2\t501.2\t1.2\tPEPTIDE\t191890",
                "3\t-1\t3\t502.2\t502.2\t0.8\tPEPTIDE\tDECOY_191889",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    datasets = mokapot.read_pin(
        [pin],
        max_workers=2,
        temp_dir=tmp_path,
        auto_parquet=True,
        auto_parquet_min_bytes=1,
        auto_parquet_min_files=1,
        auto_parquet_workers=2,
    )
    assert len(datasets) == 1
    proteins = datasets[0].read_data(columns=["Proteins"])["Proteins"]
    assert proteins.iloc[-1] == "DECOY_191889"
    assert pd.api.types.is_string_dtype(proteins.dtype)


def test_read_pin_auto_parquet_traditional_ragged_proteins(tmp_path):
    pin = tmp_path / "ragged_traditional.pin"
    pin.write_text(
        "\n".join(
            [
                "SpecId\tLabel\tScanNr\tExpMass\tMass\tfeature_1\tPeptide\tProteins",
                "1001\t1\t11\t500.2\t500.2\t2.1\tPEPTIDE\t191889\t191890",
                "1002\t-1\t12\t501.2\t501.2\t1.2\tPEPTIDE\t191891",
                "1003\t-1\t13\t502.2\t502.2\t0.8\tPEPTIDE\tDECOY_191889\t191892",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    datasets = mokapot.read_pin(
        [pin],
        max_workers=2,
        temp_dir=tmp_path,
        auto_parquet=True,
        auto_parquet_min_bytes=1,
        auto_parquet_min_files=1,
        auto_parquet_workers=2,
    )
    assert len(datasets) == 1
    proteins = datasets[0].read_data(columns=["Proteins"])["Proteins"]
    assert proteins.iloc[0] == "191889:191890"
    assert proteins.iloc[2] == "DECOY_191889:191892"
    assert pd.api.types.is_string_dtype(proteins.dtype)


def test_make_fold_ids_dtype_and_range():
    datasets = mokapot.read_pin(
        Path("data", "10k_psms_test.pin"),
        max_workers=1,
        auto_parquet=False,
    )
    dataset = datasets[0]
    out = np.empty(len(dataset.spectra_dataframe), dtype=np.uint8)
    fold_ids = dataset.make_fold_ids(
        folds=3,
        rng=np.random.default_rng(1),
        dtype=np.dtype("uint8"),
        out=out,
    )
    assert fold_ids.dtype == np.uint8
    assert len(fold_ids) == len(dataset.spectra_dataframe)
    assert int(fold_ids.min()) >= 0
    assert int(fold_ids.max()) < 3


def test_brew_with_temp_dir_and_memmap(tmp_path):
    datasets = mokapot.read_pin(
        Path("data", "10k_psms_test.parquet"),
        max_workers=1,
        auto_parquet=False,
    )
    models, scores = mokapot.brew(
        datasets=datasets,
        model=mokapot.PercolatorModel(rng=1, n_jobs=1),
        max_workers=1,
        folds=3,
        rng=1,
        test_fdr=1.0,
        temp_dir=tmp_path,
        memmap_threshold_psms=1,
    )
    assert len(models) == 3
    assert len(scores) == 1
    assert len(scores[0]) == len(datasets[0].spectra_dataframe)
    assert (tmp_path / "memmap").exists()


def test_brew_cleans_temp_train_buckets(tmp_path):
    datasets = mokapot.read_pin(
        Path("data", "10k_psms_test.parquet"),
        max_workers=1,
        auto_parquet=False,
    )
    _models, _scores = mokapot.brew(
        datasets=datasets,
        model=mokapot.PercolatorModel(rng=1, n_jobs=1),
        max_workers=1,
        folds=3,
        rng=1,
        test_fdr=1.0,
        temp_dir=tmp_path,
        memmap_threshold_psms=0,
    )
    bucket_dir = tmp_path / "fold_buckets"
    assert not bucket_dir.exists()
