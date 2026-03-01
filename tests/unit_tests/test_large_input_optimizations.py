import gzip
from pathlib import Path
import shutil

import numpy as np

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
    assert all(dataset.get_default_extension() == ".parquet" for dataset in datasets)
    parquet_files = sorted((tmp_path / "input_parquet").glob("*.parquet"))
    assert len(parquet_files) == 2


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
