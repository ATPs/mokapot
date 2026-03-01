"""Test that parsing Percolator input files works correctly"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mokapot
from mokapot.parsers.pin import parse_in_chunks
from mokapot.utils import make_bool_trarget


@pytest.fixture
def std_pin(tmp_path):
    """Create a standard pin file"""
    out_file = tmp_path / "std.pin"
    with open(str(out_file), "w+") as pin:
        dat = (
            "sPeCid\tLaBel\tpepTide\tsCore\tscanNR\tpRoteins\n"
            "a\t1\tABC\t5\t2\tprotein1:protein2\n"
            "b\t-1\tCBA\t10\t3\tdecoy_protein1:decoy_protein2"
        )
        pin.write(dat)

    return out_file


@pytest.fixture
def traditional_pin(tmp_path):
    """Create a traditional pin file with ragged protein columns."""
    out_file = tmp_path / "traditional.pin"
    with open(str(out_file), "w+") as pin:
        dat = (
            "sPeCid\tLaBel\tpepTide\tsCore\tscanNR\tpRoteins\n"
            "a\t1\tABC\t5\t2\tprotein1\tprotein2\n"
            "b\t-1\tCBA\t10\t3\tdecoy_protein1\tdecoy_protein2"
        )
        pin.write(dat)

    return out_file


def gzip_file(source: Path, target: Path):
    with open(source, "rb") as in_ref:
        with gzip.open(target, "wb") as out_ref:
            out_ref.write(in_ref.read())


def test_pin_parsing(std_pin):
    """Test pin parsing"""
    datasets = mokapot.read_pin(
        std_pin,
        max_workers=4,
    )
    df = pd.read_csv(std_pin, sep="\t")
    assert len(datasets) == 1

    pd.testing.assert_frame_equal(
        df.loc[:, ("sCore",)], df.loc[:, datasets[0].feature_columns]
    )
    pd.testing.assert_series_equal(
        df.loc[:, "sPeCid"],
        df.loc[:, datasets[0].column_groups.optional_columns.id],
    )
    pd.testing.assert_series_equal(
        df.loc[:, "pRoteins"],
        df.loc[:, datasets[0].column_groups.optional_columns.protein],
    )
    pd.testing.assert_frame_equal(
        df.loc[:, ("scanNR",)], df.loc[:, datasets[0].spectrum_columns]
    )


def test_pin_wo_dir():
    """Test a PIN file without a DefaultDirection line"""
    mokapot.read_pin(Path("data", "scope2_FP97AA.pin"), max_workers=4)


def test_pin_parsing_gz(std_pin, tmp_path):
    """Test pin parsing from gzip-compressed input."""
    gz_pin = tmp_path / "std.pin.gz"
    gzip_file(std_pin, gz_pin)

    datasets = mokapot.read_pin(gz_pin, max_workers=4)
    df = pd.read_csv(gz_pin, sep="\t")
    assert len(datasets) == 1

    pd.testing.assert_frame_equal(
        df.loc[:, ("sCore",)], df.loc[:, datasets[0].feature_columns]
    )
    pd.testing.assert_series_equal(
        df.loc[:, "sPeCid"],
        df.loc[:, datasets[0].column_groups.optional_columns.id],
    )
    pd.testing.assert_series_equal(
        df.loc[:, "pRoteins"],
        df.loc[:, datasets[0].column_groups.optional_columns.protein],
    )
    pd.testing.assert_frame_equal(
        df.loc[:, ("scanNR",)], df.loc[:, datasets[0].spectrum_columns]
    )


def test_traditional_pin_parsing_gz(traditional_pin, tmp_path):
    """Test traditional pin parsing from gzip-compressed input."""
    gz_pin = tmp_path / "traditional.pin.gz"
    gzip_file(traditional_pin, gz_pin)

    datasets = mokapot.read_pin(gz_pin, max_workers=4)
    protein_column = datasets[0].column_groups.optional_columns.protein
    proteins = datasets[0].read_data(columns=[protein_column])[protein_column]
    assert proteins.tolist() == [
        "protein1:protein2",
        "decoy_protein1:decoy_protein2",
    ]


def test_traditional_pin_parsing_plain(traditional_pin):
    """Test traditional pin parsing from plain text input."""
    datasets = mokapot.read_pin(traditional_pin, max_workers=4)
    protein_column = datasets[0].column_groups.optional_columns.protein
    proteins = datasets[0].read_data(columns=[protein_column])[protein_column]
    assert proteins.tolist() == [
        "protein1:protein2",
        "decoy_protein1:decoy_protein2",
    ]


def test_missing_feature_values_are_dropped(tmp_path):
    """A feature with at least one missing value should be dropped."""
    pin_file = tmp_path / "missing_feature.pin"
    with open(pin_file, "w+", encoding="utf-8") as pin:
        pin.write(
            "SpecId\tLabel\tPeptide\tScanNr\tfeature_keep\tfeature_drop\tProteins\n"
            "a\t1\tABC\t1\t0.5\t1.0\tprotein1\n"
            "b\t-1\tXYZ\t2\t0.1\t\tdecoy_protein1\n"
        )

    datasets = mokapot.read_pin(pin_file, max_workers=2)
    dataset = datasets[0]
    assert "feature_keep" in dataset.feature_columns
    assert "feature_drop" not in dataset.feature_columns


def test_parse_in_chunks_matches_expected_rows():
    datasets = mokapot.read_pin(Path("data", "scope2_FP97AA.pin"), max_workers=2)
    dataset = datasets[0]
    num_rows = len(dataset.spectra_dataframe)
    all_idx = np.arange(num_rows)
    train_idx = [
        [all_idx[all_idx % 3 != fold].tolist()] for fold in range(3)
    ]

    parsed = parse_in_chunks(
        datasets=datasets,
        train_idx=train_idx,
        chunk_size=17,
        max_workers=2,
    )

    full_df = dataset.read_data(columns=dataset.columns)
    for fold, parsed_df in enumerate(parsed):
        expected_idx = train_idx[fold][0]
        expected_df = full_df.iloc[expected_idx].copy()
        expected_df[dataset.target_column] = make_bool_trarget(
            expected_df[dataset.target_column]
        )

        assert parsed_df.index.tolist() == expected_df.index.tolist()
        pd.testing.assert_frame_equal(
            parsed_df.loc[:, expected_df.columns], expected_df
        )
