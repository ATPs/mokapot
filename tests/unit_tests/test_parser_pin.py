"""Test that parsing Percolator input files works correctly"""

import gzip
from pathlib import Path

import pandas as pd
import pytest

import mokapot


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
