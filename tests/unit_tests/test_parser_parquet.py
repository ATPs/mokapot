"""Test that parsing Percolator input files (parquet) works correctly"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import mokapot
from mokapot.tabular_data.parquet import ParquetFileWriter


@pytest.fixture
def std_parquet(tmp_path):
    """Create a standard pin file"""
    out_file = tmp_path / "std_pin.parquet"
    df = pd.DataFrame([
        # {
        #     "sPeCid": "DefaultDirection",
        #     "LaBel": "0",
        #     "pepTide": "-",
        #     "sCore": "-",
        #     "scanNR": "1",
        #     "pRoteins": "-",
        # },
        {
            "sPeCid": "a",
            "LaBel": "1",
            "pepTide": "ABC",
            "sCore": "5",
            "scanNR": "2",
            "pRoteins": "protein1",
        },
        {
            "sPeCid": "b",
            "LaBel": "-1",
            "pepTide": "CBA",
            "sCore": "10",
            "scanNR": "3",
            "pRoteins": "decoy_protein1",
        },
    ])

    # 2025-01-02
    # Rn this fails bc the label column is expected to
    # be either -1/1 or 0/1 BUT not 0/-1/1. What should be the
    # default behavior?
    # Op1: promote 0 to 1,
    # Op2: demote 0 to -1 (aka promote -1 to 0)
    df.to_parquet(out_file, index=False)
    return out_file


def test_parquet_parsing(std_parquet):
    """Test pin parsing"""
    datasets = mokapot.read_pin(
        std_parquet,
        max_workers=4,
    )
    df = pq.read_table(std_parquet).to_pandas()
    assert len(datasets) == 1

    # Q: How is this test different from just doing
    # >>> assert datasets[0].feature_columns == ("sCore",)
    pd.testing.assert_frame_equal(
        df.loc[:, ("sCore",)], df.loc[:, datasets[0].feature_columns]
    )
    # pd.testing.assert_series_equal(
    #     df.loc[:, "sPeCid"], df.loc[:, datasets[0].specId_column]
    # )
    # pd.testing.assert_series_equal(
    #     df.loc[:, "pRoteins"], df.loc[:, datasets[0].protein_column]
    # )
    pd.testing.assert_frame_equal(
        df.loc[:, ("scanNR",)], df.loc[:, datasets[0].spectrum_columns]
    )


def test_parquet_writer_forwards_from_pandas_nthreads(tmp_path, monkeypatch):
    out_file = tmp_path / "writer_nthreads.parquet"
    writer = ParquetFileWriter(
        out_file,
        columns=["a", "b"],
        column_types=[np.dtype("int64"), np.dtype("O")],
        from_pandas_nthreads=1,
    )
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    calls = []
    import mokapot.tabular_data.parquet as parquet_module

    real_table = parquet_module.pa.Table

    class _TableProxy:
        @staticmethod
        def from_pandas(*args, **kwargs):
            calls.append(kwargs.get("nthreads"))
            return real_table.from_pandas(*args, **kwargs)

    monkeypatch.setattr(parquet_module.pa, "Table", _TableProxy)
    with writer:
        writer.append_data(df)

    assert calls
    assert all(n == 1 for n in calls)
