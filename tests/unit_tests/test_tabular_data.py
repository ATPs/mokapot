import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy import dtype

from mokapot.tabular_data import (
    ColumnMappedReader,
    ColumnSelectReader,
    ComputedTabularDataReader,
    CSVFileReader,
    CSVFileWriter,
    DataFrameReader,
    JoinedTabularDataReader,
    MergedTabularDataReader,
    ParquetFileReader,
    TabularDataReader,
    auto_finalize,
)
from mokapot.tabular_data.traditional_pin import TraditionalPINReader


def test_from_path(tmp_path):
    reader = TabularDataReader.from_path(Path(tmp_path, "test.csv"))
    assert isinstance(reader, CSVFileReader)

    reader = TabularDataReader.from_path(Path(tmp_path, "test.parquet"))
    assert isinstance(reader, ParquetFileReader)

    with pytest.warns(UserWarning):
        reader = TabularDataReader.from_path(Path(tmp_path, "test.blah"))
    assert isinstance(reader, CSVFileReader)


@pytest.mark.filterwarnings("ignore::pandas.errors.ParserWarning")
@pytest.mark.parametrize(
    "file_use", ["phospho_rep1.pin", "phospho_rep1.traditional.pin"]
)
def test_csv_file_reader(file_use):
    # Note: this pin file is kinda "non-standard" which is why I put the
    # filterwarnings decorator before the test function

    path = Path("data", file_use)
    reader = TabularDataReader.from_path(path)
    names = reader.get_column_names()
    types = reader.get_column_types()
    column_to_types = dict(zip(names, types))

    expected_column_to_types = {
        "SpecId": dtype("O"),
        "Label": dtype("int64"),
        "ScanNr": dtype("int64"),
        "ExpMass": dtype("float64"),
        "CalcMass": dtype("float64"),
        "lnrSp": dtype("float64"),
        "deltLCn": dtype("float64"),
        "deltCn": dtype("float64"),
        "Sp": dtype("float64"),
        "IonFrac": dtype("float64"),
        "RefactoredXCorr": dtype("float64"),
        "NegLog10PValue": dtype("float64"),
        "NegLog10ResEvPValue": dtype("float64"),
        "NegLog10CombinePValue": dtype("float64"),
        "PepLen": dtype("int64"),
        "Charge1": dtype("int64"),
        "Charge2": dtype("int64"),
        "Charge3": dtype("int64"),
        "Charge4": dtype("int64"),
        "Charge5": dtype("int64"),
        "enzN": dtype("int64"),
        "enzC": dtype("int64"),
        "enzInt": dtype("int64"),
        "lnNumDSP": dtype("float64"),
        "dM": dtype("float64"),
        "absdM": dtype("float64"),
        "Peptide": dtype("O"),
        "Proteins": dtype("O"),
    }

    for name, type in expected_column_to_types.items():
        assert column_to_types[name] == type

    df = reader.read(["ScanNr", "SpecId", "Proteins"])

    assert (
        "sp|Q96QR8|PURB_HUMAN:sp|Q00577|PURA_HUMAN" in df["Proteins"].tolist()
    )
    assert df.columns.tolist() == ["ScanNr", "SpecId", "Proteins"]
    assert len(df) == 55398
    assert all(df.index == range(len(df)))

    chunk_iterator = reader.get_chunked_data_iterator(
        chunk_size=20000, columns=["ScanNr", "SpecId"]
    )
    chunks = [chunk for chunk in chunk_iterator]
    sizes = [len(chunk) for chunk in chunks]
    assert sizes == [20000, 20000, 15398]
    df_from_chunks = pd.concat(chunks)
    assert all(df_from_chunks.index == range(len(df_from_chunks)))


@pytest.mark.parametrize("compressed", [False, True])
def test_traditional_pin_reader_chunked_matches_full(tmp_path, compressed):
    pin_path = tmp_path / "small.traditional.pin"
    with open(pin_path, "w+", encoding="utf-8") as pin:
        pin.write(
            "SpecId\tLabel\tPeptide\tScore\tScanNr\tProteins\n"
            "a\t1\tABC\t5.0\t2\tprotein1\tprotein2\n"
            "b\t-1\tDEF\t1.5\t3\tdecoy_protein1\tdecoy_protein2\n"
            "c\t1\tGHI\t2.5\t4\tprotein3\n"
        )

    reader_path = pin_path
    if compressed:
        gz_path = tmp_path / "small.traditional.pin.gz"
        with open(pin_path, "rb") as in_ref, gzip.open(gz_path, "wb") as out_ref:
            out_ref.write(in_ref.read())
        reader_path = gz_path

    reader = TabularDataReader.from_path(reader_path)
    assert isinstance(reader, TraditionalPINReader)

    selected_columns = ["SpecId", "ScanNr", "Proteins", "Label", "Score"]
    full = reader.read(columns=selected_columns)
    chunks = list(
        reader.get_chunked_data_iterator(chunk_size=2, columns=selected_columns)
    )
    from_chunks = pd.concat(chunks)

    assert all(from_chunks.index == range(len(from_chunks)))
    assert full["Proteins"].tolist() == [
        "protein1:protein2",
        "decoy_protein1:decoy_protein2",
        "protein3",
    ]
    pd.testing.assert_frame_equal(from_chunks, full)


def test_parquet_file_reader():
    path = Path("data", "10k_psms_test.parquet")
    reader = TabularDataReader.from_path(path)
    names = reader.get_column_names()
    types = reader.get_column_types()
    column_to_types = dict(zip(names, types))

    expected_column_to_types = {
        "SpecId": dtype("int64"),
        "Label": dtype("int64"),
        "ScanNr": dtype("int64"),
        "ExpMass": dtype("float64"),
        "Mass": dtype("float64"),
        "MS8_feature_5": dtype("int64"),
        "missedCleavages": dtype("int64"),
        "MS8_feature_7": dtype("float64"),
        "MS8_feature_13": dtype("float64"),
        "MS8_feature_20": dtype("float64"),
        "MS8_feature_21": dtype("float64"),
        "MS8_feature_22": dtype("float64"),
        "MS8_feature_24": dtype("int64"),
        "MS8_feature_29": dtype("float64"),
        "MS8_feature_30": dtype("float64"),
        "MS8_feature_32": dtype("float64"),
        "Peptide": dtype("O"),
        "Proteins": dtype("O"),
    }

    for name, type in expected_column_to_types.items():
        assert column_to_types[name] == type

    df = reader.read(["ScanNr", "SpecId"])
    assert df.columns.tolist() == ["ScanNr", "SpecId"]
    assert len(df) == 10000
    assert all(df.index == range(len(df)))

    chunk_iterator = reader.get_chunked_data_iterator(
        chunk_size=3300, columns=["ScanNr", "SpecId"]
    )
    chunks = list(chunk_iterator)
    sizes = [len(chunk) for chunk in chunks]
    assert sizes == [3300, 3300, 3300, 100]
    df_from_chunks = pd.concat(chunks)
    assert all(df_from_chunks.index == range(len(df_from_chunks)))


def test_dataframe_reader(psm_df_6):
    reader = DataFrameReader(psm_df_6)
    names = reader.get_column_names()
    types = reader.get_column_types()
    column_to_types = dict(zip(names, types))

    expected_column_to_types = {
        "target": dtype("bool"),
        "spectrum": dtype("int64"),
        "peptide": dtype("O"),
        "protein": dtype("O"),
        "feature_1": dtype("int64"),
        "feature_2": dtype("int64"),
    }

    for name, type in expected_column_to_types.items():
        assert column_to_types[name] == type

    assert len(reader.read()) == 6
    chunk_iterator = reader.get_chunked_data_iterator(
        chunk_size=4, columns=["peptide"]
    )
    chunks = [chunk for chunk in chunk_iterator]
    sizes = [len(chunk) for chunk in chunks]
    assert sizes == [4, 2]
    pd.testing.assert_frame_equal(
        chunks[0], pd.DataFrame({"peptide": ["a", "b", "a", "c"]})
    )
    pd.testing.assert_frame_equal(
        chunks[1], pd.DataFrame({"peptide": ["d", "e"]}, index=[4, 5])
    )

    assert reader.read(["feature_1", "spectrum"]).columns.tolist() == [
        "feature_1",
        "spectrum",
    ]

    # Test whether we can create a reader from a Series
    reader = DataFrameReader.from_series(
        pd.Series(data=[1, 2, 3], name="test")
    )
    pd.testing.assert_frame_equal(
        reader.read(), pd.DataFrame({"test": [1, 2, 3]})
    )

    reader = DataFrameReader.from_series(
        pd.Series(data=[1, 2, 3]), name="test"
    )
    pd.testing.assert_frame_equal(
        reader.read(), pd.DataFrame({"test": [1, 2, 3]})
    )

    reader = DataFrameReader.from_array(
        np.array([1, 2, 3], dtype=np.int32), name="test"
    )
    pd.testing.assert_frame_equal(
        reader.read(), pd.DataFrame({"test": [1, 2, 3]}, dtype=np.int32)
    )


def test_column_renaming(psm_df_6):
    orig_reader = DataFrameReader(psm_df_6)
    reader = ColumnMappedReader(
        orig_reader, {"target": "T", "peptide": "Pep", "Targ": "T"}
    )
    names = reader.get_column_names()
    types = reader.get_column_types()
    column_to_types = dict(zip(names, types))

    expected_column_to_types = {
        "T": dtype("bool"),
        "spectrum": dtype("int64"),
        "Pep": dtype("O"),
        "protein": dtype("O"),
        "feature_1": dtype("int64"),
        "feature_2": dtype("int64"),
    }

    for name, type in expected_column_to_types.items():
        assert column_to_types[name] == type

    assert (reader.read().values == orig_reader.read().values).all()
    assert (
        reader.read(["Pep", "protein", "T", "feature_1"]).values
        == orig_reader.read([
            "peptide",
            "protein",
            "target",
            "feature_1",
        ]).values
    ).all()

    renamed_chunk = next(
        reader.get_chunked_data_iterator(
            chunk_size=4, columns=["Pep", "protein", "T", "feature_1"]
        )
    )
    orig_chunk = next(
        orig_reader.get_chunked_data_iterator(
            chunk_size=4, columns=["peptide", "protein", "target", "feature_1"]
        )
    )
    assert (renamed_chunk.values == orig_chunk.values).all()


def test_default_extension_forwarding(tmp_path):
    csv_path = tmp_path / "ext_test.tsv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(
        csv_path, sep="\t", index=False
    )
    csv_reader = TabularDataReader.from_path(csv_path)
    assert csv_reader.get_default_extension() == ".tsv"

    mapped = ColumnMappedReader(csv_reader, {"a": "A"})
    assert mapped.get_default_extension() == ".tsv"

    selected = ColumnSelectReader(csv_reader, ["a"])
    assert selected.get_default_extension() == ".tsv"

    computed = ComputedTabularDataReader(
        reader=csv_reader,
        column="c",
        dtype=np.dtype("int64"),
        func=lambda df: df["a"] + df["b"],
    )
    assert computed.get_default_extension() == ".tsv"

    joined = JoinedTabularDataReader([csv_reader, selected])
    assert joined.get_default_extension() == ".tsv"

    sorted1_path = tmp_path / "sorted1.tsv"
    sorted2_path = tmp_path / "sorted2.tsv"
    pd.DataFrame({"score": [2.0, 1.0], "x": [10, 11]}).to_csv(
        sorted1_path, sep="\t", index=False
    )
    pd.DataFrame({"score": [1.5, 0.5], "x": [12, 13]}).to_csv(
        sorted2_path, sep="\t", index=False
    )
    merged = MergedTabularDataReader(
        readers=[
            TabularDataReader.from_path(sorted1_path),
            TabularDataReader.from_path(sorted2_path),
        ],
        priority_column="score",
    )
    assert merged.get_default_extension() == ".tsv"


# todo: nice to have: tests for writers (CSV, Parquet, buffering) are still
#  missing


def test_tabular_writer_context_manager(tmp_path):
    # Create a mock class that checks whether it will be correctly initialized
    # and finalized
    class MockWriter(CSVFileWriter):
        initialized = False
        finalized = False

        def __init__(self):
            super().__init__(
                tmp_path / "test.csv", columns=["a", "b"], column_types=[]
            )

        def initialize(self):
            super().initialize()
            self.initialized = True

        def finalize(self):
            super().finalize()
            self.finalized = True

    # Check that context manager works for one file
    with MockWriter() as writer:
        assert writer.initialized
        assert not writer.finalized
    assert writer.finalized

    # Check that it works when an exception is thrown
    try:
        with MockWriter() as writer:
            assert writer.initialized
            assert not writer.finalized
            raise RuntimeError("Just testing")
    except RuntimeError:
        pass  # ignore the exception
    finally:
        assert writer.finalized

    # Check that context manager convenience method (auto_finalize) works for
    # multiple files
    writers = [
        MockWriter(),
        MockWriter(),
    ]

    assert not writers[0].initialized
    assert not writers[1].initialized
    with auto_finalize(writers):
        assert writers[0].initialized
        assert writers[1].initialized
        assert not writers[0].finalized
        assert not writers[1].finalized
    assert writers[0].finalized
    assert writers[1].finalized

    # Now with an exception
    writers = [
        MockWriter(),
        MockWriter(),
    ]

    try:
        with auto_finalize(writers):
            raise RuntimeError("Just testing")
    except RuntimeError:
        pass
    finally:
        assert writers[0].finalized
        assert writers[1].finalized
