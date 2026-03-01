from io import StringIO
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from typeguard import typechecked

from mokapot.tabular_data.base import TabularDataReader
from mokapot.utils import open_file


def is_traditional_pin(path: Path) -> bool:
    """Check if the PIN file is a traditional PIN file.

    The traditional PIN file uses tabs both as field delimiters
    and as the separator for multiple proteins in the last column.

    So it can be identified if:
    1. The last column name is `protein(s)?`.
    2. The header is tab delimited.
    3. The rest of the file is tab delimited.
    4. The number of delimiters in the other rows is >= number of columns.

    Parameters
    ----------
    path : Path
        The path to the PIN file.

    Returns
    -------
    bool
        True if the PIN file is a traditional PIN file
        (with ragged protein ends); False otherwise (readable as a tsv).

    Raises
    ------
    ValueError
        If the PIN file is not a PIN file. (or is corrupted)
    """
    with open_file(path, mode="rt") as f:
        nread = 0
        header = f.readline().strip()
        nread += 1
        header = header.split("\t")
        if len(header) == 1:
            raise ValueError(f"File '{path}' file is not a PIN file.")

        if not header[-1].lower().startswith("protein"):
            raise ValueError(
                f"File '{path}' is not a PIN file "
                f"(last column '{header[-1]}' is not 'protein')."
                " Which is expected from the traditional PIN file format."
            )

        num_fields = len(header)
        for line in f:
            nread += 1
            line = line.strip()
            if line.startswith("#") or line.startswith("DefaultDirection"):
                continue

            local_num_fields = len(line.split("\t"))
            if local_num_fields < num_fields:
                raise ValueError(
                    f"File '{path}' is not a PIN file: "
                    "The number of fields is less than number of columns"
                    f" on line {nread}, expected {num_fields} but "
                    f"got {local_num_fields}"
                )

            if local_num_fields > num_fields:
                return True

        return False


@typechecked
class TraditionalPINReader(TabularDataReader):
    """Reader for traditional PIN files with ragged trailing protein columns."""

    file_name: Path

    def __init__(self, file_name: Path, sep: str = "\t"):
        self.file_name = file_name
        self.sep = sep
        self._header_names: list[str] | None = None
        self._default_chunk_size = 50000

    def __str__(self):
        return f"TraditionalPINReader({self.file_name=})"

    def __repr__(self):
        return f"TraditionalPINReader({self.file_name=},{self.sep=})"

    def _get_header_names(self) -> list[str]:
        if self._header_names is not None:
            return self._header_names

        with open_file(self.file_name, mode="rt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                header_names = line.split(self.sep)
                if len(header_names) < 2:
                    raise RuntimeError(
                        f"Error parsing PIN file '{self.file_name}':"
                        f" expected tab-delimited header but got '{line}'."
                    )
                self._header_names = header_names
                return header_names

        raise RuntimeError(
            f"Error parsing PIN file '{self.file_name}': no header found."
        )

    def _iter_normalized_lines(self):
        header_names = self._get_header_names()
        num_cols = len(header_names)

        found_header = False
        with open_file(self.file_name, mode="rt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                if not found_header:
                    found_header = True
                    continue

                line_data = line.split(self.sep)
                line_data = line_data[: num_cols - 1] + [
                    ":".join(line_data[num_cols - 1 :])
                ]
                if len(line_data) != num_cols:
                    raise RuntimeError(
                        "Error parsing PIN file. "
                        f"Line: {line}"
                        f" Expected: {num_cols} columns"
                    )

                yield self.sep.join(line_data)

    def _validate_columns(self, columns: list[str] | None) -> list[str]:
        all_columns = self.get_column_names()
        if columns is None:
            return all_columns

        missing_columns = [column for column in columns if column not in all_columns]
        if missing_columns:
            raise ValueError(
                f"Columns {missing_columns} are not present in '{self.file_name}'."
            )
        return columns

    def _lines_to_dataframe(
        self, lines: list[str], columns: list[str] | None
    ) -> pd.DataFrame:
        with StringIO("\n".join(lines)) as stream:
            df = pd.read_csv(
                stream,
                sep=self.sep,
                header=None,
                names=self.get_column_names(),
                usecols=columns,
            )
        return df if columns is None else df[columns]

    def get_column_names(self) -> list[str]:
        return self._get_header_names()

    def get_column_types(self) -> list[np.dtype]:
        iterator = self.get_chunked_data_iterator(chunk_size=2)
        sample = next(iterator, None)
        if sample is None:
            return [np.dtype("O")] * len(self.get_column_names())

        out = []
        for dtype in sample.dtypes.tolist():
            numpy_dtype = getattr(dtype, "numpy_dtype", None)
            if numpy_dtype is not None:
                out.append(np.dtype(numpy_dtype))
                continue

            try:
                out.append(np.dtype(dtype))
            except TypeError:
                out.append(np.dtype("O"))

        return out

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        selected_columns = self._validate_columns(columns)
        chunks = list(
            self.get_chunked_data_iterator(
                chunk_size=self._default_chunk_size,
                columns=selected_columns,
            )
        )
        if chunks:
            return pd.concat(chunks)
        return pd.DataFrame(columns=selected_columns)

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")

        selected_columns = self._validate_columns(columns)
        row_offset = 0
        buffer: list[str] = []
        for line in self._iter_normalized_lines():
            buffer.append(line)
            if len(buffer) == chunk_size:
                chunk = self._lines_to_dataframe(buffer, selected_columns)
                chunk.index = range(row_offset, row_offset + len(chunk))
                row_offset += len(chunk)
                yield chunk
                buffer = []

        if buffer:
            chunk = self._lines_to_dataframe(buffer, selected_columns)
            chunk.index = range(row_offset, row_offset + len(chunk))
            yield chunk

    def get_default_extension(self) -> str:
        return ".tsv"


def read_traditional_pin(path: Path) -> pd.DataFrame:
    """Reads the file in memory and bundles the proteins.

    The PIN file is assumed to be a traditional PIN file.
    The PIN file is read in memory and the proteins are bundled into a single
    column.

    Parameters
    ----------
    path : Path
        The path to the PIN file.

    Returns
    -------
    pd.DataFrame
        The PIN file as a pandas DataFrame.
    """
    return TraditionalPINReader(path).read()
