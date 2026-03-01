"""
This module contains the parsers for reading in PSMs
"""

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
from pathlib import Path
from pprint import pformat
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typeguard import typechecked

from mokapot.column_defs import (
    ColumnGroups,
)
from mokapot.constants import (
    CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS,
)
from mokapot.dataset import OnDiskPsmDataset, PsmDataset
from mokapot.tabular_data import TabularDataReader, TabularDataWriter
from mokapot.utils import (
    make_bool_trarget,
    tuplize,
)

LOGGER = logging.getLogger(__name__)
IMPORT_SCAN_CELL_BUDGET = 8_000_000
IMPORT_SCAN_MIN_ROWS = 50_000
AUTO_PARQUET_CHUNK_SIZE_ROWS = 200_000
TEXT_INPUT_SUFFIXES = {".pin", ".tsv", ".tab", ".csv"}
PIN_TEXT_COLUMNS = {
    "specid",
    "psmid",
    "id",
    "peptide",
    "proteins",
    "modifiedpeptide",
    "precursor",
    "peptidegroup",
    "filename",
}


def _input_suffix(path: Path) -> str:
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-1] == ".gz":
        return suffixes[-2]
    return path.suffix


def _is_text_input(path: Path) -> bool:
    return _input_suffix(path) in TEXT_INPUT_SUFFIXES


def _converted_path(
    source_path: Path,
    destination_dir: Path,
    position: int,
) -> Path:
    path_hash = sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
    source_stem = source_path.name.replace("/", "_")
    file_name = f"{position:06d}.{source_stem}.{path_hash}.parquet"
    return destination_dir / file_name


def _convert_one_text_input_to_parquet(
    source_path: Path,
    destination_path: Path,
    chunk_size_rows: int,
):
    reader = TabularDataReader.from_path(source_path)
    columns = reader.get_column_names()
    text_columns = _infer_textual_pin_columns(columns)
    column_types = _coerce_textual_pin_columns_to_string(
        columns=columns,
        column_types=reader.get_column_types(),
    )

    writer = TabularDataWriter.from_suffix(
        destination_path,
        columns=columns,
        column_types=column_types,
    )
    with writer:
        for chunk in reader.get_chunked_data_iterator(
            chunk_size=chunk_size_rows,
            columns=columns,
        ):
            for column in text_columns:
                if column in chunk.columns:
                    chunk[column] = chunk[column].astype("string")
            writer.append_data(chunk)


def _coerce_textual_pin_columns_to_string(
    *,
    columns: list[str],
    column_types: list[np.dtype],
) -> list[np.dtype]:
    if len(columns) != len(column_types):  # pragma: no cover - defensive path
        return column_types

    text_columns = _infer_textual_pin_columns(columns)
    out = list(column_types)
    for idx, column_name in enumerate(columns):
        if column_name in text_columns:
            out[idx] = np.dtype("O")
    return out


def _infer_textual_pin_columns(columns: list[str]) -> set[str]:
    out: set[str] = set()
    for column_name in columns:
        name = column_name.casefold()
        if name in PIN_TEXT_COLUMNS or name.startswith("protein"):
            out.add(column_name)
    return out


def _conversion_metadata_path(destination_path: Path) -> Path:
    return destination_path.with_suffix(destination_path.suffix + ".meta.json")


def _conversion_status_path(destination_dir: Path) -> Path:
    return destination_dir / "_auto_parquet_status.json"


def _conversion_complete_path(destination_dir: Path) -> Path:
    return destination_dir / "_auto_parquet_complete"


def _source_signature(source_path: Path) -> dict[str, int | str]:
    stats = source_path.stat()
    return {
        "source_path": str(source_path.resolve()),
        "source_size": int(stats.st_size),
        "source_mtime_ns": int(stats.st_mtime_ns),
    }


def _write_conversion_metadata(
    source_path: Path, destination_path: Path
) -> None:
    payload = _source_signature(source_path)
    payload["converted_path"] = str(destination_path.resolve())
    meta_path = _conversion_metadata_path(destination_path)
    with open(meta_path, "w", encoding="utf-8") as out:
        json.dump(payload, out, sort_keys=True)


def _can_reuse_converted_file(
    source_path: Path, destination_path: Path
) -> bool:
    if not destination_path.exists():
        return False
    try:
        if destination_path.stat().st_size <= 0:
            return False
    except OSError:
        return False

    meta_path = _conversion_metadata_path(destination_path)
    if not meta_path.exists():
        return False

    try:
        with open(meta_path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False

    signature = _source_signature(source_path)
    return (
        payload.get("source_path") == signature["source_path"]
        and payload.get("source_size") == signature["source_size"]
        and payload.get("source_mtime_ns") == signature["source_mtime_ns"]
    )


def _write_conversion_status(
    destination_dir: Path,
    *,
    status: str,
    workers: int,
    total_files: int,
    reused: int,
    converted: int,
    failed: int,
    started_at: float,
) -> None:
    payload = {
        "status": status,
        "workers": workers,
        "total_files": total_files,
        "reused": reused,
        "converted": converted,
        "failed": failed,
        "started_at_epoch_s": started_at,
        "updated_at_epoch_s": time.time(),
    }
    with open(
        _conversion_status_path(destination_dir),
        "w",
        encoding="utf-8",
    ) as out:
        json.dump(payload, out, sort_keys=True)


def _auto_convert_text_inputs_to_parquet(
    pin_files: list[Path],
    *,
    max_workers: int,
    temp_dir: Path,
    auto_parquet: bool,
    auto_parquet_min_bytes: int,
    auto_parquet_min_files: int,
    auto_parquet_workers: int | None,
    force: bool,
) -> list[Path]:
    if not auto_parquet:
        return pin_files

    text_inputs = [pin for pin in pin_files if _is_text_input(pin)]
    if not text_inputs:
        return pin_files

    total_bytes = sum(pin.stat().st_size for pin in text_inputs)
    should_convert = (
        total_bytes >= auto_parquet_min_bytes
        or len(text_inputs) >= auto_parquet_min_files
    )
    if not should_convert:
        return pin_files

    workers = auto_parquet_workers
    if workers is None:
        workers = max_workers
    workers = max(1, workers)
    workers = min(workers, len(text_inputs))

    destination_dir = Path(temp_dir) / "input_parquet"
    destination_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status_started = time.time()
    _conversion_complete_path(destination_dir).unlink(missing_ok=True)

    LOGGER.info(
        "Auto-parquet enabled: converting %d text input files "
        "(total %.2f GiB) using %d workers.",
        len(text_inputs),
        total_bytes / 1024**3,
        workers,
    )
    LOGGER.info(
        "Auto-parquet conversion backend: ThreadPoolExecutor "
        "(per-file parallel conversion)."
    )
    _write_conversion_status(
        destination_dir,
        status="in_progress",
        workers=workers,
        total_files=len(text_inputs),
        reused=0,
        converted=0,
        failed=0,
        started_at=status_started,
    )

    failed_paths: list[tuple[Path, Exception]] = []
    converted_map: dict[Path, Path] = {}
    converted_bytes = 0
    finished = 0
    reused = 0
    text_input_positions = {path: i for i, path in enumerate(text_inputs)}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for source_path in text_inputs:
            out_path = _converted_path(
                source_path=source_path,
                destination_dir=destination_dir,
                position=text_input_positions[source_path],
            )
            if not force and _can_reuse_converted_file(source_path, out_path):
                converted_map[source_path] = out_path
                reused += 1
                finished += 1
                converted_bytes += source_path.stat().st_size
                elapsed = time.perf_counter() - started
                LOGGER.info(
                    "Auto-parquet conversion progress: %d/%d files, %.2f GiB "
                    "processed, elapsed %.1fs.",
                    finished,
                    len(text_inputs),
                    converted_bytes / 1024**3,
                    elapsed,
                )
                _write_conversion_status(
                    destination_dir,
                    status="in_progress",
                    workers=workers,
                    total_files=len(text_inputs),
                    reused=reused,
                    converted=finished - reused,
                    failed=0,
                    started_at=status_started,
                )
                continue

            if force:
                out_path.unlink(missing_ok=True)
                _conversion_metadata_path(out_path).unlink(missing_ok=True)

            futures[
                executor.submit(
                    _convert_one_text_input_to_parquet,
                    source_path,
                    out_path,
                    AUTO_PARQUET_CHUNK_SIZE_ROWS,
                )
            ] = (source_path, out_path)

        LOGGER.info(
            "Auto-parquet dispatch summary: submitted=%d, reused=%d.",
            len(futures),
            reused,
        )
        for future in as_completed(futures):
            source_path, out_path = futures[future]
            try:
                future.result()
                converted_map[source_path] = out_path
                _write_conversion_metadata(source_path, out_path)
            except Exception as exc:  # pragma: no cover - defensive path
                failed_paths.append((source_path, exc))

            finished += 1
            converted_bytes += source_path.stat().st_size
            elapsed = time.perf_counter() - started
            LOGGER.info(
                "Auto-parquet conversion progress: %d/%d files, %.2f GiB "
                "processed, elapsed %.1fs.",
                finished,
                len(text_inputs),
                converted_bytes / 1024**3,
                elapsed,
            )
            _write_conversion_status(
                destination_dir,
                status="in_progress",
                workers=workers,
                total_files=len(text_inputs),
                reused=reused,
                converted=finished - reused,
                failed=len(failed_paths),
                started_at=status_started,
            )

    if failed_paths:
        _write_conversion_status(
            destination_dir,
            status="failed",
            workers=workers,
            total_files=len(text_inputs),
            reused=reused,
            converted=finished - reused,
            failed=len(failed_paths),
            started_at=status_started,
        )
        LOGGER.error(
            "Auto-parquet conversion failed for %d file(s):",
            len(failed_paths),
        )
        for source_path, exc in failed_paths:
            LOGGER.error("  - %s: %s", source_path, exc)
        raise RuntimeError("Auto-parquet conversion failed.")

    converted_files = [converted_map.get(pin, pin) for pin in pin_files]
    LOGGER.info(
        "Auto-parquet reuse summary: reused=%d, converted=%d.",
        reused,
        len(text_inputs) - reused,
    )
    LOGGER.info(
        "Auto-parquet conversion finished in %.1fs.",
        time.perf_counter() - started,
    )
    _write_conversion_status(
        destination_dir,
        status="complete",
        workers=workers,
        total_files=len(text_inputs),
        reused=reused,
        converted=finished - reused,
        failed=0,
        started_at=status_started,
    )
    _conversion_complete_path(destination_dir).touch(exist_ok=True)
    LOGGER.info(
        "Auto-parquet completion marker: %s",
        _conversion_complete_path(destination_dir),
    )
    return converted_files


# Functions -------------------------------------------------------------------
def read_pin(
    pin_files,
    max_workers: int,
    temp_dir: Path | None = None,
    auto_parquet: bool = True,
    auto_parquet_min_bytes: int = 8 * 1024**3,
    auto_parquet_min_files: int = 256,
    auto_parquet_workers: int | None = None,
    filename_column=None,
    calcmass_column=None,
    expmass_column=None,
    rt_column=None,
    charge_column=None,
    read_workers: int | None = None,
    force: bool = False,
) -> list[OnDiskPsmDataset]:
    """Read Percolator input (PIN) tab-delimited files.

    Read PSMs from one or more Percolator input (PIN) tab-delmited files,
    aggregating them into a single
    :py:class:`~mokapot.dataset.LinearPsmDataset`. For more details about the
    PIN file format, see the `Percolator documentation
    <https://github.com/percolator/percolator/
    wiki/Interface#tab-delimited-file-format>`_.

    Specifically, mokapot requires specific columns in the tab-delmited files:
    `specid`, `scannr`, `peptide`, `proteins`, and `label`. Note that these
    column names are case insensitive. In addition to these special columns
    defined for the PIN format, mokapot also looks for additional columns that
    specify the MS data file names, theoretical monoisotopic peptide masses,
    the measured mass, retention times, and charge states, which are necessary
    to create specific output formats for downstream tools, such as FlashLFQ.

    In addition to PIN tab-delimited files, the `pin_files` argument can be a
    :py:class:`pandas.DataFrame` containing the above columns.

    Finally, mokapot does not currently support specifying a default direction
    or feature weights in the PIN file itself. If these are present, they
    will be ignored.

    Parameters
    ----------
    pin_files : str, tuple of str, or pandas.DataFrame
        One or more PIN files to read or a :py:class:`pandas.DataFrame`.
    max_workers: int
        Maximum number of parallel processes to use.
    read_workers : int, optional
        Number of workers used to parse multiple input files in parallel.
        Defaults to ``max_workers``.
    force : bool, optional
        Force regeneration of temporary conversion artifacts in ``temp_dir``.
    filename_column : str, optional
        The column specifying the MS data file. If :code:`None`, mokapot will
        look for a column called "filename" (case insensitive). This is
        required for some output formats, such as FlashLFQ.
    calcmass_column : str, optional
        The column specifying the theoretical monoisotopic mass of the peptide
        including modifications. If :code:`None`, mokapot will look for a
        column called "calcmass" (case insensitive). This is required for some
        output formats, such as FlashLFQ.
    expmass_column : str, optional
        The column specifying the measured neutral precursor mass. If
        :code:`None`, mokapot will look for a column call "expmass" (case
        insensitive). This is required for some output formats.
    rt_column : str, optional
        The column specifying the retention time in seconds. If :code:`None`,
        mokapot will look for a column called "ret_time" (case insensitive).
        This is required for some output formats, such as FlashLFQ.
    charge_column : str, optional
        The column specifying the charge state of each peptide. If
        :code:`None`, mokapot will look for a column called "charge" (case
        insensitive). This is required for some output formats, such as
        FlashLFQ.

    Returns
    -------
        A list of :py:class:`~mokapot.dataset.OnDiskPsmDataset` objects
        containing the PSMs from all of the PIN files.
    """
    logging.info("Parsing PSMs...")
    started = time.perf_counter()
    pin_paths = [Path(pin_file) for pin_file in tuplize(pin_files)]
    if read_workers is None:
        read_workers = max_workers
    read_workers = max(1, int(read_workers))

    if temp_dir is None:
        temp_dir = (
            Path.cwd() / ".mokapot-temp" / f"readpin-{uuid.uuid4().hex[:12]}"
        )

    pin_paths = _auto_convert_text_inputs_to_parquet(
        pin_paths,
        max_workers=max_workers,
        temp_dir=temp_dir,
        auto_parquet=auto_parquet,
        auto_parquet_min_bytes=auto_parquet_min_bytes,
        auto_parquet_min_files=auto_parquet_min_files,
        auto_parquet_workers=auto_parquet_workers,
        force=force,
    )

    if len(pin_paths) <= 1 or read_workers <= 1:
        datasets = [
            read_percolator(
                pin_file,
                max_workers=max_workers,
                filename_column=filename_column,
                calcmass_column=calcmass_column,
                expmass_column=expmass_column,
                rt_column=rt_column,
                charge_column=charge_column,
            )
            for pin_file in pin_paths
        ]
    else:
        workers = min(read_workers, len(pin_paths))
        LOGGER.info(
            "Reading %d input files in parallel with %d workers.",
            len(pin_paths),
            workers,
        )
        datasets_ordered: list[OnDiskPsmDataset | None] = [
            None
        ] * len(pin_paths)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    read_percolator,
                    pin_file,
                    max_workers,
                    filename_column,
                    calcmass_column,
                    expmass_column,
                    rt_column,
                    charge_column,
                ): idx
                for idx, pin_file in enumerate(pin_paths)
            }
            for future in as_completed(futures):
                idx = futures[future]
                datasets_ordered[idx] = future.result()

        datasets = []
        for dataset in datasets_ordered:
            if dataset is None:  # pragma: no cover - defensive path
                raise RuntimeError("Failed to parse one or more input files.")
            datasets.append(dataset)

    LOGGER.info(
        "Finished parsing %d input file(s) in %.1fs.",
        len(pin_paths),
        time.perf_counter() - started,
    )
    return datasets


def read_percolator(
    perc_file: Path,
    max_workers,
    filename_column=None,
    calcmass_column=None,
    expmass_column=None,
    rt_column=None,
    charge_column=None,
) -> OnDiskPsmDataset:
    """
    Read a Percolator tab-delimited file.

    Percolator input format (PIN) files and the Percolator result files
    are tab-delimited, but also have a tab-delimited protein list as the
    final column. This function parses the file and returns a Dataset.

    Parameters
    ----------
    perc_file : Path
        The file to parse.

    Returns
    -------
    OnDiskPsmDataset
    """

    LOGGER.info("Reading %s...", perc_file)
    reader = TabularDataReader.from_path(perc_file)
    columns = reader.get_column_names()
    prelim_columns = ColumnGroups.infer_from_colnames(
        columns,
        filename_column=filename_column,
        calcmass_column=calcmass_column,
        expmass_column=expmass_column,
        rt_column=rt_column,
        charge_column=charge_column,
    )
    features = prelim_columns.feature_columns
    spectra = prelim_columns.spectrum_columns
    labels = prelim_columns.target_column

    # Check that features don't have missing values in a single chunked pass.
    df_spectra, features_to_drop = _scan_spectra_and_missing_features(
        reader=reader,
        features=features,
        spectra=spectra,
        target_column=labels,
    )
    tmp_labels = make_bool_trarget(df_spectra.loc[:, labels])
    # Deleting the column solves a deprecation warning that mentions
    # "assiging column with incompatible dtype"
    del df_spectra[labels]
    df_spectra.loc[:, labels] = tmp_labels

    if len(features_to_drop) > 1:
        LOGGER.warning("Missing values detected in the following features:")
        for col in features_to_drop:
            LOGGER.warning("  - %s", col)

        LOGGER.warning("Dropping features with missing values...")
    _feature_columns = [
        feature for feature in features if feature not in features_to_drop
    ]

    LOGGER.info("Using %i features:", len(_feature_columns))
    for i, feat in enumerate(_feature_columns):
        LOGGER.info("  (%i)\t%s", i + 0, feat)

    prelim_columns = prelim_columns.update(
        feature_columns=_feature_columns,
    )
    column_groups = prelim_columns
    LOGGER.info(f"Infered column grouping: {pformat(column_groups)}")

    return OnDiskPsmDataset(
        perc_file,
        column_groups=column_groups,
        spectra_dataframe=df_spectra,
    )


# Utility Functions -----------------------------------------------------------
def _get_adaptive_import_chunk_size(
    n_requested_columns: int,
) -> int:
    if CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS < 1:
        raise ValueError("CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS must be >= 1.")

    if n_requested_columns < 1:
        return CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS

    rows_from_budget = max(
        IMPORT_SCAN_MIN_ROWS,
        IMPORT_SCAN_CELL_BUDGET // n_requested_columns,
    )
    return max(
        1,
        min(CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS, rows_from_budget),
    )


def _scan_spectra_and_missing_features(
    reader: TabularDataReader,
    features: tuple[str, ...],
    spectra: tuple[str, ...],
    target_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    spectra_columns = list(spectra + (target_column,))
    scan_columns = list(dict.fromkeys(spectra_columns + list(features)))
    chunk_size = _get_adaptive_import_chunk_size(len(scan_columns))
    feature_na_mask = np.zeros(len(features), dtype=bool)
    spectra_chunks: list[pd.DataFrame] = []

    file_iterator = reader.get_chunked_data_iterator(
        chunk_size=chunk_size,
        columns=scan_columns,
    )
    for chunk in file_iterator:
        spectra_chunks.append(chunk[spectra_columns])
        if len(features):
            feature_na_mask |= (
                chunk[list(features)].isna().any(axis=0).to_numpy()
            )

    if spectra_chunks:
        df_spectra = pd.concat(spectra_chunks)
    else:
        df_spectra = pd.DataFrame(columns=spectra_columns)

    features_to_drop = [
        feature for feature, is_na in zip(features, feature_na_mask) if is_na
    ]
    return df_spectra, features_to_drop


@typechecked
def get_rows_from_dataframe(
    idx_sets: Iterable[set[int]],
    chunk: pd.DataFrame,
    train_psms,
    target_column: str,
    file_idx: int,
):
    """
    extract rows from a chunk of a dataframe

    Parameters
    ----------
    idx_sets : list of sets of indexes
        The indexes to select from dataframe.
    train_psms : list of list of dataframes
        Contains subsets of dataframes that are already extracted.
    chunk : dataframe
        Subset of a dataframe.
    target_column : str
        The target column name, expected to be in the dataframe.
    file_idx : the index of the file being searched

    Returns
    -------
    None
        The function modifies the `train_psms` list in place.
    """
    tmp = make_bool_trarget(chunk.loc[:, target_column])
    # Deleting the column solves a deprecation warning that
    # mentions "assiging column with incompatible dtype"
    del chunk[target_column]
    chunk.loc[:, target_column] = tmp
    chunk_idx_set = set(chunk.index)
    for k, train_idx_set in enumerate(idx_sets):
        idx_ = list(train_idx_set & chunk_idx_set)
        train_psms[file_idx][k].append(chunk.loc[idx_])


def concat_and_reindex_chunks(df, orig_idx):
    return [
        pd.concat(df_fold).reindex(orig_idx_fold)
        for df_fold, orig_idx_fold in zip(df, orig_idx)
    ]


@typechecked
def parse_in_chunks(
    datasets: list[PsmDataset],
    train_idx: list[list[list[int]]],
    chunk_size: int,
    max_workers: int,
) -> list[pd.DataFrame]:
    """
    Parse a file in chunks

    Parameters
    ----------
    datasets : OnDiskPsmDataset
        A collection of PSMs.
    train_idx : list of a list of a list of indexes
        - first level are training splits,
        - second one is the number of input files
        - third level the actual idexes The indexes to select from data.
        Thus if you have 3 splits, 2 files and 10 PSMs per file, the
        "shape" of the list is [3,2,10]
    chunk_size : int
        The chunk size in bytes.
    max_workers: int
            Number of workers for Parallel

    Returns
    -------
    List
        list of dataframes
    """

    train_psms = [
        [[] for _ in range(len(train_idx))] for _ in range(len(datasets))
    ]
    for dataset, idx, file_idx in zip(
        datasets, zip(*train_idx), range(len(datasets))
    ):
        idx_sets = [set(train_fold_idx) for train_fold_idx in idx]
        # Note: Here idx is a tuple of len == number of folds
        #       each element is a list of ints, so each list is
        #       the indices for each split of the dataset.

        # Note2: Technically the file_idx is not a file but a dataset
        #        index.
        if hasattr(dataset, "reader"):
            # Handle OnDiskPsmDataset
            reader = dataset.reader
            file_iterator = reader.get_chunked_data_iterator(
                chunk_size=chunk_size, columns=dataset.columns
            )
            # Q: Is it really a good idea to modify a list in place
            #    in a parallel loop?
            Parallel(n_jobs=max_workers, require="sharedmem")(
                delayed(get_rows_from_dataframe)(
                    idx_sets,
                    chunk,
                    train_psms,
                    dataset.target_column,
                    file_idx,
                )
                for chunk in file_iterator
            )
        else:
            # Handle LinearPsmDataset
            chunk = dataset.data
            get_rows_from_dataframe(
                idx_sets,
                chunk,
                train_psms=train_psms,
                target_column=dataset.target_column,
                file_idx=file_idx,
            )
    train_psms_reordered = Parallel(n_jobs=max_workers, require="sharedmem")(
        delayed(concat_and_reindex_chunks)(df=df, orig_idx=orig_idx)
        for df, orig_idx in zip(train_psms, zip(*train_idx))
    )
    return [pd.concat(df) for df in zip(*train_psms_reordered)]
