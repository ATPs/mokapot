"""
This module contains the parsers for reading in PSMs
"""

import logging
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
from mokapot.tabular_data import TabularDataReader
from mokapot.utils import (
    make_bool_trarget,
    tuplize,
)

LOGGER = logging.getLogger(__name__)
IMPORT_SCAN_CELL_BUDGET = 8_000_000
IMPORT_SCAN_MIN_ROWS = 50_000


# Functions -------------------------------------------------------------------
def read_pin(
    pin_files,
    max_workers: int,
    filename_column=None,
    calcmass_column=None,
    expmass_column=None,
    rt_column=None,
    charge_column=None,
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
    return [
        read_percolator(
            pin_file,
            max_workers=max_workers,
            filename_column=filename_column,
            calcmass_column=calcmass_column,
            expmass_column=expmass_column,
            rt_column=rt_column,
            charge_column=charge_column,
        )
        for pin_file in tuplize(pin_files)
    ]


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
            feature_na_mask |= chunk[list(features)].isna().any(axis=0).to_numpy()

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
