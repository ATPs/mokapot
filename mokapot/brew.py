"""
Defines a function to run the Percolator algorithm.
"""

import copy
import logging
import tempfile
from pathlib import Path
from operator import itemgetter
from typing import Generator, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typeguard import typechecked

from mokapot import utils
from mokapot.constants import (
    CHUNK_SIZE_READ_ALL_DATA,
    CHUNK_SIZE_ROWS_PREDICTION,
)
from mokapot.dataset import (
    LinearPsmDataset,
    PsmDataset,
    calibrate_scores,
    update_labels,
)
from mokapot.model import (
    BestFeatureIsBetterError,
    Model,
    ModelIterationError,
    PercolatorModel,
)
from mokapot.tabular_data import TabularDataWriter
from mokapot.utils import strictzip

LOGGER = logging.getLogger(__name__)


def _fold_dtype(folds: int) -> np.dtype:
    if folds <= np.iinfo(np.uint8).max:
        return np.dtype("uint8")
    if folds <= np.iinfo(np.uint16).max:
        return np.dtype("uint16")
    return np.dtype("uint32")


def _allocate_large_array(
    *,
    length: int,
    dtype: np.dtype,
    temp_dir: Path | None,
    memmap_threshold_psms: int,
    file_name: str,
) -> np.ndarray:
    if (
        temp_dir is not None
        and memmap_threshold_psms > 0
        and length >= memmap_threshold_psms
    ):
        memmap_path = Path(temp_dir) / "memmap"
        memmap_path.mkdir(parents=True, exist_ok=True)
        return np.memmap(
            memmap_path / file_name,
            mode="w+",
            dtype=dtype,
            shape=(length,),
        )
    return np.empty(length, dtype=dtype)


# Functions -------------------------------------------------------------------
@typechecked
def resolve_parallelism(
    max_workers: int,
    folds: int,
    model_n_jobs: int | str = "auto",
) -> tuple[int, int]:
    """
    Resolve fold-level and model-level parallelism settings.

    Parameters
    ----------
    max_workers : int
        Total requested worker budget.
    folds : int
        Number of cross-validation folds.
    model_n_jobs : int | str
        Model-level job count or ``"auto"``.
    """
    if max_workers < 1:
        raise ValueError("'max_workers' must be >= 1.")

    if folds < 1:
        raise ValueError("'folds' must be >= 1.")

    outer_workers = min(folds, max_workers)
    if model_n_jobs == "auto":
        inner_workers = max(1, max_workers // outer_workers)
    else:
        if model_n_jobs < 1:
            raise ValueError("'model_n_jobs' must be >= 1 or 'auto'.")
        inner_workers = model_n_jobs

    return outer_workers, inner_workers


@typechecked
def brew(
    datasets: list[PsmDataset],
    model: None | Model | list[Model] = None,
    test_fdr: float = 0.01,
    folds: int = 3,
    max_workers: int = 1,
    rng=None,
    subset_max_train: int | None = None,
    ensemble: bool = False,
    temp_dir: Path | None = None,
    memmap_threshold_psms: int = 100_000_000,
) -> tuple[list[Model], list[np.ndarray[np.float64]]]:
    """
    Re-score one or more collection of PSMs.

    The provided PSMs analyzed using the semi-supervised learning
    algorithm that was introduced by
    `Percolator <http://percolator.ms>`_. Cross-validation is used to
    ensure that the learned models to not overfit to the PSMs used for
    model training. If a multiple collections of PSMs are provided, they
    are aggregated for model training, but the confidence estimates are
    calculated separately for each collection.

    A list of previously trained models can be provided to the ``model``
    argument to rescore the PSMs in each fold. Note that the number of
    models must match ``folds``. Furthermore, it is valid to use the
    learned models on the same dataset from which they were trained,
    but they must be provided in the same order, such that the
    relationship of the cross-validation folds is maintained.

    Parameters
    ----------
    datasets : PsmDataset object or list of PsmDataset objects
        One or more :doc:`collections of PSMs <dataset>` objects.
        PSMs are aggregated across all of the collections for model
        training, but the confidence estimates are calculated and
        returned separately.
        Can also be instances of OnDiskPsmDatasets.
    model: Model object or list of Model objects, optional
        The :py:class:`mokapot.Model` object to be fit. The default is
        :code:`None`, which attempts to mimic the same support vector
        machine models used by Percolator. If a list of
        :py:class:`mokapot.Model` objects is provided, they are assumed
        to be previously trained models and will and one will be
        used to rescore each fold.
    test_fdr : float, optional
        The false-discovery rate threshold at which to evaluate
        the learned models.
    folds : int, optional
        The number of cross-validation folds to use. PSMs originating
        from the same mass spectrum are always in the same fold.
    max_workers : int, optional
        The number of processes to use for model training. More workers
        will require more memory, but will typically decrease the total
        run time. An integer exceeding the number of folds will have
        no additional effect. Note that logging messages will be garbled
        if more than one worker is enabled.
    temp_dir : pathlib.Path, optional
        Directory to store large temporary arrays and fold training buckets.
    memmap_threshold_psms : int, optional
        Enable numpy.memmap for selected large arrays when dataset length is
        at least this threshold.
    rng : int, np.random.Generator, optional
        A seed or generator used to generate splits, or None to use the
        default random number generator state.

    Returns
    -------
    models : list[Model]
        The learned :py:class:`~mokapot.model.Model` objects, one
        for each fold.
    scores : list[np.array[float]]
        The scores
    """
    outer_workers, _ = resolve_parallelism(
        max_workers=max_workers,
        folds=folds,
    )
    temp_dir = Path(temp_dir) if temp_dir is not None else None

    rng = np.random.default_rng(rng)
    if model is None:
        model = PercolatorModel()

    try:
        # Q: what is this doing? Why does the randon number
        # generater get set only if the model has an estimator?
        # Shouldn't it assign it to all the models if they are passed?
        model.estimator
        model.rng = rng
    except AttributeError:
        pass

    # Check that all of the datasets have the same features:
    feat_set = set(datasets[0].feature_columns)
    if not all([
        set(dataset.feature_columns) == feat_set for dataset in datasets
    ]):
        raise ValueError("All collections of PSMs must use the same features.")

    data_size = [len(datasets.spectra_dataframe) for datasets in datasets]
    if sum(data_size) > 1:
        LOGGER.info("Found %i total PSMs.", sum(data_size))
        num_targets = 0
        num_decoys = 0
        for dataset in datasets:
            trg = dataset.target_values
            local_ntargets = trg.sum()
            local_ndecoys = len(trg) - local_ntargets
            num_targets += local_ntargets
            num_decoys += local_ndecoys

        LOGGER.info(
            "  - %i target PSMs and %i decoy PSMs detected.",
            num_targets,
            num_decoys,
        )
    LOGGER.info("Splitting PSMs into %i folds...", folds)
    fold_dtype = _fold_dtype(folds)
    fold_ids_list = []
    for ds_idx, (dataset, ds_size) in enumerate(zip(datasets, data_size)):
        out = _allocate_large_array(
            length=ds_size,
            dtype=fold_dtype,
            temp_dir=temp_dir,
            memmap_threshold_psms=memmap_threshold_psms,
            file_name=f"fold_ids.dataset-{ds_idx}.mmap",
        )
        fold_ids = dataset.make_fold_ids(
            folds=folds,
            rng=rng,
            dtype=fold_dtype,
            out=out,
        )
        fold_ids_list.append(fold_ids)

    # If trained models are provided, use them as-is.
    # If the model is not iterable, it means that a single model is pased, thus
    # It is more of a template for training. (or was generated within this
    # code, thus "None" was passed)
    is_mod_iterable = hasattr(model, "__iter__")
    if is_mod_iterable:
        # Q: Is this branch ever used?
        # JSPP 2024-12-06 I think it makes sense to split this function
        # To remve the trained case ... which adds a lot of clutter.
        # Furthermore, that function can fall back to this one if its
        # Not actually trained.
        fitted = [[m, False] for m in model if m.is_trained]

        if len(model) != folds:
            raise ValueError(
                f"The number of trained models ({len(model)}) "
                f"must match the number of folds ({folds})."
            )

        if len(fitted) != len(model):  # Test that all models are fitted.
            raise RuntimeError(
                "One or more of the provided models was not previously trained"
            )

    else:
        train_bucket_paths = _write_train_buckets(
            datasets=datasets,
            fold_ids_list=fold_ids_list,
            folds=folds,
            chunk_size=CHUNK_SIZE_READ_ALL_DATA,
            rng=rng,
            subset_max_train=subset_max_train,
            temp_dir=temp_dir,
        )
        fitted = Parallel(n_jobs=outer_workers, require="sharedmem")(
            delayed(_fit_model)(bucket_path, datasets, copy.deepcopy(model), f)
            for f, bucket_path in enumerate(train_bucket_paths)
        )

    # Sort models to have deterministic results with multithreading.
    fitted.sort(key=lambda x: x[0].fold)
    models, resets = list(zip(*fitted))

    # Determine if the models need to be reset:
    reset = any(resets)

    # If we reset, just use the original model on all the folds:
    if reset:
        scores = [
            dataset.calibrate_scores(
                _predict_with_ensemble(
                    dataset=dataset,
                    models=[model],
                    max_workers=outer_workers,
                ),
                test_fdr,
            )
            for dataset in datasets
        ]

    # If we don't reset, assign scores to each fold:
    elif all([m.is_trained for m in models]):
        if ensemble:
            scores = [
                _predict_with_ensemble(
                    dataset=dataset,
                    models=models,
                    max_workers=outer_workers,
                )
                for dataset in datasets
            ]
        else:
            scores = list(
                _predict(
                    fold_ids_list=fold_ids_list,
                    datasets=datasets,
                    models=models,
                    test_fdr=test_fdr,
                    max_workers=outer_workers,
                    temp_dir=temp_dir,
                    memmap_threshold_psms=memmap_threshold_psms,
                )
            )
    else:
        num_passed = sum([m.is_trained for m in models])
        num_failed = len(models) - num_passed
        LOGGER.warning(
            f"Model training failed on {num_failed}/{len(models)}."
            " Setting scores to zero."
        )
        scores = [np.zeros(x) for x in data_size]
    # Find which is best: the learned model, the best feature, or
    # a pretrained model.
    if not all([m.override for m in models]):
        best_feats = [[m.best_feat, m.feat_pass, m.desc] for m in models]
        best_feat_idx, feat_total = max(
            enumerate(map(itemgetter(1), best_feats)), key=itemgetter(1)
        )
    else:
        feat_total = 0

    preds = [
        update_labels(
            scores=score,
            targets=dataset.target_values,
            eval_fdr=test_fdr,
        )
        for dataset, score in zip(datasets, scores)
    ]

    pred_total = sum([(pred == 1).sum() for pred in preds])

    # Here, f[0] is the name of the best feature, and f[3] is a boolean
    if feat_total > pred_total:
        using_best_feat = True
        feat, _, desc = best_feats[best_feat_idx]
        descs = [desc] * len(datasets)
        scores = [
            dataset.read_data(
                columns=[feat],
            ).values[:, 0]
            for dataset in datasets
        ]
    else:
        using_best_feat = False
        descs = [True] * len(datasets)

    if using_best_feat:
        LOGGER.warning(
            "Learned model did not improve over the best feature. "
            "Now scoring by the best feature for each collection "
            "of PSMs."
        )
    elif reset:
        LOGGER.warning(
            "Learned model did not improve upon the pretrained "
            "input model. Now re-scoring each collection of PSMs "
            "using the original model."
        )

    # Reverse all scores for which desc is False (this way, we don't have to
    # return `descs` from this function
    # Q: why dont we just return a class that denotes if its descending?
    #    JSPP 2024-12-15
    for idx, desc in enumerate(descs):
        if not desc:
            scores[idx] = -scores[idx]
            descs[idx] = not descs[idx]

    LOGGER.info("Assigning scores to PSMs...")
    for score, dataset in strictzip(scores, datasets):
        if str(score.dtype) == "bool":
            # On the rare occasions where the scores are bools
            # This can happen if the model falls back to the best feature
            # and the best feature is a bool.
            scores = np.array(score, dtype=float)
        dataset.scores = score

    return list(models), scores


# Utility Functions -----------------------------------------------------------
def _write_train_buckets(
    *,
    datasets: list[PsmDataset],
    fold_ids_list: list[np.ndarray],
    folds: int,
    chunk_size: int,
    rng,
    subset_max_train: int | None,
    temp_dir: Path | None,
) -> list[Path]:
    if temp_dir is None:
        bucket_dir = Path(
            tempfile.mkdtemp(
                prefix=".mokapot-fold-buckets-",
                dir=Path.cwd(),
            )
        )
    else:
        bucket_dir = Path(temp_dir) / "fold_buckets"
        bucket_dir.mkdir(parents=True, exist_ok=True)

    columns = datasets[0].columns
    column_types = datasets[0].reader.get_column_types()

    bucket_paths = [
        bucket_dir / f"train_fold_{fold}.parquet" for fold in range(folds)
    ]
    writers = [
        TabularDataWriter.from_suffix(
            path,
            columns=columns,
            column_types=column_types,
        )
        for path in bucket_paths
    ]

    subset_max_train_per_file = []
    if subset_max_train is not None:
        subset_max_train_per_file = [
            subset_max_train // len(datasets) for _ in range(len(datasets))
        ]
        subset_max_train_per_file[-1] += subset_max_train - sum(
            subset_max_train_per_file
        )

    per_file_fold_remaining = None
    if subset_max_train_per_file:
        per_file_fold_remaining = np.zeros((len(datasets), folds), dtype=np.int64)
        for file_idx, fold_ids in enumerate(fold_ids_list):
            desired = subset_max_train_per_file[file_idx]
            for fold in range(folds):
                available = int(np.count_nonzero(fold_ids != fold))
                per_file_fold_remaining[file_idx, fold] = min(desired, available)

    rows_per_fold = np.zeros(folds, dtype=np.int64)
    for writer in writers:
        writer.initialize()

    try:
        for file_idx, (dataset, fold_ids) in enumerate(
            zip(datasets, fold_ids_list)
        ):
            offset = 0
            file_iterator = dataset.read_data_chunked(
                chunk_size=chunk_size,
                columns=dataset.columns,
            )
            for chunk in file_iterator:
                # Keep target representation aligned with the legacy path.
                chunk[dataset.target_column] = utils.make_bool_trarget(
                    chunk[dataset.target_column]
                )
                chunk_rows = len(chunk.index)
                chunk_fold_ids = np.asarray(fold_ids[offset : offset + chunk_rows])
                offset += chunk_rows
                for fold in range(folds):
                    mask_idx = np.flatnonzero(chunk_fold_ids != fold)
                    if len(mask_idx) == 0:
                        continue

                    if per_file_fold_remaining is not None:
                        keep = int(per_file_fold_remaining[file_idx, fold])
                        if keep <= 0:
                            continue
                        if len(mask_idx) > keep:
                            mask_idx = rng.choice(
                                mask_idx,
                                keep,
                                replace=False,
                            )
                        per_file_fold_remaining[file_idx, fold] -= len(mask_idx)

                    out_chunk = chunk.iloc[mask_idx]
                    if out_chunk.columns.tolist() != list(columns):
                        out_chunk = out_chunk.reindex(columns=columns)
                    writers[fold].append_data(out_chunk)
                    rows_per_fold[fold] += len(out_chunk.index)
    finally:
        for writer in writers:
            writer.finalize()

    LOGGER.info(
        "Prepared fold training buckets: %s",
        ", ".join(f"fold{idx}={rows}" for idx, rows in enumerate(rows_per_fold)),
    )
    return bucket_paths


def make_train_sets(
    test_idx, subset_max_train, data_size, rng
) -> Generator[list[list[int]], None, None]:
    """
    Parameters
    ----------
    test_idx : list of list of numpy.ndarray
        The indicies of the test sets
    subset_max_train : int or None
        The number of PSMs for training.
    data_size : list[int]
        size of the input data

    Yields
    ------
    list of list of int
        The training set. Each element is a list of ints.
    """
    subset_max_train_per_file = []
    if subset_max_train is not None:
        subset_max_train_per_file = [
            subset_max_train // len(data_size) for _ in range(len(data_size))
        ]
        subset_max_train_per_file[-1] += subset_max_train - sum(
            subset_max_train_per_file
        )
    for fold_idx in zip(*test_idx):
        train_idx = [None for _ in data_size]
        train_idx_size = 0
        for file_idx, idx in enumerate(fold_idx):
            ds = data_size[file_idx]
            mask = np.ones(ds, dtype=bool)
            mask[np.asarray(idx, dtype=int)] = False
            train_idx_file = np.flatnonzero(mask)
            train_idx[file_idx] = train_idx_file
            train_idx_size += len(train_idx_file)
        if len(subset_max_train_per_file) > 0 and train_idx_size > sum(
            subset_max_train_per_file
        ):
            LOGGER.info(
                "Subsetting PSMs (%i) to (%i).",
                train_idx_size,
                subset_max_train,
            )
            for i, current_subset_max_train in enumerate(
                subset_max_train_per_file
            ):
                if current_subset_max_train < len(train_idx[i]):
                    train_idx[i] = rng.choice(
                        train_idx[i], current_subset_max_train, replace=False
                    )
        yield [idx.tolist() for idx in train_idx]


@typechecked
def _create_linear_dataset(
    dataset: PsmDataset, psms: pd.DataFrame, enforce_checks: bool = True
):
    psms[dataset.target_column] = utils.make_bool_trarget(
        psms.loc[:, dataset.target_column]
    )
    return LinearPsmDataset(
        psms=psms,
        column_groups=dataset.column_groups,
        copy_data=False,
        enforce_checks=enforce_checks,
    )


@typechecked
def _predict_fold_chunk(
    model: Model,
    fold: int,
    row_idx: np.ndarray,
    chunk: pd.DataFrame,
    dataset: PsmDataset,
):
    if len(row_idx) == 0:
        return fold, row_idx, np.array([], dtype=float)
    psm_slice = chunk.iloc[row_idx].copy()
    dataset_slice = _create_linear_dataset(
        dataset,
        psm_slice,
        enforce_checks=False,
    )
    return fold, row_idx, model.predict(dataset_slice)


@typechecked
def _predict(
    fold_ids_list: list[np.ndarray],
    datasets: Iterable[PsmDataset],
    models: Iterable[Model],
    test_fdr: float,
    max_workers: int,
    temp_dir: Path | None,
    memmap_threshold_psms: int,
):
    """
    Return the new scores for the dataset

    Parameters
    ----------
    datasets : Dict
        Contains all required info about the dataset to rescore
    fold_ids_list : list of numpy.ndarray
        Fold assignment for each row in each dataset.
    models : list of Model
        The models for each dataset and whether it
        was reset or not.
    test_fdr : the fdr to calibrate at.
    max_workers : maximum threads for parallelism

    Returns
    -------
    numpy.ndarray
        A :py:class:`numpy.ndarray` containing the new scores.
    """
    for ds_idx, (dataset, fold_ids) in enumerate(zip(datasets, fold_ids_list)):
        n_rows = len(fold_ids)
        worker_count = min(max_workers, len(models))
        raw_scores = _allocate_large_array(
            length=n_rows,
            dtype=np.dtype(float),
            temp_dir=temp_dir,
            memmap_threshold_psms=memmap_threshold_psms,
            file_name=f"raw_scores.dataset-{ds_idx}.mmap",
        )
        raw_scores[:] = 0.0
        targets = _allocate_large_array(
            length=n_rows,
            dtype=np.dtype(bool),
            temp_dir=temp_dir,
            memmap_threshold_psms=memmap_threshold_psms,
            file_name=f"targets.dataset-{ds_idx}.mmap",
        )
        targets[:] = False
        n_folds = len(models)
        offset = 0
        file_iterator = dataset.read_data_chunked(
            columns=dataset.columns,
            chunk_size=CHUNK_SIZE_ROWS_PREDICTION,
        )
        for psms_chunk in file_iterator:
            chunk_size = len(psms_chunk.index)
            chunk_fold_ids = np.asarray(fold_ids[offset : offset + chunk_size])
            chunk_targets = utils.make_bool_trarget(psms_chunk[dataset.target_column])
            targets[offset : offset + chunk_size] = np.asarray(chunk_targets)
            row_indices = [
                np.flatnonzero(chunk_fold_ids == fold_idx)
                for fold_idx in range(n_folds)
            ]
            fold_results = Parallel(n_jobs=worker_count, require="sharedmem")(
                delayed(_predict_fold_chunk)(
                    model=models[fold_idx],
                    fold=fold_idx,
                    row_idx=row_indices[fold_idx],
                    chunk=psms_chunk,
                    dataset=dataset,
                )
                for fold_idx in range(n_folds)
            )
            covered = 0
            for _, row_idx, fold_scores in fold_results:
                if len(row_idx) == 0:
                    continue
                covered += len(row_idx)
                raw_scores[offset + row_idx] = fold_scores
            if covered != chunk_size:
                raise RuntimeError(
                    f"Prediction fold routing mismatch at offset {offset}: "
                    f"covered {covered} rows for chunk size {chunk_size}."
                )
            offset += chunk_size

        if offset != n_rows:
            raise RuntimeError(
                f"Prediction row count mismatch for dataset {ds_idx}: "
                f"expected {n_rows}, got {offset}."
            )

        scores = _allocate_large_array(
            length=n_rows,
            dtype=np.dtype(float),
            temp_dir=temp_dir,
            memmap_threshold_psms=memmap_threshold_psms,
            file_name=f"calibrated_scores.dataset-{ds_idx}.mmap",
        )
        scores[:] = 0.0
        for fold_idx in range(n_folds):
            fold_rows = np.flatnonzero(np.asarray(fold_ids) == fold_idx)
            if len(fold_rows) == 0:
                continue
            try:
                scores[fold_rows] = calibrate_scores(
                    np.asarray(raw_scores[fold_rows]),
                    np.asarray(targets[fold_rows]),
                    test_fdr,
                )
            except RuntimeError:
                raise RuntimeError(
                    "Failed to calibrate scores between cross-validation "
                    "folds, because no target PSMs could be found below "
                    "'test_fdr'. Try raising 'test_fdr'."
                )

        yield np.asarray(scores)


@typechecked
def _predict_with_ensemble(
    dataset: PsmDataset,
    models: Iterable[Model],
    max_workers: int,
):
    """
    Return the new scores for the dataset using ensemble of all trained models

    Parameters
    ----------
    max_workers
    dataset : Dict
        Contains all required info about the dataset to rescore
    models : list of Model
        The models for each dataset and whether it
        was reset or not.
    """
    scores = [[] for _ in range(len(models))]
    worker_count = min(max_workers, len(models))
    file_iterator = dataset.read_data_chunked(
        columns=dataset.columns, chunk_size=CHUNK_SIZE_ROWS_PREDICTION
    )
    for psms_chunk in file_iterator:
        linear_dataset = _create_linear_dataset(
            dataset, psms_chunk, enforce_checks=False
        )
        fold_scores = Parallel(n_jobs=worker_count, require="sharedmem")(
            delayed(model.predict)(dataset=linear_dataset) for model in models
        )
        [score.append(fs) for score, fs in zip(scores, fold_scores)]
        del fold_scores
    scores = [np.hstack(score) for score in scores]
    return np.mean(scores, axis=0)


def _fit_model(train_set, psms, model: Model, fold):
    """
    Fit the estimator using the training data.

    Parameters
    ----------
    train_set : PsmDataset
        A PsmDataset that specifies the training data
    model : a mokapot.model.Model
        A Classifier to train.
    fold : int
        The fold number. Only used for logging.

    Returns
    -------
    model : mokapot.model.Model
        The trained model.
    reset : bool
        Whether the models should be reset to their original parameters.
    """
    model.fold = fold + 1
    LOGGER.debug("")
    LOGGER.debug("=== Analyzing Fold %i ===", fold + 1)
    reset = False
    if isinstance(train_set, (str, Path)):
        train_set = pd.read_parquet(train_set)
    train_set = _create_linear_dataset(psms[0], train_set)
    try:
        model.fit(train_set)
    except BestFeatureIsBetterError as msg:
        LOGGER.debug(msg)
        if "Model performs worse after training." not in str(msg):
            LOGGER.info(f"Fold {fold + 1}: {msg}")
            raise

        if model.is_trained:
            reset = True
    except ModelIterationError as msg:
        # Q: should we handle this differently?
        #    the model getting better and then worse is different than
        #    it never getting better.
        LOGGER.debug(msg)
        if "Model performs worse after training." not in str(msg):
            LOGGER.info(f"Fold {fold + 1}: {msg}")
            raise

        if model.is_trained:
            reset = True

    return model, reset
