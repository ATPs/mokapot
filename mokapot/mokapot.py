"""
This is the command line interface for mokapot
"""

import datetime
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from . import __version__
from .brew import brew, resolve_parallelism
from .confidence import assign_confidence
from .config import Config
from .model import (
    PercolatorModel,
    load_models as load_models_file,
    save_percolator_models,
)
from .parsers.fasta import read_fasta
from .parsers.pin import read_pin
from .temp_workspace import TempWorkspace


def main(main_args=None):
    """The CLI entry point"""
    start = time.time()

    # Get command line arguments
    parser = Config().parser
    config = Config(parser, main_args=main_args)

    # Setup logging
    verbosity_dict = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }

    logging.basicConfig(
        format=("[{levelname}] {message}"),
        style="{",
        level=verbosity_dict[config.verbosity],
    )
    logging.captureWarnings(True)
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)

    # Suppress warning if asked for
    if config.suppress_warnings:
        warnings.filterwarnings("ignore")

    # Write header
    logging.info("mokapot version %s", str(__version__))
    logging.info("Written by William E. Fondrie (wfondrie@uw.edu) in the")
    logging.info(
        "Department of Genome Sciences at the University of Washington."
    )

    # Check config parameter validity
    if config.stream_confidence and config.peps_algorithm != "hist_nnls":
        raise ValueError(
            f"Streaming and PEPs algorithm `{config.peps_algorithm}` not "
            "compatible. Use `--peps_algorithm=hist_nnls` instead.`"
        )

    # Start analysis
    logging.info("Command issued:")
    logging.info("%s", " ".join(sys.argv))
    logging.info("")
    logging.info("Starting Analysis")
    logging.info("=================")

    np.random.seed(config.seed)

    if config.dest_dir is not None:
        config.dest_dir.mkdir(parents=True, exist_ok=True)

    temp_base_dir = config.temp
    if temp_base_dir is None:
        temp_base_dir = config.dest_dir or Path.cwd()
    workspace = TempWorkspace(temp_base_dir)
    logging.info("Temporary workspace: %s", workspace.path)

    try:
        # Parse
        datasets = read_pin(
            config.psm_files,
            max_workers=config.max_workers,
            temp_dir=workspace.path,
            auto_parquet=config.auto_parquet,
            auto_parquet_min_bytes=config.auto_parquet_min_bytes,
            auto_parquet_min_files=config.auto_parquet_min_files,
            auto_parquet_workers=config.auto_parquet_workers,
        )
        if config.aggregate or len(config.psm_files) == 1:
            prefixes = ["" for f in config.psm_files]
        else:
            prefixes = [f.stem for f in config.psm_files]

        # Parse FASTA, if required:
        if config.proteins is not None:
            logging.info("Protein-level confidence estimates enabled.")
            proteins = read_fasta(
                config.proteins,
                enzyme=config.enzyme,
                missed_cleavages=config.missed_cleavages,
                clip_nterm_methionine=config.clip_nterm_methionine,
                min_length=config.min_length,
                max_length=config.max_length,
                semi=config.semi,
                decoy_prefix=config.decoy_prefix,
            )
        else:
            proteins = None

        outer_workers, inner_workers = resolve_parallelism(
            max_workers=config.max_workers,
            folds=config.folds,
            model_n_jobs=config.model_n_jobs,
        )
        logging.debug(
            "Resolved parallelism: outer_workers=%d, model_n_jobs=%d",
            outer_workers,
            inner_workers,
        )

        # Define a model:
        model = None
        if config.load_models:
            model = []
            for model_file in config.load_models:
                model.extend(load_models_file(model_file))
            for i, loaded_model in enumerate(model):
                if loaded_model.fold is None:
                    loaded_model.fold = i

        if model is None:
            logging.debug("Loading Percolator model.")
            model = PercolatorModel(
                train_fdr=config.train_fdr,
                max_iter=config.max_iter,
                direction=config.direction,
                override=config.override,
                n_jobs=inner_workers,
                rng=config.seed,
            )

        # Fit the models:
        models, scores = brew(
            datasets,
            model=model,
            test_fdr=config.test_fdr,
            folds=config.folds,
            max_workers=outer_workers,
            subset_max_train=config.subset_max_train,
            ensemble=config.ensemble,
            rng=config.seed,
            temp_dir=workspace.path,
            memmap_threshold_psms=config.memmap_threshold_psms,
        )
        logging.info("")

        if config.file_root is not None:
            file_root = f"{config.file_root}."
        else:
            file_root = ""

        assign_confidence(
            datasets=datasets,
            max_workers=config.max_workers,
            scores_list=scores,
            eval_fdr=config.test_fdr,
            dest_dir=config.dest_dir,
            file_root=file_root,
            prefixes=prefixes,
            write_decoys=config.keep_decoys,
            deduplication=not config.skip_deduplication,
            do_rollup=not config.skip_rollup,
            proteins=proteins,
            peps_error=config.peps_error,
            peps_algorithm=config.peps_algorithm,
            qvalue_algorithm=config.qvalue_algorithm,
            sqlite_path=config.sqlite_db_path,
            stream_confidence=config.stream_confidence,
        )

        if config.save_models:
            logging.info("Saving models...")
            for i, trained_model in enumerate(models):
                out_file = Path(f"mokapot.model_fold-{i + 1}.pkl")

                if config.file_root is not None:
                    out_file = Path(config.file_root + "." + out_file.name)

                if config.dest_dir is not None:
                    out_file = config.dest_dir / out_file

                trained_model.save(out_file)

        if config.save_percolator_models:
            logging.info("Saving Percolator-style models...")
            out_file = Path("mokapot.model.weights.txt")

            if config.file_root is not None:
                out_file = Path(config.file_root + "." + out_file.name)

            if config.dest_dir is not None:
                out_file = config.dest_dir / out_file

            save_percolator_models(models, out_file)

        total_time = round(time.time() - start)
        total_time = str(datetime.timedelta(seconds=total_time))

        logging.info("")
        logging.info("=== DONE! ===")
        logging.info("mokapot analysis completed in %s", total_time)
    finally:
        workspace.cleanup()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except RuntimeError as _e:
        logging.error(f"[Error] {traceback.format_exc()}")
        sys.exit(250)  # input failure
    except ValueError as _e:
        logging.error(f"[Error] {traceback.format_exc()}")
        sys.exit(250)  # input failure
    except Exception as _e:
        logging.error(f"[Error] {traceback.format_exc()}")
        sys.exit(252)
